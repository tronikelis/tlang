use anyhow::Result;
use std::{env, fs, process};

use tlang::{ast, compiler, ir, lexer, linker, vm};

mod cgen;

#[derive(Debug)]
struct TmpDir {
    pub path: String,
}

impl TmpDir {
    fn new() -> Result<Self> {
        let mut buffer: [u8; 16] = [0; 16];
        rand::fill(&mut buffer);

        let hex =
            buffer
                .iter()
                .map(|v| format!("{:02X}", v))
                .fold(String::new(), |mut acc, curr| {
                    acc.push_str(&curr);
                    return acc;
                });

        let path = format!("/tmp/{hex}");

        fs::create_dir(&path)?;

        Ok(Self { path })
    }
}

impl Drop for TmpDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

const STD_FUNCTIONS: &str = r#"
    fn len(slice Type) int {}
    fn append(slice Type, value Type) void {}
    fn new(typ Type, args Type...) Type {}

    fn dll_open(path string) ptr {}
    fn ffi_create(dll ptr, function string, return_param string, args string...) ptr {}
    fn ffi_call(ffi ptr, args ptr...) ptr {}
"#;

fn main() {
    let argv = env::args_os();
    let mut argv_iter = argv.into_iter();
    // skip executable path
    argv_iter.next();

    let mut input_file: Option<String> = None;
    let mut out_file = "a.out".to_string();
    let mut lib_dir = "std".to_string();

    while let Some(value) = argv_iter.next() {
        match value.to_str().expect("Valid utf8") {
            "-o" | "--output" => {
                out_file = argv_iter
                    .next()
                    .expect("Specify output")
                    .to_str()
                    .expect("Valid utf8")
                    .to_string();
            }
            "-l" | "--library" => {
                lib_dir = argv_iter
                    .next()
                    .expect("Specify library path")
                    .to_str()
                    .expect("Valid utf8")
                    .to_string();
            }
            input => {
                input_file = Some(input.to_string());
            }
        }
    }

    let input_file = input_file.expect("Input file provided");

    let mut code = STD_FUNCTIONS.to_string();
    code.push_str(&fs::read_to_string(input_file).unwrap());

    println!("tokenizing");
    let tokens = lexer::Lexer::new(&code).run().unwrap();
    println!("generating ast");
    let ast = ast::Ast::new(&tokens).unwrap();
    println!("generating ir");
    let ir = ir::Ir::new(&ast).unwrap();

    println!("compiling");
    let compiled = compiler::compile(ir).unwrap();
    let static_memory_len = compiled.static_memory.data.len();
    println!("linking");
    let linked = linker::link(compiled.functions).unwrap();

    let tmp_dir = TmpDir::new().unwrap();

    println!("cgen");
    cgen::gen_instructions_c_file(
        vm::Instructions::new(linked),
        &format!("{}/instructions.c", tmp_dir.path),
        "instructions",
    )
    .unwrap();

    cgen::gen_static_memory_c_file(
        compiled.static_memory,
        &format!("{}/static_memory.c", tmp_dir.path),
        "static_memory",
    )
    .unwrap();

    cgen::gen_static_memory_len_c_file(
        static_memory_len,
        &format!("{}/static_memory_len.c", tmp_dir.path),
        "static_memory_len",
    )
    .unwrap();

    println!("building binary with clang");
    let clang = process::Command::new("clang")
        .current_dir(lib_dir)
        .args([
            &format!("{}/instructions.c", tmp_dir.path),
            &format!("{}/static_memory.c", tmp_dir.path),
            &format!("{}/static_memory_len.c", tmp_dir.path),
            "run.c",
            "vm.a",
            "-o",
            &out_file,
        ])
        .output()
        .unwrap();

    if !clang.status.success() {
        let stdout = String::from_utf8(clang.stdout).unwrap();
        let stderr = String::from_utf8(clang.stderr).unwrap();
        println!("{}", stdout);
        println!("{}", stderr);
    }
}
