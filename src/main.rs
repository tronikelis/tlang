mod vm;

mod ast;
mod compiler;
mod lexer;
mod linker;

fn main() {
    let code = String::from(
        "
                fn add(a int, b int) int {
                    return a + b
                }
                fn main() void {
                    let a int = 0
                    let b int = 1
                    let c int = a + b + 37 + 200
                    let d int = b + add(a, b)
                }
                fn add3(a int, b int, c int) int {
                    let abc int = a + b + c
                    return abc
                }
            ",
    );

    let instructions: Vec<vm::Instruction>;

    {
        let tokens = lexer::Lexer::new(&code).run().unwrap();
        let ast = ast::Ast::new(&tokens).unwrap();
        let functions = compiler::FunctionCompiler::compile_functions(&ast.functions).unwrap();
        instructions = linker::link(&functions).unwrap();
    }

    println!("{:#?}", instructions);

    vm::Vm::new(instructions).run();
}
