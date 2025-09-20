mod vm;

mod ast;
mod compiler;
mod ir;
mod lexer;
mod linker;

fn main() {
    let code = String::from(
        "
                type bool _
                type int _
                type int32 _
                type int16 _
                type ptr _
                type string _
                type uint _
                type uint8 _
                type uint16 _
                type uint32 _
                type void _
                type Type _

                fn libc_write(fd int, slice uint8[]) int {}
                fn len(slice Type) int {}
                fn append(slice Type, value Type) void {}

                fn dll_open(path string) ptr {}
                fn ffi_create(dll ptr, function string, return_param string, args string...) ptr {}
                fn ffi_call(ffi ptr, args ptr...) ptr {}

                fn write(fd int32, data uint8[]) int32 {
                    let dll ptr = dll_open(\"libc.so.6\")
                    let ffi ptr = ffi_create(dll, \"write\", \"i32\", \"i32\", \"pointer\", \"i32\")
                    return *(ffi_call(ffi, &fd, &(data as ptr), &(len(data) as int32)) as *int32)
                }

                fn getenv(name string) string {
                    let dll ptr = dll_open(\"libc.so.6\")
                    let ffi ptr = ffi_create(dll, \"getenv\" \"c_string\", \"c_string\")
                    return *(ffi_call(ffi, &name) as *string)
                }

                fn main() void {
                    let f int32 = write(1, \"\n\" as uint8[])
                    let n int32 = write(1, (getenv(\"HOME\") + \"\n\") as uint8[])

                    write(f, \"NICENICE\n\" as uint8[])
                    return
                }
            ",
    );

    let tokens = lexer::Lexer::new(&code).run().unwrap();
    println!("{tokens:#?}");
    let ast = ast::Ast::new(&tokens).unwrap();
    println!("{:#?}", ast);

    let ir = ir::Ir::new(&ast).unwrap();
    println!("{ir:#?}");

    let compiled = compiler::compile(ir).unwrap();
    let linked = linker::link(compiled.functions).unwrap();
    let linked_with_index = linked.iter().enumerate().collect::<Vec<_>>();
    println!("{linked_with_index:#?}");

    vm::Vm::new(linked, compiled.static_memory).run();
}
