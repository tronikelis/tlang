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
                type ptr _
                type string _
                type uint _
                type uint8 _
                type void _
                type Type _

                fn libc_write(fd int, slice uint8[]) int {}
                fn len(slice Type) int {}
                fn append(slice Type, value Type) void {}

                fn dll_open(path string) ptr {}
                fn ffi_create(dll ptr, function string, return_param string, args string...) ptr {}
                fn ffi_call(ffi ptr, args ptr...) ptr {}

                fn write(fd int, data uint8[]) int {
                    let dll ptr = dll_open(\"libc.so.6\")
                    let ffi ptr = ffi_create(dll, \"write\", \"c_int\", \"c_int\", \"c_void\", \"c_int\")
                    return *(ffi_call(ffi, &fd, data as ptr, &len(data)) as *int)
                }

                fn main() void {
                    let foo int = 25
                    let foo_uint uint = foo as uint
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
