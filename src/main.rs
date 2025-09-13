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

                fn main() void {
                    let foo int = 25
                    let foo_uint uint = *foo as *uint
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
