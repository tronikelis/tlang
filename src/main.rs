mod vm;

mod ast;
mod compiler;
mod ir;
mod lexer;
mod linker;

fn main() {
    let code = String::from(
        "
                type bool bool
                type int int
                type ptr ptr
                type string string
                type uint uint
                type uint8 uint8
                type void void
                type Type Type
                fn libc_write(fd int, slice uint8[]) int {}

                fn main() void {
                    let foo int = 25

                    let nice fn()void = fn()void {
                        foo = 28
                    }
                    nice()

                    for let i int = 0; i < 100; i++ {
                        foo = foo + 1
                    }
                    __debug__

                    libc_write(1, uint8[](\"foo\"))
                    
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
    println!("{linked:#?}");

    vm::Vm::new(linked, compiled.static_memory).run();
}
