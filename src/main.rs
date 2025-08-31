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

                type a int

                impl a {
                    fn get(*self) a {
                        return *self
                    }
                }

                impl int {
                    fn get(*self) int {
                        return *self
                    }
                }

                fn main() void {
                    let foo int = 26

                    let ok a = 27
                    ok.get()

                    foo.get()

                    let ok fn() void = fn() void {
                        foo = 29
                        return
                    }
                }
            ",
    );

    let tokens = lexer::Lexer::new(&code).run().unwrap();
    println!("{tokens:#?}");
    let ast = ast::Ast::new(&tokens).unwrap();
    println!("{:#?}", ast);

    let ir = ir::Ir::new(&ast).unwrap();
    println!("{ir:#?}");
}
