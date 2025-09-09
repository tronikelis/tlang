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

                type User struct {
                    name string
                    email string
                }

                impl User {
                    fn set_name(*self, name string) *User {
                        self.name = name
                        return self
                    }

                    fn set_email(*self, email string) *User {
                        self.email = email
                        return self
                    }
                }

                impl int {
                    fn set(*self, value int) *int {
                        *self = value
                        return self
                    }
                }

                fn main() void {
                    let nice fn()void = fn()void {}
                    nice()

                    let u User = User{
                        name: \"nice\",
                        email: \"lolok\",
                    }
                    u.set_name(\"nicenice\").set_email(\"okokok\")

                    let i int = 0
                    i.set(22).set(100).set(127).set(1338)

                    let i1 int = i
                    let foo User = u
                    __debug__
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
