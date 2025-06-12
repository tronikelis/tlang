use std::collections::HashMap;

mod vm;

mod ast;
mod compiler;
mod lexer;
mod linker;

fn main() {
    let code = String::from(
        "
                fn len(slice Type) int {
                    // compiler builtin
                }
                fn append(slice Type, value Type) void {
                    // compiler builtin
                }
                fn syscall_write(fd int, slice uint8[]) void {
                    // compiler builtin
                }

                fn factorial(n int) int {
                    if n == 1 {
                        return n
                    }

                    return n*factorial(n-1)
                }

                fn main() void {
                    let SIZE int = 20

                    for let i int = 0; i < SIZE; i++ {

                        for let j int = SIZE-i-1; j > 0; j-- {
                            syscall_write(1, uint8[](\" \"))
                        }

                        for let j int = i; j > 0; j-- {
                            syscall_write(1, uint8[](\"#\"))
                        }

                        for let j int = i; j > 0; j-- {
                            syscall_write(1, uint8[](\"#\"))
                        }

                        syscall_write(1, uint8[](\"\\n\"))
                    }

                    let foo int = factorial(5)
                    __debug__
                }
            ",
    );

    let tokens = lexer::Lexer::new(&code).run().unwrap();
    println!("{tokens:#?}");
    let ast = ast::Ast::new(&tokens).unwrap();
    println!("{:#?}", ast);

    let mut functions = HashMap::<String, Vec<Vec<compiler::Instruction>>>::new();
    let mut static_memory = vm::StaticMemory::new();

    for v in &ast.functions {
        let compiled = compiler::FunctionCompiler::new(v, &mut static_memory)
            .compile()
            .unwrap();
        println!("{:#?}", compiled);
        functions.insert(v.identifier.clone(), compiled);
    }

    let instructions = linker::link(&functions).unwrap();

    println!("{:#?}", instructions.iter().enumerate().collect::<Vec<_>>());

    vm::Vm::new(instructions, static_memory).run();
}
