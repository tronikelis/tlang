use std::collections::HashMap;

mod vm;

mod ast;
mod compiler;
mod instructions;
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

                fn foobar(first_arg uint8, second_arg int) int {
                    return second_arg
                }

                // fn itoa(x int) string {
                //     let str uint8[] = {}
                //     let div int = 1
                //     for {
                //         x = x / div
                //         div = div * 10
                //         append(str, 48 + x % 10)
                //     }
                //
                //     return string(str)
                // }

                fn main() void {
                    let f int = 0
                    for let i int = 0; i < 1337; i++ {
                        if i == 10 {
                            f = i
                            break
                        }
                    }
                    __debug__
                }
            ",
    );

    let tokens = lexer::Lexer::new(&code).run().unwrap();
    println!("{tokens:#?}");
    let ast = ast::Ast::new(&tokens).unwrap();
    println!("{:#?}", ast);

    let mut functions = HashMap::<String, Vec<Vec<instructions::Instruction>>>::new();
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
