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

                fn main() void {
                    let foo int = -5-10 + 15
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
