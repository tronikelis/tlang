use std::collections::HashMap;

mod vm;

mod ast;
mod compiler;
mod lexer;
mod linker;

fn main() {
    let code = String::from(
        "
                fn main() void {
                    let slice2 int[][] = {{1,2,3}, {1,2,3}}
                    for let i int = 0; i < 2; i++ {
                        for let j int = 0; j < 3; j++ {
                            let foo int = slice2[i][j]
                            __debug__
                        }
                    }
                }
            ",
    );

    let tokens = lexer::Lexer::new(&code).run().unwrap();
    println!("{tokens:#?}");
    let ast = ast::Ast::new(&tokens).unwrap();
    println!("{:#?}", ast);

    let mut functions = HashMap::<String, Vec<Vec<compiler::Instruction>>>::new();

    for v in &ast.functions {
        let compiled = compiler::FunctionCompiler::new(v).compile().unwrap();
        println!("{:#?}", compiled);
        functions.insert(v.identifier.clone(), compiled);
    }

    let instructions = linker::link(&functions).unwrap();

    println!("{:#?}", instructions.iter().enumerate().collect::<Vec<_>>());

    vm::Vm::new(instructions).run();
}
