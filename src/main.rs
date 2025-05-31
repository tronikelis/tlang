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
                    let slice int[] = {1, 2, 3}
                    for let i int = 0; i < 3; i++ {
                        let foo int = 99999
                        let b int = slice[i]
                        __debug__
                    }
                    let foo int = 69
                    __debug__
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
