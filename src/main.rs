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
                    let a int = 57
                    if a == 57 {
                        a = 20
                        __debug__
                    }
                }
            ",
    );

    let tokens = lexer::Lexer::new(&code).run().unwrap();
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
