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
                    let a int = 0
                    if a == 0 {
                        a = 30
                    }
                }
            ",
    );

    let tokens = lexer::Lexer::new(&code).run().unwrap();
    let ast = ast::Ast::new(&tokens).unwrap();
    println!("{:#?}", ast);

    let mut functions = HashMap::<String, Vec<Vec<compiler::Instruction>>>::new();

    for v in &ast.functions {
        functions.insert(
            v.identifier.clone(),
            compiler::FunctionCompiler::new(v).compile().unwrap(),
        );
    }

    let instructions = linker::link(&functions).unwrap();

    println!("{:#?}", instructions);

    vm::Vm::new(instructions).run();
}
