use anyhow::{anyhow, Result};
use std::{cmp, collections::HashMap};

use crate::{ast, lexer, vm};

#[derive(Debug, Clone)]
pub enum Instruction {
    Real(vm::Instruction),
    JumpAndLink(String),
}

#[derive(Debug, Clone)]
enum VarStackItem {
    Empty(usize),
    Var(VarStackItemVar),
}

#[derive(Debug, Clone)]
struct VarStackItemVar {
    identifier: String,
    _type: lexer::Type,
    size: usize,
}

struct VarStack {
    stack: Vec<VarStackItem>,
}

impl VarStack {
    fn new() -> Self {
        Self { stack: Vec::new() }
    }

    fn push(&mut self, item: VarStackItem) {
        self.stack.push(item);
    }

    fn size(self) -> usize {
        self.stack.iter().fold(0, |acc, curr| match curr {
            VarStackItem::Var(var) => acc + var.size,
            VarStackItem::Empty(size) => acc + size,
        })
    }

    fn get_var(&self, identifier: &str) -> Option<(VarStackItemVar, usize)> {
        let mut offset = 0;

        for item in self.stack.iter().rev() {
            match item {
                VarStackItem::Var(var) => {
                    if var.identifier == identifier {
                        return Some((var.clone(), offset));
                    }
                    offset += var.size;
                }
                VarStackItem::Empty(size) => offset += size,
            };
        }

        None
    }
}

pub struct FunctionCompiler {
    var_stack: VarStack,
    instructions: Vec<Instruction>,
}

impl FunctionCompiler {
    pub fn new() -> Self {
        Self {
            var_stack: VarStack::new(),
            instructions: Vec::new(),
        }
    }

    fn compile_function_call(
        &mut self,
        call: &ast::FunctionCall,
        expected_type: lexer::Type,
    ) -> usize {
        if expected_type != call.function.return_type {
            panic!("compile_function_call: expected_type does not match")
        }

        let argument_size = call.arguments.iter().enumerate().fold(0, |acc, curr| {
            let expected_type = &call
                .function
                .arguments
                .get(curr.0)
                .ok_or(anyhow!("compile_function_call: expected argument"))
                .unwrap()
                ._type;

            acc + self.compile_expression(curr.1, expected_type.clone())
        });

        let return_size = match call.function.return_type {
            lexer::Type::Int => size_of::<isize>(),
            lexer::Type::Void => 0,
        };

        let reset_size: usize;
        if argument_size < return_size {
            // the whole argument section will be used for return value
            // do not need to reset
            reset_size = 0;
            self.instructions
                .push(Instruction::Real(vm::Instruction::Increment(
                    return_size - argument_size,
                )));
        } else {
            // reset the argument section to the return size
            reset_size = argument_size - return_size;
        }

        self.instructions
            .push(Instruction::JumpAndLink(call.function.identifier.clone()));

        if reset_size != 0 {
            self.instructions
                .push(Instruction::Real(vm::Instruction::Reset(reset_size)));
        }

        return_size
    }

    fn compile_literal(&mut self, literal: &lexer::Literal, expected_type: lexer::Type) -> usize {
        match literal {
            lexer::Literal::Int(int) => {
                if expected_type != lexer::Type::Int {
                    panic!("expected type dont match");
                }

                self.instructions
                    .push(Instruction::Real(vm::Instruction::PushI(*int)));

                size_of::<isize>()
            }
        }
    }

    fn compile_identifier(&mut self, identifier: &str, expected_type: lexer::Type) -> usize {
        let (variable, offset) = self.var_stack.get_var(identifier).unwrap();

        if variable._type != expected_type {
            panic!("variable does not match expected type");
        }

        self.instructions
            .push(Instruction::Real(vm::Instruction::Increment(variable.size)));
        self.instructions
            .push(Instruction::Real(vm::Instruction::Copy(
                0,
                offset + variable.size, // we incremented by this above
                variable.size,
            )));

        variable.size
    }

    fn compile_addition(&mut self, addition: &ast::Addition, expected_type: lexer::Type) -> usize {
        self.compile_expression(&addition.left, expected_type.clone());
        self.compile_expression(&addition.right, expected_type.clone());

        let size = match expected_type {
            lexer::Type::Int => size_of::<isize>(),
            lexer::Type::Void => panic!("can't add void type"),
        };

        self.instructions
            .push(Instruction::Real(vm::Instruction::AddI(0, size)));
        self.instructions
            .push(Instruction::Real(vm::Instruction::Reset(size)));

        size
    }

    fn compile_expression(
        &mut self,
        expression: &ast::Expression,
        expected_type: lexer::Type,
    ) -> usize {
        match expression {
            ast::Expression::Literal(v) => self.compile_literal(v, expected_type),
            ast::Expression::Addition(v) => self.compile_addition(v, expected_type),
            ast::Expression::Identifier(v) => self.compile_identifier(&v, expected_type),
            ast::Expression::FunctionCall(v) => self.compile_function_call(v, expected_type),
        }
    }

    fn compile_variable_declaration(&mut self, variable: &ast::VariableDeclaration) {
        match variable._type {
            lexer::Type::Int => {
                self.compile_expression(&variable.expression, lexer::Type::Int);
                self.var_stack.push(VarStackItem::Var(VarStackItemVar {
                    identifier: variable.identifier.clone(),
                    size: variable.size,
                    _type: variable._type.clone(),
                }));
            }
            lexer::Type::Void => {
                panic!("cant declare void variable");
            }
        };
    }

    pub fn compile(mut self, function: &ast::Function) -> Result<Vec<Instruction>> {
        for arg in &function.arguments {
            self.var_stack.push(VarStackItem::Var(VarStackItemVar {
                _type: arg._type.clone(),
                identifier: arg.identifier.clone(),
                size: 8,
            }));
        }

        for v in function
            .body
            .as_ref()
            .ok_or(anyhow!("compile_function: empty function body"))?
        {
            match v {
                ast::Node::VariableDeclaration(var) => self.compile_variable_declaration(&var),
                ast::Node::Return(exp) => {
                    // write expression into arguments
                    // reset local vars
                    // return

                    self.instructions
                        .push(Instruction::Real(vm::Instruction::Return));
                }
            };
        }

        Ok(self.instructions)
    }
}

#[cfg(test)]
mod tests {
    use crate::ast::Ast;

    use super::*;

    #[test]
    fn simple() {
        let code = String::from(
            "
                fn add(a int, b int) int {
                    return a + b
                }
                fn main() void {
                    let a int = 0
                    let b int = 1
                    let c int = a + b + 37 + 200
                    let d int = b + add(a, b)
                }
            ",
        );

        let tokens = lexer::Lexer::new(&code).run().unwrap();
        let ast = Ast::new(&tokens).unwrap();

        for v in &ast.functions {
            println!("{}", v.identifier);
            println!("{:#?}", FunctionCompiler::new().compile(v).unwrap());
        }
    }
}
