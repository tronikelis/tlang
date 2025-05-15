use anyhow::{anyhow, Error, Result};

use crate::{ast, lexer, vm};

#[derive(Debug, Clone)]
pub enum Instruction {
    Real(vm::Instruction),
    JumpAndLink(String),
}

#[derive(Debug, Clone)]
enum VarStackItem {
    Anon(usize),
    Var(VarStackItemVar),
}

#[derive(Debug, Clone)]
struct VarStackItemVar {
    identifier: String,
    _type: ast::Type,
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

    fn pop(&mut self) -> Option<VarStackItem> {
        self.stack.pop()
    }

    fn size(&self) -> usize {
        self.stack.iter().fold(0, |acc, curr| match curr {
            VarStackItem::Var(var) => acc + var._type.size,
            VarStackItem::Anon(size) => acc + size,
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
                    offset += var._type.size;
                }
                VarStackItem::Anon(size) => offset += size,
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
    fn new() -> Self {
        Self {
            var_stack: VarStack::new(),
            instructions: Vec::new(),
        }
    }

    fn compile_function_call(
        &mut self,
        call: &ast::FunctionCall,
        expected_type: ast::Type,
    ) -> Result<usize> {
        if expected_type != call.function.return_type {
            return Err(anyhow!(
                "compile_function_call: expected_type does not match"
            ));
        }

        let argument_size = call.arguments.iter().enumerate().try_fold(0, |acc, curr| {
            let expected_type = &call
                .function
                .arguments
                .get(curr.0)
                .ok_or(anyhow!("compile_function_call: expected_argument"))?
                ._type;

            Ok::<usize, Error>(acc + self.compile_expression(curr.1, expected_type.clone())?)
        })?;

        let return_size = call.function.return_type.size;

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

        Ok(return_size)
    }

    fn compile_literal(
        &mut self,
        literal: &lexer::Literal,
        expected_type: ast::Type,
    ) -> Result<usize> {
        match literal {
            lexer::Literal::Int(int) => {
                if expected_type != ast::INT {
                    return Err(anyhow!("expected type dont match"));
                }

                self.instructions
                    .push(Instruction::Real(vm::Instruction::PushI(*int)));

                Ok(ast::INT.size)
            }
        }
    }

    fn compile_identifier(&mut self, identifier: &str, expected_type: ast::Type) -> Result<usize> {
        let (variable, offset) = self.var_stack.get_var(identifier).unwrap();

        if variable._type != expected_type {
            return Err(anyhow!("variable does not match expected type"));
        }

        self.instructions
            .push(Instruction::Real(vm::Instruction::Increment(
                variable._type.size,
            )));
        self.instructions
            .push(Instruction::Real(vm::Instruction::Copy(
                0,
                offset + variable._type.size, // we incremented by this above
                variable._type.size,
            )));

        Ok(variable._type.size)
    }

    fn compile_addition(
        &mut self,
        addition: &ast::Addition,
        expected_type: ast::Type,
    ) -> Result<usize> {
        self.compile_expression(&addition.left, expected_type.clone())?;
        self.compile_expression(&addition.right, expected_type.clone())?;
        self.var_stack.pop();
        self.var_stack.pop();

        if expected_type == ast::VOID {
            return Err(anyhow!("can't add void type"));
        }

        self.instructions
            .push(Instruction::Real(vm::Instruction::AddI(
                0,
                expected_type.size,
            )));
        self.instructions
            .push(Instruction::Real(vm::Instruction::Reset(
                expected_type.size,
            )));

        Ok(expected_type.size)
    }

    fn compile_expression(
        &mut self,
        expression: &ast::Expression,
        expected_type: ast::Type,
    ) -> Result<usize> {
        let size = match expression {
            ast::Expression::Literal(v) => self.compile_literal(v, expected_type),
            ast::Expression::Addition(v) => self.compile_addition(v, expected_type),
            ast::Expression::Identifier(v) => self.compile_identifier(v, expected_type),
            ast::Expression::FunctionCall(v) => self.compile_function_call(v, expected_type),
        }?;

        self.var_stack.push(VarStackItem::Anon(size));

        Ok(size)
    }

    fn compile_variable_declaration(&mut self, variable: &ast::VariableDeclaration) -> Result<()> {
        match variable._type._type {
            lexer::Type::Int => {
                self.compile_expression(&variable.expression, ast::INT)?;

                self.var_stack.pop();
                self.var_stack.push(VarStackItem::Var(VarStackItemVar {
                    identifier: variable.identifier.clone(),
                    _type: variable._type.clone(),
                }));

                Ok(())
            }
            lexer::Type::Void => Err(anyhow!("cant declare void variable")),
        }
    }

    pub fn compile_fn(function: &ast::Function) -> Result<Vec<Instruction>> {
        Self::new().compile(function)
    }

    fn compile(mut self, function: &ast::Function) -> Result<Vec<Instruction>> {
        for arg in &function.arguments {
            self.var_stack.push(VarStackItem::Var(VarStackItemVar {
                _type: arg._type.clone(),
                identifier: arg.identifier.clone(),
            }));
        }
        // return address
        self.var_stack.push(VarStackItem::Anon(size_of::<usize>()));

        let fn_arg_size = self.var_stack.size();

        for v in function
            .body
            .as_ref()
            .ok_or(anyhow!("compile_function: empty function body"))?
        {
            match v {
                ast::Node::VariableDeclaration(var) => self.compile_variable_declaration(&var)?,
                ast::Node::Return(exp) => {
                    // write expression into arguments
                    // reset local vars
                    // return

                    let size = self.compile_expression(exp, function.return_type.clone())?;
                    self.instructions
                        .push(Instruction::Real(vm::Instruction::Copy(
                            // -size because .size() gets you total stack size,
                            // while we want to index into first item
                            self.var_stack.size() - size,
                            0,
                            size,
                        )));
                    self.instructions
                        .push(Instruction::Real(vm::Instruction::Reset(
                            self.var_stack.size() - fn_arg_size,
                        )));

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
            println!("{:#?}", FunctionCompiler::compile_fn(v).unwrap());
        }
    }
}
