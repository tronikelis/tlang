use anyhow::{anyhow, Result};

use crate::{ast, instructions, lexer, vm};

pub struct FunctionCompiler<'a, 'b> {
    instructions: instructions::Instructions,
    function: &'a ast::Function,
    static_memory: &'b mut vm::StaticMemory,
}

impl<'a, 'b> FunctionCompiler<'a, 'b> {
    pub fn new(function: &'a ast::Function, static_memory: &'b mut vm::StaticMemory) -> Self {
        Self {
            static_memory,
            function,
            instructions: instructions::Instructions::new(),
        }
    }

    fn compile_function_builtin_len(&mut self, call: &ast::FunctionCall) -> Result<ast::Type> {
        let slice_arg = call
            .arguments
            .get(0)
            .ok_or(anyhow!("len: expected first argument"))?;

        // cleanup align here?
        let slice_exp = self.compile_expression(slice_arg)?;

        let ast::TypeType::Slice(_) = &slice_exp._type else {
            return Err(anyhow!("len: expected slice as the argument"));
        };

        self.instructions.instr_slice_len();

        Ok(ast::INT)
    }

    // append(slice Type, value Type) void {}
    fn compile_function_builtin_append(&mut self, call: &ast::FunctionCall) -> Result<ast::Type> {
        let slice_arg = call
            .arguments
            .get(0)
            .ok_or(anyhow!("append: expected first argument"))?;
        let value_arg = call
            .arguments
            .get(1)
            .ok_or(anyhow!("append: expected second argument"))?;

        // cleanup align here?
        let slice_exp = self.compile_expression(slice_arg)?;
        let value_exp = self.compile_expression(value_arg)?;

        let ast::TypeType::Slice(slice_item) = &slice_exp._type else {
            return Err(anyhow!("append: provide a slice as the first argument"));
        };

        if !slice_item.can_assign(&value_exp) {
            return Err(anyhow!("append: value type does not match slice type"));
        }

        self.instructions.instr_slice_append(value_exp.size);

        Ok(ast::VOID)
    }

    fn check_function_call_argument_count(call: &ast::FunctionCall) -> Result<()> {
        // todo: refactor this arguments check somehow
        if let Some(last) = call.function.arguments.last() {
            if let ast::TypeType::Variadic(_type) = &last._type._type {
                // -1 because variadic can be empty
                if call.arguments.len() < call.function.arguments.len() - 1 {
                    return Err(anyhow!(
                        "compile_function_call: variadic argument count mismatch"
                    ));
                }
            } else {
                if call.arguments.len() != call.function.arguments.len() {
                    return Err(anyhow!("compile_function_call: argument count mismatch"));
                }
            }
        } else {
            if call.arguments.len() != call.function.arguments.len() {
                return Err(anyhow!("compile_function_call: argument count mismatch"));
            }
        }

        Ok(())
    }

    fn compile_function_call(&mut self, call: &ast::FunctionCall) -> Result<ast::Type> {
        Self::check_function_call_argument_count(call)?;

        match call.function.identifier.as_str() {
            "append" => return self.compile_function_builtin_append(call),
            "len" => return self.compile_function_builtin_len(call),
            _ => {}
        }

        self.instructions
            .push_alignment(call.function.return_type.size);

        let argument_size = {
            self.instructions.push_stack_frame();
            for (i, expected_type) in call.function.arguments.iter().enumerate() {
                let arg = call
                    .arguments
                    .get(i)
                    .ok_or(anyhow!("compile_function_call: expected_argument"))?;

                let _type = self.compile_expression(arg)?;
                if expected_type._type != _type {
                    return Err(anyhow!("compile_function_call: mismatch type"));
                }
            }
            self.instructions.pop_stack_frame_size()
        };

        match call.function.identifier.as_str() {
            "syscall0" => {
                self.instructions.instr_syscall0();
                return Ok(ast::UINT);
            }
            "syscall1" => {
                self.instructions.instr_syscall1();
                return Ok(ast::UINT);
            }
            "syscall2" => {
                self.instructions.instr_syscall2();
                return Ok(ast::UINT);
            }
            "syscall3" => {
                self.instructions.instr_syscall3();
                return Ok(ast::UINT);
            }
            "syscall4" => {
                self.instructions.instr_syscall4();
                return Ok(ast::UINT);
            }
            "syscall5" => {
                self.instructions.instr_syscall5();
                return Ok(ast::UINT);
            }
            "syscall6" => {
                self.instructions.instr_syscall6();
                return Ok(ast::UINT);
            }
            _ => {}
        }

        let return_size = call.function.return_type.size;

        let reset_size: usize;
        if argument_size < return_size {
            // the whole argument section will be used for return value
            // do not need to reset
            reset_size = 0;
            self.instructions
                .instr_increment(return_size - argument_size);
        } else {
            // reset the argument section to the return size
            reset_size = argument_size - return_size;
        }

        self.instructions
            .instr_jump_and_link(call.function.identifier.clone());
        self.instructions.instr_reset(reset_size);

        Ok(call.function.return_type.clone())
    }

    fn compile_literal(&mut self, literal: &ast::Literal) -> Result<ast::Type> {
        match &literal.literal {
            lexer::Literal::Int(int) => match &literal._type {
                &ast::UINT8 => {
                    self.instructions.instr_push_u8(*int)?;
                }
                &ast::INT => {
                    self.instructions.instr_push_i(*int)?;
                }
                _type => return Err(anyhow!("can't cast int to {_type:#?}")),
            },
            lexer::Literal::Bool(bool) => match &literal._type {
                &ast::BOOL => {
                    self.instructions.instr_push_i({
                        if *bool {
                            1
                        } else {
                            0
                        }
                    })?;
                }
                _type => return Err(anyhow!("can't cast bool to {_type:#?}")),
            },
            lexer::Literal::String(string) => {
                let index = self.static_memory.push_string_slice(&string);
                self.instructions.instr_push_static(index, ast::SLICE_SIZE);
            }
        }

        Ok(literal._type.clone())
    }

    fn compile_variable(&mut self, variable: &ast::Variable) -> Result<ast::Type> {
        self.instructions.push_alignment(variable._type.size);

        let offset = self
            .instructions
            .var_get_offset(&variable.identifier)
            .ok_or(anyhow!("compile_identifier: unknown identifier"))?;

        self.instructions.instr_increment(variable._type.size);
        self.instructions
            .instr_copy(0, offset + variable._type.size, variable._type.size);

        Ok(variable._type.clone())
    }

    fn compile_arithmetic(&mut self, arithmetic: &ast::Arithmetic) -> Result<ast::Type> {
        let a = self.compile_expression(&arithmetic.left)?;
        let b = self.compile_expression(&arithmetic.right)?;

        if a != b {
            return Err(anyhow!("can't add different types"));
        }
        if a == ast::VOID {
            return Err(anyhow!("can't add void type"));
        }

        match arithmetic._type {
            ast::ArithmeticType::Minus => {
                if let ast::INT = a {
                    self.instructions.instr_minus_int();
                    self.instructions.instr_add_i();
                } else {
                    return Err(anyhow!("can only minus int"));
                }
            }
            ast::ArithmeticType::Plus => {
                if let ast::INT = a {
                    self.instructions.instr_add_i();
                } else if a == ast::STRING {
                    self.instructions.instr_add_string();
                } else {
                    return Err(anyhow!("can only plus int and string"));
                }
            }
            ast::ArithmeticType::Multiply => {
                if let ast::INT = a {
                    self.instructions.instr_multiply_i();
                } else {
                    return Err(anyhow!("can only multiply int"));
                }
            }
            ast::ArithmeticType::Divide => {
                if let ast::INT = a {
                    self.instructions.instr_divide_i();
                } else {
                    return Err(anyhow!("can only divide int"));
                }
            }
            ast::ArithmeticType::Modulo => {
                if let ast::INT = a {
                    self.instructions.instr_modulo_i();
                } else {
                    return Err(anyhow!("can only modulo int"));
                }
            }
        }

        Ok(a)
    }

    fn compile_compare(&mut self, compare: &ast::Compare) -> Result<ast::Type> {
        let a: ast::Type;
        let b: ast::Type;

        match compare.compare_type {
            // last item on the stack is smaller
            ast::CompareType::Gt => {
                b = self.compile_expression(&compare.left)?;
                a = self.compile_expression(&compare.right)?;
            }
            // last item on the stack is bigger
            ast::CompareType::Lt => {
                b = self.compile_expression(&compare.right)?;
                a = self.compile_expression(&compare.left)?;
            }
            // dont matter
            ast::CompareType::Equals | ast::CompareType::NotEquals => {
                a = self.compile_expression(&compare.right)?;
                b = self.compile_expression(&compare.left)?;
            }
        };

        if a._type != b._type {
            return Err(anyhow!("can't compare different types"));
        }

        match a {
            ast::BOOL | ast::INT => {}
            _type => return Err(anyhow!("can only compare int/bool")),
        }

        match compare.compare_type {
            ast::CompareType::Gt | ast::CompareType::Lt => {
                // a = -a
                self.instructions.instr_minus_int();

                // a + b
                self.instructions.instr_add_i();

                // >0:1 <0:0
                self.instructions.instr_to_bool();
            }
            ast::CompareType::Equals => {
                self.instructions.instr_compare_i();
            }
            ast::CompareType::NotEquals => {
                self.instructions.instr_compare_i();
                self.instructions.instr_negate_bool();
            }
        }

        Ok(ast::BOOL)
    }

    fn compile_infix(&mut self, infix: &ast::Infix) -> Result<ast::Type> {
        let exp = self.compile_expression(&infix.expression)?;
        match infix._type {
            ast::InfixType::Plus => {}
            ast::InfixType::Minus => {
                self.instructions.instr_minus_int();
            }
        }
        Ok(exp)
    }

    fn compile_list(&mut self, list: &[ast::Expression]) -> Result<ast::Type> {
        self.instructions.instr_push_slice();

        let mut curr_exp: Option<ast::Type> = None;

        for v in list {
            self.instructions.push_stack_frame();

            self.instructions.instr_increment(ast::SLICE_SIZE);
            self.instructions
                .instr_copy(0, ast::SLICE_SIZE, ast::SLICE_SIZE);

            let exp = self.compile_expression(v)?;
            if let Some(curr_exp) = curr_exp {
                if curr_exp != exp {
                    return Err(anyhow!("type mismatch"));
                }
            }
            curr_exp = Some(exp.clone());

            self.instructions.instr_slice_append(exp.size);

            self.instructions.pop_stack_frame();
        }

        Ok(ast::Type {
            size: ast::SLICE_SIZE,
            _type: curr_exp
                .map(|v| ast::TypeType::Slice(Box::new(v)))
                .unwrap_or(ast::TypeType::Slice(Box::new(ast::VOID))),
        })
    }

    fn compile_expression_index(&mut self, index: &ast::Index) -> Result<ast::Type> {
        let exp_var = self.compile_expression(&index.var)?;

        let ast::TypeType::Slice(expected_type) = exp_var._type else {
            return Err(anyhow!("can't index this type"));
        };

        let exp_index = self.compile_expression(&index.expression)?;
        if exp_index != ast::INT {
            return Err(anyhow!("cant index with {exp_index:#?}"));
        }

        self.instructions.instr_slice_index_get(expected_type.size);

        Ok(*expected_type)
    }

    fn compile_andor(&mut self, andor: &ast::AndOr) -> Result<ast::Type> {
        self.instructions.stack_instructions.jump();

        let left = self.compile_expression(&andor.left)?;
        if left != ast::BOOL {
            return Err(anyhow!("compile_andor: expected bool expression"));
        }

        match andor._type {
            ast::AndOrType::Or => {
                self.instructions.stack_instructions.back_if_true(1);

                let right = self.compile_expression(&andor.right)?;
                if right != ast::BOOL {
                    return Err(anyhow!("compile_andor: expected bool expression"));
                }

                self.instructions.instr_or();
            }
            ast::AndOrType::And => {
                self.instructions.stack_instructions.back_if_false(1);

                let right = self.compile_expression(&andor.right)?;
                if right != ast::BOOL {
                    return Err(anyhow!("compile_andor: expected bool expression"));
                }

                self.instructions.instr_and();
            }
        }

        self.instructions.stack_instructions.back(1);
        self.instructions.stack_instructions.pop_index();

        Ok(ast::BOOL)
    }

    fn compile_type_cast(&mut self, type_cast: &ast::TypeCast) -> Result<ast::Type> {
        self.instructions.push_alignment(type_cast._type.size);

        let target = self.compile_expression(&type_cast.expression)?;

        match target {
            ast::INT => match &type_cast._type {
                &ast::UINT8 => {
                    self.instructions.instr_cast_int_uint8();
                    Ok(ast::UINT8)
                }
                &ast::UINT => {
                    self.instructions.instr_cast_int_uint();
                    Ok(ast::UINT)
                }
                _type => Err(anyhow!("compile_type_cast: cant cast into {_type:#?}")),
            },
            ast::UINT8 => match &type_cast._type {
                &ast::INT => {
                    self.instructions.instr_cast_uint8_int();
                    Ok(ast::INT)
                }
                _type => Err(anyhow!("compile_type_cast: cant cast into {_type:#?}")),
            },
            ast::PTR => match &type_cast._type {
                &ast::UINT => Ok(ast::UINT),
                _type => Err(anyhow!("compile_type_cast: cant cast into {_type:#?}")),
            },
            // we have to do this because other types are runtime created
            target => {
                if target == ast::STRING {
                    if &type_cast._type == &*ast::SLICE_UINT8 {
                        Ok(ast::SLICE_UINT8.clone())
                    } else {
                        Err(anyhow!(
                            "compile_type_cast: cant cast into {:#?}",
                            &type_cast._type
                        ))
                    }
                } else if target == *ast::SLICE_UINT8 {
                    if type_cast._type == ast::STRING {
                        Ok(ast::STRING.clone())
                    } else if type_cast._type == ast::PTR {
                        self.instructions.instr_cast_slice_ptr();
                        Ok(ast::PTR)
                    } else {
                        Err(anyhow!(
                            "compile_type_cast: cant cast into {:#?}",
                            &type_cast._type
                        ))
                    }
                } else {
                    Err(anyhow!("compile_type_cast: cant cast {target:#?}"))
                }
            }
        }
    }

    fn compile_negate(&mut self, negate: &ast::Expression) -> Result<ast::Type> {
        let exp_bool = self.compile_expression(negate)?;
        if exp_bool != ast::BOOL {
            return Err(anyhow!("can only negate bools"));
        }

        self.instructions.instr_negate_bool();

        Ok(ast::BOOL)
    }

    fn compile_spread(&mut self, expression: &ast::Expression) -> Result<ast::Type> {
        let exp = self.compile_expression(expression)?;

        let ast::TypeType::Slice(slice_item) = exp._type else {
            return Err(anyhow!("compile_spread: can only spread slice types"));
        };

        Ok(ast::Type {
            size: ast::SLICE_SIZE,
            _type: ast::TypeType::Variadic(slice_item),
        })
    }

    fn compile_expression(&mut self, expression: &ast::Expression) -> Result<ast::Type> {
        let old_stack_size = self.instructions.stack_total_size();

        let exp = match expression {
            ast::Expression::AndOr(v) => self.compile_andor(v),
            ast::Expression::Literal(v) => self.compile_literal(v),
            ast::Expression::Arithmetic(v) => self.compile_arithmetic(v),
            ast::Expression::Variable(v) => self.compile_variable(v),
            ast::Expression::FunctionCall(v) => self.compile_function_call(v),
            ast::Expression::Compare(v) => self.compile_compare(v),
            ast::Expression::Infix(v) => self.compile_infix(v),
            ast::Expression::List(v) => self.compile_list(v),
            ast::Expression::Index(v) => self.compile_expression_index(v),
            ast::Expression::TypeCast(v) => self.compile_type_cast(v),
            ast::Expression::Negate(v) => self.compile_negate(v),
            ast::Expression::Spread(v) => self.compile_spread(v),
        }?;

        if exp.size != 0 {
            let new_stack_size = self.instructions.stack_total_size();
            let delta_stack_size = new_stack_size - old_stack_size;

            if old_stack_size % exp.size == 0 && delta_stack_size > exp.size {
                self.instructions
                    .instr_shift(exp.size, delta_stack_size - exp.size - 1);
            }
        }

        Ok(exp)
    }

    fn compile_variable_declaration(
        &mut self,
        declaration: &ast::VariableDeclaration,
    ) -> Result<()> {
        let exp = self.compile_expression(&declaration.expression)?;
        if exp == ast::VOID {
            return Err(anyhow!("can't declare void variable"));
        }

        if !declaration.variable._type.can_assign(&exp) {
            return Err(anyhow!("type mismatch"));
        }

        self.instructions
            .var_mark(declaration.variable.identifier.clone());

        Ok(())
    }

    fn compile_variable_assignment(&mut self, assignment: &ast::VariableAssignment) -> Result<()> {
        self.instructions.push_stack_frame();

        match &assignment.var {
            ast::Expression::Variable(var) => {
                let exp = self.compile_expression(&assignment.expression)?;

                let offset = self
                    .instructions
                    .var_get_offset(&var.identifier)
                    .ok_or(anyhow!("compile_variable_assignment: var not found"))?;

                if var._type != exp {
                    return Err(anyhow!("variable assignment type mismatch"));
                }

                self.instructions.instr_copy(offset, 0, exp.size);
            }
            ast::Expression::Index(index) => {
                let slice = self.compile_expression(&index.var)?;
                let ast::TypeType::Slice(slice_item) = &slice._type else {
                    return Err(anyhow!("can only index slices"));
                };

                let item_index = self.compile_expression(&index.expression)?;
                if item_index != ast::INT {
                    return Err(anyhow!("can only index with int type"));
                }

                let item = self.compile_expression(&assignment.expression)?;
                if !slice_item.can_assign(&item) {
                    return Err(anyhow!("slice index set type mismatch"));
                }

                self.instructions.instr_slice_index_set(item.size);
            }
            node => return Err(anyhow!("can't assign {node:#?}")),
        }

        self.instructions.pop_stack_frame();

        Ok(())
    }

    fn compile_if_block(&mut self, expression: &ast::Expression, body: &[ast::Node]) -> Result<()> {
        let exp = self.compile_expression(expression)?;
        if exp != ast::BOOL {
            return Err(anyhow!("compile_if_block: expected bool expression"));
        }

        self.instructions.stack_instructions.jump_if_true();

        self.compile_body(body)?;
        self.instructions.stack_instructions.back(2);
        self.instructions.stack_instructions.pop_index();

        Ok(())
    }

    fn compile_if(&mut self, _if: &ast::If) -> Result<()> {
        self.instructions.push_stack_frame();

        self.instructions.stack_instructions.jump();

        self.compile_if_block(&_if.expression, &_if.body)?;

        for v in &_if.elseif {
            self.compile_if_block(&v.expression, &v.body)?;
        }

        if let Some(v) = &_if._else {
            self.compile_body(&v.body)?;
        }

        self.instructions.stack_instructions.back(1);
        self.instructions.stack_instructions.pop_index();

        self.instructions.pop_stack_frame();

        Ok(())
    }

    fn compile_for(&mut self, _for: &ast::For) -> Result<()> {
        self.instructions.push_stack_frame();
        if let Some(v) = &_for.initializer {
            self.compile_node(v)?;
        }

        self.instructions
            .stack_instructions
            .label_new("for_break".to_string());

        self.instructions.stack_instructions.jump();
        self.instructions.stack_instructions.jump();

        self.instructions
            .stack_instructions
            .label_new("for_continue".to_string());

        self.instructions.stack_instructions.jump();

        self.instructions.push_stack_frame();
        self.instructions.var_mark_label();

        let bool_size = {
            if let Some(v) = &_for.expression {
                self.instructions.push_stack_frame();
                let exp = self.compile_expression(v)?;
                if exp != ast::BOOL {
                    return Err(anyhow!("compile_for: expected expression to return bool"));
                }
                let size = self.instructions.pop_stack_frame_size();

                self.instructions.instr_negate_bool();
                self.instructions.stack_instructions.back_if_true(2);

                size
            } else {
                0
            }
        };

        self.compile_body(&_for.body)?;
        self.instructions.pop_stack_frame();

        self.instructions.stack_instructions.back(1);
        self.instructions.stack_instructions.pop_index();

        // continue will jump here
        if let Some(v) = &_for.after_each {
            self.compile_node(v)?;
        }

        self.instructions.stack_instructions.again();
        self.instructions.stack_instructions.pop_index();

        // normal loop exit will jump here
        self.instructions
            .instr_reset_dangerous_not_synced(bool_size);

        self.instructions.stack_instructions.back(1);
        self.instructions.stack_instructions.pop_index();

        // break will jump here
        //
        // have to go through all this jumping,
        // because continue & break resets their stack,
        // so we have to skip normal loop exit cleanup and pop
        // only the initializer
        self.instructions.pop_stack_frame();

        self.instructions.stack_instructions.label_pop();
        self.instructions.stack_instructions.label_pop();

        Ok(())
    }

    fn compile_for_break(&mut self) -> Result<()> {
        self.instructions.var_reset_label();

        self.instructions
            .stack_instructions
            .label_jump("for_break")?;

        Ok(())
    }

    fn compile_for_continue(&mut self) -> Result<()> {
        self.instructions.var_reset_label();

        self.instructions
            .stack_instructions
            .label_jump("for_continue")?;

        Ok(())
    }

    fn compile_node(&mut self, node: &ast::Node) -> Result<()> {
        match node {
            ast::Node::VariableDeclaration(var) => self.compile_variable_declaration(var)?,
            ast::Node::Return(exp) => self.compile_return(exp.as_ref())?,
            ast::Node::Expression(exp) => {
                self.instructions.push_stack_frame();
                self.compile_expression(exp)?;
                self.instructions.pop_stack_frame();
            }
            ast::Node::VariableAssignment(assignment) => {
                self.compile_variable_assignment(assignment)?;
            }
            ast::Node::If(v) => {
                self.compile_if(v)?;
            }
            ast::Node::Debug => {
                self.instructions.instr_debug();
            }
            ast::Node::For(v) => self.compile_for(v)?,
            ast::Node::Break => {
                self.compile_for_break()?;
            }
            ast::Node::Continue => {
                self.compile_for_continue()?;
            }
        };

        Ok(())
    }

    fn compile_body(&mut self, body: &[ast::Node]) -> Result<()> {
        self.instructions.push_stack_frame();

        for node in body {
            self.compile_node(node)?;
        }

        self.instructions.pop_stack_frame();

        Ok(())
    }

    fn compile_return(&mut self, exp: Option<&ast::Expression>) -> Result<()> {
        if let Some(exp) = exp {
            let exp = self.compile_expression(exp)?;
            if exp != self.function.return_type {
                return Err(anyhow!("incorrect type"));
            }

            self.instructions.instr_copy(
                // -size because .total_size() gets you total stack size,
                // while we want to index into first item
                self.instructions.stack_total_size() - exp.size,
                0,
                exp.size,
            );
        }

        self.instructions.init_function_epilogue(self.function);

        Ok(())
    }

    pub fn compile(mut self) -> Result<Vec<Vec<instructions::Instruction>>> {
        self.instructions.init_function_prologue(self.function);

        self.compile_body(
            self.function
                .body
                .as_ref()
                .ok_or(anyhow!("compile: function body empty"))?,
        )?;

        self.compile_return(None)?;

        Ok(self.instructions.get_instructions())
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use crate::ast::Ast;
//
//     #[test]
//     fn simple() {
//         let code = String::from(
//             "
//                 fn add(a int, b int) int {
//                     return a + b
//                 }
//                 fn main() void {
//                     let a int = 0
//                     let b int = 1
//                     let c int = a + b + 37 + 200
//                     let d int = b + add(a, b)
//                 }
//                 fn add3(a int, b int, c int) int {
//                     let abc int = a + b + c
//                     return abc
//                 }
//             ",
//         );
//
//         let tokens = lexer::Lexer::new(&code).run().unwrap();
//         let ast = Ast::new(&tokens).unwrap();
//
//         for v in &ast.functions {
//             println!("{}", v.identifier);
//             println!("{:#?}", FunctionCompiler::compile_fn(v).unwrap());
//         }
//     }
// }
