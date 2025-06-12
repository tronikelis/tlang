use anyhow::{anyhow, Error, Result};

use crate::{ast, lexer, vm};

#[derive(Debug, Clone)]
pub enum Instruction {
    Real(vm::Instruction),
    JumpAndLink(String),
    Jump((usize, usize)),
    JumpIfTrue((usize, usize)),
}

#[derive(Debug, Clone)]
enum VarStackItem {
    Increment(usize),
    Reset(usize),
    Var(String),
}

struct VarStack {
    stack: Vec<Vec<VarStackItem>>,
    arg_size: Option<usize>,
}

impl VarStack {
    fn new() -> Self {
        let mut stack = Vec::new();
        stack.push(Vec::new());
        Self {
            stack,
            arg_size: None,
        }
    }

    fn set_arg_size(&mut self) {
        self.arg_size = Some(self.total_size());
    }

    fn push_frame(&mut self) {
        self.stack.push(Vec::new());
    }

    fn pop_frame(&mut self) -> usize {
        let last = self.stack.last().unwrap();
        let size = Self::size_for(last.iter());
        self.stack.pop();
        size
    }

    fn push(&mut self, item: VarStackItem) {
        self.stack.last_mut().unwrap().push(item);
    }

    fn size_for<'a>(items: impl Iterator<Item = &'a VarStackItem>) -> usize {
        items.fold(0, |acc, curr| match curr {
            VarStackItem::Var(_) => acc,
            VarStackItem::Increment(size) => acc + size,
            VarStackItem::Reset(size) => acc - size,
        })
    }

    fn total_size(&self) -> usize {
        Self::size_for(self.stack.iter().flatten())
    }

    fn get_var(&self, identifier: &str) -> Option<usize> {
        let mut offset: isize = 0;

        for item in self.stack.iter().flatten().rev() {
            match item {
                VarStackItem::Var(var) => {
                    if var == identifier {
                        return Some(offset as usize);
                    }
                }
                VarStackItem::Increment(size) => offset += *size as isize,
                VarStackItem::Reset(size) => offset -= *size as isize,
            };
        }

        None
    }
}

struct Instructions {
    instructions: Vec<Vec<Instruction>>,
    var_stack: VarStack,
    index: Vec<usize>,
}

impl Instructions {
    fn new() -> Self {
        let mut instructions = Vec::new();
        instructions.push(Vec::new());
        Self {
            index: Vec::from([0]),
            var_stack: VarStack::new(),
            instructions,
        }
    }

    fn push_stack_frame(&mut self) {
        self.var_stack.push_frame();
    }

    fn pop_stack_frame(&mut self) {
        let size = self.var_stack.pop_frame();
        self.push_instruction_no_sync(Instruction::Real(vm::Instruction::Reset(size)));
    }

    fn push_stack_identifier(&mut self, identifier: String) {
        self.var_stack.push(VarStackItem::Var(identifier));
    }

    fn get_stack_identifier(&self, identifier: &str) -> Option<usize> {
        self.var_stack.get_var(identifier)
    }

    fn push_alignment_for(&mut self, size: usize, stack_size: usize) -> usize {
        let alignment = Self::alignment_for(size, stack_size);
        if alignment != 0 {
            self.push(Instruction::Real(vm::Instruction::Increment(alignment)));
        }
        alignment
    }

    // 0 means no alignment necessary
    fn alignment_for(size: usize, stack_size: usize) -> usize {
        if size == 0 || stack_size == 0 {
            0
        } else {
            let modulo = stack_size % size;
            if modulo == 0 {
                0
            } else {
                size - modulo
            }
        }
    }

    fn stack_total_size(&self) -> usize {
        self.var_stack.total_size()
    }

    fn align_for(&mut self, instruction: &vm::Instruction) {
        let stack_size = self.stack_total_size();
        match instruction {
            vm::Instruction::PushI(_) => {
                self.push_alignment_for(ast::INT.size, stack_size);
            }
            vm::Instruction::PushU8(_) => {
                self.push_alignment_for(ast::UINT8.size, stack_size);
            }
            vm::Instruction::PushSlice => {
                self.push_alignment_for(ast::SLICE_SIZE, stack_size);
            }
            vm::Instruction::JumpAndLink(_) => {
                self.push_alignment_for(ast::PTR_SIZE, stack_size);
            }
            vm::Instruction::PushStatic(_index, size) => {
                self.push_alignment_for(*size, stack_size);
            }
            _ => {}
        };
    }

    fn push(&mut self, instruction: Instruction) {
        if let Instruction::Real(instruction) = &instruction {
            self.align_for(instruction);

            match instruction {
                vm::Instruction::PushI(_) => {
                    self.var_stack.push(VarStackItem::Increment(ast::INT.size));
                }
                vm::Instruction::PushU8(_) => {
                    self.var_stack
                        .push(VarStackItem::Increment(ast::UINT8.size));
                }
                vm::Instruction::PushSlice => {
                    self.var_stack
                        .push(VarStackItem::Increment(ast::SLICE_SIZE));
                }
                vm::Instruction::PushStatic(_index, size) => {
                    self.var_stack.push(VarStackItem::Increment(*size));
                }
                vm::Instruction::Reset(size) => self.var_stack.push(VarStackItem::Reset(*size)),
                vm::Instruction::Increment(size) => {
                    self.var_stack.push(VarStackItem::Increment(*size))
                }
                vm::Instruction::SliceIndexSet(size) => {
                    self.var_stack
                        .push(VarStackItem::Reset(ast::SLICE_SIZE + ast::INT.size + size));
                }
                vm::Instruction::SliceLen => {
                    self.var_stack.push(VarStackItem::Reset(ast::SLICE_SIZE));
                    self.var_stack.push(VarStackItem::Increment(ast::INT.size));
                }
                vm::Instruction::SliceAppend(size) => {
                    self.var_stack
                        .push(VarStackItem::Reset(ast::SLICE_SIZE + size));
                }
                vm::Instruction::SliceIndexGet(size) => {
                    self.var_stack
                        .push(VarStackItem::Reset(ast::INT.size + ast::SLICE_SIZE));
                    self.var_stack.push(VarStackItem::Increment(*size));
                }
                vm::Instruction::And => {
                    self.var_stack.push(VarStackItem::Reset(ast::BOOL.size));
                }
                vm::Instruction::Or => {
                    self.var_stack.push(VarStackItem::Reset(ast::BOOL.size));
                }
                vm::Instruction::SyscallWrite => {
                    self.var_stack
                        .push(VarStackItem::Reset(ast::SLICE_SIZE + ast::INT.size));
                }
                vm::Instruction::AddI => {
                    self.var_stack.push(VarStackItem::Reset(ast::INT.size));
                }
                vm::Instruction::MultiplyI => {
                    self.var_stack.push(VarStackItem::Reset(ast::INT.size));
                }
                vm::Instruction::DivideI => {
                    self.var_stack.push(VarStackItem::Reset(ast::INT.size));
                }
                vm::Instruction::ModuloI => {
                    self.var_stack.push(VarStackItem::Reset(ast::INT.size));
                }
                vm::Instruction::AddString => {
                    self.var_stack.push(VarStackItem::Reset(ast::SLICE_SIZE));
                }
                vm::Instruction::ToBool => {
                    // this currently does not change the bool size
                    // however I will probably change that at a later point
                }
                vm::Instruction::CompareInt => {
                    self.var_stack.push(VarStackItem::Reset(ast::INT.size));
                    self.var_stack.push(VarStackItem::Reset(ast::INT.size));
                    self.var_stack.push(VarStackItem::Increment(ast::BOOL.size));
                }
                vm::Instruction::CastIntUint8 => {
                    self.var_stack.push(VarStackItem::Reset(ast::INT.size));
                    self.var_stack
                        .push(VarStackItem::Increment(ast::UINT8.size));
                }
                vm::Instruction::CastUint8Int => {
                    self.var_stack.push(VarStackItem::Reset(ast::UINT8.size));
                    self.var_stack.push(VarStackItem::Increment(ast::INT.size));
                }
                _ => {}
            }
        }
        self.push_instruction_no_sync(instruction);
    }

    fn push_instruction_no_sync(&mut self, instruction: Instruction) {
        self.instructions[*self.index.last().unwrap()].push(instruction);
    }

    fn jump(&mut self) {
        let index = self.instructions.len();
        self.push(Instruction::Jump((index, 0)));
        self.instructions.push(Vec::new());
        self.index.push(index);
    }

    fn jump_if_true(&mut self) {
        let index = self.instructions.len();
        self.push(Instruction::JumpIfTrue((index, 0)));
        self.instructions.push(Vec::new());
        self.index.push(index);
    }

    fn back_if_true(&mut self, offset: usize) {
        let target = self.index[self.index.len() - 1 - offset];
        let target_last = self.instructions[target].len();
        self.push(Instruction::JumpIfTrue((target, target_last)));
    }

    fn again(&mut self) {
        self.push(Instruction::Jump((*self.index.last().unwrap(), 0)));
    }

    fn back(&mut self, offset: usize) {
        let target = self.index[self.index.len() - 1 - offset];
        let target_last = self.instructions[target].len();
        self.push(Instruction::Jump((target, target_last)));
    }

    fn pop_index(&mut self) {
        self.index.pop();
    }
}

pub struct FunctionCompiler<'a, 'b> {
    instructions: Instructions,
    function: &'a ast::Function,
    static_memory: &'b mut vm::StaticMemory,
}

impl<'a, 'b> FunctionCompiler<'a, 'b> {
    pub fn new(function: &'a ast::Function, static_memory: &'b mut vm::StaticMemory) -> Self {
        Self {
            static_memory,
            function,
            instructions: Instructions::new(),
        }
    }

    fn compile_function_builtin_len(&mut self, call: &ast::FunctionCall) -> Result<ast::Type> {
        let slice_arg = call
            .arguments
            .get(0)
            .ok_or(anyhow!("len: expected first argument"))?;

        let slice_exp = self.compile_expression(slice_arg)?;

        let ast::TypeType::Slice(_) = &slice_exp._type else {
            return Err(anyhow!("len: expected slice as the argument"));
        };

        self.instructions
            .push(Instruction::Real(vm::Instruction::SliceLen));

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

        let slice_exp = self.compile_expression(slice_arg)?;
        let value_exp = self.compile_expression(value_arg)?;

        let ast::TypeType::Slice(slice_item) = &slice_exp._type else {
            return Err(anyhow!("append: provide a slice as the first argument"));
        };

        if !slice_item.can_assign(&value_exp) {
            return Err(anyhow!("append: value type does not match slice type"));
        }

        self.instructions
            .push(Instruction::Real(vm::Instruction::SliceAppend(
                value_exp.size,
            )));

        Ok(ast::VOID)
    }

    fn compile_function_builtin_syscall_write(
        &mut self,
        call: &ast::FunctionCall,
    ) -> Result<ast::Type> {
        let fd = call
            .arguments
            .get(0)
            .ok_or(anyhow!("syscall_write: expected first argument"))?;
        let slice = call
            .arguments
            .get(1)
            .ok_or(anyhow!("syscall_write: expected second argument"))?;

        let exp_slice = self.compile_expression(slice)?;
        let exp_fd = self.compile_expression(fd)?;

        if exp_fd != ast::INT {
            return Err(anyhow!("syscall_write: first argument not an integer"));
        }

        if exp_slice != *ast::SLICE_UINT8 {
            return Err(anyhow!(
                "syscall_write: second argument should be uint8 slice"
            ));
        }

        self.instructions
            .push(Instruction::Real(vm::Instruction::SyscallWrite));

        Ok(ast::VOID)
    }

    fn compile_function_call(&mut self, call: &ast::FunctionCall) -> Result<ast::Type> {
        match call.function.identifier.as_str() {
            "append" => return self.compile_function_builtin_append(call),
            "len" => return self.compile_function_builtin_len(call),
            "syscall_write" => return self.compile_function_builtin_syscall_write(call),
            _ => {}
        }

        let return_alignment = self.instructions.push_alignment_for(
            call.function.return_type.size,
            self.instructions.stack_total_size(),
        );

        let argument_size = call.arguments.iter().enumerate().try_fold(0, |acc, curr| {
            let expected_type = &call
                .function
                .arguments
                .get(curr.0)
                .ok_or(anyhow!("compile_function_call: expected_argument"))?
                ._type;

            let _type = self.compile_expression(curr.1)?;
            if *expected_type != _type {
                Err(anyhow!("compile_function_call: unexpected type"))
            } else {
                Ok::<usize, Error>(acc + _type.size)
            }
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

        let return_pc_align = self
            .instructions
            .push_alignment_for(ast::PTR_SIZE, self.instructions.stack_total_size());

        self.instructions
            .push(Instruction::JumpAndLink(call.function.identifier.clone()));

        self.instructions
            .push(Instruction::Real(vm::Instruction::Reset(
                reset_size + return_pc_align + return_alignment,
            )));

        Ok(call.function.return_type.clone())
    }

    fn compile_literal(&mut self, literal: &ast::Literal) -> Result<ast::Type> {
        match &literal.literal {
            lexer::Literal::Int(int) => match &literal._type {
                &ast::UINT8 => {
                    self.instructions
                        .push(Instruction::Real(vm::Instruction::PushU8(
                            (*int).try_into()?,
                        )));
                }
                &ast::INT => {
                    self.instructions
                        .push(Instruction::Real(vm::Instruction::PushI(
                            (*int).try_into()?,
                        )));
                }
                _type => return Err(anyhow!("can't cast int to {_type:#?}")),
            },
            lexer::Literal::Bool(bool) => match &literal._type {
                &ast::BOOL => {
                    self.instructions
                        .push(Instruction::Real(vm::Instruction::PushI({
                            if *bool {
                                1
                            } else {
                                0
                            }
                        })));
                }
                _type => return Err(anyhow!("can't cast bool to {_type:#?}")),
            },
            lexer::Literal::String(string) => {
                let index = self.static_memory.push_string_slice(&string);
                self.instructions
                    .push(Instruction::Real(vm::Instruction::PushStatic(
                        index,
                        ast::SLICE_SIZE,
                    )));
            }
        }

        Ok(literal._type.clone())
    }

    fn compile_variable(&mut self, variable: &ast::Variable) -> Result<ast::Type> {
        let offset = self
            .instructions
            .get_stack_identifier(&variable.identifier)
            .ok_or(anyhow!("compile_identifier: unknown identifier"))?;

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
                    self.instructions
                        .push(Instruction::Real(vm::Instruction::MinusInt));
                    self.instructions
                        .push(Instruction::Real(vm::Instruction::AddI));
                } else {
                    return Err(anyhow!("can only minus int"));
                }
            }
            ast::ArithmeticType::Plus => {
                if let ast::INT = a {
                    self.instructions
                        .push(Instruction::Real(vm::Instruction::AddI));
                } else if a == ast::STRING {
                    self.instructions
                        .push(Instruction::Real(vm::Instruction::AddString));
                } else {
                    return Err(anyhow!("can only plus int and string"));
                }
            }
            ast::ArithmeticType::Multiply => {
                if let ast::INT = a {
                    self.instructions
                        .push(Instruction::Real(vm::Instruction::MultiplyI));
                } else {
                    return Err(anyhow!("can only multiply int"));
                }
            }
            ast::ArithmeticType::Divide => {
                if let ast::INT = a {
                    self.instructions
                        .push(Instruction::Real(vm::Instruction::DivideI));
                } else {
                    return Err(anyhow!("can only divide int"));
                }
            }
            ast::ArithmeticType::Modulo => {
                if let ast::INT = a {
                    self.instructions
                        .push(Instruction::Real(vm::Instruction::ModuloI));
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
            ast::CompareType::Equals => {
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
                self.instructions
                    .push(Instruction::Real(vm::Instruction::MinusInt));

                // a + b
                self.instructions
                    .push(Instruction::Real(vm::Instruction::AddI));

                // >0:1 <0:0
                self.instructions
                    .push(Instruction::Real(vm::Instruction::ToBool));
            }
            ast::CompareType::Equals => self
                .instructions
                .push(Instruction::Real(vm::Instruction::CompareInt)),
        }

        Ok(ast::BOOL)
    }

    fn compile_infix(&mut self, infix: &ast::Infix) -> Result<ast::Type> {
        let exp = self.compile_expression(&infix.expression)?;
        match infix._type {
            ast::InfixType::Plus => {}
            ast::InfixType::Minus => self
                .instructions
                .push(Instruction::Real(vm::Instruction::MinusInt)),
        }
        Ok(exp)
    }

    fn compile_list(&mut self, list: &[ast::Expression]) -> Result<ast::Type> {
        self.instructions
            .push(Instruction::Real(vm::Instruction::PushSlice));

        let mut curr_exp: Option<ast::Type> = None;

        for v in list {
            self.instructions
                .push(Instruction::Real(vm::Instruction::Increment(
                    ast::SLICE_SIZE,
                )));
            self.instructions
                .push(Instruction::Real(vm::Instruction::Copy(
                    0,
                    ast::SLICE_SIZE,
                    ast::SLICE_SIZE,
                )));

            let exp = self.compile_expression(v)?;
            if let Some(curr_exp) = curr_exp {
                if curr_exp != exp {
                    return Err(anyhow!("type mismatch"));
                }
            }
            curr_exp = Some(exp.clone());

            self.instructions
                .push(Instruction::Real(vm::Instruction::SliceAppend(exp.size)));
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

        self.instructions
            .push(Instruction::Real(vm::Instruction::SliceIndexGet(
                expected_type.size,
            )));

        Ok(*expected_type)
    }

    fn compile_andor(&mut self, andor: &ast::AndOr) -> Result<ast::Type> {
        self.instructions.jump();

        let left = self.compile_expression(&andor.left)?;
        if left != ast::BOOL {
            return Err(anyhow!("compile_andor: expected bool expression"));
        }

        match andor._type {
            // skip over if this is true
            ast::AndOrType::Or => {
                self.instructions.back_if_true(1);

                let right = self.compile_expression(&andor.right)?;
                if right != ast::BOOL {
                    return Err(anyhow!("compile_andor: expected bool expression"));
                }

                self.instructions
                    .push(Instruction::Real(vm::Instruction::Or));
            }
            // continue if this is true
            ast::AndOrType::And => {
                self.instructions
                    .push(Instruction::Real(vm::Instruction::NegateBool));
                self.instructions.back_if_true(1);
                self.instructions
                    .push(Instruction::Real(vm::Instruction::NegateBool));

                let right = self.compile_expression(&andor.right)?;
                if right != ast::BOOL {
                    return Err(anyhow!("compile_andor: expected bool expression"));
                }

                self.instructions
                    .push(Instruction::Real(vm::Instruction::And));
            }
        }

        self.instructions.back(1);
        self.instructions.pop_index();

        Ok(ast::BOOL)
    }

    fn compile_type_cast(&mut self, type_cast: &ast::TypeCast) -> Result<ast::Type> {
        self.instructions
            .push_alignment_for(type_cast._type.size, self.instructions.stack_total_size());

        let target = self.compile_expression(&type_cast.expression)?;

        match target {
            ast::INT => match &type_cast._type {
                &ast::UINT8 => {
                    self.instructions
                        .push(Instruction::Real(vm::Instruction::CastIntUint8));

                    Ok(ast::UINT8)
                }
                _type => Err(anyhow!("compile_type_cast: cant cast into {_type:#?}")),
            },
            ast::UINT8 => match &type_cast._type {
                &ast::INT => {
                    self.instructions
                        .push(Instruction::Real(vm::Instruction::CastUint8Int));

                    Ok(ast::INT)
                }
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

    fn compile_expression(&mut self, expression: &ast::Expression) -> Result<ast::Type> {
        match expression {
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
        }
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
            .push_stack_identifier(declaration.variable.identifier.clone());

        Ok(())
    }

    fn compile_variable_assignment(&mut self, assignment: &ast::VariableAssignment) -> Result<()> {
        match &assignment.var {
            ast::Expression::Variable(var) => {
                let exp = self.compile_expression(&assignment.expression)?;

                let offset = self
                    .instructions
                    .get_stack_identifier(&var.identifier)
                    .ok_or(anyhow!("compile_variable_assignment: var not found"))?;

                if var._type != exp {
                    return Err(anyhow!("variable assignment type mismatch"));
                }

                self.instructions
                    .push(Instruction::Real(vm::Instruction::Copy(
                        offset,
                        0,
                        var._type.size,
                    )));
                self.instructions
                    .push(Instruction::Real(vm::Instruction::Reset(var._type.size)));
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

                self.instructions
                    .push(Instruction::Real(vm::Instruction::SliceIndexSet(item.size)));
            }
            node => return Err(anyhow!("can't assign {node:#?}")),
        }

        Ok(())
    }

    fn compile_if_block(&mut self, expression: &ast::Expression, body: &[ast::Node]) -> Result<()> {
        let exp = self.compile_expression(expression)?;
        if exp != ast::BOOL {
            return Err(anyhow!("compile_if_block: expected bool expression"));
        }

        self.instructions.jump_if_true();

        self.compile_body(body)?;
        self.instructions.back(2);
        self.instructions.pop_index();

        Ok(())
    }

    fn compile_if(&mut self, _if: &ast::If) -> Result<()> {
        self.instructions.push_stack_frame();

        self.instructions.jump();

        self.compile_if_block(&_if.expression, &_if.body)?;

        for v in &_if.elseif {
            self.compile_if_block(&v.expression, &v.body)?;
        }

        if let Some(v) = &_if._else {
            self.compile_body(&v.body)?;
        }

        self.instructions.back(1);
        self.instructions.pop_index();

        self.instructions.pop_stack_frame();

        Ok(())
    }

    fn compile_for(&mut self, _for: &ast::For) -> Result<()> {
        self.instructions.push_stack_frame();
        self.compile_node(&_for.initializer)?;

        self.instructions.jump();
        self.instructions.push_stack_frame();

        let bool_alignment = self
            .instructions
            .push_alignment_for(ast::BOOL.size, self.instructions.stack_total_size());

        let exp = self.compile_expression(&_for.expression)?;
        if exp != ast::BOOL {
            return Err(anyhow!("compile_for: expected expression to return bool"));
        }
        self.instructions
            .push(Instruction::Real(vm::Instruction::NegateBool));
        self.instructions.back_if_true(1);

        self.compile_body(&_for.body)?;
        self.compile_node(&_for.after_each)?;

        self.instructions.pop_stack_frame();
        self.instructions.again();
        self.instructions.pop_index();

        self.instructions.pop_stack_frame();
        self.instructions
            .push_instruction_no_sync(Instruction::Real(vm::Instruction::Reset(
                ast::BOOL.size + bool_alignment,
            )));

        Ok(())
    }

    fn compile_node(&mut self, node: &ast::Node) -> Result<()> {
        match node {
            ast::Node::VariableDeclaration(var) => self.compile_variable_declaration(var)?,
            ast::Node::Return(exp) => self.compile_return(exp.as_ref())?,
            ast::Node::Expression(exp) => {
                let exp = self.compile_expression(exp)?;

                // no variable to store result in, reset instantly
                self.instructions
                    .push(Instruction::Real(vm::Instruction::Reset(exp.size)));
            }
            ast::Node::VariableAssignment(assignment) => {
                self.compile_variable_assignment(assignment)?;
            }
            ast::Node::If(v) => {
                self.compile_if(v)?;
            }
            ast::Node::Debug => {
                self.instructions
                    .push(Instruction::Real(vm::Instruction::Debug));
            }
            ast::Node::For(v) => self.compile_for(v)?,
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

            self.instructions
                .push(Instruction::Real(vm::Instruction::Copy(
                    // -size because .total_size() gets you total stack size,
                    // while we want to index into first item
                    self.instructions.stack_total_size() - exp.size,
                    0,
                    exp.size,
                )));
        }

        self.instructions
            .push(Instruction::Real(vm::Instruction::Reset(
                self.instructions.stack_total_size()
                    - self.instructions.var_stack.arg_size.unwrap(),
            )));

        if self.function.identifier == "main" {
            self.instructions
                .push(Instruction::Real(vm::Instruction::Exit));
        } else {
            self.instructions
                .push(Instruction::Real(vm::Instruction::Return));
        }

        Ok(())
    }

    pub fn compile(mut self) -> Result<Vec<Vec<Instruction>>> {
        let mut arguments_iter = self.function.arguments.iter();
        // align for return size, this will be "on top" of the var stack
        if let Some(arg) = arguments_iter.next() {
            self.instructions
                .var_stack
                .push(VarStackItem::Increment(Instructions::alignment_for(
                    arg._type.size,
                    self.function.return_type.size,
                )));
        }

        for arg in arguments_iter {
            let stack_size = self.instructions.stack_total_size();

            self.instructions.var_stack.push(VarStackItem::Increment(
                arg._type.size + Instructions::alignment_for(arg._type.size, stack_size),
            ));

            self.instructions
                .var_stack
                .push(VarStackItem::Var(arg.identifier.clone()));
        }

        if self.function.identifier != "main" {
            let stack_size = self.instructions.stack_total_size();
            // return address
            self.instructions.var_stack.push(VarStackItem::Increment(
                ast::PTR_SIZE + Instructions::alignment_for(ast::PTR_SIZE, stack_size),
            ));
        }

        self.instructions.var_stack.set_arg_size();

        self.compile_body(
            self.function
                .body
                .as_ref()
                .ok_or(anyhow!("compile: function body empty"))?,
        )?;

        self.compile_return(None)?;

        Ok(self.instructions.instructions)
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
