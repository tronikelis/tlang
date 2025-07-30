use anyhow::{anyhow, Result};
use std::{cell::RefCell, collections::HashMap, rc::Rc};

use crate::{
    ast::{self, Dfs, DfsMut},
    lexer, vm,
};

#[derive(Debug, Clone)]
struct VarStack<T> {
    items: Vec<Vec<T>>,
}

impl<T> VarStack<T> {
    fn new() -> Self {
        let mut items = Vec::new();
        items.push(Vec::new());
        Self { items }
    }

    fn push(&mut self, item: T) {
        self.items.last_mut().unwrap().push(item);
    }

    fn push_frame(&mut self) {
        self.items.push(Vec::new());
    }

    fn pop_frame(&mut self) -> Option<Vec<T>> {
        self.items.pop()
    }
}

struct ClosureEscapedVariables<'a> {
    closure: &'a ast::Closure,
    var_stack: VarStack<String>,
    not_found_variables: Vec<String>,
}

impl<'a> ClosureEscapedVariables<'a> {
    fn new(closure: &'a ast::Closure) -> Self {
        let mut var_stack = VarStack::new();

        for var in &closure._type.closure_err().unwrap().arguments {
            var_stack.push(var.identifier.clone());
        }

        Self {
            closure,
            var_stack,
            not_found_variables: Vec::new(),
        }
    }

    fn search(mut self) -> Vec<String> {
        self.search_body(self.closure.body.iter());
        self.not_found_variables
    }
}

impl<'a, 'b> ast::DfsMut<'b> for ClosureEscapedVariables<'a> {
    fn search_body(&mut self, body: impl Iterator<Item = &'b ast::Node>) -> ast::DfsRet {
        self.var_stack.push_frame();
        let result = ast::dfsret_search_body!(self, body);
        self.var_stack.pop_frame();
        result
    }

    fn search_expression_type(&mut self, _type: &ast::Type) -> ast::DfsRet {
        let ast::Type::Alias(identifier) = _type else {
            return ast::DfsRet::Continue;
        };

        if let None = self
            .var_stack
            .items
            .iter()
            .flatten()
            .find(|v| *v == identifier)
        {
            self.not_found_variables.push(identifier.clone());
        }

        ast::DfsRet::Continue
    }

    fn search_node_variable_declaration(
        &mut self,
        declaration: &ast::VariableDeclaration,
    ) -> ast::DfsRet {
        ast::dfsret_return_if!(self.search_expression(&declaration.expression));
        self.var_stack.push(declaration.variable.identifier.clone());
        ast::DfsRet::Continue
    }
}

struct DoesVariableEscapeInClosure<'a, 'b>(&'b DoesVariableEscape<'a>);

impl<'a, 'b, 'c> ast::Dfs<'c> for DoesVariableEscapeInClosure<'a, 'b> {
    fn search_expression_type(&self, _type: &ast::Type) -> ast::DfsRet {
        let ast::Type::Alias(identifier) = _type else {
            return ast::DfsRet::Continue;
        };

        if self.0.identifier == identifier {
            ast::DfsRet::Found
        } else {
            ast::DfsRet::Continue
        }
    }

    fn search_node_variable_declaration(
        &self,
        declaration: &ast::VariableDeclaration,
    ) -> ast::DfsRet {
        ast::dfsret_return_if!(self.search_expression(&declaration.expression));

        if declaration.variable.identifier == self.0.identifier {
            return ast::DfsRet::Break;
        }

        ast::DfsRet::Continue
    }
}

struct DoesVariableEscape<'a> {
    identifier: &'a str,
}

impl<'a> DoesVariableEscape<'a> {
    fn new(identifier: &'a str) -> Self {
        Self { identifier }
    }

    fn search(self, iter: impl Iterator<Item = &'a ast::Node>) -> bool {
        match self.search_body(iter) {
            ast::DfsRet::Found => true,
            _ => false,
        }
    }
}

impl<'a, 'b> ast::Dfs<'b> for DoesVariableEscape<'a> {
    fn search_expression_address(&self, exp: &ast::Expression) -> ast::DfsRet {
        match exp {
            ast::Expression::Type(_type) => {
                if let ast::Type::Alias(identifier) = _type {
                    if self.identifier == identifier {
                        return ast::DfsRet::Found;
                    }
                }
            }
            ast::Expression::DotAccess(dot_access) => {
                return self.search_expression_address(&dot_access.deepest().expression);
            }
            _ => {}
        }

        self.search_expression(exp)
    }

    fn search_node_variable_declaration(
        &self,
        declaration: &ast::VariableDeclaration,
    ) -> ast::DfsRet {
        ast::dfsret_return_if!(self.search_expression(&declaration.expression));

        if declaration.variable.identifier == self.identifier {
            return ast::DfsRet::Break;
        }

        ast::DfsRet::Continue
    }

    fn search_expression_closure(&self, closure: &'b ast::Closure) -> ast::DfsRet {
        DoesVariableEscapeInClosure(self).search_expression_closure(closure)
    }
}

struct CompilerBody {
    i: usize,
    body: Vec<ast::Node>,
}

impl CompilerBody {
    fn new(body: Vec<ast::Node>) -> Self {
        Self { body, i: 0 }
    }

    fn next(&mut self) {
        self.i += 1;
    }

    fn does_variable_escape(&self, identifier: &str, skip: usize) -> bool {
        DoesVariableEscape::new(identifier).search(self.body.iter().skip(self.i + skip))
    }
}

#[derive(Debug, Clone)]
pub enum CompilerInstruction {
    Real(vm::Instruction),
    JumpAndLink(String),
    Jump((usize, usize)),
    JumpIfTrue((usize, usize)),
    JumpIfFalse((usize, usize)),
    PushClosure(usize, usize),
}

#[derive(Debug, Clone)]
pub enum ScopedInstruction {
    Real(vm::Instruction),
    JumpAndLink(String),
    // offset
    Jump(usize),
    JumpIfTrue(usize),
    JumpIfFalse(usize),
    // vars count, offset (instruction index)
    PushClosure(usize, usize),
}

impl ScopedInstruction {
    pub fn add_jump_offset(&mut self, offset: usize) {
        match self {
            Self::Real(_) => {}
            Self::JumpAndLink(_) => {}
            Self::Jump(v) => *v = *v + offset,
            Self::JumpIfTrue(v) => *v = *v + offset,
            Self::JumpIfFalse(v) => *v = *v + offset,
            Self::PushClosure(_, v) => *v = *v + offset,
        }
    }
}

impl ScopedInstruction {
    pub fn from_compiled_instructions(compiled_instructions: &CompiledInstructions) -> Vec<Self> {
        let mut instructions: Vec<Self> = Vec::new();
        let mut index_to_jump = HashMap::<usize, usize>::new();
        let mut index_to_closure = HashMap::<usize, usize>::new();
        let mut folded: Vec<CompilerInstruction> = Vec::new();

        for (i, v) in compiled_instructions.closures.iter().enumerate() {
            index_to_closure.insert(i, instructions.len());
            instructions.append(&mut Self::from_compiled_instructions(v));
        }

        for (i, v) in compiled_instructions.instructions.iter().enumerate() {
            index_to_jump.insert(i, folded.len());
            folded.append(&mut v.clone());
        }

        let mut new_instructions: Vec<Self> = Vec::new();
        let folded_len = folded.len();
        for v in folded {
            new_instructions.push(match v {
                CompilerInstruction::Real(v) => ScopedInstruction::Real(v),
                CompilerInstruction::Jump((index, offset)) => {
                    ScopedInstruction::Jump(index_to_jump.get(&index).unwrap() + offset)
                }
                CompilerInstruction::JumpIfTrue((index, offset)) => {
                    ScopedInstruction::JumpIfTrue(index_to_jump.get(&index).unwrap() + offset)
                }
                CompilerInstruction::JumpIfFalse((index, offset)) => {
                    ScopedInstruction::JumpIfFalse(index_to_jump.get(&index).unwrap() + offset)
                }
                CompilerInstruction::JumpAndLink(v) => ScopedInstruction::JumpAndLink(v),
                CompilerInstruction::PushClosure(vars_count, index) => {
                    ScopedInstruction::PushClosure(
                        vars_count,
                        *index_to_closure.get(&index).unwrap() + folded_len,
                    )
                }
            });
        }

        instructions
            .iter_mut()
            .for_each(|v| v.add_jump_offset(new_instructions.len()));

        [new_instructions, instructions].concat()
    }
}

#[derive(Debug, Clone)]
enum VarStackItem {
    Increment(usize),
    Reset(usize),
    Var(Variable),
    Label,
}

#[derive(Debug, Clone)]
struct CompilerVarStack {
    stack: VarStack<VarStackItem>,
    arg_size: Option<usize>,
}

impl CompilerVarStack {
    fn new() -> Self {
        Self {
            stack: VarStack::new(),
            arg_size: None,
        }
    }

    fn set_arg_size(&mut self) {
        self.arg_size = Some(self.total_size());
    }

    fn total_size(&self) -> usize {
        Self::size_for(self.stack.items.iter().flatten())
    }

    fn inc_offset_item(offset: &mut isize, item: &VarStackItem) {
        match item {
            VarStackItem::Label | VarStackItem::Var(_) => {}
            VarStackItem::Increment(size) => *offset += *size as isize,
            VarStackItem::Reset(size) => *offset -= *size as isize,
        };
    }

    fn iter_rev(&self) -> impl Iterator<Item = &VarStackItem> {
        self.stack.items.iter().flatten().rev()
    }

    fn size_for<'a>(items: impl Iterator<Item = &'a VarStackItem>) -> usize {
        items.fold(0, |acc, curr| match curr {
            VarStackItem::Var(_) => acc,
            VarStackItem::Increment(size) => acc + size,
            VarStackItem::Reset(size) => acc - size,
            VarStackItem::Label => acc,
        })
    }

    fn get_var_offset(&self, identifier: &str) -> Option<(usize, &Variable)> {
        let mut offset: isize = 0;
        for item in self.iter_rev() {
            match item {
                VarStackItem::Var(v) => {
                    if v.identifier == identifier {
                        return Some((offset as usize, v));
                    }
                }
                item => Self::inc_offset_item(&mut offset, item),
            }
        }

        None
    }

    fn get_label_offset(&self) -> Option<usize> {
        let mut offset: isize = 0;
        for item in self.iter_rev() {
            match item {
                VarStackItem::Label => return Some(offset as usize),
                item => Self::inc_offset_item(&mut offset, item),
            }
        }

        None
    }
}

#[derive(Debug, Clone)]
struct StackLabel {
    identifier: String,
    index: usize,
}

#[derive(Debug, Clone)]
struct StackInstructions {
    instructions: Vec<Vec<CompilerInstruction>>,
    index: Vec<usize>,
    labels: Vec<StackLabel>,
}

impl StackInstructions {
    fn new() -> Self {
        let mut instructions = Vec::new();
        instructions.push(Vec::new());
        Self {
            index: Vec::from([0]),
            instructions,
            labels: Vec::new(),
        }
    }

    fn push(&mut self, instruction: CompilerInstruction) {
        self.instructions[*self.index.last().unwrap()].push(instruction);
    }

    fn jump(&mut self) {
        let index = self.instructions.len();
        self.push(CompilerInstruction::Jump((index, 0)));
        self.instructions.push(Vec::new());
        self.index.push(index);
    }

    fn label_new(&mut self, identifier: String) {
        let index = *self.index.last().unwrap();
        self.labels.push(StackLabel { index, identifier });
    }

    fn label_pop(&mut self) {
        self.labels.pop();
    }

    fn label_jump(&mut self, identifier: &str) -> Result<()> {
        let label = self
            .labels
            .iter()
            .rev()
            .find(|v| &v.identifier == identifier)
            .ok_or(anyhow!("label_jump: {:#?} not found", identifier))?;

        self.push(CompilerInstruction::Jump((
            label.index,
            self.instructions[label.index].len(),
        )));

        Ok(())
    }

    fn jump_if_true(&mut self) {
        let index = self.instructions.len();
        self.push(CompilerInstruction::JumpIfTrue((index, 0)));
        self.instructions.push(Vec::new());
        self.index.push(index);
    }

    fn back_if_true(&mut self, offset: usize) {
        let target = self.index[self.index.len() - 1 - offset];
        let target_last = self.instructions[target].len();
        self.push(CompilerInstruction::JumpIfTrue((target, target_last)));
    }

    fn back_if_false(&mut self, offset: usize) {
        let target = self.index[self.index.len() - 1 - offset];
        let target_last = self.instructions[target].len();
        self.push(CompilerInstruction::JumpIfFalse((target, target_last)));
    }

    fn again(&mut self) {
        self.push(CompilerInstruction::Jump((*self.index.last().unwrap(), 0)));
    }

    fn back(&mut self, offset: usize) {
        let target = self.index[self.index.len() - 1 - offset];
        let target_last = self.instructions[target].len();
        self.push(CompilerInstruction::Jump((target, target_last)));
    }

    fn pop_index(&mut self) {
        self.index.pop();
    }
}

fn align(alignment: usize, stack_size: usize) -> usize {
    if alignment == 0 || stack_size == 0 {
        0
    } else {
        let modulo = stack_size % alignment;
        if modulo == 0 {
            0
        } else {
            alignment - modulo
        }
    }
}

#[derive(Debug, Clone)]
struct Instructions {
    stack_instructions: StackInstructions,
    var_stack: CompilerVarStack,
    static_var_stack: CompilerVarStack,
}

impl Instructions {
    fn new(static_var_stack: CompilerVarStack) -> Self {
        Self {
            stack_instructions: StackInstructions::new(),
            var_stack: static_var_stack.clone(),
            static_var_stack,
        }
    }

    fn var_mark(&mut self, var: Variable) {
        self.var_stack.stack.push(VarStackItem::Var(var));
    }

    fn var_mark_label(&mut self) {
        self.var_stack.stack.push(VarStackItem::Label);
    }

    fn var_get_offset(&self, identifier: &str) -> Option<(usize, &Variable)> {
        self.var_stack.get_var_offset(identifier)
    }

    fn var_reset_label(&mut self) {
        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::Reset(
                self.var_stack.get_label_offset().unwrap(),
            )));
    }

    fn instr_offset(&mut self, size: usize) {
        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::Offset(size)));
    }

    fn instr_alloc(&mut self, size: usize, alignment: usize) {
        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::Alloc(
                size, alignment,
            )));
        self.var_stack.stack.push(VarStackItem::Reset(size));
        self.var_stack.stack.push(VarStackItem::Increment(PTR_SIZE));
    }

    fn instr_deref(&mut self, size: usize) {
        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::Deref(size)));
        self.var_stack.stack.push(VarStackItem::Reset(PTR_SIZE));
        self.var_stack.stack.push(VarStackItem::Increment(size));
    }

    fn instr_deref_assign(&mut self, size: usize) {
        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::DerefAssign(
                size,
            )));
        self.var_stack
            .stack
            .push(VarStackItem::Reset(PTR_SIZE + size));
    }

    fn instr_slice_index_set(&mut self, size: usize) {
        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::SliceIndexSet(
                size,
            )));
        self.var_stack
            .stack
            .push(VarStackItem::Reset(size + SLICE_SIZE + INT.size));
    }

    fn instr_slice_index_get(&mut self, size: usize) {
        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::SliceIndexGet(
                size,
            )));
        self.var_stack
            .stack
            .push(VarStackItem::Reset(INT.size + SLICE_SIZE));
        self.var_stack.stack.push(VarStackItem::Increment(size));
    }

    fn instr_slice_len(&mut self) {
        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::SliceLen));
        self.var_stack.stack.push(VarStackItem::Reset(SLICE_SIZE));
        self.var_stack.stack.push(VarStackItem::Increment(INT.size));
    }

    fn instr_slice_append(&mut self, size: usize) {
        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::SliceAppend(
                size,
            )));
        self.var_stack
            .stack
            .push(VarStackItem::Reset(SLICE_SIZE + size));
    }

    fn instr_and(&mut self) {
        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::And));
        self.var_stack.stack.push(VarStackItem::Reset(BOOL.size));
    }

    fn instr_or(&mut self) {
        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::Or));
        self.var_stack.stack.push(VarStackItem::Reset(BOOL.size));
    }

    fn instr_negate_bool(&mut self) {
        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::NegateBool));
    }

    fn instr_increment(&mut self, size: usize) {
        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::Increment(size)));
        self.var_stack.stack.push(VarStackItem::Increment(size));
    }

    fn instr_reset_dangerous_not_synced(&mut self, size: usize) {
        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::Reset(size)));
    }

    fn instr_reset(&mut self, size: usize) {
        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::Reset(size)));
        self.var_stack.stack.push(VarStackItem::Reset(size));
    }

    fn instr_push_closure(&mut self, escaped_count: usize, function_index: usize) {
        self.stack_instructions
            .push(CompilerInstruction::PushClosure(
                escaped_count,
                function_index,
            ));
        self.var_stack
            .stack
            .push(VarStackItem::Reset(escaped_count * PTR_SIZE));
        self.var_stack.stack.push(VarStackItem::Increment(PTR_SIZE));
    }

    fn instr_push_slice_new_len(&mut self, size: usize) {
        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::PushSliceNewLen(
                size,
            )));
        self.var_stack
            .stack
            .push(VarStackItem::Reset(size + INT.size - SLICE_SIZE));
    }

    fn instr_push_slice(&mut self) {
        self.push_alignment(SLICE_SIZE);

        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::PushSlice));
        self.var_stack
            .stack
            .push(VarStackItem::Increment(SLICE_SIZE));
    }

    fn instr_push_u8(&mut self, int: usize) -> Result<()> {
        self.push_alignment(UINT8.alignment);
        let uint8: u8 = int.try_into()?;

        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::PushU8(uint8)));
        self.var_stack
            .stack
            .push(VarStackItem::Increment(UINT8.size));

        Ok(())
    }

    fn instr_push_i(&mut self, int: usize) -> Result<()> {
        self.push_alignment(INT.alignment);
        let int: isize = int.try_into()?;

        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::PushI(int)));
        self.var_stack.stack.push(VarStackItem::Increment(INT.size));

        Ok(())
    }

    fn instr_minus_int(&mut self) {
        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::MinusInt));
    }

    fn instr_add_i(&mut self) {
        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::AddI));
        self.var_stack.stack.push(VarStackItem::Reset(INT.size));
    }

    fn instr_multiply_i(&mut self) {
        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::MultiplyI));
        self.var_stack.stack.push(VarStackItem::Reset(INT.size));
    }

    fn instr_divide_i(&mut self) {
        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::DivideI));
        self.var_stack.stack.push(VarStackItem::Reset(INT.size));
    }

    fn instr_modulo_i(&mut self) {
        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::ModuloI));
        self.var_stack.stack.push(VarStackItem::Reset(INT.size));
    }

    fn instr_to_bool(&mut self) {
        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::ToBoolI));
        self.var_stack.stack.push(VarStackItem::Reset(INT.size));
        self.var_stack
            .stack
            .push(VarStackItem::Increment(BOOL.size));
    }

    fn instr_compare_i(&mut self) {
        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::CompareI));
        self.var_stack.stack.push(VarStackItem::Reset(INT.size * 2));
        self.var_stack
            .stack
            .push(VarStackItem::Increment(BOOL.size));
    }

    fn instr_add_string(&mut self) {
        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::AddString));
        self.var_stack.stack.push(VarStackItem::Reset(STRING.size));
    }

    fn instr_push_static(&mut self, index: usize, size: usize, alignment: usize) {
        self.push_alignment(alignment);

        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::PushStatic(
                index, size,
            )));
        self.var_stack.stack.push(VarStackItem::Increment(size));
    }

    fn instr_copy(&mut self, dst: usize, src: usize, size: usize) {
        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::Copy(
                dst, src, size,
            )));
    }

    // align before calling this
    fn instr_cast_int_uint8(&mut self) {
        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::CastIntUint8));

        self.var_stack.stack.push(VarStackItem::Reset(INT.size));
        self.var_stack
            .stack
            .push(VarStackItem::Increment(UINT8.size));
    }

    // align before calling this
    fn instr_cast_uint8_int(&mut self) {
        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::CastUint8Int));

        self.var_stack.stack.push(VarStackItem::Reset(UINT8.size));
        self.var_stack.stack.push(VarStackItem::Increment(INT.size));
    }

    fn instr_cast_slice_ptr(&mut self) {
        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::CastSlicePtr));
    }

    fn instr_cast_int_uint(&mut self) {
        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::CastIntUint));
    }

    fn instr_debug(&mut self) {
        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::Debug));
    }

    fn instr_jump_and_link(&mut self, identifier: String) {
        self.stack_instructions
            .push(CompilerInstruction::JumpAndLink(identifier));
    }

    fn instr_jump_and_link_closure(&mut self) {
        self.stack_instructions.push(CompilerInstruction::Real(
            vm::Instruction::JumpAndLinkClosure,
        ));
        self.var_stack.stack.push(VarStackItem::Reset(PTR_SIZE));
    }

    fn instr_shift(&mut self, size: usize, amount: usize) {
        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::Shift(
                size, amount,
            )));
        self.var_stack.stack.push(VarStackItem::Reset(amount));
    }

    fn instr_libc_write(&mut self) {
        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::LibcWrite));
        self.var_stack
            .stack
            .push(VarStackItem::Reset(INT.size + SLICE_SIZE));
        self.var_stack.stack.push(VarStackItem::Increment(INT.size));
    }

    fn push_alignment(&mut self, alignment: usize) -> usize {
        let alignment = align(alignment, self.var_stack.total_size());
        if alignment != 0 {
            self.instr_increment(alignment);
        }
        alignment
    }

    fn stack_total_size(&self) -> usize {
        self.var_stack.total_size()
    }

    fn push_stack_frame(&mut self) {
        self.var_stack.stack.push_frame();
    }

    fn pop_stack_frame(&mut self) {
        let frame = self.var_stack.stack.pop_frame().unwrap();
        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::Reset(
                CompilerVarStack::size_for(frame.iter()),
            )));
    }

    fn pop_stack_frame_size(&mut self) -> usize {
        let frame = self.var_stack.stack.pop_frame().unwrap();
        let size = CompilerVarStack::size_for(frame.iter());
        frame.into_iter().for_each(|v| self.var_stack.stack.push(v));
        size
    }

    fn init_function_prologue(&mut self, function: &Function) -> Result<()> {
        let mut escaped_variables: Vec<(String, Type)> = Vec::new();
        let mut argument_size: usize = 0;

        for arg in &function.arguments {
            let (escaped, _type) = {
                if let TypeType::Escaped(_type) = arg._type._type.clone() {
                    (true, *_type)
                } else {
                    (false, arg._type.clone())
                }
            };
            let return_type = function.return_type.clone();

            argument_size += _type.size;

            let alignment = align(
                _type.alignment,
                self.var_stack.total_size() + return_type.size,
            );

            if alignment != 0 {
                self.var_stack
                    .stack
                    .push(VarStackItem::Increment(alignment));
            }

            self.var_stack
                .stack
                .push(VarStackItem::Increment(_type.size));
            self.var_mark(arg.clone());

            if escaped {
                escaped_variables.push((arg.identifier.clone(), _type));
            }
        }

        if argument_size < function.return_type.size {
            self.var_stack.stack.push(VarStackItem::Increment(
                function.return_type.size - argument_size,
            ))
        }

        // return address
        if function.identifier != "main" {
            let alignment = align(PTR_SIZE, self.var_stack.total_size());
            if alignment != 0 {
                self.var_stack
                    .stack
                    .push(VarStackItem::Increment(alignment));
            }

            self.var_stack.stack.push(VarStackItem::Increment(PTR_SIZE));
        }

        self.var_stack.set_arg_size();

        // already aligned because of return address
        for arg in &function.closure_arguments {
            assert!(arg._type._type.is_escaped());
            self.var_stack.stack.push(VarStackItem::Increment(PTR_SIZE));
            self.var_mark(arg.clone());
        }

        for (identifier, _type) in escaped_variables {
            let (offset, _) = self.var_get_offset(&identifier).unwrap();
            let alignment = self.push_alignment(PTR_SIZE);
            self.instr_increment(_type.size);
            self.instr_copy(0, offset + _type.size + alignment, _type.size);
            self.instr_alloc(_type.size, _type.alignment);
            self.var_mark(Variable {
                _type: Type::create_escaped(_type),
                identifier,
            });
        }

        Ok(())
    }

    fn init_function_epilogue(&mut self, function: &Function) {
        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::Reset(
                self.var_stack.total_size() - self.var_stack.arg_size.unwrap(),
            )));

        if function.identifier == "main" {
            self.stack_instructions
                .push(CompilerInstruction::Real(vm::Instruction::Exit));
        } else {
            self.stack_instructions
                .push(CompilerInstruction::Real(vm::Instruction::Return));
        }
    }

    fn get_instructions(self) -> Vec<Vec<CompilerInstruction>> {
        self.stack_instructions.instructions
    }
}

#[derive(Debug, Clone, PartialEq)]
enum TypeStructField {
    Type(String, Type),
    Padding(usize),
}

#[derive(Debug, Clone, PartialEq)]
struct TypeStruct {
    fields: Vec<TypeStructField>,
}

impl TypeStruct {
    fn get_field_offset(&self, identifier: &str) -> Option<(usize, &Type)> {
        let mut offset = 0;

        for field in self.fields.iter().rev() {
            match field {
                TypeStructField::Padding(padding) => offset += padding,
                TypeStructField::Type(iden, _type) => {
                    if iden == identifier {
                        return Some((offset, _type));
                    }
                    offset += _type.size;
                }
            }
        }

        None
    }

    fn get_field_offset_err(&self, identifier: &str) -> Result<(usize, &Type)> {
        self.get_field_offset(identifier)
            .ok_or(anyhow!("get_field_offset_err: not found {identifier}"))
    }

    fn identifier_field_count(&self) -> usize {
        // - 1 there is padding at the end
        // / 2 every field has padding before
        (self.fields.len() - 1) / 2
    }
}

lazy_static::lazy_static! {
    static ref NIL: Type = Type {
        id: Some("nil".to_string()),
        size: PTR_SIZE,
        alignment: PTR_SIZE,
        _type: TypeType::Builtin(TypeBuiltin::Nil),
    };
    static ref UINT: Type = Type {
        id: Some("uint".to_string()),
        size: size_of::<usize>(),
        alignment: size_of::<usize>(),
        _type: TypeType::Builtin(TypeBuiltin::Uint),
    };
    static ref UINT8: Type = Type {
        id: Some("uint8".to_string()),
        size: 1,
        alignment: 1,
        _type: TypeType::Builtin(TypeBuiltin::Uint8),
    };
    static ref INT: Type = Type {
        id: Some("int".to_string()),
        size: size_of::<isize>(),
        alignment: size_of::<isize>(),
        _type: TypeType::Builtin(TypeBuiltin::Int),
    };
    static ref BOOL: Type = Type {
        id: Some("bool".to_string()),
        size: size_of::<usize>(),      // for now
        alignment: size_of::<usize>(), // for now
        _type: TypeType::Builtin(TypeBuiltin::Bool),
    };
    static ref STRING: Type = Type {
        id: Some("string".to_string()),
        size: size_of::<usize>(),
        alignment: size_of::<usize>(),
        _type: TypeType::Builtin(TypeBuiltin::String),
    };
    static ref COMPILER_TYPE: Type = Type {
        id: Some("Type".to_string()),
        size: 0,
        alignment: 0,
        _type: TypeType::Builtin(TypeBuiltin::CompilerType),
    };
    static ref VOID: Type = Type {
        id: Some("void".to_string()),
        size: 0,
        alignment: 0,
        _type: TypeType::Builtin(TypeBuiltin::Void),
    };
    static ref PTR: Type = Type {
        id: Some("ptr".to_string()),
        size: PTR_SIZE,
        alignment: PTR_SIZE,
        _type: TypeType::Builtin(TypeBuiltin::Ptr),
    };
}

const SLICE_SIZE: usize = size_of::<usize>();
const PTR_SIZE: usize = size_of::<usize>();

#[derive(Debug, Clone, PartialEq)]
enum TypeBuiltin {
    Uint,
    Uint8,
    Int,
    String,
    Bool,
    Void,
    CompilerType,
    Ptr,
    Nil,
}

#[derive(Debug, Clone, PartialEq)]
struct TypeClosure {
    arguments: Vec<(String, Type)>,
    return_type: Type,
}

#[derive(Debug, Clone, PartialEq)]
enum TypeType {
    Struct(TypeStruct),
    Variadic(Box<Type>),
    Slice(Box<Type>),
    Builtin(TypeBuiltin),
    Address(Box<Type>),
    Lazy(String),
    Closure(Box<TypeClosure>),
    Escaped(Box<Type>),
}

impl TypeType {
    fn closure_err(self) -> Result<TypeClosure> {
        match self {
            Self::Closure(v) => Ok(*v),
            _type => Err(anyhow!("closure_err: got {_type:#?}")),
        }
    }

    fn is_escaped(&self) -> bool {
        match self {
            Self::Escaped(_) => true,
            _ => false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct Type {
    // None for inline types
    id: Option<String>,
    size: usize,
    alignment: usize,
    _type: TypeType,
}

impl Type {
    fn create_variadic(item: Self) -> Self {
        Self {
            id: item.id.as_ref().map(|id| id.clone() + "..."),
            size: SLICE_SIZE,
            alignment: SLICE_SIZE,
            _type: TypeType::Variadic(Box::new(item)),
        }
    }

    fn create_address(item: Self) -> Self {
        Self {
            id: item.id.as_ref().map(|id| id.clone() + "&"),
            size: PTR_SIZE,
            alignment: PTR_SIZE,
            _type: TypeType::Address(Box::new(item)),
        }
    }

    fn create_escaped(item: Self) -> Self {
        Self {
            id: item.id.clone(),
            size: PTR_SIZE,
            alignment: PTR_SIZE,
            _type: TypeType::Escaped(Box::new(item)),
        }
    }

    fn equals(&self, other: &Self) -> Result<()> {
        // todo: do this the other way around
        if let TypeType::Builtin(builtin) = &self._type {
            if let TypeBuiltin::Nil = builtin {
                if let TypeType::Address(_) = &other._type {
                    return Ok(());
                }
            }
        }

        match (&self.id, &other.id) {
            (Some(self_id), Some(other_id)) => {
                return match self_id == other_id {
                    true => Ok(()),
                    false => Err(anyhow!("equals: {self:#?} != {other:#?}")),
                }
            }
            _ => {}
        }

        let mut self_clone = self.clone();
        let mut other_clone = other.clone();

        self_clone.id = None;
        other_clone.id = None;

        match other == self {
            true => Ok(()),
            false => Err(anyhow!("equals: {self:#?} != {other:#?}")),
        }
    }

    fn extract_variadic(&self) -> Option<&Self> {
        match &self._type {
            TypeType::Variadic(item) => Some(&item),
            _ => None,
        }
    }

    fn resolve_lazy(self, type_resolver: &TypeResolver) -> Result<Self> {
        match self._type {
            TypeType::Lazy(alias) => type_resolver.resolve(&ast::Type::Alias(alias)),
            _ => Ok(self),
        }
    }
}

#[derive(Debug)]
pub struct TypeResolver {
    type_declarations: HashMap<String, ast::Type>,
}

impl TypeResolver {
    pub fn new(type_declarations: HashMap<String, ast::Type>) -> Self {
        Self { type_declarations }
    }

    fn resolve(&self, _type: &ast::Type) -> Result<Type> {
        self.resolve_with_alias(_type, None)
    }

    fn resolve_with_alias(&self, _type: &ast::Type, alias: Option<&str>) -> Result<Type> {
        match _type {
            ast::Type::Alias(inner_alias) => {
                match inner_alias.as_str() {
                    "uint" => return Ok(UINT.clone()),
                    "uint8" => return Ok(UINT8.clone()),
                    "int" => return Ok(INT.clone()),
                    "bool" => return Ok(BOOL.clone()),
                    "string" => return Ok(STRING.clone()),
                    "Type" => return Ok(COMPILER_TYPE.clone()),
                    "void" => return Ok(VOID.clone()),
                    "ptr" => return Ok(PTR.clone()),
                    _ => {}
                };

                let inner = self
                    .type_declarations
                    .get(inner_alias)
                    .ok_or(anyhow!("can't resolve {inner_alias:#?}"))?;

                if let Some(alias) = alias {
                    if alias == inner_alias {
                        match inner {
                            ast::Type::Struct(_) => {}
                            _ => panic!("recursive non struct?"),
                        }

                        return Ok(Type {
                            id: Some(alias.to_string() + "{}"),
                            size: 0,
                            alignment: 0,
                            _type: TypeType::Lazy(alias.to_string()),
                        });
                    }
                }

                self.resolve_with_alias(&inner, Some(inner_alias))
            }
            ast::Type::Slice(_type) => {
                let nested = self.resolve_with_alias(_type, alias)?;
                Ok(Type {
                    id: nested.id.as_ref().map(|id| id.clone() + "[]"),
                    size: SLICE_SIZE,
                    alignment: SLICE_SIZE,
                    _type: TypeType::Slice(Box::new(nested)),
                })
            }
            ast::Type::Variadic(_type) => {
                let nested = self.resolve_with_alias(_type, alias)?;
                Ok(Type {
                    id: nested.id.as_ref().map(|id| id.clone() + "..."),
                    size: size_of::<usize>(),
                    alignment: size_of::<usize>(),
                    _type: TypeType::Variadic(Box::new(nested)),
                })
            }
            ast::Type::Struct(type_struct) => {
                let mut fields: Vec<TypeStructField> = Vec::new();
                let mut size: usize = 0;
                let mut highest_alignment: usize = 0;

                for var in &type_struct.fields {
                    let resolved = self.resolve_with_alias(&var._type, alias)?;
                    if resolved.alignment > highest_alignment {
                        highest_alignment = resolved.alignment;
                    }

                    let alignment = align(resolved.alignment, size);
                    size += resolved.size;
                    size += alignment;
                    fields.push(TypeStructField::Padding(alignment));
                    fields.push(TypeStructField::Type(var.identifier.clone(), resolved));
                }

                let end_padding = align(highest_alignment, size);
                size += end_padding;
                fields.push(TypeStructField::Padding(end_padding));

                Ok(Type {
                    id: alias.map(|id| id.to_string() + "{}"),
                    size,
                    alignment: highest_alignment,
                    _type: TypeType::Struct(TypeStruct { fields }),
                })
            }
            ast::Type::Address(_type) => {
                let nested = self.resolve_with_alias(_type, alias)?;
                Ok(Type {
                    id: nested.id.as_ref().map(|id| id.clone() + "&"),
                    size: PTR_SIZE,
                    alignment: PTR_SIZE,
                    _type: TypeType::Address(Box::new(nested)),
                })
            }
            ast::Type::Closure(type_closure) => {
                let mut arguments = Vec::new();

                let mut id = String::from("fn (");
                for var in &type_closure.arguments {
                    let resolved = self.resolve_with_alias(&var._type, alias)?;
                    id.push_str(
                        resolved
                            .id
                            .as_ref()
                            .ok_or(anyhow!("resolve_closure: type without id"))?,
                    );
                    id.push(',');
                    arguments.push((var.identifier.clone(), resolved));
                }
                id.push(')');

                let resolved_return_type =
                    self.resolve_with_alias(&type_closure.return_type, alias)?;

                id.push_str(
                    resolved_return_type
                        .id
                        .as_ref()
                        .ok_or(anyhow!("resolve_closure: return type without id"))?,
                );

                Ok(Type {
                    id: Some(id),
                    size: PTR_SIZE,
                    alignment: PTR_SIZE,
                    _type: TypeType::Closure(Box::new(TypeClosure {
                        arguments,
                        return_type: resolved_return_type,
                    })),
                })
            }
        }
    }
}

struct FunctionCall {
    // None means closure call
    declaration: Option<ast::FunctionDeclaration>,
    call: ast::Call,
}

struct TypeCast {
    expression: ast::Expression,
    _type: Type,
}

#[derive(Debug, Clone)]
struct Variable {
    _type: Type,
    identifier: String,
}

#[derive(Debug)]
enum DotAccessField {
    // offset from the stack
    Stack(usize, Type),
    // offset address is on top of the stack
    Heap(Type),
}

impl DotAccessField {
    fn _type(&self) -> &Type {
        match self {
            Self::Stack(_, _type) => _type,
            Self::Heap(_type) => _type,
        }
    }
}

pub struct Function {
    identifier: String,
    arguments: Vec<Variable>,
    closure_arguments: Vec<Variable>,
    return_type: Type,
    body: Vec<ast::Node>,
}

impl Function {
    pub fn from_declaration(
        type_resolver: &TypeResolver,
        declaration: &ast::FunctionDeclaration,
    ) -> Result<Self> {
        Ok(Self {
            identifier: declaration.identifier.clone(),
            body: declaration.body.clone(),
            return_type: type_resolver.resolve(&declaration.return_type)?,
            closure_arguments: Vec::new(),
            arguments: declaration
                .arguments
                .iter()
                .map(|v| {
                    let escaped =
                        DoesVariableEscape::new(&v.identifier).search(declaration.body.iter());

                    let mut _type = type_resolver.resolve(&v._type)?;
                    if escaped {
                        _type = Type::create_escaped(_type);
                    }

                    Ok(Variable {
                        _type,
                        identifier: v.identifier.clone(),
                    })
                })
                .collect::<Result<_, anyhow::Error>>()?,
        })
    }

    fn from_closure(
        type_resolver: &TypeResolver,
        closure: &ast::Closure,
        closure_arguments: Vec<Variable>,
    ) -> Result<Self> {
        let closure_type = type_resolver
            .resolve(&closure._type)?
            ._type
            .closure_err()
            .unwrap();

        Ok(Self {
            identifier: "".to_string(), // todo??? what to do
            body: closure.body.clone(),
            return_type: closure_type.return_type.clone(),
            closure_arguments,
            arguments: closure_type
                .arguments
                .iter()
                .map(|(identifier, _type)| {
                    let escaped = DoesVariableEscape::new(identifier).search(closure.body.iter());

                    let mut _type = _type.clone();
                    if escaped {
                        _type = Type::create_escaped(_type);
                    }

                    Variable {
                        _type,
                        identifier: identifier.clone(),
                    }
                })
                .collect(),
        })
    }
}

struct ExpressionCompiler<'a> {
    instructions: &'a mut Instructions,
    closures: &'a mut Vec<CompiledInstructions>,
    type_resolver: Rc<TypeResolver>,
    function_declarations: Rc<HashMap<String, ast::FunctionDeclaration>>,
    static_memory: Rc<RefCell<vm::StaticMemory>>,
}

impl<'a> ExpressionCompiler<'a> {
    fn new(
        instructions: &'a mut Instructions,
        closures: &'a mut Vec<CompiledInstructions>,
        type_resolver: Rc<TypeResolver>,
        function_declarations: Rc<HashMap<String, ast::FunctionDeclaration>>,
        static_memory: Rc<RefCell<vm::StaticMemory>>,
    ) -> Self {
        Self {
            instructions,
            closures,
            type_resolver,
            function_declarations,
            static_memory,
        }
    }

    fn compile_nil(&mut self) -> Result<Type> {
        self.instructions.instr_push_i(0)?;
        Ok(NIL.clone())
    }

    fn resolve_variable(&self, _type: &ast::Type) -> Result<(usize, Variable)> {
        let ast::Type::Alias(identifier) = _type else {
            return Err(anyhow!("resolve_variable: cant resolve non alias"));
        };

        let variable = self
            .instructions
            .var_get_offset(identifier)
            .ok_or(anyhow!("resolve_variable: variable {identifier} not found"))?;

        Ok((variable.0, variable.1.clone()))
    }

    fn resolve_type(&self, _type: &ast::Type) -> Result<Type> {
        self.type_resolver.resolve(_type)
    }

    fn compile_closure(&mut self, closure: &ast::Closure) -> Result<Type> {
        self.instructions.push_alignment(PTR_SIZE);

        // right now all nested closures capture everything inside them
        // todo: find solution for this
        //
        // let foo = 20
        // fn() void { <- this closure does not need to capture "foo" (BUT IT DOES)
        //  fn() void { <- only this closure has to capture "foo"
        //    foo = 50
        //  }
        // }
        //
        // is it even possible to fix this? closures are created lazily right now, so outer HAS to
        // have inner values for it to copy them from the stack into nested closure
        let escaped_variables = ClosureEscapedVariables::new(closure)
            .search()
            .into_iter()
            .map(|v| {
                let (offset, variable) = self.resolve_variable(&ast::Type::Alias(v))?;
                assert!(
                    variable._type._type.is_escaped(),
                    "found non escaped variable when compiling closure",
                );

                self.instructions.instr_increment(PTR_SIZE);
                self.instructions.instr_copy(0, PTR_SIZE + offset, PTR_SIZE);

                Ok(variable)
            })
            .collect::<Result<Vec<_>, anyhow::Error>>()?;

        self.instructions
            .instr_push_closure(escaped_variables.len(), self.closures.len());

        let closure_function =
            Function::from_closure(&self.type_resolver, &closure, escaped_variables).unwrap();

        self.closures.push(
            FunctionCompiler::new(
                closure_function,
                self.instructions.static_var_stack.clone(),
                self.static_memory.clone(),
                self.type_resolver.clone(),
                self.function_declarations.clone(),
            )
            .compile()?,
        );

        Ok(self.resolve_type(&closure._type)?)
    }

    fn compile_variable(&mut self, mut offset: usize, variable: &Variable) -> Result<Type> {
        if let TypeType::Escaped(_type) = &variable._type._type {
            // this will leak alignment
            let alignment = self.instructions.push_alignment(PTR_SIZE);
            self.instructions.instr_increment(PTR_SIZE);
            self.instructions
                .instr_copy(0, offset + alignment + PTR_SIZE, PTR_SIZE);
            self.instructions.instr_deref(_type.size);

            return Ok(*_type.clone());
        }

        offset += self.instructions.push_alignment(variable._type.alignment);
        self.instructions.instr_increment(variable._type.size);
        self.instructions
            .instr_copy(0, offset + variable._type.size, variable._type.size);

        Ok(variable._type.clone())
    }

    fn compile_dot_access_field_offset_base_heap(
        &mut self,
        dot_access: &ast::DotAccess,
        type_struct: &TypeStruct,
        offset: usize,
    ) -> Result<Type> {
        let alignment = self.instructions.push_alignment(PTR_SIZE);
        self.instructions.instr_increment(PTR_SIZE);
        self.instructions
            .instr_copy(0, alignment + PTR_SIZE + offset, PTR_SIZE);

        let (field_offset, field_type) =
            type_struct.get_field_offset_err(&dot_access.identifier)?;

        self.instructions.instr_offset(field_offset);

        Ok(field_type.clone())
    }

    fn compile_dot_access_field_offset(
        &mut self,
        dot_access: &ast::DotAccess,
    ) -> Result<DotAccessField> {
        if let ast::Expression::DotAccess(inner) = &dot_access.expression {
            let target_field = self.compile_dot_access_field_offset(inner)?;
            let target_type = target_field._type();

            match &target_field {
                DotAccessField::Heap(_type) => match &target_type._type {
                    // target heap -> current stack = offset address
                    TypeType::Struct(type_struct) => {
                        let (offset, field_type) =
                            type_struct.get_field_offset_err(&dot_access.identifier)?;
                        self.instructions.instr_offset(offset);

                        return Ok(DotAccessField::Heap(field_type.clone()));
                    }
                    // target heap -> current heap = dereference + offset
                    TypeType::Address(address_type) => {
                        let address_type =
                            address_type.clone().resolve_lazy(&self.type_resolver)?;

                        let TypeType::Struct(type_struct) = &address_type._type else {
                            return Err(anyhow!("cant dot access non struct type"));
                        };

                        let (offset, field_type) =
                            type_struct.get_field_offset_err(&dot_access.identifier)?;

                        self.instructions.instr_deref(PTR_SIZE);
                        self.instructions.instr_offset(offset);

                        return Ok(DotAccessField::Heap(field_type.clone()));
                    }
                    _type => {
                        return Err(anyhow!("dot access on non struct/address type {_type:#?}"));
                    }
                },
                DotAccessField::Stack(stack_offset, _type) => match &target_type._type {
                    // target stack -> current stack = offset stack
                    TypeType::Struct(type_struct) => {
                        let (offset, field_type) =
                            type_struct.get_field_offset_err(&dot_access.identifier)?;

                        return Ok(DotAccessField::Stack(
                            *stack_offset + offset,
                            field_type.clone(),
                        ));
                    }
                    // target stack -> current heap = offset
                    TypeType::Address(address_type) => {
                        let alignment = self.instructions.push_alignment(PTR_SIZE);
                        self.instructions.instr_increment(PTR_SIZE);
                        self.instructions.instr_copy(
                            0,
                            alignment + PTR_SIZE + *stack_offset,
                            PTR_SIZE,
                        );

                        let TypeType::Struct(type_struct) = &address_type._type else {
                            return Err(anyhow!("dot access on non struct/address type"));
                        };

                        let (field_offset, field_type) =
                            type_struct.get_field_offset_err(&dot_access.identifier)?;
                        self.instructions.instr_offset(field_offset);

                        return Ok(DotAccessField::Heap(field_type.clone()));
                    }
                    _type => {
                        return Err(anyhow!("dot access on non struct/address type {_type:#?}"));
                    }
                },
            }
        }

        let ast::Expression::Type(_type) = &dot_access.expression else {
            return Err(anyhow!("cant dot access non Type"));
        };
        let (offset, variable) = self.resolve_variable(_type)?;

        if let TypeType::Escaped(_type) = variable._type._type {
            match _type._type {
                TypeType::Struct(type_struct) => Ok(DotAccessField::Heap(
                    self.compile_dot_access_field_offset_base_heap(
                        dot_access,
                        &type_struct,
                        offset,
                    )?,
                )),
                TypeType::Address(_type) => match _type._type {
                    TypeType::Struct(type_struct) => {
                        let alignment = self.instructions.push_alignment(PTR_SIZE);
                        self.instructions.instr_increment(PTR_SIZE);
                        self.instructions
                            .instr_copy(0, offset + PTR_SIZE + alignment, PTR_SIZE);
                        self.instructions.instr_deref(PTR_SIZE);

                        Ok(DotAccessField::Heap(
                            self.compile_dot_access_field_offset_base_heap(
                                dot_access,
                                &type_struct,
                                0,
                            )?,
                        ))
                    }
                    _type => Err(anyhow!("cant access non struct type: {_type:#?}")),
                },
                _type => Err(anyhow!("cant access non struct/address type: {_type:#?}")),
            }
        } else {
            match variable._type._type {
                TypeType::Struct(type_struct) => {
                    let (field_offset, field_type) =
                        type_struct.get_field_offset_err(&dot_access.identifier)?;

                    Ok(DotAccessField::Stack(
                        offset + field_offset,
                        field_type.clone(),
                    ))
                }
                TypeType::Address(_type) => match _type._type {
                    TypeType::Struct(type_struct) => Ok(DotAccessField::Heap(
                        self.compile_dot_access_field_offset_base_heap(
                            dot_access,
                            &type_struct,
                            offset,
                        )?,
                    )),
                    _type => Err(anyhow!("cant access non struct type: {_type:#?}")),
                },
                _type => Err(anyhow!("cant access non struct/address type: {_type:#?}")),
            }
        }
    }

    fn compile_type(&mut self, _type: &ast::Type) -> Result<Type> {
        let (offset, variable) = self.resolve_variable(_type)?;
        self.compile_variable(offset, &variable)
    }

    fn compile_address(&mut self, expression: &ast::Expression) -> Result<Type> {
        match expression {
            ast::Expression::Type(_type) => {
                let (offset, var) = self.resolve_variable(_type)?;

                let TypeType::Escaped(_type) = var._type._type else {
                    return Err(anyhow!("compile_address: rn all variables are escaped"));
                };

                let alignment = self.instructions.push_alignment(PTR_SIZE);
                self.instructions.instr_increment(PTR_SIZE);
                self.instructions
                    .instr_copy(0, offset + PTR_SIZE + alignment, PTR_SIZE);

                Ok(Type::create_address(*_type))
            }
            ast::Expression::DotAccess(dot_access) => {
                let field = self.compile_dot_access_field_offset(dot_access)?;
                let DotAccessField::Heap(_type) = field else {
                    return Err(anyhow!(
                        "compile_address: cant take non heap address dot access"
                    ));
                };

                Ok(Type::create_address(_type))
            }
            ast::Expression::Address(_) => {
                Err(anyhow!("compile_address: cant take address of this"))
            }
            expression => {
                self.instructions.push_alignment(PTR_SIZE);
                let exp = self.compile_expression(expression)?;
                self.instructions.instr_alloc(exp.size, exp.alignment);

                Ok(Type::create_address(exp))
            }
        }
    }

    fn compile_deref(&mut self, expression: &ast::Expression) -> Result<Type> {
        let exp = self.compile_expression(expression)?;
        let TypeType::Address(_type) = exp._type else {
            return Err(anyhow!(
                "compile_deref: can't dereference non address types"
            ));
        };

        self.instructions.instr_deref(exp.size);

        Ok(*_type)
    }

    fn compile_dot_access(&mut self, dot_access: &ast::DotAccess) -> Result<Type> {
        let field = self.compile_dot_access_field_offset(dot_access)?;

        match field {
            DotAccessField::Heap(_type) => {
                // already aligned on 8 because of address
                self.instructions.instr_deref(_type.size);

                Ok(_type)
            }
            DotAccessField::Stack(offset, _type) => {
                let alignment = self.instructions.push_alignment(_type.alignment);

                self.instructions.instr_increment(_type.size);
                self.instructions
                    .instr_copy(0, offset + _type.size + alignment, _type.size);

                Ok(_type)
            }
        }
    }

    fn compile_struct_init(&mut self, struct_init: &ast::StructInit) -> Result<Type> {
        let resolved_type = self.resolve_type(&struct_init._type)?;

        let TypeType::Struct(type_struct) = &resolved_type._type else {
            panic!("compile_struct_init: incorrect type wrong ast parser");
        };

        if type_struct.identifier_field_count() != struct_init.fields.len() {
            return Err(anyhow!(
                "compile_struct_init: field initialization count mismatch"
            ));
        }

        for field in &type_struct.fields {
            match field {
                TypeStructField::Padding(padding) => {
                    self.instructions.instr_increment(*padding);
                }
                TypeStructField::Type(identifier, _type) => {
                    let exp = struct_init.fields.get(identifier).ok_or(anyhow!(
                        "compile_struct_init: initialization field not found"
                    ))?;

                    let exp_type = self.compile_expression(exp)?;
                    exp_type.equals(_type)?;
                }
            }
        }

        Ok(resolved_type)
    }

    fn compile_spread(&mut self, expression: &ast::Expression) -> Result<Type> {
        let exp = self.compile_expression(expression)?;

        let TypeType::Slice(slice_item) = exp._type else {
            return Err(anyhow!("compile_spread: can only spread slice types"));
        };

        Ok(Type::create_variadic(*slice_item))
    }

    fn compile_negate(&mut self, negate: &ast::Expression) -> Result<Type> {
        let exp_bool = self.compile_expression(negate)?;
        if exp_bool != *BOOL {
            return Err(anyhow!("can only negate bools"));
        }

        self.instructions.instr_negate_bool();

        Ok(BOOL.clone())
    }

    fn compile_expression_index(&mut self, index: &ast::Index) -> Result<Type> {
        let exp_var = self.compile_expression(&index.var)?;

        let TypeType::Slice(expected_type) = exp_var._type else {
            return Err(anyhow!("can't index this type"));
        };

        let exp_index = self.compile_expression(&index.expression)?;
        if exp_index != *INT {
            return Err(anyhow!("cant index with {exp_index:#?}"));
        }

        self.instructions.instr_slice_index_get(expected_type.size);

        Ok(*expected_type)
    }

    fn compile_slice_init(&mut self, slice_init: &ast::SliceInit) -> Result<Type> {
        let resolved_type = self.resolve_type(&slice_init._type)?;

        let TypeType::Slice(slice_item) = &resolved_type._type else {
            panic!("compile_slice_init: incorrect type, wrong ast parser");
        };

        self.instructions.instr_push_slice();

        for v in &slice_init.expressions {
            self.instructions.push_stack_frame();

            self.instructions.instr_increment(SLICE_SIZE);
            self.instructions.instr_copy(0, SLICE_SIZE, SLICE_SIZE);

            let exp = self.compile_expression(v)?;
            if exp != **slice_item {
                return Err(anyhow!("compile_slice_init: slice item type mismatch"));
            }

            self.instructions.instr_slice_append(exp.size);

            self.instructions.pop_stack_frame();
        }

        Ok(resolved_type)
    }

    fn compile_infix(&mut self, infix: &ast::Infix) -> Result<Type> {
        let exp = self.compile_expression(&infix.expression)?;
        match infix._type {
            ast::InfixType::Plus => {}
            ast::InfixType::Minus => {
                self.instructions.instr_minus_int();
            }
        }
        Ok(exp)
    }

    fn compile_compare(&mut self, compare: &ast::Compare) -> Result<Type> {
        let a: Type;
        let b: Type;

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
        if a != *BOOL && a != *INT {
            return Err(anyhow!("can only compare int/bool"));
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

        Ok(BOOL.clone())
    }

    fn compile_type_init(&mut self, type_init: &ast::TypeInit) -> Result<Type> {
        match self.resolve_type(&type_init._type)?._type {
            TypeType::Slice(_) => self.compile_slice_init(&ast::SliceInit {
                _type: type_init._type.clone(),
                expressions: Vec::new(),
            }),
            TypeType::Struct(_) => self.compile_struct_init(&ast::StructInit {
                _type: type_init._type.clone(),
                fields: HashMap::new(),
            }),
            _ => Err(anyhow!("compile_type_init: cant compile this type")),
        }
    }

    fn resolve_type_cast(&self, call: &ast::Call) -> Result<TypeCast> {
        let ast::Expression::Type(_type) = &call.expression else {
            return Err(anyhow!(
                "resolve_type_cast: cant type cast non Type expression"
            ));
        };

        if call.arguments.len() != 1 {
            return Err(anyhow!("resolve_type_cast: 1 argument required"));
        }

        let exp = call.arguments[0].clone();
        let _type = self.resolve_type(_type)?;

        Ok(TypeCast {
            expression: exp,
            _type,
        })
    }

    fn compile_type_cast(&mut self, type_cast: &TypeCast) -> Result<Type> {
        self.instructions.push_alignment(type_cast._type.alignment);

        let target = self.compile_expression(&type_cast.expression)?;

        match target._type {
            TypeType::Builtin(builtin) => match builtin {
                TypeBuiltin::Int => match &type_cast._type._type {
                    TypeType::Builtin(builtin_dest) => match builtin_dest {
                        TypeBuiltin::Uint8 => {
                            self.instructions.instr_cast_int_uint8();
                        }
                        TypeBuiltin::Uint => {
                            self.instructions.instr_cast_int_uint();
                        }
                        _ => return Err(anyhow!("compile_type_cast: cant cast")),
                    },
                    _ => return Err(anyhow!("compile_type_cast: cant cast")),
                },
                TypeBuiltin::Uint8 => match &type_cast._type._type {
                    TypeType::Builtin(builtin_dest) => match builtin_dest {
                        TypeBuiltin::Int => {
                            self.instructions.instr_cast_uint8_int();
                        }
                        _ => return Err(anyhow!("compile_type_cast: cant cast")),
                    },
                    _ => return Err(anyhow!("compile_type_cast: cant cast")),
                },
                TypeBuiltin::Ptr => match &type_cast._type._type {
                    TypeType::Builtin(builtin_dest) => match builtin_dest {
                        TypeBuiltin::Uint => {}
                        _ => return Err(anyhow!("compile_type_cast: cant cast")),
                    },
                    _ => return Err(anyhow!("compile_type_cast: cant cast")),
                },
                TypeBuiltin::String => match &type_cast._type._type {
                    TypeType::Slice(item) => {
                        if **item != *UINT8 {
                            return Err(anyhow!(
                                "compile_type_cast: can only cast string to uint8[]"
                            ));
                        }
                    }
                    _ => return Err(anyhow!("compile_type_cast: cant cast")),
                },
                _ => return Err(anyhow!("compile_type_cast: cant cast")),
            },
            TypeType::Slice(item_target) => match &type_cast._type._type {
                TypeType::Builtin(builtin_dest) => match builtin_dest {
                    TypeBuiltin::String => {
                        if *item_target != *UINT8 {
                            return Err(anyhow!(
                                "compile_type_cast: can only cast string to uint8[]"
                            ));
                        }
                    }
                    TypeBuiltin::Ptr => {
                        self.instructions.instr_cast_slice_ptr();
                    }
                    _ => return Err(anyhow!("compile_type_cast: cant cast")),
                },
                _ => return Err(anyhow!("compile_type_cast: cant cast")),
            },
            TypeType::Variadic(item_target) => match &type_cast._type._type {
                TypeType::Slice(item_dest) => {
                    if item_target != *item_dest {
                        return Err(anyhow!(
                            "compile_type_cast: cant cast variadic into slice different types"
                        ));
                    }
                }
                _ => return Err(anyhow!("compile_type_cast: cant cast")),
            },
            _ => return Err(anyhow!("compile_type_cast: cant cast")),
        }

        Ok(type_cast._type.clone())
    }

    fn resolve_expression(&mut self, expression: &ast::Expression) -> Result<Type> {
        ExpressionCompiler::new(
            &mut self.instructions.clone(),
            &mut self.closures.clone(),
            self.type_resolver.clone(),
            self.function_declarations.clone(),
            self.static_memory.clone(),
        )
        .compile_expression(expression)
    }

    fn check_function_call_argument_count(&mut self, call: &FunctionCall) -> Result<()> {
        let arguments = match &call.declaration {
            Some(declaration) => declaration
                .arguments
                .iter()
                .map(|v| self.resolve_type(&v._type))
                .collect::<Result<Vec<_>, anyhow::Error>>()?,
            None => self
                .resolve_expression(&call.call.expression)?
                ._type
                .closure_err()?
                .arguments
                .into_iter()
                .map(|v| v.1)
                .collect(),
        };

        // todo: refactor this arguments check somehow
        if let Some(last) = arguments.last() {
            if let TypeType::Variadic(_type) = &last._type {
                // -1 because variadic can be empty
                if call.call.arguments.len() < arguments.len() - 1 {
                    return Err(anyhow!(
                        "compile_function_call: variadic argument count mismatch"
                    ));
                }
            } else {
                if call.call.arguments.len() != arguments.len() {
                    return Err(anyhow!("compile_function_call: argument count mismatch"));
                }
            }
        } else {
            if call.call.arguments.len() != arguments.len() {
                return Err(anyhow!("compile_function_call: argument count mismatch"));
            }
        }

        Ok(())
    }

    // append(slice Type, value Type) void
    fn compile_function_builtin_append(&mut self, call: &FunctionCall) -> Result<Type> {
        let slice_arg = call
            .call
            .arguments
            .get(0)
            .ok_or(anyhow!("append: expected first argument"))?;
        let value_arg = call
            .call
            .arguments
            .get(1)
            .ok_or(anyhow!("append: expected second argument"))?;

        // cleanup align here?
        let slice_exp = self.compile_expression(slice_arg)?;
        let value_exp = self.compile_expression(value_arg)?;

        let TypeType::Slice(slice_item) = &slice_exp._type else {
            return Err(anyhow!("append: provide a slice as the first argument"));
        };

        if **slice_item != value_exp {
            return Err(anyhow!("append: value type does not match slice type"));
        }

        self.instructions.instr_slice_append(value_exp.size);

        Ok(VOID.clone())
    }

    // len(slice Type) int
    fn compile_function_builtin_len(&mut self, call: &FunctionCall) -> Result<Type> {
        let slice_arg = call
            .call
            .arguments
            .get(0)
            .ok_or(anyhow!("len: expected first argument"))?;

        // cleanup align here?
        let slice_exp = self.compile_expression(slice_arg)?;
        let TypeType::Slice(_) = &slice_exp._type else {
            return Err(anyhow!(
                "len: expected slice as the argument, got {:#?}",
                slice_exp._type
            ));
        };

        self.instructions.instr_slice_len();

        Ok(INT.clone())
    }

    // new(_ Type, args Type...) Type
    fn compile_function_builtin_new(&mut self, call: &FunctionCall) -> Result<Type> {
        let type_arg = call
            .call
            .arguments
            .get(0)
            .ok_or(anyhow!("new: expected first argument"))?;

        let ast::Expression::Type(_type) = type_arg else {
            return Err(anyhow!("new: expected first argument to be type"));
        };
        let _type = self.resolve_type(_type)?;

        match &_type._type {
            TypeType::Slice(slice_item) => {
                let def_val = call
                    .call
                    .arguments
                    .get(1)
                    .ok_or(anyhow!("new: second argument expected"))?;

                let len_val = call
                    .call
                    .arguments
                    .get(2)
                    .ok_or(anyhow!("new: third argument expected"))?;

                let len_exp = self.compile_expression(len_val)?;
                if len_exp != *INT {
                    return Err(anyhow!("new: length should be of type int"));
                }

                let def_exp = self.compile_expression(def_val)?;
                if def_exp != **slice_item {
                    return Err(anyhow!("new: expression does not match slice type"));
                }

                self.instructions.instr_push_slice_new_len(def_exp.size);

                Ok(_type.clone())
            }
            _type => return Err(anyhow!("new: {_type:#?} not supported")),
        }
    }

    fn compile_function_call(&mut self, call: &FunctionCall) -> Result<Type> {
        self.check_function_call_argument_count(call)?;

        if let Some(declaration) = &call.declaration {
            match declaration.identifier.as_str() {
                "append" => return self.compile_function_builtin_append(call),
                "len" => return self.compile_function_builtin_len(call),
                "new" => return self.compile_function_builtin_new(call),
                _ => {}
            }
        }

        let (return_alignment, expected_arguments): (usize, Vec<Type>) = match &call.declaration {
            Some(declaration) => (
                self.resolve_type(&declaration.return_type)?.alignment,
                declaration
                    .arguments
                    .iter()
                    .map(|v| self.resolve_type(&v._type))
                    .collect::<Result<_, anyhow::Error>>()?,
            ),
            None => (
                self.resolve_expression(&call.call.expression)?
                    ._type
                    .closure_err()?
                    .return_type
                    .alignment,
                self.resolve_expression(&call.call.expression)?
                    ._type
                    .closure_err()?
                    .arguments
                    .into_iter()
                    .map(|v| v.1)
                    .collect(),
            ),
        };

        self.instructions.push_alignment(return_alignment);

        let argument_size = {
            self.instructions.push_stack_frame();
            for (i, expected_type) in expected_arguments.iter().enumerate() {
                let arg = call.call.arguments.get(i);

                let Some(arg) = arg else {
                    if expected_type.extract_variadic().is_some() {
                        self.instructions.instr_push_slice();
                        continue;
                    }
                    return Err(anyhow!("compile_function_call: argument missing"));
                };

                let Some(inner) = expected_type.extract_variadic() else {
                    let exp = self.compile_expression(arg)?;
                    if exp != *expected_type {
                        return Err(anyhow!("function call type mismatch"));
                    }
                    continue;
                };

                if let ast::Expression::Spread(_) = arg {
                    let exp = self.compile_expression(arg)?;
                    if exp != *expected_type {
                        return Err(anyhow!("function call type mismatch"));
                    }
                    // this check is weird, because ast actually does not allow creating
                    // declarations where spread is not last, but this probably should be in
                    // compiler
                    if expected_arguments.len() != call.call.arguments.len() {
                        return Err(anyhow!("spread must be last argument"));
                    }

                    continue;
                }

                self.instructions.instr_push_slice();
                for arg in call.call.arguments.iter().skip(i) {
                    self.instructions.instr_increment(SLICE_SIZE);
                    self.instructions.instr_copy(0, SLICE_SIZE, SLICE_SIZE);

                    let value_exp = self.compile_expression(arg)?;
                    if value_exp != *inner {
                        return Err(anyhow!("variadic argument type mismatch"));
                    }

                    self.instructions.instr_slice_append(value_exp.size);
                }
            }
            self.instructions.pop_stack_frame_size()
        };

        if let Some(declaration) = &call.declaration {
            match declaration.identifier.as_str() {
                "libc_write" => {
                    self.instructions.instr_libc_write();
                    return Ok(INT.clone());
                }
                _ => {}
            }
        }

        let return_type = match &call.declaration {
            Some(declaration) => self.resolve_type(&declaration.return_type)?,
            None => {
                self.resolve_expression(&call.call.expression)?
                    ._type
                    .closure_err()?
                    .return_type
            }
        };
        let return_size = return_type.size;

        let mut reset_size: usize;
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

        // aligning for pushing of return address
        reset_size += self.instructions.push_alignment(PTR_SIZE);

        match &call.declaration {
            Some(declaration) => {
                self.instructions
                    .instr_jump_and_link(declaration.identifier.clone());
            }
            None => {
                self.compile_expression(&call.call.expression)?;
                self.instructions.instr_jump_and_link_closure();
            }
        }

        self.instructions.instr_reset(reset_size);

        Ok(return_type)
    }

    fn resolve_function_call(&self, call: &ast::Call) -> FunctionCall {
        if let ast::Expression::Type(_type) = &call.expression {
            if let ast::Type::Alias(identifier) = _type {
                return FunctionCall {
                    call: call.clone(),
                    declaration: self
                        .function_declarations
                        .get(identifier)
                        .map(|v| v.clone()),
                };
            }
        };

        FunctionCall {
            call: call.clone(),
            declaration: None,
        }
    }

    fn compile_call(&mut self, call: &ast::Call) -> Result<Type> {
        if let Ok(v) = self.resolve_type_cast(call) {
            return self.compile_type_cast(&v);
        }

        self.compile_function_call(&self.resolve_function_call(call))
    }

    fn compile_arithmetic(&mut self, arithmetic: &ast::Arithmetic) -> Result<Type> {
        let a = self.compile_expression(&arithmetic.left)?;
        let b = self.compile_expression(&arithmetic.right)?;

        if a != b {
            return Err(anyhow!("can't add different types"));
        }
        if a == *VOID {
            return Err(anyhow!("can't add void type"));
        }

        match arithmetic._type {
            ast::ArithmeticType::Minus => {
                if *INT == a {
                    self.instructions.instr_minus_int();
                    self.instructions.instr_add_i();
                } else {
                    return Err(anyhow!("can only minus int"));
                }
            }
            ast::ArithmeticType::Plus => {
                if *INT == a {
                    self.instructions.instr_add_i();
                } else if a == *STRING {
                    self.instructions.instr_add_string();
                } else {
                    return Err(anyhow!("can only plus int and string"));
                }
            }
            ast::ArithmeticType::Multiply => {
                if *INT == a {
                    self.instructions.instr_multiply_i();
                } else {
                    return Err(anyhow!("can only multiply int"));
                }
            }
            ast::ArithmeticType::Divide => {
                if *INT == a {
                    self.instructions.instr_divide_i();
                } else {
                    return Err(anyhow!("can only divide int"));
                }
            }
            ast::ArithmeticType::Modulo => {
                if *INT == a {
                    self.instructions.instr_modulo_i();
                } else {
                    return Err(anyhow!("can only modulo int"));
                }
            }
        }

        Ok(a)
    }

    fn compile_literal(&mut self, literal: &ast::Literal) -> Result<Type> {
        let literal_type = match literal.literal {
            lexer::Literal::Int(_) => INT.clone(),
            lexer::Literal::Bool(_) => BOOL.clone(),
            lexer::Literal::String(_) => STRING.clone(),
        };

        match &literal.literal {
            lexer::Literal::Int(int) => {
                if literal_type == *UINT8 {
                    self.instructions.instr_push_u8(*int)?;
                } else if literal_type == *INT {
                    self.instructions.instr_push_i(*int)?;
                } else {
                    return Err(anyhow!("can't cast int to {literal_type:#?}"));
                }
            }
            lexer::Literal::Bool(bool) => {
                if literal_type == *BOOL {
                    self.instructions.instr_push_i({
                        if *bool {
                            1
                        } else {
                            0
                        }
                    })?;
                } else {
                    return Err(anyhow!("can't cast bool to {literal_type:#?}"));
                }
            }
            lexer::Literal::String(string) => {
                let index = self.static_memory.borrow_mut().push_string_slice(&string);
                self.instructions
                    .instr_push_static(index, SLICE_SIZE, SLICE_SIZE);
            }
        }

        Ok(literal_type)
    }

    fn compile_andor(&mut self, andor: &ast::AndOr) -> Result<Type> {
        self.instructions.stack_instructions.jump();

        let left = self.compile_expression(&andor.left)?;
        if left != *BOOL {
            return Err(anyhow!("compile_andor: expected bool expression"));
        }

        match andor._type {
            ast::AndOrType::Or => {
                self.instructions.stack_instructions.back_if_true(1);

                let right = self.compile_expression(&andor.right)?;
                if right != *BOOL {
                    return Err(anyhow!("compile_andor: expected bool expression"));
                }

                self.instructions.instr_or();
            }
            ast::AndOrType::And => {
                self.instructions.stack_instructions.back_if_false(1);

                let right = self.compile_expression(&andor.right)?;
                if right != *BOOL {
                    return Err(anyhow!("compile_andor: expected bool expression"));
                }

                self.instructions.instr_and();
            }
        }

        self.instructions.stack_instructions.back(1);
        self.instructions.stack_instructions.pop_index();

        Ok(BOOL.clone())
    }

    fn compile_expression(&mut self, expression: &ast::Expression) -> Result<Type> {
        let old_stack_size = self.instructions.stack_total_size();

        let exp = match expression {
            ast::Expression::AndOr(v) => self.compile_andor(v),
            ast::Expression::Literal(v) => self.compile_literal(v),
            ast::Expression::Arithmetic(v) => self.compile_arithmetic(v),
            ast::Expression::Call(v) => self.compile_call(v),
            ast::Expression::TypeInit(v) => self.compile_type_init(v),
            ast::Expression::Compare(v) => self.compile_compare(v),
            ast::Expression::Infix(v) => self.compile_infix(v),
            ast::Expression::SliceInit(v) => self.compile_slice_init(v),
            ast::Expression::Index(v) => self.compile_expression_index(v),
            ast::Expression::Negate(v) => self.compile_negate(v),
            ast::Expression::Spread(v) => self.compile_spread(v),
            ast::Expression::StructInit(v) => self.compile_struct_init(v),
            ast::Expression::DotAccess(v) => self.compile_dot_access(v),
            ast::Expression::Deref(v) => self.compile_deref(v),
            ast::Expression::Address(v) => self.compile_address(v),
            ast::Expression::Type(v) => self.compile_type(v),
            ast::Expression::Closure(v) => self.compile_closure(v),
            ast::Expression::Nil => self.compile_nil(),
        }?;

        if exp.alignment != 0 {
            let new_stack_size = self.instructions.stack_total_size();
            let delta_stack_size = new_stack_size - old_stack_size;

            if old_stack_size % exp.alignment == 0 && delta_stack_size > exp.size {
                self.instructions
                    .instr_shift(exp.size, delta_stack_size - exp.size);
            }
        }

        Ok(exp)
    }
}

pub struct Compiled {
    pub functions: HashMap<String, Vec<ScopedInstruction>>,
    pub static_instructions: Vec<ScopedInstruction>,
    pub static_memory: vm::StaticMemory,
}

pub fn compile(ast: ast::Ast) -> Result<Compiled> {
    let mut functions = HashMap::<String, Vec<ScopedInstruction>>::new();
    let static_memory = Rc::new(RefCell::new(vm::StaticMemory::new()));

    let type_resolver = Rc::new(TypeResolver::new(ast.type_declarations));
    let function_declarations = Rc::new(ast.function_declarations);

    let (static_var_stack, static_instructions) = compile_static_vars(
        ast.static_var_declarations,
        type_resolver.clone(),
        function_declarations.clone(),
        static_memory.clone(),
    )?;

    for (identifier, declaration) in function_declarations.iter() {
        let compiled = FunctionCompiler::new(
            Function::from_declaration(&type_resolver, declaration).unwrap(),
            static_var_stack.clone(),
            static_memory.clone(),
            type_resolver.clone(),
            function_declarations.clone(),
        )
        .compile()
        .unwrap();
        println!("{:#?}", compiled);

        functions.insert(
            identifier.clone(),
            ScopedInstruction::from_compiled_instructions(&compiled),
        );
    }

    let static_memory = static_memory.borrow().clone();
    Ok(Compiled {
        static_memory,
        functions,
        static_instructions,
    })
}

fn compile_static_vars(
    vars: Vec<ast::StaticVarDeclaration>,
    type_resolver: Rc<TypeResolver>,
    function_declarations: Rc<HashMap<String, ast::FunctionDeclaration>>,
    static_memory: Rc<RefCell<vm::StaticMemory>>,
) -> Result<(CompilerVarStack, Vec<ScopedInstruction>)> {
    let mut instructions = Instructions::new(CompilerVarStack::new());
    let mut closures = Vec::new();

    for v in vars {
        let exp = ExpressionCompiler::new(
            &mut instructions,
            &mut closures,
            type_resolver.clone(),
            function_declarations.clone(),
            static_memory.clone(),
        )
        .compile_expression(&v.expression)?;
        type_resolver.resolve(&v._type)?.equals(&exp)?;
        instructions.var_mark(Variable {
            _type: exp,
            identifier: v.identifier.clone(),
        });
    }

    Ok((
        instructions.var_stack.clone(),
        ScopedInstruction::from_compiled_instructions(&CompiledInstructions {
            instructions: instructions.get_instructions(),
            closures,
        }),
    ))
}

pub struct FunctionCompiler {
    instructions: Instructions,
    function: Function,
    closures: Vec<CompiledInstructions>,
    current_body: Vec<CompilerBody>,
    type_resolver: Rc<TypeResolver>,
    function_declarations: Rc<HashMap<String, ast::FunctionDeclaration>>,
    static_memory: Rc<RefCell<vm::StaticMemory>>,
}

#[derive(Debug, Clone)]
pub struct CompiledInstructions {
    pub instructions: Vec<Vec<CompilerInstruction>>,
    pub closures: Vec<CompiledInstructions>,
}

impl FunctionCompiler {
    fn new(
        function: Function,
        static_var_stack: CompilerVarStack,
        static_memory: Rc<RefCell<vm::StaticMemory>>,
        type_resolver: Rc<TypeResolver>,
        function_declarations: Rc<HashMap<String, ast::FunctionDeclaration>>,
    ) -> Self {
        let mut current_body = Vec::new();
        current_body.push(CompilerBody::new(function.body.clone()));
        Self {
            function_declarations,
            static_memory,
            function,
            closures: Vec::new(),
            instructions: Instructions::new(static_var_stack),
            type_resolver,
            current_body,
        }
    }

    fn resolve_type(&self, _type: &ast::Type) -> Result<Type> {
        self.type_resolver.resolve(_type)
    }

    fn resolve_variable(&self, _type: &ast::Type) -> Result<(usize, Variable)> {
        let ast::Type::Alias(identifier) = _type else {
            return Err(anyhow!("resolve_variable: cant resolve non alias"));
        };

        let variable = self
            .instructions
            .var_get_offset(identifier)
            .ok_or(anyhow!("resolve_variable: variable {identifier} not found"))?;

        Ok((variable.0, variable.1.clone()))
    }

    fn expression_compiler(&mut self) -> ExpressionCompiler {
        ExpressionCompiler::new(
            &mut self.instructions,
            &mut self.closures,
            self.type_resolver.clone(),
            self.function_declarations.clone(),
            self.static_memory.clone(),
        )
    }

    fn compile_expression(&mut self, expression: &ast::Expression) -> Result<Type> {
        self.expression_compiler().compile_expression(expression)
    }

    fn compile_dot_access_field_offset(
        &mut self,
        dot_access: &ast::DotAccess,
    ) -> Result<DotAccessField> {
        self.expression_compiler()
            .compile_dot_access_field_offset(dot_access)
    }

    fn compile_variable_declaration(
        &mut self,
        declaration: &ast::VariableDeclaration,
    ) -> Result<()> {
        let escaped = self
            .current_body
            .last()
            .unwrap()
            .does_variable_escape(&declaration.variable.identifier, 1);
        if escaped {
            self.instructions.push_alignment(PTR_SIZE);
        }

        let mut exp = self.compile_expression(&declaration.expression)?;
        if exp == *VOID {
            return Err(anyhow!("can't declare void variable"));
        }

        self.resolve_type(&declaration.variable._type)?
            .equals(&exp)?;

        if escaped {
            self.instructions.instr_alloc(exp.size, exp.alignment);
            exp = Type::create_escaped(exp);
        }

        self.instructions.var_mark(Variable {
            identifier: declaration.variable.identifier.clone(),
            _type: exp,
        });

        Ok(())
    }

    fn compile_variable_assignment(&mut self, assignment: &ast::VariableAssignment) -> Result<()> {
        self.instructions.push_stack_frame();

        match &assignment.var {
            ast::Expression::Type(_type) => {
                let (offset, variable) = self.resolve_variable(_type)?;

                if let TypeType::Escaped(_type) = &variable._type._type {
                    let alignment = self.instructions.push_alignment(PTR_SIZE);
                    self.instructions.instr_increment(PTR_SIZE);
                    self.instructions
                        .instr_copy(0, offset + alignment + PTR_SIZE, PTR_SIZE);

                    // no alignment because of PTR_SIZE align above
                    let exp = self.compile_expression(&assignment.expression)?;
                    if **_type != exp {
                        return Err(anyhow!("variable assignment type mismatch"));
                    }

                    self.instructions.instr_deref_assign(exp.size);
                } else {
                    self.instructions.push_stack_frame();
                    let exp = self.compile_expression(&assignment.expression)?;
                    if variable._type != exp {
                        return Err(anyhow!("variable assignment type mismatch"));
                    }
                    let size = self.instructions.pop_stack_frame_size();

                    self.instructions.instr_copy(offset + size, 0, exp.size);
                }
            }
            ast::Expression::Index(index) => {
                let slice = self.compile_expression(&index.var)?;
                let TypeType::Slice(slice_item) = &slice._type else {
                    return Err(anyhow!("can only index slices"));
                };

                let item_index = self.compile_expression(&index.expression)?;
                if item_index != *INT {
                    return Err(anyhow!("can only index with int type"));
                }

                let item = self.compile_expression(&assignment.expression)?;
                if **slice_item != item {
                    return Err(anyhow!("slice index set type mismatch"));
                }

                self.instructions.instr_slice_index_set(item.size);
            }
            ast::Expression::DotAccess(dot_access) => {
                let field = self.compile_dot_access_field_offset(dot_access)?;

                self.instructions.push_stack_frame();
                let exp = self.compile_expression(&assignment.expression)?;
                let exp_size = self.instructions.pop_stack_frame_size();

                exp.equals(field._type())?;

                match field {
                    DotAccessField::Heap(_type) => {
                        self.instructions.instr_deref_assign(_type.size);
                    }
                    DotAccessField::Stack(offset, _type) => {
                        self.instructions.instr_copy(offset + exp_size, 0, exp.size);
                    }
                }
            }
            ast::Expression::Deref(expression) => {
                let dst = self.compile_expression(expression)?;
                let TypeType::Address(_type) = dst._type else {
                    return Err(anyhow!("can not dereference non address"));
                };

                let src = self.compile_expression(&assignment.expression)?;
                if src != *_type {
                    return Err(anyhow!("variable assignment type mismatch"));
                }

                self.instructions.instr_deref_assign(src.size);
            }
            node => return Err(anyhow!("can't assign {node:#?}")),
        }

        self.instructions.pop_stack_frame();

        Ok(())
    }

    fn compile_if_block(&mut self, expression: &ast::Expression, body: &[ast::Node]) -> Result<()> {
        let exp = self.compile_expression(expression)?;
        if exp != *BOOL {
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
                if exp != *BOOL {
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
        self.current_body.push(CompilerBody::new(body.to_vec()));
        self.instructions.push_stack_frame();

        for node in body {
            self.compile_node(node)?;
            self.current_body.last_mut().unwrap().next();
        }

        self.current_body.pop();
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

        self.instructions.init_function_epilogue(&self.function);

        Ok(())
    }

    pub fn compile(mut self) -> Result<CompiledInstructions> {
        self.instructions.init_function_prologue(&self.function)?;

        self.compile_body(&self.function.body.clone())?;

        self.compile_return(None)?;

        Ok(CompiledInstructions {
            instructions: self.instructions.get_instructions(),
            closures: self.closures,
        })
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
