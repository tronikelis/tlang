use anyhow::{anyhow, Result};
use std::{cell::RefCell, collections::HashMap, rc::Rc};

use crate::{ir, vm};

#[cfg(test)]
mod tests;

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
    Var(ir::Variable),
    Label,
}

#[derive(Debug, Clone)]
struct CompilerVarStack {
    stack: ir::Stack<VarStackItem>,
    arg_size: Option<usize>,
}

impl CompilerVarStack {
    fn new() -> Self {
        Self {
            stack: ir::Stack::new(),
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

    fn get_var_offset(&self, identifier: &str) -> Option<(usize, &ir::Variable)> {
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

#[derive(Debug)]
enum Function<'a> {
    Function(&'a ir::Function),
    Closure(&'a ir::Closure),
}

impl<'a> Function<'a> {
    fn return_type(&self) -> &ir::Type {
        match &self {
            Self::Function(v) => &v.return_type,
            Self::Closure(v) => &v._type.expect_closure().unwrap().return_type,
        }
    }

    fn actions(&self) -> &[ir::Action] {
        match &self {
            Self::Function(v) => &v.actions,
            Self::Closure(v) => &v.actions,
        }
    }
}

#[derive(Debug, Clone)]
struct Instructions {
    stack_instructions: StackInstructions,
    var_stack: CompilerVarStack,
}

impl ir::VariableStack for Instructions {
    fn get_type(&self, identifier: &str) -> Option<ir::Type> {
        let var = self.var_get_offset(identifier);
        var.map(|v| v.1._type.clone())
    }
}

impl Instructions {
    fn new() -> Self {
        Self {
            stack_instructions: StackInstructions::new(),
            var_stack: CompilerVarStack::new(),
        }
    }

    fn var_mark(&mut self, var: ir::Variable) {
        self.var_stack.stack.push(VarStackItem::Var(var));
    }

    fn var_mark_label(&mut self) {
        self.var_stack.stack.push(VarStackItem::Label);
    }

    fn var_get_offset(&self, identifier: &str) -> Option<(usize, &ir::Variable)> {
        self.var_stack.get_var_offset(identifier)
    }

    fn var_get_offset_err(&self, identifier: &str) -> Result<(usize, &ir::Variable)> {
        self.var_get_offset(identifier)
            .ok_or(anyhow!("var_get_offset_err: {identifier}"))
    }

    fn var_reset_label(&mut self) {
        self.instr_reset_dangerous_not_synced(self.var_stack.get_label_offset().unwrap());
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
        self.var_stack
            .stack
            .push(VarStackItem::Increment(ir::PTR_SIZE));
    }

    fn instr_deref(&mut self, size: usize) {
        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::Deref(size)));
        self.var_stack.stack.push(VarStackItem::Reset(ir::PTR_SIZE));
        self.var_stack.stack.push(VarStackItem::Increment(size));
    }

    fn instr_deref_assign(&mut self, size: usize) {
        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::DerefAssign(
                size,
            )));
        self.var_stack
            .stack
            .push(VarStackItem::Reset(ir::PTR_SIZE + size));
    }

    fn instr_slice_index_set(&mut self, size: usize) {
        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::SliceIndexSet(
                size,
            )));
        self.var_stack
            .stack
            .push(VarStackItem::Reset(size + ir::SLICE_SIZE + ir::INT.size));
    }

    fn instr_slice_index_get(&mut self, size: usize) {
        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::SliceIndexGet(
                size,
            )));
        self.var_stack
            .stack
            .push(VarStackItem::Reset(ir::INT.size + ir::SLICE_SIZE));
        self.var_stack.stack.push(VarStackItem::Increment(size));
    }

    fn instr_slice_len(&mut self) {
        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::SliceLen));
        self.var_stack
            .stack
            .push(VarStackItem::Reset(ir::SLICE_SIZE));
        self.var_stack
            .stack
            .push(VarStackItem::Increment(ir::INT.size));
    }

    fn instr_slice_append(&mut self, size: usize) {
        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::SliceAppend(
                size,
            )));
        self.var_stack
            .stack
            .push(VarStackItem::Reset(ir::SLICE_SIZE + size));
    }

    fn instr_and(&mut self) {
        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::And));
        self.var_stack
            .stack
            .push(VarStackItem::Reset(ir::BOOL.size));
    }

    fn instr_or(&mut self) {
        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::Or));
        self.var_stack
            .stack
            .push(VarStackItem::Reset(ir::BOOL.size));
    }

    fn instr_negate_bool(&mut self) {
        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::NegateBool));
    }

    fn instr_increment(&mut self, size: usize) {
        if size == 0 {
            return;
        }
        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::Increment(size)));
        self.var_stack.stack.push(VarStackItem::Increment(size));
    }

    fn instr_reset_dangerous_not_synced(&mut self, size: usize) {
        if size == 0 {
            return;
        }
        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::Reset(size)));
    }

    fn instr_reset(&mut self, size: usize) {
        if size == 0 {
            return;
        }
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
            .push(VarStackItem::Reset(escaped_count * ir::PTR_SIZE));
        self.var_stack
            .stack
            .push(VarStackItem::Increment(ir::PTR_SIZE));
    }

    fn instr_push_slice_new_len(&mut self, size: usize) {
        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::PushSliceNewLen(
                size,
            )));
        self.var_stack
            .stack
            .push(VarStackItem::Reset(size + ir::INT.size - ir::SLICE_SIZE));
    }

    fn instr_push_slice(&mut self) {
        self.push_alignment(ir::SLICE_SIZE);

        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::PushSlice));
        self.var_stack
            .stack
            .push(VarStackItem::Increment(ir::SLICE_SIZE));
    }

    fn instr_push_u(&mut self, int: usize) -> Result<()> {
        self.push_alignment(ir::UINT.alignment);
        let uint: usize = int.try_into()?;

        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::PushU(uint)));
        self.var_stack
            .stack
            .push(VarStackItem::Increment(ir::UINT.size));

        Ok(())
    }

    fn instr_push_u8(&mut self, int: usize) -> Result<()> {
        self.push_alignment(ir::UINT8.alignment);
        let uint8: u8 = int.try_into()?;

        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::PushU8(uint8)));
        self.var_stack
            .stack
            .push(VarStackItem::Increment(ir::UINT8.size));

        Ok(())
    }

    fn instr_push_u16(&mut self, int: usize) -> Result<()> {
        self.push_alignment(ir::UINT16.alignment);
        let uint16: u16 = int.try_into()?;

        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::PushU16(uint16)));
        self.var_stack
            .stack
            .push(VarStackItem::Increment(ir::UINT16.size));

        Ok(())
    }

    fn instr_push_u32(&mut self, int: usize) -> Result<()> {
        self.push_alignment(ir::UINT32.alignment);
        let uint32: u32 = int.try_into()?;

        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::PushU32(uint32)));
        self.var_stack
            .stack
            .push(VarStackItem::Increment(ir::UINT32.size));

        Ok(())
    }

    fn instr_push_u64(&mut self, int: usize) -> Result<()> {
        self.push_alignment(ir::UINT64.alignment);
        let uint64: u64 = int.try_into()?;

        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::PushU64(uint64)));
        self.var_stack
            .stack
            .push(VarStackItem::Increment(ir::UINT64.size));

        Ok(())
    }

    fn instr_push_i(&mut self, int: usize) -> Result<()> {
        self.push_alignment(ir::INT.alignment);
        let int: isize = int.try_into()?;

        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::PushI(int)));
        self.var_stack
            .stack
            .push(VarStackItem::Increment(ir::INT.size));

        Ok(())
    }

    fn instr_push_i8(&mut self, int: usize) -> Result<()> {
        self.push_alignment(ir::INT8.alignment);
        let int8: i8 = int.try_into()?;

        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::PushI8(int8)));
        self.var_stack
            .stack
            .push(VarStackItem::Increment(ir::INT8.size));

        Ok(())
    }

    fn instr_push_i16(&mut self, int: usize) -> Result<()> {
        self.push_alignment(ir::INT16.alignment);
        let int16: i16 = int.try_into()?;

        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::PushI16(int16)));
        self.var_stack
            .stack
            .push(VarStackItem::Increment(ir::INT16.size));

        Ok(())
    }

    fn instr_push_i32(&mut self, int: usize) -> Result<()> {
        self.push_alignment(ir::INT32.alignment);
        let int32: i32 = int.try_into()?;

        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::PushI32(int32)));
        self.var_stack
            .stack
            .push(VarStackItem::Increment(ir::INT32.size));

        Ok(())
    }

    fn instr_push_i64(&mut self, int: usize) -> Result<()> {
        self.push_alignment(ir::INT64.alignment);
        let int64: i64 = int.try_into()?;

        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::PushI64(int64)));
        self.var_stack
            .stack
            .push(VarStackItem::Increment(ir::INT64.size));

        Ok(())
    }

    fn instr_minus_int(&mut self) {
        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::MinusInt));
    }

    fn instr_add_i(&mut self) {
        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::AddI));
        self.var_stack.stack.push(VarStackItem::Reset(ir::INT.size));
    }

    fn instr_multiply_i(&mut self) {
        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::MultiplyI));
        self.var_stack.stack.push(VarStackItem::Reset(ir::INT.size));
    }

    fn instr_divide_i(&mut self) {
        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::DivideI));
        self.var_stack.stack.push(VarStackItem::Reset(ir::INT.size));
    }

    fn instr_modulo_i(&mut self) {
        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::ModuloI));
        self.var_stack.stack.push(VarStackItem::Reset(ir::INT.size));
    }

    fn instr_to_bool(&mut self) {
        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::ToBoolI));
        self.var_stack.stack.push(VarStackItem::Reset(ir::INT.size));
        self.var_stack
            .stack
            .push(VarStackItem::Increment(ir::BOOL.size));
    }

    fn instr_compare(&mut self, size: u8) {
        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::Compare(size)));
        self.var_stack
            .stack
            .push(VarStackItem::Reset(size as usize * 2));
        self.var_stack
            .stack
            .push(VarStackItem::Increment(ir::BOOL.size));
    }

    fn instr_compare_string(&mut self) {
        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::CompareString));
        self.var_stack
            .stack
            .push(VarStackItem::Reset(ir::SLICE_SIZE * 2));
        self.var_stack
            .stack
            .push(VarStackItem::Increment(ir::BOOL.size));
    }

    fn instr_cast_int(&mut self, mut from: u8, mut to: u8) {
        if from == to {
            return;
        }
        if from == 0 {
            from = size_of::<usize>() as u8;
        }
        if to == 0 {
            to = size_of::<usize>() as u8;
        }

        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::CastInt(
                from, to,
            )));
        self.var_stack
            .stack
            .push(VarStackItem::Reset(from as usize));
        self.var_stack
            .stack
            .push(VarStackItem::Increment(to as usize));
    }

    fn instr_cast_uint(&mut self, mut from: u8, mut to: u8) {
        if from == to {
            return;
        }
        if from == 0 {
            from = size_of::<usize>() as u8;
        }
        if to == 0 {
            to = size_of::<usize>() as u8;
        }

        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::CastUint(
                from, to,
            )));
        self.var_stack
            .stack
            .push(VarStackItem::Reset(from as usize));
        self.var_stack
            .stack
            .push(VarStackItem::Increment(to as usize));
    }

    fn instr_add_string(&mut self) {
        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::AddString));
        self.var_stack
            .stack
            .push(VarStackItem::Reset(ir::STRING.size));
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

    fn instr_cast_slice_ptr(&mut self) {
        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::CastSlicePtr));
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
        self.var_stack.stack.push(VarStackItem::Reset(ir::PTR_SIZE));
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
            .push(VarStackItem::Reset(ir::INT.size + ir::SLICE_SIZE));
        self.var_stack
            .stack
            .push(VarStackItem::Increment(ir::INT.size));
    }

    fn instr_dll_open(&mut self) {
        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::FfiDllOpen));
        // will pop 1 argument
        // will push 1 argument so no further instructions needed
    }

    fn instr_ffi_create(&mut self) {
        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::FfiCreate));
        self.var_stack
            .stack
            .push(VarStackItem::Reset(ir::PTR_SIZE * 4));
        self.var_stack
            .stack
            .push(VarStackItem::Increment(ir::PTR_SIZE));
    }

    fn instr_ffi_call(&mut self) {
        self.stack_instructions
            .push(CompilerInstruction::Real(vm::Instruction::FfiCall));
        self.var_stack
            .stack
            .push(VarStackItem::Reset(ir::PTR_SIZE * 2));
        self.var_stack
            .stack
            .push(VarStackItem::Increment(ir::PTR_SIZE));
    }

    fn push_alignment(&mut self, alignment: usize) -> usize {
        let alignment = ir::align(alignment, self.var_stack.total_size());
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
        self.instr_reset_dangerous_not_synced(CompilerVarStack::size_for(frame.iter()));
    }

    fn pop_stack_frame_size(&mut self) -> usize {
        let frame = self.var_stack.stack.pop_frame().unwrap();
        let size = CompilerVarStack::size_for(frame.iter());
        frame.into_iter().for_each(|v| self.var_stack.stack.push(v));
        size
    }

    fn init_function_prologue(&mut self, function: &Function) -> Result<()> {
        let mut argument_size: usize = 0;

        let return_type = match function {
            Function::Function(function) => function.return_type.clone(),
            Function::Closure(closure) => closure._type.expect_closure()?.return_type.clone(),
        };

        let arguments: Vec<ir::Variable> = match function {
            Function::Function(function) => function.arguments.iter().map(|v| v.clone()).collect(),
            Function::Closure(closure) => closure
                .arguments
                .iter()
                .map(|v| v.borrow().clone())
                .collect(),
        };

        // function calls are aligned on ptr size

        // arguments are pushed into the stack
        for arg in &arguments {
            // skipping escape here, because at this point,
            // variable is not allocated on the heap yet
            let mut arg = arg.clone();
            arg._type = arg._type.skip_escaped().clone();

            argument_size += arg._type.size;

            let alignment = ir::align(arg._type.alignment, self.var_stack.total_size());
            argument_size += alignment;

            if alignment != 0 {
                self.var_stack
                    .stack
                    .push(VarStackItem::Increment(alignment));
            }

            self.var_stack
                .stack
                .push(VarStackItem::Increment(arg._type.size));
            self.var_mark(arg.clone());
        }

        // return value is set in the arguments
        // space is allocated if return value does not fit into arguments
        if argument_size < return_type.size {
            self.var_stack
                .stack
                .push(VarStackItem::Increment(return_type.size - argument_size))
        }

        // add return address for all functions except for main function
        // it does not have anywhere to return
        let add_return_address = match function {
            Function::Closure(_) => true,
            Function::Function(function) => function.identifier != "main",
        };

        if add_return_address {
            let alignment = ir::align(ir::PTR_SIZE, self.var_stack.total_size());
            if alignment != 0 {
                self.var_stack
                    .stack
                    .push(VarStackItem::Increment(alignment));
            }

            self.var_stack
                .stack
                .push(VarStackItem::Increment(ir::PTR_SIZE));
        }

        self.var_stack.set_arg_size();

        // for closures:
        // escaped variables are pushed now
        // these are already allocated to the heap
        // already aligned because of return address
        if let Function::Closure(closure) = function {
            for arg in &closure.escaped_variables {
                assert!(arg.borrow()._type._type.is_escaped());
                self.var_stack
                    .stack
                    .push(VarStackItem::Increment(ir::PTR_SIZE));
                self.var_mark(arg.borrow().clone());
            }
        }

        // any escaped function arguments are allocated now, after return address
        for arg in &arguments {
            if let ir::TypeType::Escaped(_type) = &arg._type._type {
                let (offset, _) = self.var_get_offset(&arg.identifier).unwrap();
                let alignment = self.push_alignment(ir::PTR_SIZE);
                self.instr_increment(_type.size);
                self.instr_copy(0, offset + _type.size + alignment, _type.size);
                self.instr_alloc(_type.size, _type.alignment);
                self.var_mark(arg.clone());
            }
        }

        Ok(())
    }

    fn init_function_epilogue(&mut self, function: &Function) {
        self.instr_reset_dangerous_not_synced(
            self.var_stack.total_size() - self.var_stack.arg_size.unwrap(),
        );

        let is_main = match function {
            Function::Function(function) => function.identifier == "main",
            Function::Closure(_) => false,
        };

        if is_main {
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

enum FunctionCallType<'a> {
    Function(&'a ir::ExpFunction),
    Closure(&'a ir::Expression),
    Method(&'a ir::Method),
}

struct FunctionCall<'a> {
    arguments: &'a [ir::Expression],
    call_type: FunctionCallType<'a>,
}

#[derive(Debug)]
enum DotAccessField {
    // offset from the stack
    Stack(usize, ir::Type),
    // offset address is on top of the stack
    Heap(ir::Type),
}

impl DotAccessField {
    fn _type(&self) -> &ir::Type {
        match self {
            Self::Stack(_, _type) => _type,
            Self::Heap(_type) => _type,
        }
    }
}

struct ExpressionCompiler<'a, 'b> {
    instructions: &'a mut Instructions,
    closures: &'a mut Vec<CompiledInstructions>,
    ir: &'b ir::Ir<'b>,
    static_memory: Rc<RefCell<vm::StaticMemory>>,
}

impl<'a, 'b> ExpressionCompiler<'a, 'b> {
    fn new(
        instructions: &'a mut Instructions,
        closures: &'a mut Vec<CompiledInstructions>,
        ir: &'b ir::Ir<'b>,
        static_memory: Rc<RefCell<vm::StaticMemory>>,
    ) -> Self {
        Self {
            instructions,
            closures,
            ir,
            static_memory,
        }
    }

    fn compile_nil(&mut self) -> Result<ir::Type> {
        self.instructions.instr_push_i(0)?;
        Ok(ir::NIL.clone())
    }

    fn compile_closure(&mut self, closure: &ir::Closure) -> Result<ir::Type> {
        self.instructions.push_alignment(ir::PTR_SIZE);

        // right now all nested closures capture everything inside them
        //
        // let foo = 20
        // fn() void { <- this closure does not need to capture "foo" (BUT IT DOES)
        //  fn() void { <- only this closure has to capture "foo"
        //    foo = 50
        //  }
        // }

        for escaped in &closure.escaped_variables {
            assert!(
                escaped.borrow()._type._type.is_escaped(),
                "found non escaped variable when compiling closure",
            );

            let (offset, _) = self
                .instructions
                .var_get_offset_err(&escaped.borrow().identifier)?;

            self.instructions.instr_increment(ir::PTR_SIZE);
            self.instructions
                .instr_copy(0, ir::PTR_SIZE + offset, ir::PTR_SIZE);
        }

        self.instructions
            .instr_push_closure(closure.escaped_variables.len(), self.closures.len());

        let closure_type = closure._type.clone();

        self.closures.push(
            FunctionCompiler::new(
                Function::Closure(closure),
                self.static_memory.clone(),
                self.ir,
            )
            .compile()?,
        );

        Ok(closure_type)
    }

    fn compile_variable(&mut self, identifier: &str) -> Result<ir::Type> {
        let (_, variable) = self.instructions.var_get_offset_err(identifier)?;
        let variable = variable.clone();

        if let ir::TypeType::Escaped(_type) = &variable._type._type {
            // this will leak alignment
            self.instructions.push_alignment(ir::PTR_SIZE);
            self.instructions.instr_increment(ir::PTR_SIZE);

            let (offset, _) = self.instructions.var_get_offset_err(&variable.identifier)?;
            self.instructions.instr_copy(0, offset, ir::PTR_SIZE);
            self.instructions.instr_deref(_type.size);

            return Ok(*_type.clone());
        }

        self.instructions.push_alignment(variable._type.alignment);
        self.instructions.instr_increment(variable._type.size);

        let (offset, _) = self.instructions.var_get_offset_err(&variable.identifier)?;
        self.instructions.instr_copy(0, offset, variable._type.size);

        Ok(variable._type.clone())
    }

    fn compile_dot_access_field_offset_base_heap(
        &mut self,
        dot_access: &ir::DotAccess,
        type_struct: &ir::TypeStruct,
    ) -> Result<ir::Type> {
        let alignment = self.instructions.push_alignment(ir::PTR_SIZE);
        self.instructions.instr_increment(ir::PTR_SIZE);
        self.instructions
            .instr_copy(0, alignment + ir::PTR_SIZE, ir::PTR_SIZE);

        let (field_offset, field_type) =
            type_struct.get_field_offset_err(&dot_access.identifier)?;

        self.instructions.instr_offset(field_offset);

        Ok(field_type.clone())
    }

    fn compile_dot_access_field_offset(
        &mut self,
        dot_access: &ir::DotAccess,
    ) -> Result<DotAccessField> {
        if let ir::Expression::DotAccess(inner) = &dot_access.expression {
            let target_field = self.compile_dot_access_field_offset(inner)?;
            let target_type = target_field._type();

            match &target_field {
                DotAccessField::Heap(_type) => match &target_type._type {
                    // target heap -> current stack = offset address
                    ir::TypeType::Struct(type_struct) => {
                        let (offset, field_type) =
                            type_struct.get_field_offset_err(&dot_access.identifier)?;
                        self.instructions.instr_offset(offset);

                        return Ok(DotAccessField::Heap(field_type.clone()));
                    }
                    // target heap -> current heap = dereference + offset
                    ir::TypeType::Address(address_type) => {
                        let address_type =
                            address_type.clone().resolve_lazy(&self.ir.type_resolver)?;

                        let ir::TypeType::Struct(type_struct) = &address_type._type else {
                            return Err(anyhow!("cant dot access non struct type"));
                        };

                        let (offset, field_type) =
                            type_struct.get_field_offset_err(&dot_access.identifier)?;

                        self.instructions.instr_deref(ir::PTR_SIZE);
                        self.instructions.instr_offset(offset);

                        return Ok(DotAccessField::Heap(field_type.clone()));
                    }
                    _type => {
                        return Err(anyhow!("dot access on non struct/address type {_type:#?}"));
                    }
                },
                DotAccessField::Stack(stack_offset, _type) => match &target_type._type {
                    // target stack -> current stack = offset stack
                    ir::TypeType::Struct(type_struct) => {
                        let (offset, field_type) =
                            type_struct.get_field_offset_err(&dot_access.identifier)?;

                        return Ok(DotAccessField::Stack(
                            *stack_offset + offset,
                            field_type.clone(),
                        ));
                    }
                    // target stack -> current heap = offset
                    ir::TypeType::Address(address_type) => {
                        let alignment = self.instructions.push_alignment(ir::PTR_SIZE);
                        self.instructions.instr_increment(ir::PTR_SIZE);
                        self.instructions.instr_copy(
                            0,
                            alignment + ir::PTR_SIZE + *stack_offset,
                            ir::PTR_SIZE,
                        );

                        let ir::TypeType::Struct(type_struct) = &address_type._type else {
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

        let exp = match &dot_access.expression {
            // compile expression derefs escaped vars automatically,
            // however we don't want this here
            ir::Expression::Variable(iden) => {
                let (offset, variable) = self.instructions.var_get_offset_err(iden)?;
                let variable = variable.clone();

                let alignment = self.instructions.push_alignment(variable._type.alignment);

                self.instructions.instr_increment(variable._type.size);
                self.instructions.instr_copy(
                    0,
                    offset + alignment + variable._type.size,
                    variable._type.size,
                );

                variable._type
            }
            exp => self.compile_expression(exp)?,
        };

        if let ir::TypeType::Escaped(_type) = exp._type {
            match _type._type {
                ir::TypeType::Struct(type_struct) => Ok(DotAccessField::Heap(
                    self.compile_dot_access_field_offset_base_heap(dot_access, &type_struct)?,
                )),
                ir::TypeType::Address(_type) => match _type._type {
                    ir::TypeType::Struct(type_struct) => {
                        let alignment = self.instructions.push_alignment(ir::PTR_SIZE);
                        self.instructions.instr_increment(ir::PTR_SIZE);
                        self.instructions
                            .instr_copy(0, ir::PTR_SIZE + alignment, ir::PTR_SIZE);
                        self.instructions.instr_deref(ir::PTR_SIZE);

                        Ok(DotAccessField::Heap(
                            self.compile_dot_access_field_offset_base_heap(
                                dot_access,
                                &type_struct,
                            )?,
                        ))
                    }
                    _type => Err(anyhow!("cant access non struct type: {_type:#?}")),
                },
                _type => Err(anyhow!("cant access non struct/address type: {_type:#?}")),
            }
        } else {
            match exp._type {
                ir::TypeType::Struct(type_struct) => {
                    let (field_offset, field_type) =
                        type_struct.get_field_offset_err(&dot_access.identifier)?;

                    Ok(DotAccessField::Stack(field_offset, field_type.clone()))
                }
                ir::TypeType::Address(_type) => match _type._type {
                    ir::TypeType::Struct(type_struct) => Ok(DotAccessField::Heap(
                        self.compile_dot_access_field_offset_base_heap(dot_access, &type_struct)?,
                    )),
                    _type => Err(anyhow!("cant access non struct type: {_type:#?}")),
                },
                _type => Err(anyhow!("cant access non struct/address type: {_type:#?}")),
            }
        }
    }

    fn compile_address(&mut self, expression: &ir::Expression) -> Result<ir::Type> {
        match expression {
            ir::Expression::Variable(identifier) => {
                let (offset, var) = self.instructions.var_get_offset_err(identifier)?;
                let var = var.clone();

                let ir::TypeType::Escaped(_type) = &var._type._type else {
                    return Err(anyhow!("compile_address: rn all variables are escaped"));
                };

                let alignment = self.instructions.push_alignment(ir::PTR_SIZE);
                self.instructions.instr_increment(ir::PTR_SIZE);
                self.instructions
                    .instr_copy(0, offset + ir::PTR_SIZE + alignment, ir::PTR_SIZE);

                Ok(ir::Type::create_address(*_type.clone()))
            }
            ir::Expression::DotAccess(dot_access) => {
                let field = self.compile_dot_access_field_offset(dot_access)?;
                let DotAccessField::Heap(_type) = field else {
                    return Err(anyhow!(
                        "compile_address: cant take non heap address dot access"
                    ));
                };

                Ok(ir::Type::create_address(_type))
            }
            expression => {
                self.instructions.push_alignment(ir::PTR_SIZE);
                let exp = self.compile_expression(expression)?;
                self.instructions.instr_alloc(exp.size, exp.alignment);

                Ok(ir::Type::create_address(exp))
            }
        }
    }

    fn compile_deref(&mut self, expression: &ir::Expression) -> Result<ir::Type> {
        let exp = self.compile_expression(expression)?;
        let ir::TypeType::Address(_type) = exp._type else {
            return Err(anyhow!(
                "compile_deref: can't dereference non address types"
            ));
        };

        self.instructions.instr_deref(exp.size);

        Ok(*_type)
    }

    fn compile_dot_access(&mut self, dot_access: &ir::DotAccess) -> Result<ir::Type> {
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

    fn compile_struct_init(&mut self, struct_init: &ir::StructInit) -> Result<ir::Type> {
        let ir::TypeType::Struct(type_struct) = &struct_init._type._type else {
            panic!("compile_struct_init: incorrect type wrong ast parser");
        };

        if type_struct.identifier_field_count() != struct_init.fields.len() {
            return Err(anyhow!(
                "compile_struct_init: field initialization count mismatch"
            ));
        }

        for field in type_struct.fields.iter().rev() {
            match field {
                ir::TypeStructField::Padding(padding) => {
                    self.instructions.instr_increment(*padding);
                }
                ir::TypeStructField::Type(identifier, _type) => {
                    let exp = struct_init.fields.get(identifier).ok_or(anyhow!(
                        "compile_struct_init: initialization field not found"
                    ))?;

                    let exp_type = self.compile_expression(exp)?;
                    exp_type.must_equal(_type)?;
                }
            }
        }

        Ok(struct_init._type.clone())
    }

    fn compile_spread(&mut self, expression: &ir::Expression) -> Result<ir::Type> {
        let exp = self.compile_expression(expression)?;

        let ir::TypeType::Slice(slice_item) = exp._type else {
            return Err(anyhow!("compile_spread: can only spread slice types"));
        };

        Ok(ir::Type::create_variadic(*slice_item))
    }

    fn compile_negate(&mut self, negate: &ir::Expression) -> Result<ir::Type> {
        let exp_bool = self.compile_expression(negate)?;
        if exp_bool != *ir::BOOL {
            return Err(anyhow!("can only negate bools"));
        }

        self.instructions.instr_negate_bool();

        Ok(ir::BOOL.clone())
    }

    fn compile_expression_index(&mut self, index: &ir::Index) -> Result<ir::Type> {
        let exp_var = self.compile_expression(&index.var)?;

        let ir::TypeType::Slice(expected_type) = exp_var._type else {
            return Err(anyhow!("can't index this type"));
        };

        let exp_index = self.compile_expression(&index.expression)?;
        if exp_index != *ir::INT {
            return Err(anyhow!("cant index with {exp_index:#?}"));
        }

        self.instructions.instr_slice_index_get(expected_type.size);

        Ok(*expected_type)
    }

    fn compile_slice_init(&mut self, slice_init: &ir::SliceInit) -> Result<ir::Type> {
        let ir::TypeType::Slice(slice_item) = &slice_init._type._type else {
            panic!("compile_slice_init: incorrect type, wrong ast parser");
        };

        self.instructions.instr_push_slice();

        for v in &slice_init.expressions {
            self.instructions.push_stack_frame();

            self.instructions.instr_increment(ir::SLICE_SIZE);
            self.instructions
                .instr_copy(0, ir::SLICE_SIZE, ir::SLICE_SIZE);

            let exp = self.compile_expression(v)?;
            if exp != **slice_item {
                return Err(anyhow!("compile_slice_init: slice item type mismatch"));
            }

            self.instructions.instr_slice_append(exp.size);

            self.instructions.pop_stack_frame();
        }

        Ok(slice_init._type.clone())
    }

    fn compile_infix(&mut self, infix: &ir::Infix) -> Result<ir::Type> {
        let exp = self.compile_expression(&infix.expression)?;
        match infix._type {
            ir::InfixType::Plus => {}
            ir::InfixType::Minus => {
                self.instructions.instr_minus_int();
            }
        }
        Ok(exp)
    }

    fn compile_compare(&mut self, compare: &ir::Compare) -> Result<ir::Type> {
        let a: ir::Type;
        let b: ir::Type;

        match compare.compare_type {
            // last item on the stack is smaller
            ir::CompareType::Gt => {
                b = self.compile_expression(&compare.left)?;
                a = self.compile_expression_compact(&compare.right, b.alignment)?;
            }
            // last item on the stack is bigger
            ir::CompareType::Lt => {
                b = self.compile_expression(&compare.right)?;
                a = self.compile_expression_compact(&compare.left, b.alignment)?;
            }
            // dont matter
            ir::CompareType::Equals | ir::CompareType::NotEquals => {
                a = self.compile_expression(&compare.right)?;
                b = self.compile_expression_compact(&compare.left, a.alignment)?;
            }
        };

        a.must_equal(&b)?;

        match &a._type {
            ir::TypeType::Builtin(type_builtin) => match type_builtin {
                ir::TypeBuiltin::Int
                | ir::TypeBuiltin::Int8
                | ir::TypeBuiltin::Int16
                | ir::TypeBuiltin::Int32
                | ir::TypeBuiltin::Int64
                | ir::TypeBuiltin::Uint8
                | ir::TypeBuiltin::Uint16
                | ir::TypeBuiltin::Uint32
                | ir::TypeBuiltin::Uint64
                | ir::TypeBuiltin::Uint
                | ir::TypeBuiltin::Bool
                | ir::TypeBuiltin::String => {}
                _type => return Err(anyhow!("cant compare {_type:#?}")),
            },
            _type => return Err(anyhow!("cant compare {_type:#?}")),
        }

        match compare.compare_type {
            ir::CompareType::Gt | ir::CompareType::Lt => {
                // a = -a
                self.instructions.instr_minus_int();

                // a + b
                self.instructions.instr_add_i();

                // >0:1 <0:0
                self.instructions.instr_to_bool();
            }
            ir::CompareType::Equals => match &a._type {
                ir::TypeType::Builtin(type_builtin) => match type_builtin {
                    ir::TypeBuiltin::String => self.instructions.instr_compare_string(),
                    _ => {
                        self.instructions.instr_compare(a.size as u8);
                    }
                },
                _ => unreachable!(),
            },
            ir::CompareType::NotEquals => {
                match &a._type {
                    ir::TypeType::Builtin(type_builtin) => match type_builtin {
                        ir::TypeBuiltin::String => self.instructions.instr_compare_string(),
                        _ => {
                            self.instructions.instr_compare(a.size as u8);
                        }
                    },
                    _ => unreachable!(),
                }
                self.instructions.instr_negate_bool();
            }
        }

        Ok(ir::BOOL.clone())
    }

    fn compile_type_cast(&mut self, type_cast: &ir::TypeCast) -> Result<ir::Type> {
        let from = self.compile_expression_compact(&type_cast.expression, size_of::<usize>())?;

        match from._type.clone() {
            ir::TypeType::Builtin(builtin) => match builtin {
                ir::TypeBuiltin::Uint8 => match &type_cast._type._type {
                    ir::TypeType::Builtin(builtin_dest) => match builtin_dest {
                        ir::TypeBuiltin::Uint16 => self.instructions.instr_cast_uint(1, 2),
                        ir::TypeBuiltin::Uint32 => self.instructions.instr_cast_uint(1, 4),
                        ir::TypeBuiltin::Uint64 => self.instructions.instr_cast_uint(1, 8),
                        ir::TypeBuiltin::Int8 => self.instructions.instr_cast_uint(1, 1),
                        ir::TypeBuiltin::Int16 => self.instructions.instr_cast_uint(1, 2),
                        ir::TypeBuiltin::Int32 => self.instructions.instr_cast_uint(1, 4),
                        ir::TypeBuiltin::Int64 => self.instructions.instr_cast_uint(1, 8),
                        ir::TypeBuiltin::Int => self.instructions.instr_cast_uint(1, 0),
                        ir::TypeBuiltin::Uint => self.instructions.instr_cast_uint(1, 0),
                        _type => return Err(anyhow!("compile_type_cast: cant cast")),
                    },
                    _type => return Err(anyhow!("compile_type_cast: cant cast")),
                },
                ir::TypeBuiltin::Uint16 => match &type_cast._type._type {
                    ir::TypeType::Builtin(builtin_dest) => match builtin_dest {
                        ir::TypeBuiltin::Uint8 => self.instructions.instr_cast_uint(2, 1),
                        ir::TypeBuiltin::Uint32 => self.instructions.instr_cast_uint(2, 4),
                        ir::TypeBuiltin::Uint64 => self.instructions.instr_cast_uint(2, 8),
                        ir::TypeBuiltin::Int8 => self.instructions.instr_cast_uint(2, 1),
                        ir::TypeBuiltin::Int16 => self.instructions.instr_cast_uint(2, 2),
                        ir::TypeBuiltin::Int32 => self.instructions.instr_cast_uint(2, 4),
                        ir::TypeBuiltin::Int64 => self.instructions.instr_cast_uint(2, 8),
                        ir::TypeBuiltin::Int => self.instructions.instr_cast_uint(2, 0),
                        ir::TypeBuiltin::Uint => self.instructions.instr_cast_uint(2, 0),
                        _type => return Err(anyhow!("compile_type_cast: cant cast")),
                    },
                    _type => return Err(anyhow!("compile_type_cast: cant cast")),
                },
                ir::TypeBuiltin::Uint32 => match &type_cast._type._type {
                    ir::TypeType::Builtin(builtin_dest) => match builtin_dest {
                        ir::TypeBuiltin::Uint8 => self.instructions.instr_cast_uint(4, 1),
                        ir::TypeBuiltin::Uint16 => self.instructions.instr_cast_uint(4, 2),
                        ir::TypeBuiltin::Uint64 => self.instructions.instr_cast_uint(4, 8),
                        ir::TypeBuiltin::Int8 => self.instructions.instr_cast_uint(4, 1),
                        ir::TypeBuiltin::Int16 => self.instructions.instr_cast_uint(4, 2),
                        ir::TypeBuiltin::Int32 => self.instructions.instr_cast_uint(4, 4),
                        ir::TypeBuiltin::Int64 => self.instructions.instr_cast_uint(4, 8),
                        ir::TypeBuiltin::Int => self.instructions.instr_cast_uint(4, 0),
                        ir::TypeBuiltin::Uint => self.instructions.instr_cast_uint(4, 0),
                        _type => return Err(anyhow!("compile_type_cast: cant cast")),
                    },
                    _type => return Err(anyhow!("compile_type_cast: cant cast")),
                },
                ir::TypeBuiltin::Uint64 => match &type_cast._type._type {
                    ir::TypeType::Builtin(builtin_dest) => match builtin_dest {
                        ir::TypeBuiltin::Uint8 => self.instructions.instr_cast_uint(8, 1),
                        ir::TypeBuiltin::Uint16 => self.instructions.instr_cast_uint(8, 2),
                        ir::TypeBuiltin::Uint32 => self.instructions.instr_cast_uint(8, 4),
                        ir::TypeBuiltin::Int8 => self.instructions.instr_cast_uint(8, 1),
                        ir::TypeBuiltin::Int16 => self.instructions.instr_cast_uint(8, 2),
                        ir::TypeBuiltin::Int32 => self.instructions.instr_cast_uint(8, 4),
                        ir::TypeBuiltin::Int64 => self.instructions.instr_cast_uint(8, 8),
                        ir::TypeBuiltin::Int => self.instructions.instr_cast_uint(8, 0),
                        ir::TypeBuiltin::Uint => self.instructions.instr_cast_uint(8, 0),
                        _type => return Err(anyhow!("compile_type_cast: cant cast")),
                    },
                    _type => return Err(anyhow!("compile_type_cast: cant cast")),
                },
                ir::TypeBuiltin::Int8 => match &type_cast._type._type {
                    ir::TypeType::Builtin(builtin_dest) => match builtin_dest {
                        ir::TypeBuiltin::Uint8 => self.instructions.instr_cast_uint(1, 1),
                        ir::TypeBuiltin::Uint16 => self.instructions.instr_cast_uint(1, 2),
                        ir::TypeBuiltin::Uint32 => self.instructions.instr_cast_uint(1, 4),
                        ir::TypeBuiltin::Uint64 => self.instructions.instr_cast_uint(1, 8),
                        ir::TypeBuiltin::Int16 => self.instructions.instr_cast_int(1, 2),
                        ir::TypeBuiltin::Int32 => self.instructions.instr_cast_int(1, 4),
                        ir::TypeBuiltin::Int64 => self.instructions.instr_cast_int(1, 8),
                        ir::TypeBuiltin::Int => self.instructions.instr_cast_int(1, 0),
                        ir::TypeBuiltin::Uint => self.instructions.instr_cast_uint(1, 0),
                        _type => return Err(anyhow!("compile_type_cast: cant cast")),
                    },
                    _type => return Err(anyhow!("compile_type_cast: cant cast")),
                },
                ir::TypeBuiltin::Int16 => match &type_cast._type._type {
                    ir::TypeType::Builtin(builtin_dest) => match builtin_dest {
                        ir::TypeBuiltin::Uint8 => self.instructions.instr_cast_uint(2, 1),
                        ir::TypeBuiltin::Uint16 => self.instructions.instr_cast_uint(2, 2),
                        ir::TypeBuiltin::Uint32 => self.instructions.instr_cast_uint(2, 4),
                        ir::TypeBuiltin::Uint64 => self.instructions.instr_cast_uint(2, 8),
                        ir::TypeBuiltin::Int8 => self.instructions.instr_cast_int(2, 1),
                        ir::TypeBuiltin::Int32 => self.instructions.instr_cast_int(2, 4),
                        ir::TypeBuiltin::Int64 => self.instructions.instr_cast_int(2, 8),
                        ir::TypeBuiltin::Int => self.instructions.instr_cast_int(2, 0),
                        ir::TypeBuiltin::Uint => self.instructions.instr_cast_uint(2, 0),
                        _type => return Err(anyhow!("compile_type_cast: cant cast")),
                    },
                    _type => return Err(anyhow!("compile_type_cast: cant cast")),
                },
                ir::TypeBuiltin::Int32 => match &type_cast._type._type {
                    ir::TypeType::Builtin(builtin_dest) => match builtin_dest {
                        ir::TypeBuiltin::Uint8 => self.instructions.instr_cast_uint(4, 1),
                        ir::TypeBuiltin::Uint16 => self.instructions.instr_cast_uint(4, 2),
                        ir::TypeBuiltin::Uint32 => self.instructions.instr_cast_uint(4, 4),
                        ir::TypeBuiltin::Uint64 => self.instructions.instr_cast_uint(4, 8),
                        ir::TypeBuiltin::Int8 => self.instructions.instr_cast_int(4, 1),
                        ir::TypeBuiltin::Int16 => self.instructions.instr_cast_int(4, 2),
                        ir::TypeBuiltin::Int64 => self.instructions.instr_cast_int(4, 8),
                        ir::TypeBuiltin::Int => self.instructions.instr_cast_int(4, 0),
                        ir::TypeBuiltin::Uint => self.instructions.instr_cast_uint(4, 0),
                        _type => return Err(anyhow!("compile_type_cast: cant cast")),
                    },
                    _type => return Err(anyhow!("compile_type_cast: cant cast")),
                },
                ir::TypeBuiltin::Int64 => match &type_cast._type._type {
                    ir::TypeType::Builtin(builtin_dest) => match builtin_dest {
                        ir::TypeBuiltin::Uint8 => self.instructions.instr_cast_uint(8, 1),
                        ir::TypeBuiltin::Uint16 => self.instructions.instr_cast_uint(8, 2),
                        ir::TypeBuiltin::Uint32 => self.instructions.instr_cast_uint(8, 4),
                        ir::TypeBuiltin::Uint64 => self.instructions.instr_cast_uint(8, 8),
                        ir::TypeBuiltin::Int8 => self.instructions.instr_cast_int(8, 1),
                        ir::TypeBuiltin::Int16 => self.instructions.instr_cast_int(8, 2),
                        ir::TypeBuiltin::Int32 => self.instructions.instr_cast_int(8, 4),
                        ir::TypeBuiltin::Int => self.instructions.instr_cast_int(8, 0),
                        ir::TypeBuiltin::Uint => self.instructions.instr_cast_uint(8, 0),
                        _type => return Err(anyhow!("compile_type_cast: cant cast")),
                    },
                    _type => return Err(anyhow!("compile_type_cast: cant cast")),
                },
                ir::TypeBuiltin::Int => match &type_cast._type._type {
                    ir::TypeType::Builtin(builtin_dest) => match builtin_dest {
                        ir::TypeBuiltin::Uint8 => self.instructions.instr_cast_uint(0, 1),
                        ir::TypeBuiltin::Uint16 => self.instructions.instr_cast_uint(0, 2),
                        ir::TypeBuiltin::Uint32 => self.instructions.instr_cast_uint(0, 4),
                        ir::TypeBuiltin::Uint64 => self.instructions.instr_cast_uint(0, 8),
                        ir::TypeBuiltin::Int8 => self.instructions.instr_cast_int(0, 1),
                        ir::TypeBuiltin::Int16 => self.instructions.instr_cast_int(0, 2),
                        ir::TypeBuiltin::Int32 => self.instructions.instr_cast_int(0, 4),
                        ir::TypeBuiltin::Int64 => self.instructions.instr_cast_int(0, 8),
                        ir::TypeBuiltin::Uint => self.instructions.instr_cast_uint(0, 0),
                        _type => return Err(anyhow!("compile_type_cast: cant cast")),
                    },
                    _type => return Err(anyhow!("compile_type_cast: cant cast")),
                },
                ir::TypeBuiltin::Uint => match &type_cast._type._type {
                    ir::TypeType::Builtin(builtin_dest) => match builtin_dest {
                        ir::TypeBuiltin::Uint8 => self.instructions.instr_cast_uint(0, 1),
                        ir::TypeBuiltin::Uint16 => self.instructions.instr_cast_uint(0, 2),
                        ir::TypeBuiltin::Uint32 => self.instructions.instr_cast_uint(0, 4),
                        ir::TypeBuiltin::Uint64 => self.instructions.instr_cast_uint(0, 8),
                        ir::TypeBuiltin::Int8 => self.instructions.instr_cast_uint(0, 1),
                        ir::TypeBuiltin::Int16 => self.instructions.instr_cast_uint(0, 2),
                        ir::TypeBuiltin::Int32 => self.instructions.instr_cast_uint(0, 4),
                        ir::TypeBuiltin::Int64 => self.instructions.instr_cast_uint(0, 8),
                        ir::TypeBuiltin::Int => self.instructions.instr_cast_uint(0, 0),
                        _type => return Err(anyhow!("compile_type_cast: cant cast")),
                    },
                    _type => return Err(anyhow!("compile_type_cast: cant cast")),
                },

                ir::TypeBuiltin::String => match &type_cast._type._type {
                    ir::TypeType::Slice(slice_item) => {
                        slice_item.must_equal(&ir::UINT8)?;
                    }
                    _ => return Err(anyhow!("compile_type_cast: cant cast")),
                },
                ir::TypeBuiltin::Ptr => match &type_cast._type._type {
                    ir::TypeType::Address(_) => {}
                    _ => return Err(anyhow!("compile_type_cast: cant cast")),
                },
                _ => return Err(anyhow!("compile_type_cast: cant cast")),
            },
            ir::TypeType::Variadic(variadic_item) => match &type_cast._type._type {
                ir::TypeType::Slice(slice_item) => {
                    slice_item.must_equal(&variadic_item)?;
                }
                _ => return Err(anyhow!("compile_type_cast: cant cast")),
            },
            ir::TypeType::Slice(slice_item) => match &type_cast._type._type {
                ir::TypeType::Variadic(variadic_item) => {
                    slice_item.must_equal(variadic_item)?;
                }
                ir::TypeType::Builtin(builtin_dest) => match builtin_dest {
                    ir::TypeBuiltin::Uint8 => {}
                    _ => return Err(anyhow!("compile_type_cast: cant cast")),
                },
                _ => return Err(anyhow!("compile_type_cast: cant cast")),
            },
            ir::TypeType::Address(_) => match &type_cast._type._type {
                ir::TypeType::Builtin(builtin_dest) => match builtin_dest {
                    ir::TypeBuiltin::Ptr => {}
                    _ => return Err(anyhow!("compile_type_cast: cant cast")),
                },
                _ => return Err(anyhow!("compile_type_cast: cant cast")),
            },
            _ => return Err(anyhow!("compile_type_cast: cant cast")),
        }

        Ok(type_cast._type.clone())
    }

    // append(slice ir::Type, value ir::Type) void
    fn compile_function_builtin_append(&mut self, call: &FunctionCall) -> Result<ir::Type> {
        let slice_arg = call
            .arguments
            .get(0)
            .ok_or(anyhow!("append: expected first argument"))?;
        let value_arg = call
            .arguments
            .get(1)
            .ok_or(anyhow!("append: expected second argument"))?;

        let slice_exp = self.compile_expression(slice_arg)?;

        let ir::TypeType::Slice(slice_item) = &slice_exp._type else {
            return Err(anyhow!("append: provide a slice as the first argument"));
        };

        let value_exp = self.compile_expression_compact(value_arg, slice_item.alignment)?;

        if **slice_item != value_exp {
            return Err(anyhow!("append: value type does not match slice type"));
        }

        self.instructions.instr_slice_append(value_exp.size);

        Ok(ir::VOID.clone())
    }

    // len(slice ir::Type) int
    fn compile_function_builtin_len(&mut self, call: &FunctionCall) -> Result<ir::Type> {
        let slice_arg = call
            .arguments
            .get(0)
            .ok_or(anyhow!("len: expected first argument"))?;

        // cleanup align here?
        let slice_exp = self.compile_expression(slice_arg)?;
        let ir::TypeType::Slice(_) = &slice_exp._type else {
            return Err(anyhow!(
                "len: expected slice as the argument, got {:#?}",
                slice_exp._type
            ));
        };

        self.instructions.instr_slice_len();

        Ok(ir::INT.clone())
    }

    // new(_ ir::Type, args ir::Type...) ir::Type
    fn compile_function_builtin_new(&mut self, call: &FunctionCall) -> Result<ir::Type> {
        let type_arg = call
            .arguments
            .get(0)
            .ok_or(anyhow!("new: expected first argument"))?;

        let ir::Expression::Type(_type) = type_arg else {
            return Err(anyhow!("new: expected first argument to be type"));
        };

        match &_type._type {
            ir::TypeType::Slice(slice_item) => {
                let def_val = call
                    .arguments
                    .get(1)
                    .ok_or(anyhow!("new: second argument expected"))?;

                let len_val = call
                    .arguments
                    .get(2)
                    .ok_or(anyhow!("new: third argument expected"))?;

                let len_exp = self.compile_expression(len_val)?;
                if len_exp != *ir::INT {
                    return Err(anyhow!("new: length should be of type int"));
                }

                let def_exp = self.compile_expression_compact(def_val, slice_item.alignment)?;
                if def_exp != **slice_item {
                    return Err(anyhow!("new: expression does not match slice type"));
                }

                self.instructions.instr_push_slice_new_len(def_exp.size);

                Ok(_type.clone())
            }
            _type => return Err(anyhow!("new: {_type:#?} not supported")),
        }
    }

    fn compile_function_call(&mut self, call: &FunctionCall) -> Result<ir::Type> {
        // todo: check these
        // self.check_function_call_argument_count(call)?;

        if let FunctionCallType::Function(function) = &call.call_type {
            match function.identifier.as_str() {
                "append" => return self.compile_function_builtin_append(call),
                "len" => return self.compile_function_builtin_len(call),
                "new" => return self.compile_function_builtin_new(call),
                _ => {}
            }
        }

        let expected_arguments = match &call.call_type {
            FunctionCallType::Function(v) => v._type.expect_function()?.arguments.clone(),
            FunctionCallType::Method(v) => v.function._type.expect_function()?.arguments.clone(),
            FunctionCallType::Closure(v) => v
                ._type(self.instructions)?
                .expect_closure()?
                .arguments
                .clone(),
        };

        self.instructions.push_alignment(ir::PTR_SIZE);

        let argument_size = {
            self.instructions.push_stack_frame();
            for (i, expected_arg) in expected_arguments.iter().enumerate() {
                let mut arg = call.arguments.get(i);
                if i == 0 {
                    if let FunctionCallType::Method(method) = &call.call_type {
                        arg = Some(&method._self);
                    }
                }

                let Some(arg) = arg else {
                    // calling push(int...) -> like push()
                    if expected_arg._type.extract_variadic().is_some() {
                        self.instructions.instr_push_slice();
                        continue;
                    }
                    return Err(anyhow!("compile_function_call: argument missing"));
                };

                let Some(inner) = expected_arg._type.extract_variadic() else {
                    // calling push(a int) -> like push(20)
                    let exp = self.compile_expression_compact(arg, expected_arg._type.alignment)?;
                    exp.must_equal(&expected_arg._type)?;
                    continue;
                };

                if let ir::Expression::Spread(_) = arg {
                    let exp = self.compile_expression_compact(arg, expected_arg._type.alignment)?;
                    if exp != expected_arg._type {
                        return Err(anyhow!("function call type mismatch"));
                    }
                    // this check is weird, because ast actually does not allow creating
                    // declarations where spread is not last, but this probably should be in
                    // compiler
                    if expected_arguments.len() != call.arguments.len() {
                        return Err(anyhow!("spread must be last argument"));
                    }

                    continue;
                }

                self.instructions.instr_push_slice();
                for arg in call.arguments.iter().skip(i) {
                    self.instructions.instr_increment(ir::SLICE_SIZE);
                    self.instructions
                        .instr_copy(0, ir::SLICE_SIZE, ir::SLICE_SIZE);

                    let value_exp = self.compile_expression_compact(arg, inner.alignment)?;
                    value_exp.must_equal(inner)?;

                    self.instructions.instr_slice_append(value_exp.size);
                }
            }
            self.instructions.pop_stack_frame_size()
        };

        if let FunctionCallType::Function(function) = &call.call_type {
            match function.identifier.as_str() {
                "libc_write" => {
                    self.instructions.instr_libc_write();
                    return Ok(ir::INT.clone());
                }
                "dll_open" => {
                    self.instructions.instr_dll_open();
                    return Ok(ir::PTR.clone());
                }
                "ffi_create" => {
                    self.instructions.instr_ffi_create();
                    return Ok(ir::PTR.clone());
                }
                "ffi_call" => {
                    self.instructions.instr_ffi_call();
                    return Ok(ir::PTR.clone());
                }
                _ => {}
            }
        }

        let return_type = match &call.call_type {
            FunctionCallType::Function(v) => v._type.expect_function()?.return_type.clone(),
            FunctionCallType::Method(v) => v.function._type.expect_function()?.return_type.clone(),
            FunctionCallType::Closure(v) => v
                ._type(self.instructions)?
                .expect_closure()?
                .return_type
                .clone(),
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
        reset_size += self.instructions.push_alignment(ir::PTR_SIZE);

        match &call.call_type {
            FunctionCallType::Function(function) => {
                self.instructions
                    .instr_jump_and_link(function.identifier.clone());
            }
            FunctionCallType::Method(method) => {
                self.instructions
                    .instr_jump_and_link(method.function.identifier.clone());
            }
            FunctionCallType::Closure(expression) => {
                self.compile_expression(expression)?;
                self.instructions.instr_jump_and_link_closure();
            }
        }

        self.instructions.instr_reset(reset_size);

        Ok(return_type)
    }

    fn compile_call(&mut self, call: &ir::Call) -> Result<ir::Type> {
        match &call.call_type {
            ir::CallType::Closure(expression) => self.compile_function_call(&FunctionCall {
                arguments: &call.arguments,
                call_type: FunctionCallType::Closure(expression),
            }),
            ir::CallType::Function(function) => self.compile_function_call(&FunctionCall {
                arguments: &call.arguments,
                call_type: FunctionCallType::Function(function),
            }),
            ir::CallType::Method(method) => self.compile_function_call(&FunctionCall {
                arguments: &call.arguments,
                call_type: FunctionCallType::Method(method),
            }),
        }
    }

    fn compile_arithmetic(&mut self, arithmetic: &ir::Arithmetic) -> Result<ir::Type> {
        let a = self.compile_expression(&arithmetic.left)?;
        let b = self.compile_expression(&arithmetic.right)?;

        if a != b {
            return Err(anyhow!("can't add different types"));
        }
        if a == *ir::VOID {
            return Err(anyhow!("can't add void type"));
        }

        match arithmetic._type {
            ir::ArithmeticType::Minus => {
                if *ir::INT == a {
                    self.instructions.instr_minus_int();
                    self.instructions.instr_add_i();
                } else {
                    return Err(anyhow!("can only minus int"));
                }
            }
            ir::ArithmeticType::Plus => {
                if *ir::INT == a {
                    self.instructions.instr_add_i();
                } else if a == *ir::STRING {
                    self.instructions.instr_add_string();
                } else {
                    return Err(anyhow!("can only plus int and string"));
                }
            }
            ir::ArithmeticType::Multiply => {
                if *ir::INT == a {
                    self.instructions.instr_multiply_i();
                } else {
                    return Err(anyhow!("can only multiply int"));
                }
            }
            ir::ArithmeticType::Divide => {
                if *ir::INT == a {
                    self.instructions.instr_divide_i();
                } else {
                    return Err(anyhow!("can only divide int"));
                }
            }
            ir::ArithmeticType::Modulo => {
                if *ir::INT == a {
                    self.instructions.instr_modulo_i();
                } else {
                    return Err(anyhow!("can only modulo int"));
                }
            }
        }

        Ok(a)
    }

    fn compile_literal(&mut self, literal: &ir::Literal) -> Result<ir::Type> {
        match &literal.literal_type {
            ir::LiteralType::Int(int) => {
                if literal._type == *ir::INT {
                    self.instructions.instr_push_i(*int)?;
                } else if literal._type == *ir::INT8 {
                    self.instructions.instr_push_i8(*int)?;
                } else if literal._type == *ir::INT16 {
                    self.instructions.instr_push_i16(*int)?;
                } else if literal._type == *ir::INT32 {
                    self.instructions.instr_push_i32(*int)?;
                } else if literal._type == *ir::INT64 {
                    self.instructions.instr_push_i64(*int)?;
                } else if literal._type == *ir::UINT {
                    self.instructions.instr_push_u(*int)?;
                } else if literal._type == *ir::UINT8 {
                    self.instructions.instr_push_u8(*int)?;
                } else if literal._type == *ir::UINT16 {
                    self.instructions.instr_push_u16(*int)?;
                } else if literal._type == *ir::UINT32 {
                    self.instructions.instr_push_u32(*int)?;
                } else if literal._type == *ir::UINT64 {
                    self.instructions.instr_push_u64(*int)?;
                } else {
                    return Err(anyhow!("can't cast int to {:#?}", literal._type));
                }
            }
            ir::LiteralType::Bool(bool) => {
                if literal._type == *ir::BOOL {
                    self.instructions.instr_push_i({
                        if *bool {
                            1
                        } else {
                            0
                        }
                    })?;
                } else {
                    return Err(anyhow!("can't cast bool to {:#?}", literal._type));
                }
            }
            ir::LiteralType::String(string) => {
                let index = self.static_memory.borrow_mut().push_string_slice(&string);
                self.instructions
                    .instr_push_static(index, ir::SLICE_SIZE, ir::SLICE_SIZE);
            }
        }

        Ok(literal._type.clone())
    }

    fn compile_andor(&mut self, andor: &ir::AndOr) -> Result<ir::Type> {
        self.instructions.stack_instructions.jump();

        let left = self.compile_expression(&andor.left)?;
        if left != *ir::BOOL {
            return Err(anyhow!("compile_andor: expected bool expression"));
        }

        match andor._type {
            ir::AndOrType::Or => {
                self.instructions.stack_instructions.back_if_true(1);

                let right = self.compile_expression(&andor.right)?;
                if right != *ir::BOOL {
                    return Err(anyhow!("compile_andor: expected bool expression"));
                }

                self.instructions.instr_or();
            }
            ir::AndOrType::And => {
                self.instructions.stack_instructions.back_if_false(1);

                let right = self.compile_expression(&andor.right)?;
                if right != *ir::BOOL {
                    return Err(anyhow!("compile_andor: expected bool expression"));
                }

                self.instructions.instr_and();
            }
        }

        self.instructions.stack_instructions.back(1);
        self.instructions.stack_instructions.pop_index();

        Ok(ir::BOOL.clone())
    }

    fn compile_to_closure(&mut self, expression: &ir::Expression) -> Result<ir::Type> {
        todo!();
    }

    fn compile_expression_compact(
        &mut self,
        expression: &ir::Expression,
        base_alignment: usize,
    ) -> Result<ir::Type> {
        self.instructions.push_alignment(base_alignment);

        let old_stack_size = self.instructions.stack_total_size();

        let exp = self.compile_expression(expression)?;

        let new_stack_size = self.instructions.stack_total_size();
        let delta_stack_size = new_stack_size - old_stack_size;

        if delta_stack_size > exp.size {
            self.instructions
                .instr_shift(exp.size, delta_stack_size - exp.size);
        }

        Ok(exp)
    }

    fn compile_expression(&mut self, expression: &ir::Expression) -> Result<ir::Type> {
        Ok(match expression {
            ir::Expression::AndOr(v) => self.compile_andor(v),
            ir::Expression::Literal(v) => self.compile_literal(v),
            ir::Expression::Arithmetic(v) => self.compile_arithmetic(v),
            ir::Expression::Call(v) => self.compile_call(v),
            ir::Expression::Compare(v) => self.compile_compare(v),
            ir::Expression::Infix(v) => self.compile_infix(v),
            ir::Expression::SliceInit(v) => self.compile_slice_init(v),
            ir::Expression::Index(v) => self.compile_expression_index(v),
            ir::Expression::Negate(v) => self.compile_negate(v),
            ir::Expression::Spread(v) => self.compile_spread(v),
            ir::Expression::StructInit(v) => self.compile_struct_init(v),
            ir::Expression::DotAccess(v) => self.compile_dot_access(v),
            ir::Expression::Deref(v) => self.compile_deref(v),
            ir::Expression::Address(v) => self.compile_address(v),
            ir::Expression::Closure(v) => self.compile_closure(v),
            ir::Expression::Nil => self.compile_nil(),
            ir::Expression::Variable(v) => self.compile_variable(v),
            ir::Expression::Type(_type) => Ok(_type.clone()),
            ir::Expression::ToClosure(v) => self.compile_to_closure(v),
            ir::Expression::TypeCast(v) => self.compile_type_cast(v),
            ir::Expression::Function(_) => {
                panic!("cant compile function expression, expected call expression")
            }
            ir::Expression::Method(_) => {
                panic!("cant compile method, expected call expression")
            }
        }?)
    }
}

pub struct Compiled {
    pub functions: HashMap<String, Vec<ScopedInstruction>>,
    pub static_memory: vm::StaticMemory,
}

pub fn compile(ir: ir::Ir) -> Result<Compiled> {
    let mut functions = HashMap::<String, Vec<ScopedInstruction>>::new();
    let static_memory = Rc::new(RefCell::new(vm::StaticMemory::new()));

    for (identifier, declaration) in &ir.functions {
        let compiled =
            FunctionCompiler::new(Function::Function(declaration), static_memory.clone(), &ir)
                .compile()?;

        functions.insert(
            identifier.clone(),
            ScopedInstruction::from_compiled_instructions(&compiled),
        );
    }

    let static_memory = static_memory.borrow().clone();
    Ok(Compiled {
        static_memory,
        functions,
    })
}

pub struct FunctionCompiler<'a> {
    instructions: Instructions,
    function: Function<'a>,
    closures: Vec<CompiledInstructions>,
    ir: &'a ir::Ir<'a>,
    static_memory: Rc<RefCell<vm::StaticMemory>>,
}

#[derive(Debug, Clone)]
pub struct CompiledInstructions {
    pub instructions: Vec<Vec<CompilerInstruction>>,
    pub closures: Vec<CompiledInstructions>,
}

impl<'a> FunctionCompiler<'a> {
    fn new(
        function: Function<'a>,
        static_memory: Rc<RefCell<vm::StaticMemory>>,
        ir: &'a ir::Ir<'a>,
    ) -> Self {
        Self {
            static_memory,
            function,
            closures: Vec::new(),
            instructions: Instructions::new(),
            ir,
        }
    }

    fn expression_compiler(&mut self) -> ExpressionCompiler {
        ExpressionCompiler::new(
            &mut self.instructions,
            &mut self.closures,
            self.ir,
            self.static_memory.clone(),
        )
    }

    fn compile_expression(&mut self, expression: &ir::Expression) -> Result<ir::Type> {
        self.expression_compiler().compile_expression(expression)
    }

    fn compile_expression_compact(
        &mut self,
        expression: &ir::Expression,
        base_alignment: usize,
    ) -> Result<ir::Type> {
        self.expression_compiler()
            .compile_expression_compact(expression, base_alignment)
    }

    fn compile_dot_access_field_offset(
        &mut self,
        dot_access: &ir::DotAccess,
    ) -> Result<DotAccessField> {
        self.expression_compiler()
            .compile_dot_access_field_offset(dot_access)
    }

    fn compile_variable_declaration(
        &mut self,
        declaration: &ir::VariableDeclaration,
    ) -> Result<()> {
        let escaped = declaration.variable.borrow()._type._type.is_escaped();
        if escaped {
            self.instructions.push_alignment(ir::PTR_SIZE);
        }

        let exp = self.compile_expression_compact(
            &declaration.expression,
            declaration.variable.borrow()._type.alignment,
        )?;
        if exp == *ir::VOID {
            return Err(anyhow!("can't declare void variable"));
        }

        exp.must_equal(&declaration.variable.borrow()._type)?;

        if escaped {
            self.instructions.instr_alloc(exp.size, exp.alignment);
        }

        self.instructions
            .var_mark(declaration.variable.borrow().clone());

        Ok(())
    }

    fn compile_variable_assignment(&mut self, assignment: &ir::VariableAssignment) -> Result<()> {
        self.instructions.push_stack_frame();

        match &assignment.var {
            ir::Expression::Variable(identifier) => {
                let (offset, variable) = self.instructions.var_get_offset_err(identifier)?;
                let variable = variable.clone();

                if let ir::TypeType::Escaped(_type) = &variable._type._type {
                    let alignment = self.instructions.push_alignment(ir::PTR_SIZE);
                    self.instructions.instr_increment(ir::PTR_SIZE);
                    self.instructions.instr_copy(
                        0,
                        offset + alignment + ir::PTR_SIZE,
                        ir::PTR_SIZE,
                    );

                    // no alignment because of ir::PTR_SIZE align above
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
            ir::Expression::Index(index) => {
                let slice = self.compile_expression(&index.var)?;
                let ir::TypeType::Slice(slice_item) = &slice._type else {
                    return Err(anyhow!("can only index slices"));
                };

                let item_index = self.compile_expression(&index.expression)?;
                if item_index != *ir::INT {
                    return Err(anyhow!("can only index with int type"));
                }

                let item = self.compile_expression(&assignment.expression)?;
                if **slice_item != item {
                    return Err(anyhow!("slice index set type mismatch"));
                }

                self.instructions.instr_slice_index_set(item.size);
            }
            ir::Expression::DotAccess(dot_access) => {
                let field = self.compile_dot_access_field_offset(dot_access)?;

                self.instructions.push_stack_frame();
                let exp = self.compile_expression(&assignment.expression)?;
                let exp_size = self.instructions.pop_stack_frame_size();

                exp.must_equal(field._type())?;

                match field {
                    DotAccessField::Heap(_type) => {
                        self.instructions.instr_deref_assign(_type.size);
                    }
                    DotAccessField::Stack(offset, _type) => {
                        self.instructions.instr_copy(offset + exp_size, 0, exp.size);
                    }
                }
            }
            ir::Expression::Deref(expression) => {
                let dst = self.compile_expression(expression)?;
                let ir::TypeType::Address(_type) = dst._type else {
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

    fn compile_if_block(
        &mut self,
        expression: &ir::Expression,
        actions: &[ir::Action],
    ) -> Result<()> {
        let exp = self.compile_expression(expression)?;
        if exp != *ir::BOOL {
            return Err(anyhow!("compile_if_block: expected bool expression"));
        }

        self.instructions.stack_instructions.jump_if_true();

        self.compile_actions(actions)?;
        self.instructions.stack_instructions.back(2);
        self.instructions.stack_instructions.pop_index();

        Ok(())
    }

    fn compile_if(&mut self, _if: &ir::If) -> Result<()> {
        self.instructions.push_stack_frame();

        self.instructions.stack_instructions.jump();

        self.compile_if_block(&_if.expression, &_if.actions)?;

        for v in &_if.elseif {
            self.compile_if_block(&v.expression, &v.actions)?;
        }

        if let Some(v) = &_if._else {
            self.compile_actions(&v.actions)?;
        }

        self.instructions.stack_instructions.back(1);
        self.instructions.stack_instructions.pop_index();

        self.instructions.pop_stack_frame();

        Ok(())
    }

    fn compile_for(&mut self, _for: &ir::For) -> Result<()> {
        self.instructions.push_stack_frame();
        if let Some(v) = &_for.initializer {
            self.compile_action(v)?;
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
                if exp != *ir::BOOL {
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

        self.compile_actions(&_for.actions)?;
        self.instructions.pop_stack_frame();

        self.instructions.stack_instructions.back(1);
        self.instructions.stack_instructions.pop_index();

        // continue will jump here
        if let Some(v) = &_for.after_each {
            self.compile_action(v)?;
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

    fn compile_action(&mut self, action: &ir::Action) -> Result<()> {
        match action {
            ir::Action::VariableDeclaration(var) => self.compile_variable_declaration(var)?,
            ir::Action::Return(exp) => self.compile_return(exp.as_ref())?,
            ir::Action::Expression(exp) => {
                self.instructions.push_stack_frame();
                self.compile_expression(exp)?;
                self.instructions.pop_stack_frame();
            }
            ir::Action::VariableAssignment(assignment) => {
                self.compile_variable_assignment(assignment)?;
            }
            ir::Action::If(v) => {
                self.compile_if(v)?;
            }
            ir::Action::Debug => {
                self.instructions.instr_debug();
            }
            ir::Action::For(v) => self.compile_for(v)?,
            ir::Action::Break => {
                self.compile_for_break()?;
            }
            ir::Action::Continue => {
                self.compile_for_continue()?;
            }
        };

        Ok(())
    }

    fn compile_actions(&mut self, actions: &[ir::Action]) -> Result<()> {
        self.instructions.push_stack_frame();

        for action in actions {
            self.compile_action(action)?;
        }

        self.instructions.pop_stack_frame();

        Ok(())
    }

    fn compile_return(&mut self, exp: Option<&ir::Expression>) -> Result<()> {
        if let Some(exp) = exp {
            let exp = self.compile_expression(exp)?;
            if exp != *self.function.return_type() {
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

        self.compile_actions(&self.function.actions().to_vec())?;

        self.compile_return(None)?;

        Ok(CompiledInstructions {
            instructions: self.instructions.get_instructions(),
            closures: self.closures,
        })
    }
}
