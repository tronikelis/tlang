use anyhow::{anyhow, Result};

use crate::{ast, vm};

#[derive(Debug, Clone)]
pub enum Instruction {
    Real(vm::Instruction),
    JumpAndLink(String),
    Jump((usize, usize)),
    JumpIfTrue((usize, usize)),
}

#[derive(Debug, Clone, PartialEq)]
enum VarStackItem {
    Increment(usize),
    Reset(usize),
    Var(String),
    Label,
}

#[derive(Debug, Clone)]
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

    fn push_frame(&mut self, frame: Vec<VarStackItem>) {
        self.stack.push(frame);
    }

    fn pop_frame(&mut self) -> Vec<VarStackItem> {
        self.stack.pop().unwrap_or(Vec::new())
    }

    fn push(&mut self, item: VarStackItem) {
        self.stack.last_mut().unwrap().push(item);
    }

    fn total_size(&self) -> usize {
        Self::size_for(self.stack.iter().flatten())
    }

    fn offset_for(&self, target: &VarStackItem) -> Option<usize> {
        let mut offset: isize = 0;

        for item in self.stack.iter().flatten().rev() {
            if item == target {
                return Some(offset.try_into().unwrap());
            }

            match item {
                VarStackItem::Label | VarStackItem::Var(_) => {}
                VarStackItem::Increment(size) => offset += *size as isize,
                VarStackItem::Reset(size) => offset -= *size as isize,
            }
        }

        None
    }

    fn size_for<'a>(items: impl Iterator<Item = &'a VarStackItem>) -> usize {
        items.fold(0, |acc, curr| match curr {
            VarStackItem::Var(_) => acc,
            VarStackItem::Increment(size) => acc + size,
            VarStackItem::Reset(size) => acc - size,
            VarStackItem::Label => acc,
        })
    }

    fn get_var_offset(&self, identifier: &str) -> Option<usize> {
        self.offset_for(&VarStackItem::Var(identifier.to_string()))
    }

    fn get_label_offset(&self) -> Option<usize> {
        self.offset_for(&VarStackItem::Label)
    }
}

#[derive(Debug, PartialEq)]
struct StackLabel {
    identifier: String,
    index: usize,
}

pub struct StackInstructions {
    instructions: Vec<Vec<Instruction>>,
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

    fn push(&mut self, instruction: Instruction) {
        self.instructions[*self.index.last().unwrap()].push(instruction);
    }

    pub fn jump(&mut self) {
        let index = self.instructions.len();
        self.push(Instruction::Jump((index, 0)));
        self.instructions.push(Vec::new());
        self.index.push(index);
    }

    pub fn label_new(&mut self, identifier: String) {
        let index = *self.index.last().unwrap();
        self.labels.push(StackLabel { index, identifier });
    }

    pub fn label_pop(&mut self) {
        self.labels.pop();
    }

    pub fn label_jump(&mut self, identifier: &str) -> Result<()> {
        let label = self
            .labels
            .iter()
            .rev()
            .find(|v| &v.identifier == identifier)
            .ok_or(anyhow!("label_jump: {:#?} not found", identifier))?;

        self.push(Instruction::Jump((
            label.index,
            self.instructions[label.index].len(),
        )));

        Ok(())
    }

    pub fn jump_if_true(&mut self) {
        let index = self.instructions.len();
        self.push(Instruction::JumpIfTrue((index, 0)));
        self.instructions.push(Vec::new());
        self.index.push(index);
    }

    pub fn back_if_true(&mut self, offset: usize) {
        let target = self.index[self.index.len() - 1 - offset];
        let target_last = self.instructions[target].len();
        self.push(Instruction::JumpIfTrue((target, target_last)));
    }

    pub fn again(&mut self) {
        self.push(Instruction::Jump((*self.index.last().unwrap(), 0)));
    }

    pub fn back(&mut self, offset: usize) {
        let target = self.index[self.index.len() - 1 - offset];
        let target_last = self.instructions[target].len();
        self.push(Instruction::Jump((target, target_last)));
    }

    pub fn pop_index(&mut self) {
        self.index.pop();
    }
}

fn align(size: usize, stack_size: usize) -> usize {
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

pub struct Instructions {
    pub stack_instructions: StackInstructions,
    var_stack: VarStack,
}

impl Instructions {
    pub fn new() -> Self {
        Self {
            stack_instructions: StackInstructions::new(),
            var_stack: VarStack::new(),
        }
    }

    pub fn var_mark(&mut self, identifier: String) {
        self.var_stack.push(VarStackItem::Var(identifier));
    }

    pub fn var_mark_label(&mut self) {
        self.var_stack.push(VarStackItem::Label);
    }

    pub fn var_get_offset(&mut self, identifier: &str) -> Option<usize> {
        self.var_stack.get_var_offset(identifier)
    }

    pub fn var_reset_label(&mut self) {
        self.stack_instructions
            .push(Instruction::Real(vm::Instruction::Reset(
                self.var_stack.get_label_offset().unwrap(),
            )));
    }

    pub fn instr_slice_index_set(&mut self, size: usize) {
        self.stack_instructions
            .push(Instruction::Real(vm::Instruction::SliceIndexSet(size)));
        self.var_stack
            .push(VarStackItem::Reset(size + ast::SLICE_SIZE + ast::INT.size));
    }

    pub fn instr_slice_index_get(&mut self, size: usize) {
        self.stack_instructions
            .push(Instruction::Real(vm::Instruction::SliceIndexGet(size)));
        self.var_stack
            .push(VarStackItem::Reset(ast::INT.size + ast::SLICE_SIZE));
        self.var_stack.push(VarStackItem::Increment(size));
    }

    pub fn instr_slice_len(&mut self) {
        self.stack_instructions
            .push(Instruction::Real(vm::Instruction::SliceLen));
        self.var_stack.push(VarStackItem::Reset(ast::SLICE_SIZE));
        self.var_stack.push(VarStackItem::Increment(ast::INT.size));
    }

    pub fn instr_slice_append(&mut self, size: usize) {
        self.stack_instructions
            .push(Instruction::Real(vm::Instruction::SliceAppend(size)));
        self.var_stack
            .push(VarStackItem::Reset(ast::SLICE_SIZE + size));
    }

    pub fn instr_syscall_write(&mut self) {
        self.stack_instructions
            .push(Instruction::Real(vm::Instruction::SyscallWrite));
        self.var_stack
            .push(VarStackItem::Reset(ast::SLICE_SIZE + ast::INT.size));
    }

    pub fn instr_and(&mut self) {
        self.stack_instructions
            .push(Instruction::Real(vm::Instruction::And));
        self.var_stack.push(VarStackItem::Reset(ast::BOOL.size));
    }

    pub fn instr_or(&mut self) {
        self.stack_instructions
            .push(Instruction::Real(vm::Instruction::Or));
        self.var_stack.push(VarStackItem::Reset(ast::BOOL.size));
    }

    pub fn instr_negate_bool(&mut self) {
        self.stack_instructions
            .push(Instruction::Real(vm::Instruction::NegateBool));
    }

    pub fn instr_increment(&mut self, size: usize) {
        self.stack_instructions
            .push(Instruction::Real(vm::Instruction::Increment(size)));
        self.var_stack.push(VarStackItem::Increment(size));
    }

    pub fn instr_reset_dangerous_not_synced(&mut self, size: usize) {
        self.stack_instructions
            .push(Instruction::Real(vm::Instruction::Reset(size)));
    }

    pub fn instr_reset(&mut self, size: usize) {
        self.stack_instructions
            .push(Instruction::Real(vm::Instruction::Reset(size)));
        self.var_stack.push(VarStackItem::Reset(size));
    }

    pub fn instr_push_slice(&mut self) {
        self.push_alignment(ast::SLICE_SIZE);

        self.stack_instructions
            .push(Instruction::Real(vm::Instruction::PushSlice));
        self.var_stack
            .push(VarStackItem::Increment(ast::SLICE_SIZE));
    }

    pub fn instr_push_u8(&mut self, int: usize) -> Result<()> {
        self.push_alignment(ast::UINT8.size);
        let uint8: u8 = int.try_into()?;

        self.stack_instructions
            .push(Instruction::Real(vm::Instruction::PushU8(uint8)));
        self.var_stack
            .push(VarStackItem::Increment(ast::UINT8.size));

        Ok(())
    }

    pub fn instr_push_i(&mut self, int: usize) -> Result<()> {
        self.push_alignment(ast::INT.size);
        let int: isize = int.try_into()?;

        self.stack_instructions
            .push(Instruction::Real(vm::Instruction::PushI(int)));
        self.var_stack.push(VarStackItem::Increment(ast::INT.size));

        Ok(())
    }

    pub fn instr_minus_int(&mut self) {
        self.stack_instructions
            .push(Instruction::Real(vm::Instruction::MinusInt));
    }

    pub fn instr_add_i(&mut self) {
        self.stack_instructions
            .push(Instruction::Real(vm::Instruction::AddI));
        self.var_stack.push(VarStackItem::Reset(ast::INT.size));
    }

    pub fn instr_multiply_i(&mut self) {
        self.stack_instructions
            .push(Instruction::Real(vm::Instruction::MultiplyI));
        self.var_stack.push(VarStackItem::Reset(ast::INT.size));
    }

    pub fn instr_divide_i(&mut self) {
        self.stack_instructions
            .push(Instruction::Real(vm::Instruction::DivideI));
        self.var_stack.push(VarStackItem::Reset(ast::INT.size));
    }

    pub fn instr_modulo_i(&mut self) {
        self.stack_instructions
            .push(Instruction::Real(vm::Instruction::ModuloI));
        self.var_stack.push(VarStackItem::Reset(ast::INT.size));
    }

    pub fn instr_to_bool(&mut self) {
        self.stack_instructions
            .push(Instruction::Real(vm::Instruction::ToBoolI));
        self.var_stack.push(VarStackItem::Reset(ast::INT.size));
        self.var_stack.push(VarStackItem::Increment(ast::BOOL.size));
    }

    pub fn instr_compare_i(&mut self) {
        self.stack_instructions
            .push(Instruction::Real(vm::Instruction::CompareI));
        self.var_stack.push(VarStackItem::Reset(ast::INT.size * 2));
        self.var_stack.push(VarStackItem::Increment(ast::BOOL.size));
    }

    pub fn instr_add_string(&mut self) {
        self.stack_instructions
            .push(Instruction::Real(vm::Instruction::AddString));
        self.var_stack.push(VarStackItem::Reset(ast::STRING.size));
    }

    pub fn instr_push_static(&mut self, index: usize, size: usize) {
        self.push_alignment(size);

        self.stack_instructions
            .push(Instruction::Real(vm::Instruction::PushStatic(index, size)));
        self.var_stack.push(VarStackItem::Increment(size));
    }

    pub fn instr_copy(&mut self, dst: usize, src: usize, size: usize) {
        self.stack_instructions
            .push(Instruction::Real(vm::Instruction::Copy(dst, src, size)));
    }

    // align before calling this
    pub fn instr_cast_int_uint8(&mut self) {
        self.stack_instructions
            .push(Instruction::Real(vm::Instruction::CastIntUint8));

        self.var_stack.push(VarStackItem::Reset(ast::INT.size));
        self.var_stack
            .push(VarStackItem::Increment(ast::UINT8.size));
    }

    // align before calling this
    pub fn instr_cast_uint8_int(&mut self) {
        self.stack_instructions
            .push(Instruction::Real(vm::Instruction::CastUint8Int));

        self.var_stack.push(VarStackItem::Reset(ast::UINT8.size));
        self.var_stack.push(VarStackItem::Increment(ast::INT.size));
    }

    pub fn instr_debug(&mut self) {
        self.stack_instructions
            .push(Instruction::Real(vm::Instruction::Debug));
    }

    pub fn instr_jump_and_link(&mut self, identifier: String) {
        self.stack_instructions
            .push(Instruction::JumpAndLink(identifier));
    }

    pub fn instr_shift(&mut self, size: usize, amount: usize) {
        for i in 0..amount {
            self.stack_instructions
                .push(Instruction::Real(vm::Instruction::Shift(size + amount - i)));
        }
        self.var_stack.push(VarStackItem::Reset(amount));
    }

    pub fn push_alignment(&mut self, size: usize) {
        let alignment = align(size, self.var_stack.total_size());
        if alignment != 0 {
            self.instr_increment(alignment);
        }
    }

    pub fn stack_total_size(&self) -> usize {
        self.var_stack.total_size()
    }

    pub fn push_stack_frame(&mut self) {
        self.var_stack.push_frame(Vec::new());
    }

    pub fn pop_stack_frame(&mut self) {
        self.stack_instructions
            .push(Instruction::Real(vm::Instruction::Reset(
                VarStack::size_for(self.var_stack.pop_frame().iter()),
            )));
    }

    pub fn pop_stack_frame_size(&mut self) -> usize {
        let frame = self.var_stack.pop_frame();
        let size = VarStack::size_for(frame.iter());
        frame.into_iter().for_each(|v| self.var_stack.push(v));
        size
    }

    pub fn init_function_prologue(&mut self, function: &ast::Function) {
        // push arguments to var_stack, they are already in the stack
        // push return address to var_stack

        for arg in function.arguments.iter() {
            let alignment = align(
                arg._type.size,
                self.var_stack.total_size() + function.return_type.size,
            );
            if alignment != 0 {
                self.var_stack.push(VarStackItem::Increment(alignment));
            }

            self.var_stack.push(VarStackItem::Increment(arg._type.size));
            self.var_mark(arg.identifier.clone());
        }

        // return address
        if function.identifier != "main" {
            let alignment = align(ast::PTR_SIZE, self.var_stack.total_size());
            if alignment != 0 {
                self.var_stack.push(VarStackItem::Increment(alignment));
            }

            self.var_stack.push(VarStackItem::Increment(ast::PTR_SIZE));
        }

        self.var_stack.set_arg_size();
    }

    pub fn init_function_epilogue(&mut self, function: &ast::Function) {
        self.stack_instructions
            .push(Instruction::Real(vm::Instruction::Reset(
                self.var_stack.total_size() - self.var_stack.arg_size.unwrap(),
            )));

        if function.identifier == "main" {
            self.stack_instructions
                .push(Instruction::Real(vm::Instruction::Exit));
        } else {
            self.stack_instructions
                .push(Instruction::Real(vm::Instruction::Return));
        }
    }

    pub fn get_instructions(self) -> Vec<Vec<Instruction>> {
        self.stack_instructions.instructions
    }
}
