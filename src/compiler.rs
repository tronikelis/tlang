use anyhow::{anyhow, Result};

use crate::{ast, lexer, vm};

#[derive(Debug, Clone)]
pub enum Instruction {
    Real(vm::Instruction),
    JumpAndLink(String),
    Jump((usize, usize)),
    JumpIfTrue((usize, usize)),
    JumpIfFalse((usize, usize)),
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

struct StackInstructions {
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

    fn jump(&mut self) {
        let index = self.instructions.len();
        self.push(Instruction::Jump((index, 0)));
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

        self.push(Instruction::Jump((
            label.index,
            self.instructions[label.index].len(),
        )));

        Ok(())
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

    fn back_if_false(&mut self, offset: usize) {
        let target = self.index[self.index.len() - 1 - offset];
        let target_last = self.instructions[target].len();
        self.push(Instruction::JumpIfFalse((target, target_last)));
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

struct Instructions {
    stack_instructions: StackInstructions,
    var_stack: VarStack,
}

impl Instructions {
    fn new() -> Self {
        Self {
            stack_instructions: StackInstructions::new(),
            var_stack: VarStack::new(),
        }
    }

    fn var_mark(&mut self, identifier: String) {
        self.var_stack.push(VarStackItem::Var(identifier));
    }

    fn var_mark_label(&mut self) {
        self.var_stack.push(VarStackItem::Label);
    }

    fn var_get_offset(&self, identifier: &str) -> Option<usize> {
        self.var_stack.get_var_offset(identifier)
    }

    fn var_reset_label(&mut self) {
        self.stack_instructions
            .push(Instruction::Real(vm::Instruction::Reset(
                self.var_stack.get_label_offset().unwrap(),
            )));
    }

    fn instr_slice_index_set(&mut self, size: usize) {
        self.stack_instructions
            .push(Instruction::Real(vm::Instruction::SliceIndexSet(size)));
        self.var_stack
            .push(VarStackItem::Reset(size + SLICE_SIZE + INT.size));
    }

    fn instr_slice_index_get(&mut self, size: usize) {
        self.stack_instructions
            .push(Instruction::Real(vm::Instruction::SliceIndexGet(size)));
        self.var_stack
            .push(VarStackItem::Reset(INT.size + SLICE_SIZE));
        self.var_stack.push(VarStackItem::Increment(size));
    }

    fn instr_slice_len(&mut self) {
        self.stack_instructions
            .push(Instruction::Real(vm::Instruction::SliceLen));
        self.var_stack.push(VarStackItem::Reset(SLICE_SIZE));
        self.var_stack.push(VarStackItem::Increment(INT.size));
    }

    fn instr_slice_append(&mut self, size: usize) {
        self.stack_instructions
            .push(Instruction::Real(vm::Instruction::SliceAppend(size)));
        self.var_stack.push(VarStackItem::Reset(SLICE_SIZE + size));
    }

    fn instr_and(&mut self) {
        self.stack_instructions
            .push(Instruction::Real(vm::Instruction::And));
        self.var_stack.push(VarStackItem::Reset(BOOL.size));
    }

    fn instr_or(&mut self) {
        self.stack_instructions
            .push(Instruction::Real(vm::Instruction::Or));
        self.var_stack.push(VarStackItem::Reset(BOOL.size));
    }

    fn instr_negate_bool(&mut self) {
        self.stack_instructions
            .push(Instruction::Real(vm::Instruction::NegateBool));
    }

    fn instr_increment(&mut self, size: usize) {
        self.stack_instructions
            .push(Instruction::Real(vm::Instruction::Increment(size)));
        self.var_stack.push(VarStackItem::Increment(size));
    }

    fn instr_reset_dangerous_not_synced(&mut self, size: usize) {
        self.stack_instructions
            .push(Instruction::Real(vm::Instruction::Reset(size)));
    }

    fn instr_reset(&mut self, size: usize) {
        self.stack_instructions
            .push(Instruction::Real(vm::Instruction::Reset(size)));
        self.var_stack.push(VarStackItem::Reset(size));
    }

    fn instr_push_slice_new_len(&mut self, size: usize) {
        self.stack_instructions
            .push(Instruction::Real(vm::Instruction::PushSliceNewLen(size)));
        self.var_stack
            .push(VarStackItem::Reset(size + INT.size - SLICE_SIZE));
    }

    fn instr_push_slice(&mut self) {
        self.push_alignment(SLICE_SIZE);

        self.stack_instructions
            .push(Instruction::Real(vm::Instruction::PushSlice));
        self.var_stack.push(VarStackItem::Increment(SLICE_SIZE));
    }

    fn instr_push_u8(&mut self, int: usize) -> Result<()> {
        self.push_alignment(UINT8.alignment);
        let uint8: u8 = int.try_into()?;

        self.stack_instructions
            .push(Instruction::Real(vm::Instruction::PushU8(uint8)));
        self.var_stack.push(VarStackItem::Increment(UINT8.size));

        Ok(())
    }

    fn instr_push_i(&mut self, int: usize) -> Result<()> {
        self.push_alignment(INT.alignment);
        let int: isize = int.try_into()?;

        self.stack_instructions
            .push(Instruction::Real(vm::Instruction::PushI(int)));
        self.var_stack.push(VarStackItem::Increment(INT.size));

        Ok(())
    }

    fn instr_minus_int(&mut self) {
        self.stack_instructions
            .push(Instruction::Real(vm::Instruction::MinusInt));
    }

    fn instr_add_i(&mut self) {
        self.stack_instructions
            .push(Instruction::Real(vm::Instruction::AddI));
        self.var_stack.push(VarStackItem::Reset(INT.size));
    }

    fn instr_multiply_i(&mut self) {
        self.stack_instructions
            .push(Instruction::Real(vm::Instruction::MultiplyI));
        self.var_stack.push(VarStackItem::Reset(INT.size));
    }

    fn instr_divide_i(&mut self) {
        self.stack_instructions
            .push(Instruction::Real(vm::Instruction::DivideI));
        self.var_stack.push(VarStackItem::Reset(INT.size));
    }

    fn instr_modulo_i(&mut self) {
        self.stack_instructions
            .push(Instruction::Real(vm::Instruction::ModuloI));
        self.var_stack.push(VarStackItem::Reset(INT.size));
    }

    fn instr_to_bool(&mut self) {
        self.stack_instructions
            .push(Instruction::Real(vm::Instruction::ToBoolI));
        self.var_stack.push(VarStackItem::Reset(INT.size));
        self.var_stack.push(VarStackItem::Increment(BOOL.size));
    }

    fn instr_compare_i(&mut self) {
        self.stack_instructions
            .push(Instruction::Real(vm::Instruction::CompareI));
        self.var_stack.push(VarStackItem::Reset(INT.size * 2));
        self.var_stack.push(VarStackItem::Increment(BOOL.size));
    }

    fn instr_add_string(&mut self) {
        self.stack_instructions
            .push(Instruction::Real(vm::Instruction::AddString));
        self.var_stack.push(VarStackItem::Reset(STRING.size));
    }

    fn instr_push_static(&mut self, index: usize, size: usize, alignment: usize) {
        self.push_alignment(alignment);

        self.stack_instructions
            .push(Instruction::Real(vm::Instruction::PushStatic(index, size)));
        self.var_stack.push(VarStackItem::Increment(size));
    }

    fn instr_copy(&mut self, dst: usize, src: usize, size: usize) {
        self.stack_instructions
            .push(Instruction::Real(vm::Instruction::Copy(dst, src, size)));
    }

    // align before calling this
    fn instr_cast_int_uint8(&mut self) {
        self.stack_instructions
            .push(Instruction::Real(vm::Instruction::CastIntUint8));

        self.var_stack.push(VarStackItem::Reset(INT.size));
        self.var_stack.push(VarStackItem::Increment(UINT8.size));
    }

    // align before calling this
    fn instr_cast_uint8_int(&mut self) {
        self.stack_instructions
            .push(Instruction::Real(vm::Instruction::CastUint8Int));

        self.var_stack.push(VarStackItem::Reset(UINT8.size));
        self.var_stack.push(VarStackItem::Increment(INT.size));
    }

    fn instr_cast_slice_ptr(&mut self) {
        self.stack_instructions
            .push(Instruction::Real(vm::Instruction::CastSlicePtr));
    }

    fn instr_cast_int_uint(&mut self) {
        self.stack_instructions
            .push(Instruction::Real(vm::Instruction::CastIntUint));
    }

    fn instr_debug(&mut self) {
        self.stack_instructions
            .push(Instruction::Real(vm::Instruction::Debug));
    }

    fn instr_jump_and_link(&mut self, identifier: String) {
        self.stack_instructions
            .push(Instruction::JumpAndLink(identifier));
    }

    fn instr_shift(&mut self, size: usize, amount: usize) {
        self.stack_instructions
            .push(Instruction::Real(vm::Instruction::Shift(size, amount)));
        self.var_stack.push(VarStackItem::Reset(amount));
    }

    fn instr_syscall0(&mut self) {
        self.stack_instructions
            .push(Instruction::Real(vm::Instruction::Syscall0));
    }

    fn instr_syscall1(&mut self) {
        self.stack_instructions
            .push(Instruction::Real(vm::Instruction::Syscall1));
        self.var_stack.push(VarStackItem::Reset(UINT.size * 1));
    }

    fn instr_syscall2(&mut self) {
        self.stack_instructions
            .push(Instruction::Real(vm::Instruction::Syscall2));
        self.var_stack.push(VarStackItem::Reset(UINT.size * 2));
    }

    fn instr_syscall3(&mut self) {
        self.stack_instructions
            .push(Instruction::Real(vm::Instruction::Syscall3));
        self.var_stack.push(VarStackItem::Reset(UINT.size * 3));
    }

    fn instr_syscall4(&mut self) {
        self.stack_instructions
            .push(Instruction::Real(vm::Instruction::Syscall4));
        self.var_stack.push(VarStackItem::Reset(UINT.size * 4));
    }

    fn instr_syscall5(&mut self) {
        self.stack_instructions
            .push(Instruction::Real(vm::Instruction::Syscall5));
        self.var_stack.push(VarStackItem::Reset(UINT.size * 5));
    }

    fn instr_syscall6(&mut self) {
        self.stack_instructions
            .push(Instruction::Real(vm::Instruction::Syscall6));
        self.var_stack.push(VarStackItem::Reset(UINT.size * 6));
    }

    fn push_alignment(&mut self, alignment: usize) {
        let alignment = align(alignment, self.var_stack.total_size());
        if alignment != 0 {
            self.instr_increment(alignment);
        }
    }

    fn stack_total_size(&self) -> usize {
        self.var_stack.total_size()
    }

    fn push_stack_frame(&mut self) {
        self.var_stack.push_frame(Vec::new());
    }

    fn pop_stack_frame(&mut self) {
        let frame = self.var_stack.pop_frame();
        self.stack_instructions
            .push(Instruction::Real(vm::Instruction::Reset(
                VarStack::size_for(frame.iter()),
            )));
    }

    fn pop_stack_frame_size(&mut self) -> usize {
        let frame = self.var_stack.pop_frame();
        let size = VarStack::size_for(frame.iter());
        frame.into_iter().for_each(|v| self.var_stack.push(v));
        size
    }

    fn init_function_prologue(
        &mut self,
        function: &ast::Function,
        type_declarations: &ast::TypeDeclarations,
    ) -> Result<()> {
        // push arguments to var_stack, they are already in the stack
        // push return address to var_stack

        for arg in function.arguments.iter() {
            let _type = resolve_type(type_declarations, &arg._type)?;
            let return_type = resolve_type(type_declarations, &function.return_type)?;

            let alignment = align(
                _type.alignment,
                self.var_stack.total_size() + return_type.size,
            );

            if alignment != 0 {
                self.var_stack.push(VarStackItem::Increment(alignment));
            }

            self.var_stack.push(VarStackItem::Increment(_type.size));
            self.var_mark(arg.identifier.clone());
        }

        // return address
        if function.identifier != "main" {
            let alignment = align(PTR_SIZE, self.var_stack.total_size());
            if alignment != 0 {
                self.var_stack.push(VarStackItem::Increment(alignment));
            }

            self.var_stack.push(VarStackItem::Increment(PTR_SIZE));
        }

        self.var_stack.set_arg_size();

        Ok(())
    }

    fn init_function_epilogue(&mut self, function: &ast::Function) {
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

    fn get_instructions(self) -> Vec<Vec<Instruction>> {
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

    fn identifier_field_count(&self) -> usize {
        // - 1 there is padding at the end
        // / 2 every field has padding before
        (self.fields.len() - 1) / 2
    }
}

const UINT: Type = Type {
    size: size_of::<usize>(),
    alignment: size_of::<usize>(),
    _type: TypeType::Builtin(TypeBuiltin::Uint),
};
const UINT8: Type = Type {
    size: 1,
    alignment: 1,
    _type: TypeType::Builtin(TypeBuiltin::Uint8),
};
const INT: Type = Type {
    size: size_of::<isize>(),
    alignment: size_of::<isize>(),
    _type: TypeType::Builtin(TypeBuiltin::Int),
};
const BOOL: Type = Type {
    size: size_of::<usize>(),      // for now
    alignment: size_of::<usize>(), // for now
    _type: TypeType::Builtin(TypeBuiltin::Bool),
};
const STRING: Type = Type {
    size: size_of::<usize>(),
    alignment: size_of::<usize>(),
    _type: TypeType::Builtin(TypeBuiltin::String),
};
const COMPILER_TYPE: Type = Type {
    size: 0,
    alignment: 0,
    _type: TypeType::Builtin(TypeBuiltin::CompilerType),
};
const VOID: Type = Type {
    size: 0,
    alignment: 0,
    _type: TypeType::Builtin(TypeBuiltin::Void),
};
const PTR: Type = Type {
    size: PTR_SIZE,
    alignment: PTR_SIZE,
    _type: TypeType::Builtin(TypeBuiltin::Ptr),
};
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
}

#[derive(Debug, Clone, PartialEq)]
enum TypeType {
    Struct(TypeStruct),
    Variadic(Box<Type>),
    Slice(Box<Type>),
    Builtin(TypeBuiltin),
}

#[derive(Debug, Clone, PartialEq)]
struct Type {
    size: usize,
    alignment: usize,
    _type: TypeType,
}

impl Type {
    fn can_assign(&self, other: &Self) -> bool {
        match &self._type {
            TypeType::Slice(_) => self.can_assign_slice(other),
            _ => other == self,
        }
    }

    fn can_assign_slice(&self, other: &Self) -> bool {
        let (me, me_depth) = self.extract_slice_type();
        let (other, other_depth) = other.extract_slice_type();
        // some safety checks
        if other_depth == 0 && *other == VOID {
            panic!("impossible other_depth=0 VOID type in slice");
        }

        if other_depth > me_depth {
            return false;
        }

        if other_depth < me_depth {
            return *other == VOID;
        }

        return *other == VOID || other == me;
    }

    fn extract_slice_type(&self) -> (&Type, usize) {
        let mut _type = self;
        let mut i = 0;
        loop {
            if let TypeType::Slice(v) = &_type._type {
                _type = v;
                i += 1;
            } else {
                break;
            }
        }

        (_type, i)
    }

    fn extract_variadic(&self) -> Option<Self> {
        match &self._type {
            TypeType::Variadic(item) => Some(*item.clone()),
            _ => None,
        }
    }
}

fn resolve_type(type_declarations: &ast::TypeDeclarations, _type: &ast::Type) -> Result<Type> {
    match _type {
        ast::Type::Alias(alias) => {
            match alias.as_str() {
                "uint" => return Ok(UINT),
                "uint8" => return Ok(UINT8),
                "int" => return Ok(INT),
                "bool" => return Ok(BOOL),
                "string" => return Ok(STRING),
                "Type" => return Ok(COMPILER_TYPE),
                "void" => return Ok(VOID),
                "ptr" => return Ok(PTR),
                _ => {}
            };

            let inner = type_declarations
                .0
                .get(alias)
                .ok_or(anyhow!("can't resolve {alias:#?}"))?;

            resolve_type(type_declarations, &inner)
        }
        ast::Type::Slice(_type) => Ok(Type {
            size: SLICE_SIZE,
            alignment: SLICE_SIZE,
            _type: TypeType::Slice(Box::new(resolve_type(type_declarations, _type)?)),
        }),
        ast::Type::Variadic(_type) => Ok(Type {
            size: size_of::<usize>(),
            alignment: size_of::<usize>(),
            _type: TypeType::Variadic(Box::new(resolve_type(type_declarations, _type)?)),
        }),
        ast::Type::Struct(type_struct) => {
            let mut fields: Vec<TypeStructField> = Vec::new();
            let mut size: usize = 0;
            let mut highest_alignment: usize = 0;

            for (identifier, _type) in &type_struct.fields {
                let resolved = resolve_type(type_declarations, _type)?;
                if resolved.alignment > highest_alignment {
                    highest_alignment = resolved.alignment;
                }

                let alignment = align(resolved.alignment, size);
                size += resolved.size;
                size += alignment;
                fields.push(TypeStructField::Padding(alignment));
                fields.push(TypeStructField::Type(identifier.clone(), resolved));
            }

            let end_padding = align(highest_alignment, size);
            size += end_padding;
            fields.push(TypeStructField::Padding(end_padding));

            Ok(Type {
                size,
                alignment: highest_alignment,
                _type: TypeType::Struct(TypeStruct { fields }),
            })
        }
    }
}

pub struct FunctionCompiler<'a, 'b, 'c> {
    instructions: Instructions,
    function: &'a ast::Function,
    static_memory: &'b mut vm::StaticMemory,
    type_declarations: &'c ast::TypeDeclarations,
}

impl<'a, 'b, 'c> FunctionCompiler<'a, 'b, 'c> {
    pub fn new(
        function: &'a ast::Function,
        static_memory: &'b mut vm::StaticMemory,
        type_declarations: &'c ast::TypeDeclarations,
    ) -> Self {
        Self {
            static_memory,
            function,
            instructions: Instructions::new(),
            type_declarations,
        }
    }

    fn resolve_type(&self, _type: &ast::Type) -> Result<Type> {
        resolve_type(self.type_declarations, _type)
    }

    // new(_ Type, args Type...) Type
    fn compile_function_builtin_new(&mut self, call: &ast::FunctionCall) -> Result<Type> {
        let type_arg = call
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
                    .arguments
                    .get(1)
                    .ok_or(anyhow!("new: second argument expected"))?;

                let len_val = call
                    .arguments
                    .get(2)
                    .ok_or(anyhow!("new: third argument expected"))?;

                let len_exp = self.compile_expression(len_val)?;
                if len_exp != INT {
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

    // len(slice Type) int
    fn compile_function_builtin_len(&mut self, call: &ast::FunctionCall) -> Result<Type> {
        let slice_arg = call
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

        Ok(INT)
    }

    // append(slice Type, value Type) void
    fn compile_function_builtin_append(&mut self, call: &ast::FunctionCall) -> Result<Type> {
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

        let TypeType::Slice(slice_item) = &slice_exp._type else {
            return Err(anyhow!("append: provide a slice as the first argument"));
        };

        if !slice_item.can_assign(&value_exp) {
            return Err(anyhow!("append: value type does not match slice type"));
        }

        self.instructions.instr_slice_append(value_exp.size);

        Ok(VOID)
    }

    fn check_function_call_argument_count(&self, call: &ast::FunctionCall) -> Result<()> {
        // todo: refactor this arguments check somehow
        if let Some(last) = call.function.arguments.last() {
            let last_type = self.resolve_type(&last._type)?;
            if let TypeType::Variadic(_type) = &last_type._type {
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

    fn compile_function_call(&mut self, call: &ast::FunctionCall) -> Result<Type> {
        self.check_function_call_argument_count(call)?;

        match call.function.identifier.as_str() {
            "append" => return self.compile_function_builtin_append(call),
            "len" => return self.compile_function_builtin_len(call),
            "new" => return self.compile_function_builtin_new(call),
            _ => {}
        }

        self.instructions
            .push_alignment(self.resolve_type(&call.function.return_type)?.alignment);

        let argument_size = {
            self.instructions.push_stack_frame();
            for (i, expected_type) in call.function.arguments.iter().enumerate() {
                let expected_type = self.resolve_type(&expected_type._type)?;
                let arg = call.arguments.get(i);

                let Some(arg) = arg else {
                    if expected_type.extract_variadic().is_some() {
                        self.instructions.instr_push_slice();
                        continue;
                    }
                    return Err(anyhow!("compile_function_call: argument missing"));
                };

                let Some(inner) = expected_type.extract_variadic() else {
                    let exp = self.compile_expression(arg)?;
                    if exp != expected_type {
                        return Err(anyhow!("function call type mismatch"));
                    }
                    continue;
                };

                if let ast::Expression::Spread(_) = arg {
                    let exp = self.compile_expression(arg)?;
                    if exp != expected_type {
                        return Err(anyhow!("function call type mismatch"));
                    }
                    if call.function.arguments.len() != call.arguments.len() {
                        return Err(anyhow!("spread must be last argument"));
                    }

                    continue;
                }

                self.instructions.instr_push_slice();
                for arg in call.arguments.iter().skip(i) {
                    self.instructions.instr_increment(SLICE_SIZE);
                    self.instructions.instr_copy(0, SLICE_SIZE, SLICE_SIZE);

                    let value_exp = self.compile_expression(arg)?;
                    if value_exp != inner {
                        return Err(anyhow!("variadic argument type mismatch"));
                    }

                    self.instructions.instr_slice_append(value_exp.size);
                }
            }
            self.instructions.pop_stack_frame_size()
        };

        match call.function.identifier.as_str() {
            "syscall0" => {
                self.instructions.instr_syscall0();
                return Ok(UINT);
            }
            "syscall1" => {
                self.instructions.instr_syscall1();
                return Ok(UINT);
            }
            "syscall2" => {
                self.instructions.instr_syscall2();
                return Ok(UINT);
            }
            "syscall3" => {
                self.instructions.instr_syscall3();
                return Ok(UINT);
            }
            "syscall4" => {
                self.instructions.instr_syscall4();
                return Ok(UINT);
            }
            "syscall5" => {
                self.instructions.instr_syscall5();
                return Ok(UINT);
            }
            "syscall6" => {
                self.instructions.instr_syscall6();
                return Ok(UINT);
            }
            _ => {}
        }

        let return_type = self.resolve_type(&call.function.return_type)?;
        let return_size = return_type.size;

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

        Ok(return_type)
    }

    fn compile_literal(&mut self, literal: &ast::Literal) -> Result<Type> {
        let literal_type = self.resolve_type(&literal._type)?;

        match &literal.literal {
            lexer::Literal::Int(int) => match &literal_type {
                &UINT8 => {
                    self.instructions.instr_push_u8(*int)?;
                }
                &INT => {
                    self.instructions.instr_push_i(*int)?;
                }
                _type => return Err(anyhow!("can't cast int to {_type:#?}")),
            },
            lexer::Literal::Bool(bool) => match &literal_type {
                &BOOL => {
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
                self.instructions
                    .instr_push_static(index, SLICE_SIZE, SLICE_SIZE);
            }
        }

        Ok(literal_type)
    }

    fn compile_variable(&mut self, variable: &ast::Variable) -> Result<Type> {
        let _type = self.resolve_type(&variable._type)?;
        self.instructions.push_alignment(_type.alignment);

        let offset = self
            .instructions
            .var_get_offset(&variable.identifier)
            .ok_or(anyhow!("compile_identifier: unknown identifier"))?;

        self.instructions.instr_increment(_type.size);
        self.instructions
            .instr_copy(0, offset + _type.size, _type.size);

        Ok(_type)
    }

    fn compile_arithmetic(&mut self, arithmetic: &ast::Arithmetic) -> Result<Type> {
        let a = self.compile_expression(&arithmetic.left)?;
        let b = self.compile_expression(&arithmetic.right)?;

        if a != b {
            return Err(anyhow!("can't add different types"));
        }
        if a == VOID {
            return Err(anyhow!("can't add void type"));
        }

        match arithmetic._type {
            ast::ArithmeticType::Minus => {
                if INT == a {
                    self.instructions.instr_minus_int();
                    self.instructions.instr_add_i();
                } else {
                    return Err(anyhow!("can only minus int"));
                }
            }
            ast::ArithmeticType::Plus => {
                if INT == a {
                    self.instructions.instr_add_i();
                } else if a == STRING {
                    self.instructions.instr_add_string();
                } else {
                    return Err(anyhow!("can only plus int and string"));
                }
            }
            ast::ArithmeticType::Multiply => {
                if INT == a {
                    self.instructions.instr_multiply_i();
                } else {
                    return Err(anyhow!("can only multiply int"));
                }
            }
            ast::ArithmeticType::Divide => {
                if INT == a {
                    self.instructions.instr_divide_i();
                } else {
                    return Err(anyhow!("can only divide int"));
                }
            }
            ast::ArithmeticType::Modulo => {
                if INT == a {
                    self.instructions.instr_modulo_i();
                } else {
                    return Err(anyhow!("can only modulo int"));
                }
            }
        }

        Ok(a)
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

        match a {
            BOOL | INT => {}
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

        Ok(BOOL)
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

    fn compile_list(&mut self, list: &[ast::Expression]) -> Result<Type> {
        self.instructions.instr_push_slice();

        let mut curr_exp: Option<Type> = None;

        for v in list {
            self.instructions.push_stack_frame();

            self.instructions.instr_increment(SLICE_SIZE);
            self.instructions.instr_copy(0, SLICE_SIZE, SLICE_SIZE);

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

        Ok(Type {
            size: SLICE_SIZE,
            alignment: SLICE_SIZE,
            _type: curr_exp
                .map(|v| TypeType::Slice(Box::new(v)))
                .unwrap_or(TypeType::Slice(Box::new(VOID))),
        })
    }

    fn compile_expression_index(&mut self, index: &ast::Index) -> Result<Type> {
        let exp_var = self.compile_expression(&index.var)?;

        let TypeType::Slice(expected_type) = exp_var._type else {
            return Err(anyhow!("can't index this type"));
        };

        let exp_index = self.compile_expression(&index.expression)?;
        if exp_index != INT {
            return Err(anyhow!("cant index with {exp_index:#?}"));
        }

        self.instructions.instr_slice_index_get(expected_type.size);

        Ok(*expected_type)
    }

    fn compile_andor(&mut self, andor: &ast::AndOr) -> Result<Type> {
        self.instructions.stack_instructions.jump();

        let left = self.compile_expression(&andor.left)?;
        if left != BOOL {
            return Err(anyhow!("compile_andor: expected bool expression"));
        }

        match andor._type {
            ast::AndOrType::Or => {
                self.instructions.stack_instructions.back_if_true(1);

                let right = self.compile_expression(&andor.right)?;
                if right != BOOL {
                    return Err(anyhow!("compile_andor: expected bool expression"));
                }

                self.instructions.instr_or();
            }
            ast::AndOrType::And => {
                self.instructions.stack_instructions.back_if_false(1);

                let right = self.compile_expression(&andor.right)?;
                if right != BOOL {
                    return Err(anyhow!("compile_andor: expected bool expression"));
                }

                self.instructions.instr_and();
            }
        }

        self.instructions.stack_instructions.back(1);
        self.instructions.stack_instructions.pop_index();

        Ok(BOOL)
    }

    fn compile_type_cast(&mut self, type_cast: &ast::TypeCast) -> Result<Type> {
        let type_cast_type = self.resolve_type(&type_cast._type)?;
        self.instructions.push_alignment(type_cast_type.alignment);

        let target = self.compile_expression(&type_cast.expression)?;

        match target._type {
            TypeType::Builtin(builtin) => match builtin {
                TypeBuiltin::Int => match &type_cast_type._type {
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
                TypeBuiltin::Uint8 => match &type_cast_type._type {
                    TypeType::Builtin(builtin_dest) => match builtin_dest {
                        TypeBuiltin::Int => {
                            self.instructions.instr_cast_uint8_int();
                        }
                        _ => return Err(anyhow!("compile_type_cast: cant cast")),
                    },
                    _ => return Err(anyhow!("compile_type_cast: cant cast")),
                },
                TypeBuiltin::Ptr => match &type_cast_type._type {
                    TypeType::Builtin(builtin_dest) => match builtin_dest {
                        TypeBuiltin::Uint => {}
                        _ => return Err(anyhow!("compile_type_cast: cant cast")),
                    },
                    _ => return Err(anyhow!("compile_type_cast: cant cast")),
                },
                TypeBuiltin::String => match &type_cast_type._type {
                    TypeType::Slice(item) => {
                        if **item != UINT8 {
                            return Err(anyhow!(
                                "compile_type_cast: can only cast string to uint8[]"
                            ));
                        }
                    }
                    _ => return Err(anyhow!("compile_type_cast: cant cast")),
                },
                _ => return Err(anyhow!("compile_type_cast: cant cast")),
            },
            TypeType::Slice(item_target) => match &type_cast_type._type {
                TypeType::Builtin(builtin_dest) => match builtin_dest {
                    TypeBuiltin::String => {
                        if *item_target != UINT8 {
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
            TypeType::Variadic(item_target) => match &type_cast_type._type {
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

        Ok(type_cast_type)
    }

    fn compile_negate(&mut self, negate: &ast::Expression) -> Result<Type> {
        let exp_bool = self.compile_expression(negate)?;
        if exp_bool != BOOL {
            return Err(anyhow!("can only negate bools"));
        }

        self.instructions.instr_negate_bool();

        Ok(BOOL)
    }

    fn compile_spread(&mut self, expression: &ast::Expression) -> Result<Type> {
        let exp = self.compile_expression(expression)?;

        let TypeType::Slice(slice_item) = exp._type else {
            return Err(anyhow!("compile_spread: can only spread slice types"));
        };

        Ok(Type {
            size: SLICE_SIZE,
            alignment: SLICE_SIZE,
            _type: TypeType::Variadic(slice_item),
        })
    }

    fn compile_struct_init(&mut self, struct_init: &ast::StructInit) -> Result<Type> {
        let resolved_type = self.resolve_type(&struct_init._type)?;

        let TypeType::Struct(type_struct) = &resolved_type._type else {
            return Err(anyhow!("compile_struct_init: incorrect type"));
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
                    if exp_type != *_type {
                        return Err(anyhow!("compile_struct_init: incorrect field type"));
                    }
                }
            }
        }

        Ok(resolved_type)
    }

    fn compile_expression(&mut self, expression: &ast::Expression) -> Result<Type> {
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
            ast::Expression::StructInit(v) => self.compile_struct_init(v),
            ast::Expression::Type(v) => {
                Err(anyhow!("compile_expression: cant compile type {v:#?}"))
            }
            ast::Expression::DotAccess(v) => todo!(),
        }?;

        if exp.alignment != 0 {
            let new_stack_size = self.instructions.stack_total_size();
            let delta_stack_size = new_stack_size - old_stack_size;

            if old_stack_size % exp.alignment == 0 && delta_stack_size > exp.size {
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
        if exp == VOID {
            return Err(anyhow!("can't declare void variable"));
        }

        if !self
            .resolve_type(&declaration.variable._type)?
            .can_assign(&exp)
        {
            return Err(anyhow!("type mismatch"));
        }

        self.instructions
            .var_mark(declaration.variable.identifier.clone());

        Ok(())
    }

    fn compute_dot_access_field_offset(
        &self,
        dot_access: &ast::DotAccess,
    ) -> Result<(usize, Type)> {
        if let ast::Expression::DotAccess(inner) = &dot_access.expression {
            let (offset, _type) = self.compute_dot_access_field_offset(inner)?;
            let TypeType::Struct(type_struct) = &_type._type else {
                unreachable!();
            };

            let (field_offset, field_type) = type_struct
                .get_field_offset(&dot_access.identifier)
                .ok_or(anyhow!("struct field not found"))?;

            return Ok((offset + field_offset, field_type.clone()));
        }

        let ast::Expression::Variable(variable) = &dot_access.expression else {
            return Err(anyhow!("cant dot access non variable"));
        };

        let offset = self
            .instructions
            .var_get_offset(&variable.identifier)
            .ok_or(anyhow!("variable assignment variable not found"))?;

        let _type = self.resolve_type(&variable._type)?;
        let TypeType::Struct(type_struct) = &_type._type else {
            return Err(anyhow!("cant dot access non struct type"));
        };

        let (field_offset, field_type) = type_struct
            .get_field_offset(&dot_access.identifier)
            .ok_or(anyhow!("struct field not found"))?;

        Ok((offset + field_offset, field_type.clone()))
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

                if self.resolve_type(&var._type)? != exp {
                    return Err(anyhow!("variable assignment type mismatch"));
                }

                self.instructions.instr_copy(offset, 0, exp.size);
            }
            ast::Expression::Index(index) => {
                let slice = self.compile_expression(&index.var)?;
                let TypeType::Slice(slice_item) = &slice._type else {
                    return Err(anyhow!("can only index slices"));
                };

                let item_index = self.compile_expression(&index.expression)?;
                if item_index != INT {
                    return Err(anyhow!("can only index with int type"));
                }

                let item = self.compile_expression(&assignment.expression)?;
                if !slice_item.can_assign(&item) {
                    return Err(anyhow!("slice index set type mismatch"));
                }

                self.instructions.instr_slice_index_set(item.size);
            }
            ast::Expression::DotAccess(dot_access) => {
                let exp = self.compile_expression(&assignment.expression)?;
                let (offset, _type) = self.compute_dot_access_field_offset(dot_access)?;
                if exp != _type {
                    return Err(anyhow!("dot access assign type mismatch"));
                }

                self.instructions.instr_copy(offset, 0, exp.size);
            }
            node => return Err(anyhow!("can't assign {node:#?}")),
        }

        self.instructions.pop_stack_frame();

        Ok(())
    }

    fn compile_if_block(&mut self, expression: &ast::Expression, body: &[ast::Node]) -> Result<()> {
        let exp = self.compile_expression(expression)?;
        if exp != BOOL {
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
                if exp != BOOL {
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
            if exp != self.resolve_type(&self.function.return_type)? {
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

    pub fn compile(mut self) -> Result<Vec<Vec<Instruction>>> {
        self.instructions
            .init_function_prologue(self.function, self.type_declarations)?;

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
