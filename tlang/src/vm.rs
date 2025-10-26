use core::{cell::RefCell, str};
use std::{
    alloc::{alloc, dealloc, Layout},
    collections::HashMap,
    env,
    ffi::{CStr, CString},
    mem, ptr, slice,
    str::FromStr,
};

use anyhow::Result;

#[cfg(test)]
mod tests;

macro_rules! pop_bytes {
    ($type:ident, $ptr:expr) => {{
        let mut arr: [u8; size_of::<$type>()] = [0; size_of::<$type>()];
        let slice = slice::from_raw_parts(*$ptr, size_of::<$type>());
        for (i, v) in slice.iter().enumerate() {
            arr[i] = *v;
        }
        *$ptr = $ptr.byte_offset(size_of::<$type>() as isize);
        $type::from_le_bytes(arr)
    }};
}

macro_rules! push_bytes {
    ($container:expr, $($expr:expr),*) => {{
        $({
            $expr
                .to_le_bytes()
                .into_iter()
                .for_each(|v| $container.push(v))
        })*
    }};
}

fn is_debug() -> bool {
    env::var("DEBUG").is_ok()
}

unsafe fn alloc_value<T>(value: T) -> (*mut u8, Layout) {
    let layout = Layout::for_value(&value);
    let ptr = alloc(layout);
    *ptr.cast() = value;
    (ptr, layout)
}

#[derive(Debug)]
enum GcObjectData {
    Slice(*mut Slice),
    Alloced(*mut u8, Layout),
    Cif(*mut Cif),
}

#[derive(Debug)]
struct GcObject {
    marked: bool,
    data: GcObjectData,
}

impl GcObject {
    fn new(data: GcObjectData) -> Self {
        Self {
            marked: false,
            data,
        }
    }

    fn new_closure(vars: &[*mut u8], function_index: usize) -> (Self, *mut u8) {
        // closure:
        //
        // function index
        // vars count N
        // ...var1
        // ...var2
        // ...varN

        let size = vars.len() * size_of::<usize>() + size_of::<usize>() * 2;
        let layout = Layout::from_size_align(size, size_of::<usize>()).unwrap();
        let alloced = unsafe { alloc(layout) };

        unsafe {
            let mut alloced = alloced;
            *alloced.cast() = function_index;

            alloced = alloced.byte_offset(size_of::<usize>() as isize);
            *alloced.cast() = vars.len();

            for var in vars {
                alloced = alloced.byte_offset(size_of::<usize>() as isize);
                *alloced.cast::<*mut u8>() = *var;
            }
        }

        (Self::new(GcObjectData::Alloced(alloced, layout)), alloced)
    }

    fn from_slice_val(slice: &[u8], alignment: usize) -> (Self, *mut u8) {
        let layout = Layout::from_size_align(slice.len(), alignment).unwrap();
        let ptr = unsafe { alloc(layout) };
        let data = GcObjectData::Alloced(ptr, layout);

        unsafe {
            ptr::copy_nonoverlapping(slice.as_ptr(), ptr, slice.len());
        };

        (Self::new(data), ptr)
    }
}

struct Gc {
    objects: HashMap<*const u8, RefCell<GcObject>>,
}

impl Gc {
    fn new() -> Self {
        Self {
            objects: HashMap::new(),
        }
    }

    fn add_object(&mut self, object: GcObject) {
        let addr: *const u8 = match object.data {
            GcObjectData::Slice(slice) => slice.cast(),
            GcObjectData::Alloced(ptr, _) => ptr,
            GcObjectData::Cif(v) => v.cast(),
        };
        self.objects.insert(addr, RefCell::new(object));
    }

    fn mark_ptr(&self, mut ptr: *mut u8, size: usize) {
        if size % size_of::<usize>() != 0 {
            return;
        }

        for _ in 0..size / size_of::<usize>() {
            let addr: *const u8 = unsafe {
                let val = *ptr.cast();
                ptr = ptr.byte_offset(size_of::<usize>() as isize);
                val
            };

            if let Some(obj) = self.objects.get(&addr) {
                self.mark_object(obj)
            }
        }
    }

    fn mark_object(&self, object: &RefCell<GcObject>) {
        if object.borrow().marked {
            return;
        }
        object.borrow_mut().marked = true;

        match object.borrow().data {
            GcObjectData::Slice(slice) => {
                let slice = unsafe { &mut *slice };
                self.mark_ptr(slice.data.as_mut_ptr(), slice.data.len());
            }
            GcObjectData::Alloced(ptr, layout) => {
                self.mark_ptr(ptr, layout.size());
            }
            GcObjectData::Cif(_) => {
                // Cifs cant contain nested pointers
            }
        }
    }

    // [align 8] sp_end end is pointing to the end address + 1
    // [align ?] sp is pointing to current address unknown size
    fn mark(&self, sp: *const u8, mut sp_end: *const u8) {
        self.objects.iter().for_each(|(_, obj)| {
            obj.borrow_mut().marked = false;
        });

        while unsafe { sp_end.byte_offset(-(size_of::<usize>() as isize)) } >= sp {
            unsafe {
                sp_end = sp_end.byte_offset(-(size_of::<usize>() as isize));
            };

            let addr: *const u8 = unsafe { *sp_end.cast() };
            if let Some(obj) = self.objects.get(&addr) {
                self.mark_object(obj);
            }
        }
    }

    fn sweep(&mut self) {
        let mut to_remove = Vec::with_capacity(1 << 8);

        for (addr, obj) in &self.objects {
            if !obj.borrow().marked {
                match obj.borrow().data {
                    GcObjectData::Slice(v) => {
                        let _ = unsafe { Box::from_raw(v) };
                    }
                    GcObjectData::Cif(v) => {
                        let _ = unsafe { Box::from_raw(v) };
                    }
                    GcObjectData::Alloced(ptr, layout) => unsafe { dealloc(ptr, layout) },
                }

                to_remove.push(*addr);
            }
        }

        for v in to_remove {
            self.objects.remove(&v);
        }
    }

    fn run(&mut self, sp: *const u8, sp_end: *const u8) {
        self.mark(sp, sp_end);
        self.sweep();
    }
}

#[derive(Debug)]
pub struct Slice {
    data: Vec<u8>,
    len: usize,
}

impl Slice {
    pub fn new() -> *mut Self {
        Box::into_raw(Box::new(Self {
            data: Vec::new(),
            len: 0,
        }))
    }

    pub fn from_string(v: &str) -> *mut Self {
        let mut data = Vec::new();
        data.extend_from_slice(v.as_bytes());
        Box::into_raw(Box::new(Self { len: v.len(), data }))
    }

    pub fn from_default_len(len: usize, val: &[u8]) -> *mut Self {
        let mut data = Vec::with_capacity(val.len() * len);
        for _ in 0..len {
            for v in val {
                data.push(*v);
            }
        }

        Box::into_raw(Box::new(Self { len, data }))
    }

    fn index(&self, index: isize, size: usize) -> &[u8] {
        let index = index as usize;
        let from = index * size;
        &self.data[from..(from + size)]
    }

    fn index_set(&mut self, index: isize, val: &[u8]) {
        let from = index as usize * val.len();
        for (i, v) in val.into_iter().enumerate() {
            self.data[from + i] = *v;
        }
    }

    fn concat(&mut self, other: &Self) {
        self.len += other.len;
        self.data.extend_from_slice(&other.data);
    }

    fn append(&mut self, val: &[u8]) {
        self.len += 1;
        self.data.extend_from_slice(val);
    }

    fn string(&self) -> String {
        String::from_utf8(self.data.clone()).unwrap()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Instruction {
    FfiCreate,
    FfiDllOpen,
    FfiCall,

    SliceLen,
    SliceAppend(usize),
    SliceIndexGet(usize),
    SliceIndexSet(usize),

    Increment(usize),
    // dst = src * len
    Copy(usize, usize, usize),
    // len, count
    Shift(usize, usize),
    Reset(usize),
    PushI(isize),
    PushI8(i8),
    PushI16(i16),
    PushI32(i32),
    PushI64(i64),
    PushU(usize),
    PushU8(u8),
    PushU16(u16),
    PushU32(u32),
    PushU64(u64),
    PushSlice,
    PushSliceNewLen(usize),
    // var count, function index
    PushClosure(usize, usize),

    // index
    PushStaticString(usize),

    AddString,

    AddI(u8),
    MinusI(u8),
    MulI(u8),
    DivI(u8),
    ModI(u8),
    AddU(u8),
    MinusU(u8),
    MulU(u8),
    DivU(u8),
    ModU(u8),

    Exit,
    Debug,

    JumpAndLink(usize),
    JumpAndLinkClosure,
    Jump(usize),
    Return,
    JumpIfTrue(usize),
    JumpIfFalse(usize),

    NegateBool,
    And,
    Or,

    CompareEq(u8),
    CompareEqString,
    CompareGtI(u8),
    CompareLtI(u8),
    CompareGtU(u8),
    CompareLtU(u8),

    CastSlicePtr,

    // from, to size
    CastUint(u8, u8),
    // from, to size
    CastInt(u8, u8),

    Offset(usize),
    Alloc(usize, usize),
    Deref(usize),
    DerefAssign(usize),
}

impl Instruction {
    pub fn add_jump_offset(&mut self, offset: usize) {
        match self {
            Self::JumpAndLink(v) => *v = *v + offset,
            Self::Jump(v) => *v = *v + offset,
            Self::JumpIfTrue(v) => *v = *v + offset,
            Self::JumpIfFalse(v) => *v = *v + offset,
            Self::PushClosure(_, v) => *v = *v + offset,
            _ => {}
        }
    }

    fn to_num(&self) -> InstructionNum {
        match self {
            Self::FfiCreate => InstructionNum::FfiCreate,
            Self::FfiDllOpen => InstructionNum::FfiDllOpen,
            Self::FfiCall => InstructionNum::FfiCall,
            Self::SliceLen => InstructionNum::SliceLen,
            Self::PushSlice => InstructionNum::PushSlice,
            Self::AddString => InstructionNum::AddString,
            Self::Exit => InstructionNum::Exit,
            Self::Debug => InstructionNum::Debug,
            Self::JumpAndLinkClosure => InstructionNum::JumpAndLinkClosure,
            Self::Return => InstructionNum::Return,
            Self::NegateBool => InstructionNum::NegateBool,
            Self::And => InstructionNum::And,
            Self::Or => InstructionNum::Or,
            Self::CompareEqString => InstructionNum::CompareEqString,
            Self::CastSlicePtr => InstructionNum::CastSlicePtr,
            Self::AddI(_) => InstructionNum::AddI,
            Self::AddU(_) => InstructionNum::AddU,
            Self::CompareEq(_) => InstructionNum::CompareEq,
            Self::CompareGtI(_) => InstructionNum::CompareGtI,
            Self::CompareGtU(_) => InstructionNum::CompareGtU,
            Self::CompareLtI(_) => InstructionNum::CompareLtI,
            Self::CompareLtU(_) => InstructionNum::CompareLtU,
            Self::Deref(_) => InstructionNum::Deref,
            Self::DerefAssign(_) => InstructionNum::DerefAssign,
            Self::DivI(_) => InstructionNum::DivI,
            Self::DivU(_) => InstructionNum::DivU,
            Self::Increment(_) => InstructionNum::Increment,
            Self::Jump(_) => InstructionNum::Jump,
            Self::JumpAndLink(_) => InstructionNum::JumpAndLink,
            Self::JumpIfFalse(_) => InstructionNum::JumpIfFalse,
            Self::JumpIfTrue(_) => InstructionNum::JumpIfTrue,
            Self::MinusI(_) => InstructionNum::MinusI,
            Self::MinusU(_) => InstructionNum::MinusU,
            Self::ModI(_) => InstructionNum::ModI,
            Self::ModU(_) => InstructionNum::ModU,
            Self::MulI(_) => InstructionNum::MulI,
            Self::MulU(_) => InstructionNum::MulU,
            Self::Offset(_) => InstructionNum::Offset,
            Self::PushI(_) => InstructionNum::PushI,
            Self::PushI16(_) => InstructionNum::PushI16,
            Self::PushI32(_) => InstructionNum::PushI32,
            Self::PushI64(_) => InstructionNum::PushI64,
            Self::PushI8(_) => InstructionNum::PushI8,
            Self::PushSliceNewLen(_) => InstructionNum::PushSliceNewLen,
            Self::PushStaticString(_) => InstructionNum::PushStaticString,
            Self::PushU(_) => InstructionNum::PushU,
            Self::PushU16(_) => InstructionNum::PushU16,
            Self::PushU32(_) => InstructionNum::PushU32,
            Self::PushU64(_) => InstructionNum::PushU64,
            Self::PushU8(_) => InstructionNum::PushU8,
            Self::Reset(_) => InstructionNum::Reset,
            Self::SliceAppend(_) => InstructionNum::SliceAppend,
            Self::SliceIndexGet(_) => InstructionNum::SliceIndexGet,
            Self::SliceIndexSet(_) => InstructionNum::SliceIndexSet,
            Self::Alloc(_, _) => InstructionNum::Alloc,
            Self::CastInt(_, _) => InstructionNum::CastInt,
            Self::CastUint(_, _) => InstructionNum::CastUint,
            Self::PushClosure(_, _) => InstructionNum::PushClosure,
            Self::Shift(_, _) => InstructionNum::Shift,
            Self::Copy(_, _, _) => InstructionNum::Copy,
        }
    }
}

pub struct Instructions(pub Vec<Instruction>);

impl Instructions {
    pub fn new(instructions: Vec<Instruction>) -> Self {
        Self(instructions)
    }

    pub unsafe fn from_binary(mut ptr: *const u8) -> Self {
        let len = pop_bytes!(usize, &mut ptr);

        let mut instructions = Vec::<Instruction>::new();

        for _ in 0..len {
            let num: InstructionNum = mem::transmute(pop_bytes!(u8, &mut ptr));
            instructions.push(match num {
                InstructionNum::FfiCreate => Instruction::FfiCreate,
                InstructionNum::FfiDllOpen => Instruction::FfiDllOpen,
                InstructionNum::FfiCall => Instruction::FfiCall,
                InstructionNum::SliceLen => Instruction::SliceLen,
                InstructionNum::PushSlice => Instruction::PushSlice,
                InstructionNum::AddString => Instruction::AddString,
                InstructionNum::Exit => Instruction::Exit,
                InstructionNum::Debug => Instruction::Debug,
                InstructionNum::JumpAndLinkClosure => Instruction::JumpAndLinkClosure,
                InstructionNum::Return => Instruction::Return,
                InstructionNum::NegateBool => Instruction::NegateBool,
                InstructionNum::And => Instruction::And,
                InstructionNum::Or => Instruction::Or,
                InstructionNum::CompareEqString => Instruction::CompareEqString,
                InstructionNum::CastSlicePtr => Instruction::CastSlicePtr,
                InstructionNum::PushI => Instruction::PushI(pop_bytes!(isize, &mut ptr)),
                InstructionNum::PushI8 => Instruction::PushI8(pop_bytes!(i8, &mut ptr)),
                InstructionNum::PushI16 => Instruction::PushI16(pop_bytes!(i16, &mut ptr)),
                InstructionNum::PushI32 => Instruction::PushI32(pop_bytes!(i32, &mut ptr)),
                InstructionNum::PushI64 => Instruction::PushI64(pop_bytes!(i64, &mut ptr)),
                InstructionNum::PushU => Instruction::PushU(pop_bytes!(usize, &mut ptr)),
                InstructionNum::PushU8 => Instruction::PushU8(pop_bytes!(u8, &mut ptr)),
                InstructionNum::PushU16 => Instruction::PushU16(pop_bytes!(u16, &mut ptr)),
                InstructionNum::PushU32 => Instruction::PushU32(pop_bytes!(u32, &mut ptr)),
                InstructionNum::PushU64 => Instruction::PushU64(pop_bytes!(u64, &mut ptr)),
                InstructionNum::SliceAppend => {
                    Instruction::SliceAppend(pop_bytes!(usize, &mut ptr))
                }
                InstructionNum::SliceIndexGet => {
                    Instruction::SliceIndexGet(pop_bytes!(usize, &mut ptr))
                }
                InstructionNum::SliceIndexSet => {
                    Instruction::SliceIndexSet(pop_bytes!(usize, &mut ptr))
                }
                InstructionNum::Increment => Instruction::Increment(pop_bytes!(usize, &mut ptr)),
                InstructionNum::Copy => Instruction::Copy(
                    pop_bytes!(usize, &mut ptr),
                    pop_bytes!(usize, &mut ptr),
                    pop_bytes!(usize, &mut ptr),
                ),
                InstructionNum::Shift => {
                    Instruction::Shift(pop_bytes!(usize, &mut ptr), pop_bytes!(usize, &mut ptr))
                }
                InstructionNum::Reset => Instruction::Reset(pop_bytes!(usize, &mut ptr)),
                InstructionNum::PushSliceNewLen => {
                    Instruction::PushSliceNewLen(pop_bytes!(usize, &mut ptr))
                }
                InstructionNum::PushClosure => Instruction::PushClosure(
                    pop_bytes!(usize, &mut ptr),
                    pop_bytes!(usize, &mut ptr),
                ),
                InstructionNum::PushStaticString => {
                    Instruction::PushStaticString(pop_bytes!(usize, &mut ptr))
                }
                InstructionNum::AddI => Instruction::AddI(pop_bytes!(u8, &mut ptr)),
                InstructionNum::MinusI => Instruction::MinusI(pop_bytes!(u8, &mut ptr)),
                InstructionNum::MulI => Instruction::MulI(pop_bytes!(u8, &mut ptr)),
                InstructionNum::DivI => Instruction::DivI(pop_bytes!(u8, &mut ptr)),
                InstructionNum::ModI => Instruction::ModI(pop_bytes!(u8, &mut ptr)),
                InstructionNum::AddU => Instruction::AddU(pop_bytes!(u8, &mut ptr)),
                InstructionNum::MinusU => Instruction::MinusU(pop_bytes!(u8, &mut ptr)),
                InstructionNum::MulU => Instruction::MulU(pop_bytes!(u8, &mut ptr)),
                InstructionNum::DivU => Instruction::DivU(pop_bytes!(u8, &mut ptr)),
                InstructionNum::ModU => Instruction::ModU(pop_bytes!(u8, &mut ptr)),
                InstructionNum::JumpAndLink => {
                    Instruction::JumpAndLink(pop_bytes!(usize, &mut ptr))
                }
                InstructionNum::Jump => Instruction::Jump(pop_bytes!(usize, &mut ptr)),
                InstructionNum::JumpIfTrue => Instruction::JumpIfTrue(pop_bytes!(usize, &mut ptr)),
                InstructionNum::JumpIfFalse => {
                    Instruction::JumpIfFalse(pop_bytes!(usize, &mut ptr))
                }
                InstructionNum::CompareEq => Instruction::CompareEq(pop_bytes!(u8, &mut ptr)),
                InstructionNum::CompareGtI => Instruction::CompareGtI(pop_bytes!(u8, &mut ptr)),
                InstructionNum::CompareLtI => Instruction::CompareLtI(pop_bytes!(u8, &mut ptr)),
                InstructionNum::CompareGtU => Instruction::CompareGtU(pop_bytes!(u8, &mut ptr)),
                InstructionNum::CompareLtU => Instruction::CompareLtU(pop_bytes!(u8, &mut ptr)),
                InstructionNum::CastUint => {
                    Instruction::CastUint(pop_bytes!(u8, &mut ptr), pop_bytes!(u8, &mut ptr))
                }
                InstructionNum::CastInt => {
                    Instruction::CastInt(pop_bytes!(u8, &mut ptr), pop_bytes!(u8, &mut ptr))
                }
                InstructionNum::Offset => Instruction::Offset(pop_bytes!(usize, &mut ptr)),
                InstructionNum::Alloc => {
                    Instruction::Alloc(pop_bytes!(usize, &mut ptr), pop_bytes!(usize, &mut ptr))
                }
                InstructionNum::Deref => Instruction::Deref(pop_bytes!(usize, &mut ptr)),
                InstructionNum::DerefAssign => {
                    Instruction::DerefAssign(pop_bytes!(usize, &mut ptr))
                }
            });
        }

        Self(instructions)
    }

    pub fn to_binary(self) -> Vec<u8> {
        let mut binary = Vec::<u8>::new();

        push_bytes!(binary, self.0.len());

        for v in self.0 {
            binary.push(v.to_num().to_u8());
            match v {
                Instruction::FfiCreate
                | Instruction::FfiDllOpen
                | Instruction::FfiCall
                | Instruction::And
                | Instruction::Or
                | Instruction::PushSlice
                | Instruction::AddString
                | Instruction::Exit
                | Instruction::Debug
                | Instruction::Return
                | Instruction::JumpAndLinkClosure
                | Instruction::CompareEqString
                | Instruction::CastSlicePtr
                | Instruction::NegateBool
                | Instruction::SliceLen => {
                    // no args
                }

                Instruction::PushI(v) => push_bytes!(binary, v),
                Instruction::PushI8(v) => push_bytes!(binary, v),
                Instruction::PushI16(v) => push_bytes!(binary, v),
                Instruction::PushI32(v) => push_bytes!(binary, v),
                Instruction::PushI64(v) => push_bytes!(binary, v),
                Instruction::PushU(v) => push_bytes!(binary, v),
                Instruction::PushU8(v) => push_bytes!(binary, v),
                Instruction::PushU16(v) => push_bytes!(binary, v),
                Instruction::PushU32(v) => push_bytes!(binary, v),
                Instruction::PushU64(v) => push_bytes!(binary, v),

                Instruction::AddI(v)
                | Instruction::MinusI(v)
                | Instruction::MulI(v)
                | Instruction::DivI(v)
                | Instruction::ModI(v)
                | Instruction::AddU(v)
                | Instruction::MinusU(v)
                | Instruction::MulU(v)
                | Instruction::DivU(v)
                | Instruction::CompareEq(v)
                | Instruction::CompareGtI(v)
                | Instruction::CompareLtI(v)
                | Instruction::CompareGtU(v)
                | Instruction::CompareLtU(v)
                | Instruction::ModU(v) => push_bytes!(binary, v),

                Instruction::SliceAppend(v)
                | Instruction::SliceIndexGet(v)
                | Instruction::SliceIndexSet(v)
                | Instruction::Reset(v)
                | Instruction::PushSliceNewLen(v)
                | Instruction::PushStaticString(v)
                | Instruction::JumpAndLink(v)
                | Instruction::Jump(v)
                | Instruction::JumpIfTrue(v)
                | Instruction::JumpIfFalse(v)
                | Instruction::Offset(v)
                | Instruction::Deref(v)
                | Instruction::DerefAssign(v)
                | Instruction::Increment(v) => push_bytes!(binary, v),

                Instruction::CastUint(v1, v2) | Instruction::CastInt(v1, v2) => {
                    push_bytes!(binary, v1, v2);
                }

                Instruction::Shift(v1, v2)
                | Instruction::PushClosure(v1, v2)
                | Instruction::Alloc(v1, v2) => {
                    push_bytes!(binary, v1, v2);
                }

                Instruction::Copy(v1, v2, v3) => {
                    push_bytes!(binary, v1, v2, v3);
                }
            }
        }

        binary
    }
}

pub struct Stack {
    data: *mut u8,
    sp: *mut u8,
    layout: Layout,
}

impl Drop for Stack {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.data, self.layout);
        };
    }
}

impl Stack {
    fn new(size: usize) -> Self {
        let layout = Layout::from_size_align(size, size_of::<usize>()).unwrap();
        let data = unsafe { alloc(layout) };

        Self {
            sp: unsafe { data.byte_offset(size as isize) },
            data,
            layout,
        }
    }

    fn sp_end(&self) -> *mut u8 {
        unsafe { self.data.byte_offset(self.layout.size() as isize) }
    }

    fn pop_size(&mut self, size: usize) -> &[u8] {
        unsafe {
            let slice = slice::from_raw_parts(self.sp, size);
            self.reset(size);
            slice
        }
    }

    fn push_size(&mut self, val: &[u8]) {
        unsafe {
            self.increment(val.len());
            ptr::copy_nonoverlapping(val.as_ptr(), self.sp, val.len());
        }
    }

    fn push<T: Copy>(&mut self, item: T) {
        unsafe {
            self.increment(size_of::<T>());
            *self.sp.cast() = item;
        };
    }

    fn increment(&mut self, by: usize) {
        unsafe {
            self.sp = self.sp.byte_offset(-(by as isize));
        }
    }

    fn pop<T: Copy>(&mut self) -> T {
        unsafe {
            let item = *self.sp.cast();
            self.reset(size_of::<T>());
            item
        }
    }

    fn reset(&mut self, offset: usize) {
        unsafe {
            self.sp = self.sp.byte_offset(offset as isize);
        }
    }

    fn shift(&mut self, len: usize, count: usize) {
        unsafe {
            for _ in 0..count {
                self.reset(len);
                for i in 0..len {
                    *self.sp.byte_offset(-(i as isize)) = *self.sp.byte_offset(-(i as isize) - 1);
                }
                self.increment(len - 1);
            }
        }
    }

    fn debug_print(&self) {
        unsafe {
            let mut data_end = self.data.byte_offset(self.layout.size() as isize);
            while data_end > self.sp {
                data_end = data_end.byte_offset(-8);
                let value: usize = *data_end.cast();
                println!("data_end: {data_end:?}, value: {value}");
            }
        }
    }

    #[cfg(test)]
    fn debug(&self, sp_offset: usize) -> Vec<u8> {
        let mut values = Vec::new();

        unsafe {
            let mut sp = self.sp.byte_offset(-(sp_offset as isize));
            while sp < self.sp_end() {
                values.push(*sp.cast());
                sp = sp.byte_offset(1);
            }
        }

        values
    }

    fn copy(&mut self, dst: usize, src: usize, len: usize) {
        unsafe {
            ptr::copy_nonoverlapping(
                self.sp.byte_offset(src as isize),
                self.sp.byte_offset(dst as isize),
                len,
            );
        }
    }

    fn deref(&mut self, ptr: *mut u8, size: usize) {
        self.increment(size);
        unsafe {
            ptr::copy_nonoverlapping(ptr, self.sp, size);
        };
    }
}

#[derive(Debug, Clone)]
pub struct StaticMemory {
    pub data: Vec<u8>,
}

impl StaticMemory {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    pub fn push_string_slice(&mut self, string: &str) -> usize {
        let c_string = CString::new(string).unwrap();
        self.push(c_string.as_bytes_with_nul())
    }

    fn push(&mut self, val: &[u8]) -> usize {
        let old_len = self.data.len();
        self.data.extend_from_slice(val);
        old_len
    }

    pub unsafe fn from_binary(mut ptr: *const u8, len: usize) -> Self {
        let mut data = Vec::<u8>::with_capacity(len);
        for _ in 0..len {
            data.push(*ptr);
            ptr = ptr.byte_offset(1);
        }
        Self { data }
    }

    pub fn get_data(self) -> Vec<u8> {
        self.data
    }
}

#[derive(Debug)]
enum FfiType {
    Void,
    Pointer,
    Cstring,
    U32,
    U16,
    I32,
    I16,
    Struct(Vec<FfiType>),
}

impl FfiType {
    fn from_str(from: &mut &str) -> Self {
        match from.to_string().trim() {
            v if v.starts_with("void") => Self::Void,
            v if v.starts_with("pointer") => Self::Pointer,
            v if v.starts_with("c_string") => Self::Cstring,
            v if v.starts_with("u16") => Self::U16,
            v if v.starts_with("u32") => Self::U32,
            v if v.starts_with("i16") => Self::I16,
            v if v.starts_with("i32") => Self::I32,
            // {u32, u32,{i32,},}
            v if v.starts_with("{") => {
                let mut fields = Vec::new();
                *from = &from[1..];
                loop {
                    let first = from.chars().next();
                    let Some(ch) = first else {
                        break;
                    };
                    if ch == '}' {
                        break;
                    }

                    fields.push(Self::from_str(from));

                    let comma_index = from.chars().position(|v| v == ',');
                    let Some(index) = comma_index else {
                        break;
                    };

                    *from = &from[(index + 1)..];
                }

                Self::Struct(fields)
            }
            other => panic!("{other}"),
        }
    }

    fn to_ffi_type(&self) -> libffi::middle::Type {
        match self {
            Self::I32 => libffi::middle::Type::i32(),
            Self::I16 => libffi::middle::Type::i16(),
            Self::Void => libffi::middle::Type::void(),
            Self::Pointer | Self::Cstring => libffi::middle::Type::pointer(),
            Self::U16 => libffi::middle::Type::u16(),
            Self::U32 => libffi::middle::Type::u32(),
            Self::Struct(types) => libffi::middle::Type::structure(
                types.iter().map(|v| v.to_ffi_type()).collect::<Vec<_>>(),
            ),
        }
    }
}

#[derive(Debug)]
struct Cif {
    arguments: Vec<FfiType>,
    return_type: FfiType,
    cif: libffi::middle::Cif,
    fn_ptr: *mut libc::c_void,
}

enum AnyVecItem {
    Slice(*mut u8, Layout, Layout),
    Ptr(*mut u8),
}

struct AnyVec {
    values: Vec<AnyVecItem>,
}

impl AnyVec {
    fn new() -> Self {
        Self { values: Vec::new() }
    }

    fn push_ptr(&mut self, ptr: *mut u8) {
        self.values.push(AnyVecItem::Ptr(ptr));
    }

    fn push_slice(&mut self, slice: &[u8]) {
        let slice_layout = Layout::from_size_align(slice.len(), size_of::<usize>()).unwrap();
        let slice_ptr = unsafe { alloc(slice_layout) };
        unsafe {
            ptr::copy_nonoverlapping(slice.as_ptr(), slice_ptr, slice.len());
        };

        let (ptr, layout) = unsafe { alloc_value(slice_ptr) };
        self.values
            .push(AnyVecItem::Slice(ptr, layout, slice_layout));
    }

    fn pointers(&self) -> Vec<*mut u8> {
        let mut vec = Vec::new();
        for value in &self.values {
            vec.push(match *value {
                AnyVecItem::Slice(v, _, _) => v,
                AnyVecItem::Ptr(v) => v,
            });
        }
        vec
    }
}

impl Drop for AnyVec {
    fn drop(&mut self) {
        for v in &self.values {
            unsafe {
                match *v {
                    AnyVecItem::Slice(v, v_layout, slice_layout) => {
                        let slice_ptr: *mut u8 = *v.cast();
                        dealloc(v, v_layout);
                        dealloc(slice_ptr, slice_layout);
                    }
                    AnyVecItem::Ptr(_) => {}
                }
            };
        }
    }
}

unsafe fn cstring_ptr_to_string(mut ptr: *const u8) -> Result<String> {
    let mut vec = Vec::new();
    while *ptr != 0 {
        vec.push(*ptr);
        ptr = ptr.byte_offset(1);
    }
    Ok(CString::new(vec)?.into_string()?)
}

#[repr(u8)]
enum InstructionNum {
    FfiCreate,
    FfiDllOpen,
    FfiCall,
    SliceLen,
    SliceAppend,
    SliceIndexGet,
    SliceIndexSet,
    Increment,
    Copy,
    Shift,
    Reset,
    PushI,
    PushI8,
    PushI16,
    PushI32,
    PushI64,
    PushU,
    PushU8,
    PushU16,
    PushU32,
    PushU64,
    PushSlice,
    PushSliceNewLen,
    PushClosure,
    PushStaticString,
    AddString,
    AddI,
    MinusI,
    MulI,
    DivI,
    ModI,
    AddU,
    MinusU,
    MulU,
    DivU,
    ModU,
    Exit,
    Debug,
    JumpAndLink,
    JumpAndLinkClosure,
    Jump,
    Return,
    JumpIfTrue,
    JumpIfFalse,
    NegateBool,
    And,
    Or,
    CompareEq,
    CompareEqString,
    CompareGtI,
    CompareLtI,
    CompareGtU,
    CompareLtU,
    CastSlicePtr,
    CastUint,
    CastInt,
    Offset,
    Alloc,
    Deref,
    DerefAssign,
}

impl InstructionNum {
    fn to_u8(self) -> u8 {
        unsafe { mem::transmute(self) }
    }
}

pub struct Vm {
    stack: Stack,
    instructions: Vec<Instruction>,
    static_memory: StaticMemory,
    gc: Gc,
}

impl Vm {
    pub fn new(instructions: Vec<Instruction>, static_memory: StaticMemory) -> Self {
        return Self {
            stack: Stack::new(65536),
            instructions,
            static_memory,
            gc: Gc::new(),
        };
    }

    pub fn run(&mut self) {
        let mut pc = 0;

        loop {
            #[cfg(debug_assertions)]
            if is_debug() {
                println!("executing instruction: {:#?}", self.instructions[pc]);
            }

            match self.instructions[pc] {
                Instruction::PushI(v) => self.push_i(v),
                Instruction::PushI8(v) => self.push_i8(v),
                Instruction::PushI16(v) => self.push_i16(v),
                Instruction::PushI32(v) => self.push_i32(v),
                Instruction::PushI64(v) => self.push_i64(v),
                Instruction::PushU(v) => self.push_u(v),
                Instruction::PushU8(v) => self.push_u8(v),
                Instruction::PushU16(v) => self.push_u16(v),
                Instruction::PushU32(v) => self.push_u32(v),
                Instruction::PushU64(v) => self.push_u64(v),
                Instruction::Exit => return,
                Instruction::Debug => {
                    self.stack.debug_print();
                }
                Instruction::Jump(i) => {
                    pc = i;
                    continue;
                }
                Instruction::Return => {
                    pc = self.stack.pop();
                    continue;
                }
                Instruction::JumpAndLink(i) => {
                    self.stack.push(pc + 1);
                    pc = i;
                    continue;
                }
                Instruction::Reset(offset) => {
                    self.stack.reset(offset);
                }
                Instruction::Copy(dst, src, len) => {
                    self.stack.copy(dst, src, len);
                }
                Instruction::Increment(by) => {
                    self.stack.increment(by);
                }
                Instruction::JumpIfFalse(i) => {
                    let boolean = self.stack.pop::<u8>();
                    self.stack.push(boolean);
                    if boolean != 1 {
                        pc = i;
                        continue;
                    }
                }
                Instruction::JumpIfTrue(i) => {
                    let boolean = self.stack.pop::<u8>();
                    self.stack.push(boolean);
                    if boolean == 1 {
                        pc = i;
                        continue;
                    }
                }
                Instruction::NegateBool => {
                    let int = self.stack.pop::<u8>();
                    self.stack.push::<u8>(int ^ 1);
                }
                Instruction::CompareEq(size) => {
                    let a = self.stack.pop_size(size as usize).to_vec();
                    let b = self.stack.pop_size(size as usize).to_vec();
                    if a == b {
                        self.stack.push::<u8>(1);
                    } else {
                        self.stack.push::<u8>(0)
                    }
                }
                Instruction::CompareEqString => {
                    let a = unsafe { &mut *self.stack.pop::<*mut Slice>() };
                    let b = unsafe { &mut *self.stack.pop::<*mut Slice>() };

                    self.stack.push::<u8>(if a.data == b.data { 1 } else { 0 });
                }
                Instruction::And => {
                    let a = self.stack.pop::<u8>();
                    let b = self.stack.pop::<u8>();
                    self.stack.push(a & b);
                }
                Instruction::Or => {
                    let a = self.stack.pop::<u8>();
                    let b = self.stack.pop::<u8>();
                    self.stack.push(a | b);
                }
                Instruction::PushSlice => {
                    let slice = Slice::new();
                    self.gc
                        .add_object(GcObject::new(GcObjectData::Slice(slice)));
                    self.stack.push(slice);
                }
                Instruction::PushSliceNewLen(size) => {
                    let val = self.stack.pop_size(size).to_vec();
                    let len: isize = self.stack.pop();

                    let slice = Slice::from_default_len(len as usize, &val);
                    self.stack.push(slice);
                    self.gc
                        .add_object(GcObject::new(GcObjectData::Slice(slice)));
                }
                Instruction::SliceAppend(size) => {
                    let item = self.stack.pop_size(size).to_vec();
                    let slice = unsafe { &mut *self.stack.pop::<*mut Slice>() };

                    slice.append(&item);
                }
                Instruction::SliceIndexGet(size) => {
                    let index = self.stack.pop::<isize>();
                    let slice = unsafe { &mut *self.stack.pop::<*mut Slice>() };

                    self.stack.push_size(slice.index(index, size));
                }
                Instruction::SliceIndexSet(size) => {
                    let item = self.stack.pop_size(size).to_vec();
                    let index = self.stack.pop::<isize>();
                    let slice = unsafe { &mut *self.stack.pop::<*mut Slice>() };

                    slice.index_set(index, &item);
                }
                Instruction::SliceLen => {
                    let slice = unsafe { &mut *self.stack.pop::<*mut Slice>() };
                    self.stack.push(slice.len as isize);
                }
                Instruction::CastSlicePtr => {
                    let slice = unsafe { &mut *self.stack.pop::<*mut Slice>() };
                    self.stack.push(slice.data.as_ptr());
                }
                Instruction::AddString => {
                    let a = unsafe { &mut *self.stack.pop::<*mut Slice>() };
                    let b = unsafe { &mut *self.stack.pop::<*mut Slice>() };

                    let slice =
                        unsafe { &mut *Slice::from_string(str::from_utf8_unchecked(&b.data)) };
                    slice.concat(a);
                    let slice = slice as *mut Slice;
                    self.gc
                        .add_object(GcObject::new(GcObjectData::Slice(slice)));
                    self.stack.push(slice);
                }
                Instruction::Shift(len, count) => {
                    self.stack.shift(len, count);
                }
                Instruction::Alloc(size, alignment) => {
                    let val = self.stack.pop_size(size);
                    let (obj, ptr) = GcObject::from_slice_val(val, alignment);
                    self.stack.push(ptr);
                    self.gc.add_object(obj);
                }
                Instruction::Deref(size) => {
                    let ptr = self.stack.pop::<*mut u8>();
                    self.stack.deref(ptr, size);
                }
                Instruction::DerefAssign(size) => {
                    let src = self.stack.pop_size(size).to_vec();
                    let dst = self.stack.pop::<*mut u8>();
                    unsafe {
                        ptr::copy_nonoverlapping(src.as_ptr(), dst, size);
                    };
                }
                Instruction::Offset(size) => {
                    let ptr = self.stack.pop::<*mut u8>();
                    unsafe {
                        self.stack.push(ptr.byte_offset(size as isize));
                    };
                }
                Instruction::PushClosure(var_count, function_index) => {
                    let mut vars = Vec::with_capacity(var_count);
                    for _ in 0..var_count {
                        let var = self.stack.pop::<*mut u8>();
                        vars.push(var);
                    }
                    // so the order is the same as you popped
                    vars.reverse();

                    let (obj, ptr) = GcObject::new_closure(&vars, function_index);
                    self.gc.add_object(obj);
                    self.stack.push(ptr);
                }
                Instruction::JumpAndLinkClosure => {
                    let mut closure = self.stack.pop::<*mut u8>();
                    self.stack.push(pc + 1);

                    let function_index: usize = unsafe { *closure.cast() };
                    unsafe { closure = closure.byte_offset(size_of::<usize>() as isize) }
                    pc = function_index;

                    let var_count: usize = unsafe { *closure.cast() };
                    unsafe { closure = closure.byte_offset(size_of::<usize>() as isize) }

                    for _ in 0..var_count {
                        self.stack.push::<*mut u8>(unsafe {
                            let val = *closure.cast();
                            closure = closure.byte_offset(size_of::<usize>() as isize);
                            val
                        })
                    }

                    continue;
                }
                Instruction::FfiCreate => {
                    // slice of strings
                    let arguments: &mut Slice = unsafe { &mut *self.stack.pop::<*mut Slice>() };
                    let mut argument_types: Vec<FfiType> = Vec::new();

                    for index in 0..arguments.len {
                        let slice: &mut Slice = unsafe {
                            &mut **arguments
                                .data
                                .as_mut_ptr()
                                .byte_offset((index * size_of::<usize>()) as isize)
                                .cast::<*mut Slice>()
                        };
                        argument_types.push(FfiType::from_str(&mut slice.string().as_str()));
                    }

                    let return_argument = unsafe {
                        FfiType::from_str(
                            &mut (&mut *self.stack.pop::<*mut Slice>()).string().as_str(),
                        )
                    };

                    let function_iden = unsafe { (&mut *self.stack.pop::<*mut Slice>()).string() };

                    let dll: *mut libc::c_void = self.stack.pop();

                    let mut builder = libffi::middle::Builder::new();
                    for arg in &argument_types {
                        builder = builder.arg(arg.to_ffi_type());
                    }
                    builder = builder.res(return_argument.to_ffi_type());

                    let fn_ptr = unsafe {
                        let name = CString::from_str(&function_iden).unwrap();
                        libc::dlsym(dll, name.as_ptr())
                    };

                    let cif = Box::into_raw(
                        Cif {
                            fn_ptr,
                            arguments: argument_types,
                            return_type: return_argument,
                            cif: builder.into_cif(),
                        }
                        .into(),
                    );

                    self.gc.add_object(GcObject::new(GcObjectData::Cif(cif)));
                    self.stack.push(cif);
                }
                Instruction::FfiDllOpen => {
                    let path: &mut Slice = unsafe { &mut *self.stack.pop::<*mut Slice>() };

                    let handle = unsafe {
                        let cpath = CString::from_str(&path.string()).unwrap();
                        libc::dlopen(cpath.as_ptr(), libc::RTLD_LAZY)
                    };

                    self.stack.push(handle);
                }
                Instruction::FfiCall => {
                    // slice of pointers to arguments
                    let args: &mut Slice = unsafe { &mut *self.stack.pop::<*mut Slice>() };
                    let cif: &mut Cif = unsafe { &mut *self.stack.pop::<*mut Cif>() };

                    let mut converted_args = AnyVec::new();

                    let mut arg_ptr = args.data.as_mut_ptr();
                    for arg in &cif.arguments {
                        match arg {
                            FfiType::Void => panic!("Cvoid in argument"),
                            FfiType::I32
                            | FfiType::U32
                            | FfiType::U16
                            | FfiType::I16
                            | FfiType::Pointer
                            | FfiType::Struct(_) => {
                                converted_args.push_ptr(unsafe { *arg_ptr.cast::<*mut u8>() })
                            }
                            FfiType::Cstring => {
                                let cstring: CString = unsafe {
                                    let v: &mut Slice = &mut ***arg_ptr.cast::<*mut *mut Slice>();
                                    CString::from_str(&v.string()).unwrap()
                                };
                                converted_args.push_slice(cstring.as_bytes_with_nul());
                            }
                        };

                        unsafe {
                            arg_ptr = arg_ptr.byte_offset(size_of::<usize>() as isize);
                        };
                    }

                    // will hold the result value in here,
                    // size has to be at least register size
                    let result_ptr = unsafe {
                        match &cif.return_type {
                            FfiType::Pointer
                            | FfiType::Cstring
                            | FfiType::U32
                            | FfiType::U16
                            | FfiType::I16
                            | FfiType::I32 => Some(alloc_value(0 as usize)),
                            FfiType::Struct(_) => {
                                // right now just allocate enough for most structs
                                // will pass size / alignment somehow later
                                let layout =
                                    Layout::from_size_align(256, size_of::<usize>()).unwrap();
                                Some((alloc(layout), layout))
                            }
                            FfiType::Void => None,
                        }
                    };

                    unsafe {
                        libffi::raw::ffi_call(
                            cif.cif.as_raw_ptr(),
                            Some(mem::transmute(cif.fn_ptr)),
                            result_ptr
                                .map(|v| v.0 as *mut libc::c_void)
                                .unwrap_or(0 as *mut libc::c_void),
                            converted_args.pointers().as_mut_ptr().cast(),
                        );
                    };

                    #[cfg(debug_assertions)]
                    if is_debug() {
                        if let Some((ptr, _)) = result_ptr {
                            let v = unsafe { *ptr.cast::<usize>() };
                            println!("FfiCall returned usize({})", v);
                        }
                    }

                    let result_converted = match result_ptr {
                        Some((ptr, _)) => unsafe {
                            let is_null = *ptr.cast::<usize>() == 0;

                            match cif.return_type {
                                FfiType::Void => unreachable!(),
                                FfiType::U32 | FfiType::I32 => {
                                    Some(alloc_value(*ptr.cast::<u32>()))
                                }
                                FfiType::U16 | FfiType::I16 => {
                                    Some(alloc_value(*ptr.cast::<u16>()))
                                }
                                FfiType::Struct(_) => {
                                    let layout =
                                        Layout::from_size_align(256, size_of::<usize>()).unwrap();
                                    let ret_ptr = alloc(layout);
                                    ptr::copy_nonoverlapping(ptr, ret_ptr, 256);
                                    Some((ret_ptr, layout))
                                }
                                FfiType::Pointer => Some(alloc_value(*ptr.cast::<*mut u8>())),
                                FfiType::Cstring => match is_null {
                                    false => {
                                        let cstring = CStr::from_ptr(*ptr.cast());
                                        let slice = Slice::from_string(cstring.to_str().unwrap());
                                        self.gc
                                            .add_object(GcObject::new(GcObjectData::Slice(slice)));
                                        Some(alloc_value(slice))
                                    }
                                    true => None,
                                },
                            }
                        },
                        None => None,
                    };

                    if let Some((ptr, layout)) = result_ptr {
                        unsafe {
                            dealloc(ptr, layout);
                        };
                    }

                    if let Some((ptr, layout)) = result_converted {
                        self.gc
                            .add_object(GcObject::new(GcObjectData::Alloced(ptr, layout)));
                        self.stack.push(ptr);
                    } else {
                        self.stack.push(0 as usize);
                    }
                }
                Instruction::CastUint(from, to) => self.cast_uint(from, to),
                Instruction::CastInt(from, to) => self.cast_int(from, to),
                Instruction::AddI(v) => self.add_i(v),
                Instruction::MinusI(v) => self.minus_i(v),
                Instruction::MulI(v) => self.mul_i(v),
                Instruction::DivI(v) => self.div_i(v),
                Instruction::ModI(v) => self.mod_i(v),
                Instruction::AddU(v) => self.add_u(v),
                Instruction::MinusU(v) => self.minus_u(v),
                Instruction::MulU(v) => self.mul_u(v),
                Instruction::DivU(v) => self.div_u(v),
                Instruction::ModU(v) => self.mod_u(v),
                Instruction::CompareGtI(v) => self.compare_gt_i(v),
                Instruction::CompareLtI(v) => self.compare_lt_i(v),
                Instruction::CompareGtU(v) => self.compare_gt_u(v),
                Instruction::CompareLtU(v) => self.compare_lt_u(v),
                Instruction::PushStaticString(v) => self.push_static_string(v),
            }

            self.gc.run(self.stack.sp, self.stack.sp_end());
            pc += 1;
        }
    }

    fn push_static_string(&mut self, index: usize) {
        let string = unsafe {
            cstring_ptr_to_string(self.static_memory.data.as_ptr().byte_offset(index as isize))
                .unwrap()
        };

        let slice = Slice::from_string(&string);
        self.gc
            .add_object(GcObject::new(GcObjectData::Slice(slice)));
        self.stack.push(slice);
    }

    fn cast_uint(&mut self, from: u8, to: u8) {
        debug_assert_ne!(from, 0);
        debug_assert_ne!(to, 0);

        let mut slice = self.stack.pop_size(from as usize).to_vec();

        if from > to {
            self.stack.push_size(&slice[0..(to as usize)]);
        } else {
            for _ in 0..(to - from) {
                slice.push(0);
            }
            self.stack.push_size(&slice);
        }
    }

    fn cast_int(&mut self, from: u8, to: u8) {
        debug_assert_ne!(from, 0);
        debug_assert_ne!(to, 0);

        let mut slice = self.stack.pop_size(from as usize).to_vec();

        let sign_mask = (slice.last().unwrap() & 0x80) >> 7;

        if from > to {
            let mut cast = (&slice[0..(to as usize)]).to_vec();
            let msb = *cast.last().unwrap();
            *cast.last_mut().unwrap() = if sign_mask == 1 {
                // set last bit to 1
                msb | 0x80
            } else {
                // set last bit to 0
                msb & 0x7F
            };
            self.stack.push_size(&cast);
        } else {
            let msb = if sign_mask == 1 { 0xFF } else { 0 };
            for _ in 0..(to - from) {
                slice.push(msb);
            }
            self.stack.push_size(&slice);
        }
    }

    fn push_i(&mut self, v: isize) {
        self.stack.push(v);
    }

    fn push_i8(&mut self, v: i8) {
        self.stack.push(v);
    }

    fn push_i16(&mut self, v: i16) {
        self.stack.push(v);
    }

    fn push_i32(&mut self, v: i32) {
        self.stack.push(v);
    }

    fn push_i64(&mut self, v: i64) {
        self.stack.push(v);
    }

    fn push_u(&mut self, v: usize) {
        self.stack.push(v);
    }

    fn push_u8(&mut self, v: u8) {
        self.stack.push(v);
    }

    fn push_u16(&mut self, v: u16) {
        self.stack.push(v);
    }

    fn push_u32(&mut self, v: u32) {
        self.stack.push(v);
    }

    fn push_u64(&mut self, v: u64) {
        self.stack.push(v);
    }

    fn pop_cast_i(&mut self, size: u8, cb: impl FnOnce(i64, i64) -> i64) {
        debug_assert_ne!(size, 0);

        let increment = self.stack.sp as usize % 8;
        self.stack.increment(increment + size as usize);
        self.stack.copy(0, increment + size as usize, size as usize);

        self.cast_int(size, 8);
        let a: i64 = self.stack.pop();

        self.stack.increment(size as usize);
        self.stack
            .copy(0, (size as usize) * 2 + increment, size as usize);
        self.cast_int(size, 8);
        let b: i64 = self.stack.pop();

        self.stack.push::<i64>(cb(a, b));

        self.cast_int(8, size);
        self.stack
            .shift(size as usize, increment + (size as usize) * 2);
    }

    fn pop_cast_u(&mut self, size: u8, cb: impl FnOnce(u64, u64) -> u64) {
        debug_assert_ne!(size, 0);

        let increment = self.stack.sp as usize % 8;
        self.stack.increment(increment + size as usize);
        self.stack.copy(0, increment + size as usize, size as usize);

        self.cast_int(size, 8);
        let a: u64 = self.stack.pop();

        self.stack.increment(size as usize);
        self.stack
            .copy(0, (size as usize) * 2 + increment, size as usize);
        self.cast_int(size, 8);
        let b: u64 = self.stack.pop();

        self.stack.push::<u64>(cb(a, b));

        self.cast_int(8, size);
        self.stack
            .shift(size as usize, increment + (size as usize) * 2);
    }

    fn add_i(&mut self, size: u8) {
        self.pop_cast_i(size, |a, b| b + a);
    }

    fn add_u(&mut self, size: u8) {
        self.pop_cast_u(size, |a, b| b + a);
    }

    fn minus_i(&mut self, size: u8) {
        self.pop_cast_i(size, |a, b| b - a);
    }

    fn minus_u(&mut self, size: u8) {
        self.pop_cast_u(size, |a, b| b - a);
    }

    fn mul_i(&mut self, size: u8) {
        self.pop_cast_i(size, |a, b| b * a);
    }

    fn mul_u(&mut self, size: u8) {
        self.pop_cast_u(size, |a, b| b * a);
    }

    fn div_i(&mut self, size: u8) {
        self.pop_cast_i(size, |a, b| b / a);
    }

    fn div_u(&mut self, size: u8) {
        self.pop_cast_u(size, |a, b| b / a);
    }

    fn mod_i(&mut self, size: u8) {
        self.pop_cast_i(size, |a, b| b % a);
    }

    fn mod_u(&mut self, size: u8) {
        self.pop_cast_u(size, |a, b| b % a);
    }

    fn compare_gt_i(&mut self, size: u8) {
        self.pop_cast_i(size, |a, b| if b > a { 1 } else { 0 });
        self.cast_uint(size, 1);
    }

    fn compare_lt_i(&mut self, size: u8) {
        self.pop_cast_i(size, |a, b| if b < a { 1 } else { 0 });
        self.cast_uint(size, 1);
    }

    fn compare_gt_u(&mut self, size: u8) {
        self.pop_cast_u(size, |a, b| if b > a { 1 } else { 0 });
        self.cast_uint(size, 1);
    }

    fn compare_lt_u(&mut self, size: u8) {
        self.pop_cast_u(size, |a, b| if b < a { 1 } else { 0 });
        self.cast_uint(size, 1);
    }
}
