use core::{cell::RefCell, str};
use std::{
    alloc::{alloc, dealloc, Layout},
    collections::HashMap,
    env,
    ffi::{CStr, CString},
    mem, ptr, slice,
    str::FromStr,
};

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

#[derive(Debug, Clone)]
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
    PushI16(i16),
    PushI32(i32),
    PushU8(u8),
    PushU16(u16),
    PushU32(u32),
    PushSlice,
    PushSliceNewLen(usize),
    // var count, function index
    PushClosure(usize, usize),

    // index, len
    PushStatic(usize, usize),

    MinusInt,
    AddI,
    AddString,
    MultiplyI,
    DivideI,
    ModuloI,

    Exit,
    Debug,

    JumpAndLink(usize),
    JumpAndLinkClosure,
    Jump(usize),
    Return,
    JumpIfTrue(usize),
    JumpIfFalse(usize),

    ToBoolI,
    NegateBool,
    CompareI,
    And,
    Or,

    CastIntUint,
    CastIntInt32,
    CastIntUint8,
    CastUint8Int,
    CastSlicePtr,

    LibcWrite,

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
        let layout = Layout::from_size_align(size, 8).unwrap();
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
            for i in 0..count {
                let len = len + count - i;
                self.reset(len + 1);
                for _ in 0..len {
                    self.increment(1);
                    let prev: u8 = *self.sp.byte_offset(-1);
                    *self.sp.cast() = prev;
                }
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
    data: Vec<u8>,
}

impl StaticMemory {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    pub fn push_string_slice(&mut self, string: &str) -> usize {
        let slice = Slice::from_string(string);
        let as_raw: [u8; size_of::<*mut Slice>()] = unsafe { mem::transmute(slice) };
        self.push(&as_raw)
    }

    pub fn push(&mut self, val: &[u8]) -> usize {
        let old_len = self.data.len();
        self.data.extend_from_slice(val);
        old_len
    }

    fn index(&self, index: usize, len: usize) -> &[u8] {
        &self.data[index..(index + len)]
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
}

impl FfiType {
    fn from_str(from: &str) -> Self {
        match from {
            "void" => Self::Void,
            "pointer" => Self::Pointer,
            "c_string" => Self::Cstring,
            "u16" => Self::U16,
            "u32" => Self::U32,
            "i16" => Self::I16,
            "i32" => Self::I32,
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
    Value(*mut u8, Layout),
    Slice(*mut u8, Layout, Layout),
}

struct AnyVec {
    values: Vec<AnyVecItem>,
}

impl AnyVec {
    fn new() -> Self {
        Self { values: Vec::new() }
    }

    fn push<T>(&mut self, value: T) {
        let (ptr, layout) = unsafe { alloc_value(value) };
        self.values.push(AnyVecItem::Value(ptr, layout));
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
                AnyVecItem::Value(v, _) => v,
                AnyVecItem::Slice(v, _, _) => v,
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
                    AnyVecItem::Value(v, layout) => dealloc(v, layout),
                    AnyVecItem::Slice(v, v_layout, slice_layout) => {
                        let slice_ptr: *mut u8 = *v.cast();
                        dealloc(v, v_layout);
                        dealloc(slice_ptr, slice_layout);
                    }
                }
            };
        }
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

    pub fn run(mut self) {
        let mut pc = 0;

        loop {
            #[cfg(debug_assertions)]
            if env::var("DEBUG").is_ok() {
                println!("executing instruction: {:#?}", self.instructions[pc]);
            }

            match self.instructions[pc] {
                Instruction::AddI => {
                    let a = self.stack.pop::<isize>();
                    let b = self.stack.pop::<isize>();
                    self.stack.push(a + b);
                }
                Instruction::Exit => return,
                Instruction::PushI(v) => {
                    self.stack.push(v);
                }
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
                    let boolean = self.stack.pop::<isize>();
                    self.stack.push(boolean);
                    if boolean != 1 {
                        pc = i;
                        continue;
                    }
                }
                Instruction::JumpIfTrue(i) => {
                    let boolean = self.stack.pop::<isize>();
                    self.stack.push(boolean);
                    if boolean == 1 {
                        pc = i;
                        continue;
                    }
                }
                Instruction::ToBoolI => {
                    let int = self.stack.pop::<isize>();
                    if int > 0 {
                        self.stack.push::<isize>(1);
                    } else {
                        self.stack.push::<isize>(0);
                    }
                }
                Instruction::NegateBool => {
                    let int = self.stack.pop::<isize>();
                    self.stack.push(int ^ 1);
                }
                Instruction::MinusInt => {
                    let int = self.stack.pop::<isize>();
                    self.stack.push(-int);
                }
                Instruction::CompareI => {
                    let a = self.stack.pop::<isize>();
                    let b = self.stack.pop::<isize>();
                    if a == b {
                        self.stack.push::<isize>(1);
                    } else {
                        self.stack.push::<isize>(0)
                    }
                }
                Instruction::And => {
                    let a = self.stack.pop::<isize>();
                    let b = self.stack.pop::<isize>();
                    self.stack.push(a & b);
                }
                Instruction::Or => {
                    let a = self.stack.pop::<isize>();
                    let b = self.stack.pop::<isize>();
                    self.stack.push(a | b);
                }
                Instruction::DivideI => {
                    let a = self.stack.pop::<isize>();
                    let b = self.stack.pop::<isize>();
                    self.stack.push(b / a);
                }
                Instruction::MultiplyI => {
                    let a = self.stack.pop::<isize>();
                    let b = self.stack.pop::<isize>();
                    self.stack.push(a * b);
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
                Instruction::PushU8(v) => self.stack.push(v),
                Instruction::PushU16(v) => self.stack.push(v),
                Instruction::PushU32(v) => self.stack.push(v),
                Instruction::PushI16(v) => self.stack.push(v),
                Instruction::PushI32(v) => self.stack.push(v),
                Instruction::PushStatic(index, len) => {
                    self.stack.push_size(self.static_memory.index(index, len));
                }
                Instruction::CastIntUint => {
                    let target = self.stack.pop::<isize>();
                    self.stack.push::<usize>(target as usize);
                }
                Instruction::CastIntInt32 => {
                    let target: isize = self.stack.pop();
                    self.stack.push::<i32>(target as i32);
                }
                Instruction::CastIntUint8 => {
                    let target = self.stack.pop::<isize>();
                    self.stack.push::<u8>(target as u8);
                }
                Instruction::CastUint8Int => {
                    let target = self.stack.pop::<u8>();
                    self.stack.push::<isize>(target as isize);
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
                Instruction::ModuloI => {
                    let a = self.stack.pop::<isize>();
                    let b = self.stack.pop::<isize>();

                    self.stack.push(b % a);
                }
                Instruction::Shift(len, count) => {
                    self.stack.shift(len, count);
                }
                Instruction::LibcWrite => {
                    let slice = unsafe { &mut *self.stack.pop::<*mut Slice>() };
                    let fd = self.stack.pop::<isize>();
                    let result = unsafe {
                        libc::write(
                            fd as i32,
                            slice.data.as_ptr() as *const libc::c_void,
                            slice.data.len(),
                        )
                    };
                    self.stack.push(result as isize);
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
                        argument_types.push(FfiType::from_str(&slice.string()));
                    }

                    let return_argument = unsafe {
                        FfiType::from_str(&(&mut *self.stack.pop::<*mut Slice>()).string())
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
                            FfiType::I32 | FfiType::U32 => {
                                converted_args.push::<u32>(unsafe { **arg_ptr.cast::<*mut u32>() })
                            }
                            FfiType::U16 | FfiType::I16 => {
                                converted_args.push::<u16>(unsafe { **arg_ptr.cast::<*mut u16>() })
                            }
                            FfiType::Pointer => converted_args
                                .push::<*mut u8>(unsafe { **arg_ptr.cast::<*mut *mut u8>() }),
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
                    if let Some((ptr, _)) = result_ptr {
                        let v = unsafe { *ptr.cast::<usize>() };
                        println!("FfiCall returned usize({})", v);
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
            }

            self.gc.run(self.stack.sp, self.stack.sp_end());
            pc += 1;
        }
    }
}
