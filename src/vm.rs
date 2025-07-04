use core::{cell::RefCell, str};
use std::{
    alloc::{alloc, dealloc, Layout},
    collections::HashMap,
    mem, ptr, slice,
};
use syscalls::{syscall0, syscall1, syscall2, syscall3, syscall4, syscall5, syscall6, Sysno};

#[allow(dead_code)]
#[repr(usize)]
enum PortableSysno {
    Write = 0,
}

#[derive(Debug)]
enum GcObjectData {
    Slice(*mut Slice),
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
        };
        self.objects.insert(addr, RefCell::new(object));
    }

    fn mark_object(&self, object: &RefCell<GcObject>) {
        if object.borrow().marked {
            return;
        }
        object.borrow_mut().marked = true;

        match object.borrow().data {
            GcObjectData::Slice(slice) => {
                let slice = unsafe { &mut *slice };
                let len = slice.data.len();

                // check if addresses are even possible
                if len % size_of::<usize>() != 0 {
                    return;
                }

                let mut ptr = slice.data.as_ptr();
                for _ in 0..len / size_of::<usize>() {
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
                    GcObjectData::Slice(slice) => {
                        let _ = unsafe { Box::from_raw(slice) };
                    }
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

fn layout(size: usize) -> Layout {
    Layout::from_size_align(size, size_of::<usize>()).unwrap()
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

    pub fn new_from_string(v: &str) -> *mut Self {
        let mut data = Vec::new();
        data.extend_from_slice(v.as_bytes());
        Box::into_raw(Box::new(Self { len: v.len(), data }))
    }

    pub fn new_default_len(len: usize, val: &[u8]) -> *mut Self {
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
}

#[derive(Debug, Clone)]
pub enum Instruction {
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
    PushU8(u8),
    PushSlice,
    PushSliceNewLen(usize),
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
    CastIntUint8,
    CastUint8Int,
    CastSlicePtr,

    Syscall0,
    Syscall1,
    Syscall2,
    Syscall3,
    Syscall4,
    Syscall5,
    Syscall6,
}

pub struct Stack {
    data: *mut u8,
    sp: *mut u8,
    size: usize,
}

impl Drop for Stack {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.data, layout(self.size));
        };
    }
}

impl Stack {
    fn new(size: usize) -> Self {
        let data = unsafe { alloc(layout(size)) };

        Self {
            sp: unsafe { data.byte_offset(size as isize) },
            data,
            size,
        }
    }

    fn sp_end(&self) -> *mut u8 {
        unsafe { self.data.byte_offset(self.size as isize) }
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
            let mut data_end = self.data.byte_offset(self.size as isize);
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
}

pub struct StaticMemory {
    data: Vec<u8>,
}

impl StaticMemory {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    pub fn push_string_slice(&mut self, string: &str) -> usize {
        let slice = Slice::new_from_string(string);
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

pub struct Vm {
    stack: Stack,
    instructions: Vec<Instruction>,
    static_memory: StaticMemory,
    gc: Gc,
}

impl Vm {
    pub fn new(instructions: Vec<Instruction>, static_memory: StaticMemory) -> Self {
        return Self {
            stack: Stack::new(4096),
            instructions,
            static_memory,
            gc: Gc::new(),
        };
    }

    pub unsafe fn pop_sysno(&mut self) -> Sysno {
        let sysno: usize = self.stack.pop();
        let portable_sysno: PortableSysno = mem::transmute(sysno);
        match portable_sysno {
            PortableSysno::Write => Sysno::write,
        }
    }

    pub fn run(mut self) {
        let mut pc = 0;

        loop {
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

                    let slice = Slice::new_default_len(len as usize, &val);
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
                Instruction::PushU8(v) => {
                    self.stack.push(v);
                }
                Instruction::PushStatic(index, len) => {
                    self.stack.push_size(self.static_memory.index(index, len));
                }
                Instruction::CastIntUint => {
                    let target = self.stack.pop::<isize>();
                    self.stack.push::<usize>(target as usize);
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
                        unsafe { &mut *Slice::new_from_string(str::from_utf8_unchecked(&b.data)) };
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
                Instruction::Syscall0 => {
                    let result = unsafe { syscall0(self.pop_sysno()).unwrap() };
                    self.stack.push(result)
                }
                Instruction::Syscall1 => {
                    let arg1: usize = self.stack.pop();
                    let result = unsafe { syscall1(self.pop_sysno(), arg1).unwrap() };
                    self.stack.push(result);
                }
                Instruction::Syscall2 => {
                    let arg2: usize = self.stack.pop();
                    let arg1: usize = self.stack.pop();
                    let result = unsafe { syscall2(self.pop_sysno(), arg1, arg2).unwrap() };
                    self.stack.push(result);
                }
                Instruction::Syscall3 => {
                    let arg3: usize = self.stack.pop();
                    let arg2: usize = self.stack.pop();
                    let arg1: usize = self.stack.pop();
                    let result = unsafe { syscall3(self.pop_sysno(), arg1, arg2, arg3).unwrap() };
                    self.stack.push(result);
                }
                Instruction::Syscall4 => {
                    let arg4: usize = self.stack.pop();
                    let arg3: usize = self.stack.pop();
                    let arg2: usize = self.stack.pop();
                    let arg1: usize = self.stack.pop();
                    let result =
                        unsafe { syscall4(self.pop_sysno(), arg1, arg2, arg3, arg4).unwrap() };
                    self.stack.push(result);
                }
                Instruction::Syscall5 => {
                    let arg5: usize = self.stack.pop();
                    let arg4: usize = self.stack.pop();
                    let arg3: usize = self.stack.pop();
                    let arg2: usize = self.stack.pop();
                    let arg1: usize = self.stack.pop();
                    let result = unsafe {
                        syscall5(self.pop_sysno(), arg1, arg2, arg3, arg4, arg5).unwrap()
                    };
                    self.stack.push(result);
                }
                Instruction::Syscall6 => {
                    let arg6: usize = self.stack.pop();
                    let arg5: usize = self.stack.pop();
                    let arg4: usize = self.stack.pop();
                    let arg3: usize = self.stack.pop();
                    let arg2: usize = self.stack.pop();
                    let arg1: usize = self.stack.pop();
                    let result = unsafe {
                        syscall6(self.pop_sysno(), arg1, arg2, arg3, arg4, arg5, arg6).unwrap()
                    };
                    self.stack.push(result);
                }
            }

            self.gc.run(self.stack.sp, self.stack.sp_end());
            pc += 1;
        }
    }
}
