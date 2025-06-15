use core::str;
use std::{
    alloc::{alloc, dealloc, Layout},
    fs::File,
    io::Write,
    mem,
    os::{fd::IntoRawFd, unix::io::FromRawFd},
    ptr, slice,
};

fn layout_u8(size: usize) -> Layout {
    Layout::from_size_align(size, 1).unwrap()
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

    fn index(&self, index: isize, size: usize) -> &[u8] {
        let index = index as usize;
        let from = index * size;
        &self.data[from..(from + size)]
    }

    fn index_set(&mut self, index: isize, val: Vec<u8>) {
        let from = index as usize * val.len();
        for (i, v) in val.into_iter().enumerate() {
            self.data[from + i] = v;
        }
    }

    fn concat(&mut self, other: &Self) {
        self.len += other.len;
        self.data.extend_from_slice(&other.data);
    }

    fn append(&mut self, val: Vec<u8>) {
        self.len += 1;
        self.data.extend_from_slice(&val);
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
    // len
    Shift(usize),
    Reset(usize),
    PushI(isize),
    PushU8(u8),
    PushSlice,
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

    ToBoolI,
    NegateBool,
    CompareI,
    And,
    Or,

    CastIntUint8,
    CastUint8Int,

    SyscallWrite,
}

pub struct Stack {
    data: *mut u8,
    sp: *mut u8,
    size: usize,
}

impl Drop for Stack {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.data, layout_u8(self.size));
        };
    }
}

impl Stack {
    fn new(size: usize) -> Self {
        let data = unsafe { alloc(layout_u8(size)) };

        Self {
            sp: unsafe { data.byte_offset(size as isize) },
            data,
            size,
        }
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

    fn shift(&mut self, len: usize) {
        unsafe {
            self.reset(len + 1);
            for _ in 0..len {
                self.increment(1);
                let prev: u8 = *self.sp.byte_offset(-1);
                *self.sp.cast() = prev;
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
}

impl Vm {
    pub fn new(instructions: Vec<Instruction>, static_memory: StaticMemory) -> Self {
        return Self {
            stack: Stack::new(4096),
            instructions,
            static_memory,
        };
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
                    self.stack.push(Slice::new());
                }
                Instruction::SliceAppend(size) => {
                    let item = self.stack.pop_size(size).to_vec();
                    let slice = unsafe { &mut *self.stack.pop::<*mut Slice>() };

                    slice.append(item);
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

                    slice.index_set(index, item);
                }
                Instruction::SliceLen => {
                    let slice = unsafe { &mut *self.stack.pop::<*mut Slice>() };
                    self.stack.push(slice.len as isize);
                }
                Instruction::PushU8(v) => {
                    self.stack.push(v);
                }
                Instruction::SyscallWrite => {
                    let fd = self.stack.pop::<isize>();
                    let slice = unsafe { &mut *self.stack.pop::<*mut Slice>() };

                    let mut file = unsafe { File::from_raw_fd(fd as i32) };
                    file.write_all(&slice.data).unwrap();
                    let _ = file.into_raw_fd(); // don't close the file descriptor
                }
                Instruction::PushStatic(index, len) => {
                    self.stack.push_size(self.static_memory.index(index, len));
                }
                Instruction::CastIntUint8 => {
                    let target = self.stack.pop::<isize>();
                    self.stack.push::<u8>(target as u8);
                }
                Instruction::CastUint8Int => {
                    let target = self.stack.pop::<u8>();
                    self.stack.push::<isize>(target as isize);
                }
                Instruction::AddString => {
                    let a = unsafe { &mut *self.stack.pop::<*mut Slice>() };
                    let b = unsafe { &mut *self.stack.pop::<*mut Slice>() };

                    let slice =
                        unsafe { &mut *Slice::new_from_string(str::from_utf8_unchecked(&b.data)) };
                    slice.concat(a);
                    self.stack.push(slice as *mut Slice);
                }
                Instruction::ModuloI => {
                    let a = self.stack.pop::<isize>();
                    let b = self.stack.pop::<isize>();

                    self.stack.push(b % a);
                }
                Instruction::Shift(len) => {
                    self.stack.shift(len);
                }
            }

            pc += 1;
        }
    }
}
