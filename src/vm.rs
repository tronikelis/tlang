use std::{
    alloc::{alloc, dealloc, Layout},
    ptr, slice,
};

fn layout_u8(size: usize) -> Layout {
    Layout::from_size_align(size, 1).unwrap()
}

struct Slice {
    data: Vec<u8>,
}

impl Slice {
    fn new() -> *mut Self {
        Box::into_raw(Box::new(Self { data: Vec::new() }))
    }

    fn index(&self, index: isize, size: usize) -> &[u8] {
        let index = index as usize;
        &self.data[index..(index + size)]
    }

    fn index_set(&mut self, index: isize, val: &[u8]) {
        for (i, v) in val.iter().enumerate() {
            self.data[(index as usize) + i] = *v;
        }
    }

    fn append(&mut self, val: &[u8]) {
        self.data.reserve(val.len());
        for v in val {
            self.data.push(*v);
        }
    }
}

#[derive(Debug, Clone)]
pub enum Instruction {
    SliceAppend(usize),
    SliceIndexGet(usize),
    SliceIndexSet(usize),
    PushSlice,
    Increment(usize),
    PushI(isize),
    AddI,
    MultiplyI,
    DivideI,
    // dst = src * len
    Copy(usize, usize, usize),
    Exit,
    Debug,
    Reset(usize),
    JumpAndLink(usize),
    Jump(usize),
    Return,
    JumpIfTrue(usize),
    ToBool,
    NegateBool,
    MinusInt,
    CompareInt,
    And,
    Or,
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

pub struct Vm {
    stack: Stack,
    instructions: Vec<Instruction>,
}

impl Vm {
    pub fn new(instructions: Vec<Instruction>) -> Self {
        return Self {
            stack: Stack::new(4096),
            instructions,
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
                Instruction::ToBool => {
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
                Instruction::CompareInt => {
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
                    self.stack.push(a / b);
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
                    let slice = self.stack.pop::<*mut Slice>();
                    let item = self.stack.pop_size(size);

                    let slice = unsafe { &mut *slice };
                    slice.append(item);
                }
                Instruction::SliceIndexGet(size) => {
                    let slice = self.stack.pop::<*mut Slice>();
                    let index = self.stack.pop::<isize>();

                    let slice = unsafe { &mut *slice };
                    self.stack.push_size(slice.index(index, size));
                }
                Instruction::SliceIndexSet(size) => {
                    let slice = self.stack.pop::<*mut Slice>();
                    let index = self.stack.pop::<isize>();
                    let item = self.stack.pop_size(size);

                    let slice = unsafe { &mut *slice };
                    slice.index_set(index, item);
                }
            }

            pc += 1;
        }
    }
}
