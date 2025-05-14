use std::alloc::{alloc, dealloc, Layout};

#[derive(Debug, Clone)]
pub enum Instruction {
    Increment(usize),
    PushI(isize),
    AddI(usize, usize),
    // dst = src * len
    Copy(usize, usize, usize),
    Exit,
    Debug,
    Reset(usize),
    JumpAndLink(usize),
    Jump(usize),
    Return,
}

pub struct Stack {
    data: *mut u8,
    sp: *mut u8,
    size: usize,
}

impl Drop for Stack {
    fn drop(&mut self) {
        unsafe {
            dealloc(
                self.data,
                Layout::from_size_align(self.size, Layout::new::<u8>().align()).unwrap(),
            )
        };
    }
}

impl Stack {
    fn new(size: usize) -> Self {
        let data =
            unsafe { alloc(Layout::from_size_align(size, Layout::new::<u8>().align()).unwrap()) };

        Self {
            sp: unsafe { data.byte_offset(size as isize) },
            data,
            size,
        }
    }

    fn push<T: Copy>(&mut self, item: T) {
        unsafe {
            self.sp = self.sp.byte_offset(-(size_of::<T>() as isize));
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
            self.sp = self.sp.byte_offset(size_of::<T>() as isize);
            item
        }
    }

    fn offset_write<T: Copy>(&mut self, offset: usize, write: T) {
        unsafe {
            *self.sp.byte_offset(offset as isize).cast() = write;
        }
    }

    fn offset<T: Copy>(&mut self, offset: usize) -> T {
        unsafe { *self.sp.byte_offset(offset as isize).cast() }
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
            std::ptr::copy_nonoverlapping(
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
                Instruction::AddI(a_offset, b_offset) => {
                    let a = self.stack.offset::<isize>(a_offset);
                    let b = self.stack.offset::<isize>(b_offset);
                    self.stack.offset_write(b_offset, a + b);
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
            }

            pc += 1;
        }
    }
}
