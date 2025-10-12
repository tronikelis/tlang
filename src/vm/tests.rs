use anyhow::Result;
use pretty_assertions::assert_eq;

use super::*;
use crate::{ast, compiler, ir, lexer, linker};

enum StackBuilderItem {
    Increment(Vec<u8>),
    Value(Vec<u8>),
}

struct StackBuilder {
    stack: Vec<StackBuilderItem>,
}

impl StackBuilder {
    fn new() -> Self {
        Self { stack: Vec::new() }
    }

    fn increment(mut self, by: usize) -> Self {
        let mut value = Vec::new();
        for _ in 0..by {
            value.push(0);
        }
        self.stack.push(StackBuilderItem::Increment(value));
        self
    }

    fn push<T>(mut self, value: T) -> Self {
        let mut vec = Vec::new();
        let mut value_ptr = &value as *const T as *const u8;

        for _ in 0..size_of::<T>() {
            unsafe {
                vec.push(*value_ptr);
                value_ptr = value_ptr.byte_offset(1);
            }
        }

        self.stack.push(StackBuilderItem::Value(vec));
        self
    }

    fn len(&self) -> usize {
        self.stack.iter().fold(0, |acc, curr| {
            acc + match curr {
                StackBuilderItem::Value(v) | StackBuilderItem::Increment(v) => v.len(),
            }
        })
    }

    fn assert_eq(self, stack: &Stack) {
        let mut right = stack.debug(self.len());
        assert_eq!(self.len(), right.len());

        let mut acc = Vec::<u8>::new();
        for value in self.stack.into_iter().rev() {
            match value {
                StackBuilderItem::Increment(val) => {
                    for i in 0..val.len() {
                        // have to go through all of these hoops
                        // because increment does not zero out anything,
                        // just moves the sp
                        right[acc.len() + i] = val[i];
                    }
                    acc.extend(val.iter());
                }
                StackBuilderItem::Value(val) => {
                    acc.extend(val.iter());
                }
            }
        }

        assert_eq!(acc, right);
    }
}

const BUILTIN: &str = r"
    type bool _
    type int _
    type int32 _
    type int16 _
    type ptr _
    type string _
    type uint _
    type uint8 _
    type uint16 _
    type uint32 _
    type void _
    type Type _

    fn libc_write(fd int, slice uint8[]) int {}
    fn len(slice Type) int {}
    fn append(slice Type, value Type) void {}
    fn new(typ Type, args Type...) Type {}

    fn dll_open(path string) ptr {}
    fn ffi_create(dll ptr, function string, return_param string, args string...) ptr {}
    fn ffi_call(ffi ptr, args ptr...) ptr {}

";

fn compile_test_code(code: &str) -> Result<Vec<Instruction>> {
    let tokens = lexer::Lexer::new(&code).run()?;
    let ast = ast::Ast::new(&tokens)?;
    let ir = ir::Ir::new(&ast)?;

    let compiled = compiler::compile(ir)?;
    let linked = linker::link(compiled.functions)?;

    Ok(linked)
}

fn get_test_code(code: &str) -> String {
    BUILTIN.to_string() + code
}

#[test]
fn simple_fn() -> Result<()> {
    let code = get_test_code(
        r"
            fn simple(a int, b int) int {
                return a + b
            }
            fn main() void {
                simple(25, 20)
            }
        ",
    );

    assert_eq!(
        vec![
            Instruction::PushI(25),
            Instruction::PushI(20),
            Instruction::JumpAndLink(13),
            Instruction::Reset(8),
            Instruction::Reset(8),
            Instruction::Exit,
            Instruction::Return,
            Instruction::Return,
            Instruction::Return,
            Instruction::Return,
            Instruction::Return,
            Instruction::Return,
            Instruction::Return,
            Instruction::Increment(8),
            Instruction::Copy(0, 24, 8),
            Instruction::Increment(8),
            Instruction::Copy(0, 24, 8),
            Instruction::AddI(8),
            Instruction::Copy(24, 0, 8),
            Instruction::Reset(8),
            Instruction::Return,
            Instruction::Reset(8),
            Instruction::Return
        ],
        compile_test_code(&code)?
    );

    Ok(())
}

#[test]
fn dot_access() -> Result<()> {
    let code = get_test_code(
        r"
        type TcpConn struct {
            addr uint32
            fd int32
            port uint16
        }

        fn main() void {
            let conn TcpConn = TcpConn{
                addr: 1,
                fd: 2,
                port: 3,
            }

            let fd int32 = conn.fd
            __debug__
        }
    ",
    );

    assert_eq!(
        vec![
            // bottom padding
            Instruction::Increment(2,),
            Instruction::PushU16(3,),
            Instruction::PushI32(2,),
            Instruction::PushU32(1,),
            Instruction::Increment(12,),
            Instruction::Copy(0, 12, 12,),
            Instruction::Increment(4,),
            Instruction::Copy(0, 8, 4,),
            Instruction::Shift(4, 12,),
            Instruction::Debug,
            // total var stack size 12 for struct, 4 for int32 fd
            Instruction::Reset(16,),
            Instruction::Exit,
            Instruction::Return,
            Instruction::Return,
            Instruction::Return,
            Instruction::Return,
            Instruction::Return,
            Instruction::Return,
            Instruction::Return,
        ],
        compile_test_code(&code)?
    );

    Ok(())
}

#[test]
fn fn_call() -> Result<()> {
    let code = get_test_code(
        r"
            fn foo(a int32, b uint8, b int) void {

            }

            fn main() void {
                foo(1, 2, 3)
                __debug__
            }
        ",
    );

    assert_eq!(
        vec![
            Instruction::PushI32(1,),
            Instruction::PushU8(2,),
            Instruction::Increment(3,),
            Instruction::PushI(3,),
            Instruction::JumpAndLink(11,),
            Instruction::Reset(16,),
            Instruction::Debug,
            Instruction::Exit,
            Instruction::Return,
            Instruction::Return,
            Instruction::Return,
            Instruction::Return,
            Instruction::Return,
            Instruction::Return,
            Instruction::Return,
            Instruction::Return,
        ],
        compile_test_code(&code)?
    );

    Ok(())
}

#[test]
fn fn_call_dot_access() -> Result<()> {
    let code = get_test_code(
        r"
        type TcpConn struct {
            addr uint32
            fd int32
            port uint16
        }

        fn foobar(a int32, b uint32) int32 {
            return 32 as int32
        }

        fn main() void {
            let conn TcpConn = TcpConn{
                addr: 1,
                fd: 2,
                port: 3,
            }

            foobar(conn.fd, conn.addr)
            __debug__
        }
    ",
    );

    assert_eq!(
        vec![
            Instruction::Increment(2,),
            Instruction::PushU16(3,),
            Instruction::PushI32(2,),
            Instruction::PushU32(1,),
            Instruction::Increment(4,),
            Instruction::Increment(12,),
            Instruction::Copy(0, 16, 12,),
            Instruction::Increment(4,),
            Instruction::Copy(0, 8, 4,),
            Instruction::Shift(4, 12,),
            Instruction::Increment(12,),
            Instruction::Copy(0, 20, 12,),
            Instruction::Increment(4,),
            Instruction::Copy(0, 4, 4,),
            Instruction::Shift(4, 12,),
            Instruction::JumpAndLink(24,),
            Instruction::Reset(4,),
            Instruction::Reset(8,),
            Instruction::Debug,
            Instruction::Reset(12,),
            Instruction::Exit,
            Instruction::Return,
            Instruction::Return,
            Instruction::Return,
            Instruction::PushI(32,),
            Instruction::CastInt(8, 4),
            Instruction::Copy(16, 0, 4,),
            Instruction::Reset(4,),
            Instruction::Return,
            Instruction::Reset(4,),
            Instruction::Return,
            Instruction::Return,
            Instruction::Return,
            Instruction::Return,
            Instruction::Return,
        ],
        compile_test_code(&code)?
    );

    Ok(())
}

#[test]
fn builtin_slice_new() -> Result<()> {
    let code = get_test_code(
        r"
            type TcpConn struct {
                addr uint32
                fd int32
                port uint16
            }

            fn main() void {
                let conn TcpConn = TcpConn{
                    addr: 1,
                    fd: 2,
                    port: 3,
                }
                let buf uint8[] = new(uint8[], 0 as uint8, 10)
                let f int32 = conn.fd
                let b int = len(buf)
                __debug__
            }
        ",
    );

    assert_eq!(
        vec![
            Instruction::Increment(2,),
            Instruction::PushU16(3,),
            Instruction::PushI32(2,),
            Instruction::PushU32(1,),
            Instruction::Increment(4,),
            Instruction::PushI(10,),
            Instruction::PushI(0,),
            Instruction::CastUint(8, 1),
            Instruction::PushSliceNewLen(1,),
            Instruction::Increment(12,),
            Instruction::Copy(0, 24, 12,),
            Instruction::Increment(4,),
            Instruction::Copy(0, 8, 4,),
            Instruction::Shift(4, 12,),
            Instruction::Increment(4,),
            Instruction::Increment(8,),
            Instruction::Copy(0, 16, 8,),
            Instruction::SliceLen,
            Instruction::Debug,
            Instruction::Reset(40,),
            Instruction::Exit,
            Instruction::Return,
            Instruction::Return,
            Instruction::Return,
            Instruction::Return,
            Instruction::Return,
            Instruction::Return,
            Instruction::Return,
        ],
        compile_test_code(&code)?
    );

    Ok(())
}

#[test]
fn cast_int_uint() -> Result<()> {
    let code = get_test_code(
        r"
            fn main() void {
                let foo int8 = 25
                let foo1 int = foo as int
                let foo2 uint = foo as uint
            }
        ",
    );

    let mut vm = Vm::new(compile_test_code(&code)?, StaticMemory::new());
    vm.run();

    StackBuilder::new()
        .push(25 as i8)
        .increment(7)
        .push(25 as isize)
        .push(25 as usize)
        .assert_eq(&vm.stack);

    Ok(())
}

#[test]
fn arithmetic() -> Result<()> {
    let code = get_test_code(
        r"
            fn main() void {
                let foo int8 = 25
                let foo2 int8 = 1
                let foo3 int8 = foo + foo2

                let bar uint32 = 1
                let bar1 uint32 = 29
                let bar3 uint32 = bar + bar1
            }
        ",
    );

    let mut vm = Vm::new(compile_test_code(&code)?, StaticMemory::new());
    vm.run();

    StackBuilder::new()
        .push(25 as i8)
        .push(1 as i8)
        .push(26 as i8)
        .increment(1)
        .push(1 as u32)
        .push(29 as u32)
        .push(30 as u32)
        .assert_eq(&vm.stack);

    Ok(())
}

#[test]
fn stack_shift_works() {
    let mut stack = Stack::new(256);
    stack.increment(4);
    let old_sp = stack.sp;
    stack.push(0xFFFFFFFF as u32);
    stack.shift(4, 4);

    assert_eq!(old_sp, stack.sp);
    assert_eq!(stack.pop::<u32>(), 0xFFFFFFFF);
}

#[test]
fn vm_cast_int() {
    let mut vm = Vm::new(Vec::new(), StaticMemory::new());

    vm.push_i8(-1);
    vm.cast_int(1, 4);

    let as32: u32 = vm.stack.pop();
    assert_eq!(as32, 0xFFFFFFFF);

    vm.push_u32(0x80_00_00_1F);
    vm.cast_int(4, 1);
    let as8: u8 = vm.stack.pop();
    assert_eq!(as8, 0x9F);
}

#[test]
fn vm_cast_uint() {
    let mut vm = Vm::new(Vec::new(), StaticMemory::new());
    let old_sp = vm.stack.sp;

    vm.push_u16(0xFFFF);
    vm.cast_uint(2, 1);

    assert_eq!(old_sp as usize - 1, vm.stack.sp as usize);
    assert_eq!(vm.stack.pop::<u8>(), 0xFF);
}

#[test]
fn vm_add_i32() {
    let mut vm = Vm::new(Vec::new(), StaticMemory::new());
    let old_sp = vm.stack.sp;

    vm.push_i32(10);
    vm.push_i32(25);
    vm.add_i(4);

    assert_eq!((old_sp as usize) - 4, vm.stack.sp as usize);
    assert_eq!(vm.stack.pop::<i32>(), 35);
}

#[test]
fn vm_add_i8() {
    let mut vm = Vm::new(Vec::new(), StaticMemory::new());
    let old_sp = vm.stack.sp;

    vm.push_i8(-3);
    vm.push_i8(-8);
    vm.add_i(1);

    assert_eq!((old_sp as usize) - 1, vm.stack.sp as usize);
    assert_eq!(vm.stack.pop::<i8>(), -11);
}

#[test]
fn vm_add_i16() {
    let mut vm = Vm::new(Vec::new(), StaticMemory::new());
    let old_sp = vm.stack.sp;

    vm.push_i16(8);
    vm.push_i16(-9);
    vm.add_i(2);

    assert_eq!((old_sp as usize) - 2, vm.stack.sp as usize);
    assert_eq!(vm.stack.pop::<i16>(), -1);
}

#[test]
fn vm_add_i() {
    let mut vm = Vm::new(Vec::new(), StaticMemory::new());
    let old_sp = vm.stack.sp;

    vm.push_i(-11);
    vm.push_i(12);
    vm.add_i(size_of::<isize>() as u8);

    assert_eq!((old_sp as usize) - size_of::<isize>(), vm.stack.sp as usize);
    assert_eq!(vm.stack.pop::<isize>(), 1);
}

#[test]
fn vm_minus_i() {
    let mut vm = Vm::new(Vec::new(), StaticMemory::new());
    let old_sp = vm.stack.sp;

    vm.push_i(-11);
    vm.push_i(12);
    vm.minus_i(size_of::<isize>() as u8);

    assert_eq!((old_sp as usize) - size_of::<isize>(), vm.stack.sp as usize);
    assert_eq!(vm.stack.pop::<isize>(), -23);
}

#[test]
fn vm_minus_u() {
    let mut vm = Vm::new(Vec::new(), StaticMemory::new());
    let old_sp = vm.stack.sp;

    vm.push_u(10);
    vm.push_u(5);
    vm.minus_u(size_of::<isize>() as u8);

    assert_eq!((old_sp as usize) - size_of::<isize>(), vm.stack.sp as usize);
    assert_eq!(vm.stack.pop::<usize>(), 5);
}
