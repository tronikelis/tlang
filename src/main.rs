mod vm;
use vm::Instruction;

mod lexer;

fn main() {
    let instructions = [
        // fn(a,b)
        Instruction::PushI(10),
        Instruction::PushI(20),
        Instruction::JumpAndLink(5),
        // Instruction::Reset(size_of::<isize>() * 2),
        Instruction::Debug,
        Instruction::Exit,
        // fn(a,b) => a + b
        // [ret, a, b]
        Instruction::AddI(8, 16),
        Instruction::Return,
    ];

    vm::Vm::new(&instructions).run();
}
