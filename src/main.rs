use std::collections::HashMap;

mod vm;

mod ast;
mod compiler;
mod instructions;
mod lexer;
mod linker;

fn main() {
    let code = String::from(
        "
                fn len(slice Type) int {}
                fn append(slice Type, value Type) void {}

                fn syscall0(sysno uint) uint {}
                fn syscall1(sysno uint, arg1 uint) uint {}
                fn syscall2(sysno uint, arg1 uint, arg2 uint) uint {}
                fn syscall3(sysno uint, arg1 uint, arg2 uint, arg3 uint) uint {}
                fn syscall4(sysno uint, arg1 uint, arg2 uint, arg3 uint, arg4 uint) uint {}
                fn syscall5(sysno uint, arg1 uint, arg2 uint, arg3 uint, arg4 uint, arg5 uint) uint {}
                fn syscall6(sysno uint, arg1 uint, arg2 uint, arg3 uint, arg4 uint, arg5 uint, arg6 uint) uint {}

                fn add(x int...) int {
                    let final int = 0
                    return final
                }

                fn syscall_write(fd int, slice uint8[]) uint {
                    // ssize_t write(int fd, const void buf[.count], size_t count);
                    return syscall3(uint(1), uint(fd), uint(ptr(slice)), uint(len(slice)))
                }

                fn factorial(n int) int {
                    if n == 1 {
                        return n
                    }

                    return n*factorial(n-1)
                }

                fn foobar(first_arg uint8, second_arg int) int {
                    return second_arg
                }

                fn slice_reverse(slice uint8[]) void {
                    for let i int = 0; i < len(slice) / 2; i++ {
                        let j int = len(slice)-1-i
                        let temp uint8 = slice[i]
                        slice[i] = slice[j]
                        slice[j] = temp
                    }
                }

                fn itoa(x int) string {
                    let str uint8[] = {}
                    for {
                        append(str, uint8(48 + x % 10))
                        x = x / 10
                        if x == 0 {
                            break
                        }
                    }

                    slice_reverse(str)
                    return string(str)
                }

                fn main() void {
                    let one_two_three string = itoa(69420)

                    if false && true {
                        syscall_write(1, uint8[](\"NICE GUYS\\n\"))
                    }

                    let nums int[] = {1, 2, 3, 4, 5, 6}
                    add()


                    for let i int = 0; i < 100; i++ {
                        let str string = \"\"

                        if i % 3 == 0 {
                            str = str + \"Fizz\"
                        }
                        if i % 5 == 0 {
                            str = str + \"Buzz\"
                        }

                        if i % 3 != 0 && i % 5 != 0 {
                            str = str + itoa(i)
                        }

                        syscall_write(1, uint8[](str + \"\\n\"))
                    }
                }
            ",
    );

    let tokens = lexer::Lexer::new(&code).run().unwrap();
    println!("{tokens:#?}");
    let ast = ast::Ast::new(&tokens).unwrap();
    println!("{:#?}", ast);

    let mut functions = HashMap::<String, Vec<Vec<instructions::Instruction>>>::new();
    let mut static_memory = vm::StaticMemory::new();

    for v in &ast.functions {
        let compiled = compiler::FunctionCompiler::new(v, &mut static_memory)
            .compile()
            .unwrap();
        println!("{:#?}", compiled);
        functions.insert(v.identifier.clone(), compiled);
    }

    let instructions = linker::link(&functions).unwrap();

    println!("{:#?}", instructions.iter().enumerate().collect::<Vec<_>>());

    vm::Vm::new(instructions, static_memory).run();
}
