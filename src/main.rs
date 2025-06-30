use std::collections::HashMap;

mod vm;

mod ast;
mod compiler;
mod lexer;
mod linker;

fn main() {
    let code = String::from(
        "
                type bool bool
                type int int
                type ptr ptr
                type string string
                type uint uint
                type uint8 uint8
                type void void
                type Type Type

                type UI UserInner

                type UserInner struct{
                    foo int
                    bar int
                }

                type Smol struct {
                    one uint8
                    two uint8
                    nice int
                }

                type User struct {
                    inner1 UserInner
                    inner2 UserInner
                }
                    
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
                    let x int[] = int[](x)
                    for let i int = 0; i < len(x); i++ {
                        final = final + x[i]
                    }
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

                fn new(x Type, args Type...) Type {}

                fn main() void {
                    let one_two_three string = itoa(69420)

                    let s Smol = Smol {
                        one: uint8(2),
                        two: uint8(3),
                        nice: 4,
                    }
                    __debug__

                    let u User = User{
                        inner1: UI{
                            foo: 20,
                            bar: 20,
                        },
                        inner2: UI{
                            foo: 25,
                            bar: 25,
                        },
                    }

                    if false && true {
                        syscall_write(1, uint8[](\"NICE GUYS\\n\"))
                    }

                    let buf uint8[] = new(uint8[], uint8(0), 2048)

                    let nums int[] = {1, 2, 3, 4, 5, 6}
                    let foo int = add(nums...)

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

    let mut functions = HashMap::<String, Vec<Vec<compiler::Instruction>>>::new();
    let mut static_memory = vm::StaticMemory::new();

    for function in &ast.functions {
        let compiled =
            compiler::FunctionCompiler::new(function, &mut static_memory, &ast.type_declarations)
                .compile()
                .unwrap();
        println!("{:#?}", compiled);
        functions.insert(function.identifier.clone(), compiled);
    }

    let instructions = linker::link(&functions).unwrap();

    println!("{:#?}", instructions.iter().enumerate().collect::<Vec<_>>());

    vm::Vm::new(instructions, static_memory).run();
}
