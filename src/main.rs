use std::{cell::RefCell, collections::HashMap, rc::Rc};

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
                    three uint8[]
                    nice int
                }

                type User struct {
                    inner1 UserInner
                    inner2 UserInner
                }
                    
                fn len(slice Type) int {}
                fn append(slice Type, value Type) void {}

                fn libc_write(fd int, slice uint8[]) int {}

                fn add(x int...) int {
                    let final int = 0
                    let x int[] = int[](x)
                    for let i int = 0; i < len(x); i++ {
                        final = final + x[i]
                    }
                    return final
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
                    let str uint8[] = uint8[]{}
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

                type LinkedList struct {
                    next *LinkedList
                    value int
                }

                // type Incrementer struct {
                //     get fn() int
                //     set fn(value int) void
                // }
                //
                // fn create_incrementer() Incrementer {
                //     let value int = 0
                //
                //     return Incrementer {
                //         get: fn() int {
                //             return value
                //         },
                //         set: fn(v int) void {
                //             value = v
                //         }
                //     }
                // }

                fn create_lists() void {
                    let ll *LinkedList = &LinkedList{
                        next: &LinkedList{
                            next: __nil__,
                            value: 2,
                        },
                        value: 1,
                    }

                    ll.next = &LinkedList{
                        next: __nil__,
                        value: 2,
                    }

                    ll.next.next = &LinkedList{
                        next: __nil__,
                        value: 3,
                    }

                    ll.next.next.next = &LinkedList{
                        next: __nil__,
                        value: 4,
                    }

                    print_int(ll.value)
                    print_int(ll.next.value)
                    print_int(ll.next.next.value)
                    print_int(ll.next.next.next.value)
                }

                fn print_int(x int) void {
                    libc_write(1, uint8[](itoa(x)))
                }

                fn main() void {
                    let foo1 int = 0

                    let nice fn() void = fn() void {
                        let ok fn() void = fn() void {
                            foo1 = 20
                        }
                        ok()
                    }

                    nice()
                    create_lists()

                    let one_two_three string = itoa(69420)
                    let s Smol = Smol {
                        one: uint8(65),
                        two: uint8(66),
                        nice: 4,
                        three: uint8[]{},
                    }
                    s.one = uint8(200)

                    let one *uint8 = &s.one
                    *one = uint8(254)

                    libc_write(1, uint8[](itoa(int(s.one))))

                    let nice *Smol = &s

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

                    u.inner1 = u.inner2

                    __debug__


                    if false && true {
                        libc_write(1, uint8[](\"NICE GUYS\\n\"))
                    }

                    let buf uint8[] = new(uint8[], uint8(0), 2048)

                    let nums int[] = int[]{1, 2, 3, 4, 5, 6}
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

                        libc_write(1, uint8[](str + \"\\n\"))
                    }
                }
            ",
    );

    let tokens = lexer::Lexer::new(&code).run().unwrap();
    println!("{tokens:#?}");
    let ast = ast::Ast::new(&tokens).unwrap();
    println!("{:#?}", ast);

    let mut functions = HashMap::<String, compiler::CompiledInstructions>::new();
    let static_memory = Rc::new(RefCell::new(vm::StaticMemory::new()));

    let type_resolver = Rc::new(compiler::TypeResolver::new(ast.type_declarations));
    let function_declarations = Rc::new(ast.function_declarations);

    for (identifier, declaration) in function_declarations.iter() {
        let compiled = compiler::FunctionCompiler::new(
            compiler::Function::from_declaration(&type_resolver, declaration).unwrap(),
            static_memory.clone(),
            type_resolver.clone(),
            function_declarations.clone(),
        )
        .compile()
        .unwrap();
        println!("{:#?}", compiled);
        functions.insert(identifier.clone(), compiled.instructions);
    }

    let instructions = linker::link(&functions).unwrap();

    println!("{:#?}", instructions.iter().enumerate().collect::<Vec<_>>());

    vm::Vm::new(instructions, static_memory.borrow().clone()).run();
}
