use tlang::{ast, compiler, ir, lexer, linker, vm};

mod cgen;

fn main() {
    let code = String::from(
        "
                fn libc_write(fd int, slice uint8[]) int {}
                fn len(slice Type) int {}
                fn append(slice Type, value Type) void {}
                fn new(typ Type, args Type...) Type {}

                fn dll_open(path string) ptr {}
                fn ffi_create(dll ptr, function string, return_param string, args string...) ptr {}
                fn ffi_call(ffi ptr, args ptr...) ptr {}

                fn std_read(fd int32, data uint8[]) int32 {
                    let dll ptr = dll_open(\"std/main.so\")
                    let ffi ptr = ffi_create(dll, \"std_read\", \"i32\", \"i32\", \"pointer\", \"i32\")
                    return *(ffi_call(ffi, &fd, &(data as ptr), &(len(data) as int32)) as *int32)
                }

                fn std_write(fd int32, data uint8[]) int32 {
                    let dll ptr = dll_open(\"std/main.so\")
                    let ffi ptr = ffi_create(dll, \"std_write\", \"i32\", \"i32\", \"pointer\", \"i32\")
                    return *(ffi_call(ffi, &fd, &(data as ptr), &(len(data) as int32)) as *int32)
                }

                fn std_getenv(name string) string {
                    let dll ptr = dll_open(\"std/main.so\")
                    let ffi ptr = ffi_create(dll, \"std_getenv\" \"c_string\", \"c_string\")
                    return *(ffi_call(ffi, &name) as *string)
                }

                fn std_tcp_listen(addr string, port int32) int32 {
                    let dll ptr = dll_open(\"std/main.so\")
                    let ffi ptr = ffi_create(dll, \"std_tcp_listen\" \"i32\", \"c_string\", \"i32\")
                    return *(ffi_call(ffi, &addr, &port) as *int32)
                }

                type TcpConn struct {
                    addr uint32
                    fd int32
                    port uint16
                }

                fn std_tcp_accept(fd int32) TcpConn {
                    let dll ptr = dll_open(\"std/main.so\")
                    let ffi ptr = ffi_create(dll, \"std_tcp_accept\" \"{u32,i32,u16}\", \"i32\")
                    return *(ffi_call(ffi, &fd) as *TcpConn)
                }

                fn main() void {
                    std_write(1, \"sup from binary\\n\" as uint8[])
                    let socket int32 = std_tcp_listen(\"127.0.0.1\", 8080)
                    let conn TcpConn = std_tcp_accept(socket)
                    for {
                        let buf uint8[] = new(uint8[], 0 as uint8, 10)
                        std_read(conn.fd, buf)
                        std_write(conn.fd, buf)
                        std_write(1, buf)
                    }

                    return
                }
            ",
    );

    let tokens = lexer::Lexer::new(&code).run().unwrap();
    println!("{tokens:#?}");
    let ast = ast::Ast::new(&tokens).unwrap();
    println!("{:#?}", ast);

    let ir = ir::Ir::new(&ast).unwrap();
    println!("{ir:#?}");

    let compiled = compiler::compile(ir).unwrap();
    let linked = linker::link(compiled.functions).unwrap();
    let linked_with_index = linked.iter().enumerate().collect::<Vec<_>>();
    println!("{linked_with_index:#?}");

    let static_memory_len = compiled.static_memory.data.len();

    cgen::gen_instructions_c_file(
        vm::Instructions::new(linked),
        "./build/instructions.c",
        "instructions",
    )
    .unwrap();

    cgen::gen_static_memory_c_file(
        compiled.static_memory,
        "./build/static_memory.c",
        "static_memory",
    )
    .unwrap();

    cgen::gen_static_memory_len_c_file(
        static_memory_len,
        "./build/static_memory_len.c",
        "static_memory_len",
    )
    .unwrap();
}
