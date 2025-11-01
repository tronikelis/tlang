use tlang::vm;

#[no_mangle]
pub unsafe extern "C" fn tlang_run_vm(
    instructions: *const libc::c_char,
    static_memory: *const libc::c_char,
    static_memory_len: libc::c_uint,
) -> libc::c_int {
    vm::Vm::new(
        vm::Instructions::from_binary(instructions.cast()).0,
        vm::StaticMemory::from_binary(static_memory.cast(), static_memory_len as usize),
    )
    .run();

    0
}
