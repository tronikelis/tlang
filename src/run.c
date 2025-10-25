int tlang_run_vm(const void* instructions, const void* static_memory,
                 const int static_memory_len);

const char instructions[];
const char static_memory[];
const unsigned int static_memory_len;

int main(int argc, char** argv) {
    return tlang_run_vm(instructions, static_memory, static_memory_len);
}
