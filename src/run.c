int tlang_run_vm(const char* instructions, const char* static_memory,
                 const int static_memory_len);

extern const char instructions[];
extern const char static_memory[];
extern const unsigned int static_memory_len;

int main(int argc, char** argv) {
    return tlang_run_vm(instructions, static_memory, static_memory_len);
}
