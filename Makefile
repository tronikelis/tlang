.PHONY: test
test: test_tlang_lib

.PHONY: vm
vm:
	cd tlang_static_vm && cargo build

std/main.so: std/main.c
	clang -shared -o std/main.so std/main.c

.PHONY: test_tlang_lib
test_tlang_lib:
	export RUSTFLAGS=-Awarnings && cd tlang && cargo test

