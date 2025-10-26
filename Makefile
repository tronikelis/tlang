.PHONY: run
run: build std/main.so
	./a.out

.PHONY: vm
vm:
	cd tlang_shared_vm && cargo build

.PHONY: cgen
cgen:
	cargo run

.PHONY: build
build: vm cgen
	clang build/* src/run.c tlang_shared_vm/target/debug/libtlang_shared_vm.a -o a.out

std/main.so: std/main.c
	clang -shared -o std/main.so std/main.c

.PHONY: test_tlang_lib
test_tlang_lib:
	cd tlang && cargo test

.PHONY: test
test: test_tlang_lib

.PHONY: clean
clean:
	rm -rf std/*.so
	rm -rf build/*

