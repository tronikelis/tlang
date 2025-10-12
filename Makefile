std/main.so: std/main.c
	clang -shared -o std/main.so std/main.c

.PHONY: clean
clean:
	rm -rf std/*.so
