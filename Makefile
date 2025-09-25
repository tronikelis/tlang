std/main.so: std/main.c
	clang -shared -o std/main.so std/main.c

clean:
	rm -rf std/*.so
