## Syntax

```
fn add(a int, b int) int {
    return a + b
}

fn main() {
    let a int = 0
    let b int = 1

    if a > b && a < b {
        a = 20
    } elseif a < b {
        b = 20
    } else {
        c = 200
    }

    let c int = add(a, b) + 3
}
```

## FizzBuzz

```
// compiler builtins, libc_write will be replaced with ffi
fn libc_write(fd int, slice uint8[]) int {}
fn len(slice Type) int {}
fn append(slice Type, value Type) void {}

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

fn main() void {
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
```
