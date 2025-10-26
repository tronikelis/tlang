use anyhow::Result;
use std::{
    fs,
    io::{self, Write},
};

use tlang::vm;

const TYPE_QUALIFIER: &str = "const";

pub fn gen_instructions_c_file(
    instructions: vm::Instructions,
    path: &str,
    label: &str,
) -> Result<()> {
    let mut filebuf = io::BufWriter::new(fs::File::create(path)?);

    filebuf.write_all(format!(r"{TYPE_QUALIFIER} char {label}[] = {{").as_bytes())?;

    for v in instructions.to_binary() {
        filebuf.write_all(v.to_string().as_bytes())?;
        filebuf.write_all(",\n".as_bytes())?;
    }
    filebuf.write_all("};".as_bytes())?;

    filebuf.flush()?;

    Ok(())
}

pub fn gen_static_memory_c_file(
    static_memory: vm::StaticMemory,
    path: &str,
    label: &str,
) -> Result<()> {
    let mut filebuf = io::BufWriter::new(fs::File::create(path)?);

    filebuf.write_all(format!(r"{TYPE_QUALIFIER} char {label}[] = {{").as_bytes())?;

    for v in static_memory.data {
        filebuf.write_all(v.to_string().as_bytes())?;
        filebuf.write_all(",\n".as_bytes())?;
    }
    filebuf.write_all("};".as_bytes())?;

    filebuf.flush()?;

    Ok(())
}

pub fn gen_static_memory_len_c_file(len: usize, path: &str, label: &str) -> Result<()> {
    let mut filebuf = io::BufWriter::new(fs::File::create(path)?);

    filebuf.write_all(format!(r"{TYPE_QUALIFIER} unsigned int {label} = {len};").as_bytes())?;
    filebuf.flush()?;

    Ok(())
}
