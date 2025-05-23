use anyhow::{anyhow, Result};
use std::collections::HashMap;

use crate::{compiler, vm};

fn link_jumps(
    instructions: &Vec<Vec<compiler::Instruction>>,
    offset: usize,
) -> Result<Vec<compiler::Instruction>> {
    let mut ins = Vec::<compiler::Instruction>::new();
    let mut index_to_len = HashMap::<usize, usize>::new();

    for (i, v) in instructions.iter().enumerate() {
        index_to_len.insert(i, ins.len());

        for v in v {
            ins.push(v.clone());
        }
    }

    for v in &mut ins {
        match v {
            compiler::Instruction::Jump(i) => {
                *v = compiler::Instruction::Jump(
                    *index_to_len.get(i).ok_or(anyhow!("index_to_len None"))? + offset,
                );
            }
            compiler::Instruction::JumpIfTrue(i) => {
                *v = compiler::Instruction::JumpIfTrue(
                    *index_to_len.get(i).ok_or(anyhow!("index_to_len None"))? + offset,
                );
            }
            _ => {}
        }
    }

    Ok(ins)
}

pub fn link(
    functions: &HashMap<String, Vec<Vec<compiler::Instruction>>>,
) -> Result<Vec<vm::Instruction>> {
    let main = functions
        .get("main")
        .ok_or(anyhow!("cant link without main"))?;

    let mut fake_instructions = link_jumps(main, 0)?;
    let mut fn_to_offset = HashMap::<&str, usize>::new();

    for (identifier, fn_instructions) in functions.iter().filter(|v| *v.0 != "main") {
        let offset = fake_instructions.len();
        fn_to_offset.insert(identifier, offset);

        for v in link_jumps(fn_instructions, offset)? {
            fake_instructions.push(v);
        }
    }

    let real_instructions = fake_instructions
        .iter()
        .map(|v| {
            Ok(match v {
                compiler::Instruction::Real(v) => v.clone(),
                compiler::Instruction::JumpAndLink(identifier) => {
                    let index = fn_to_offset
                        .get(identifier as &str)
                        .ok_or(anyhow!("unknown identifier"))?;
                    vm::Instruction::JumpAndLink(*index)
                }
                compiler::Instruction::Jump(v) => vm::Instruction::Jump(*v),
                compiler::Instruction::JumpIfTrue(v) => vm::Instruction::JumpIfTrue(*v),
            })
        })
        .collect::<Result<Vec<vm::Instruction>>>()?;

    Ok(real_instructions)
}
