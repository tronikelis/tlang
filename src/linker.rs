use anyhow::{anyhow, Result};
use std::collections::HashMap;

use crate::{compiler, vm};

pub fn link(
    functions: &HashMap<String, Vec<compiler::Instruction>>,
) -> Result<Vec<vm::Instruction>> {
    let main = functions
        .get("main")
        .ok_or(anyhow!("cant link without main"))?;

    let mut fake_instructions = Vec::<&compiler::Instruction>::new();
    for v in main {
        fake_instructions.push(v);
    }

    let mut fn_index_map = HashMap::<&str, usize>::new();

    for (identifier, fn_instructions) in functions.iter().filter(|v| *v.0 != "main") {
        fn_index_map.insert(identifier, fake_instructions.len());

        for v in fn_instructions {
            fake_instructions.push(v);
        }
    }

    let real_instructions = fake_instructions
        .iter()
        .map(|v| {
            Ok(match v {
                compiler::Instruction::Real(v) => v.clone(),
                compiler::Instruction::JumpAndLink(identifier) => {
                    let index = fn_index_map
                        .get(identifier as &str)
                        .ok_or(anyhow!("unknown identifier"))?;
                    vm::Instruction::JumpAndLink(*index)
                }
            })
        })
        .collect::<Result<Vec<vm::Instruction>>>()?;

    Ok(real_instructions)
}
