use anyhow::{anyhow, Result};
use std::collections::HashMap;

use crate::{compiler, vm};

pub fn link(
    static_instructions: Vec<compiler::ScopedInstruction>,
    functions: HashMap<String, Vec<compiler::ScopedInstruction>>,
) -> Result<Vec<vm::Instruction>> {
    let mut functions: Vec<(String, Vec<compiler::ScopedInstruction>)> =
        functions.into_iter().collect();
    functions.sort_by(|a, b| a.0.cmp(&b.0));

    let main_index = functions
        .iter()
        .position(|v| v.0 == "main")
        .ok_or(anyhow!("link: cant link without main"))?;

    {
        let temp = functions[0].clone();
        functions[0] = functions[main_index].clone();
        functions[main_index] = temp;
    }

    let mut folded = Vec::<compiler::ScopedInstruction>::new();
    let mut function_to_index = HashMap::<&str, usize>::new();

    for (k, v) in &functions {
        let index = folded.len();
        function_to_index.insert(&k, index);

        folded.append(
            &mut v
                .clone()
                .into_iter()
                .map(|mut v| {
                    v.add_jump_offset(index);
                    v
                })
                .collect(),
        );
    }

    let static_instructions_len = static_instructions.len();
    let static_instructions = static_instructions
        .into_iter()
        .map(|v| {
            Ok(match v {
                compiler::ScopedInstruction::Real(v) => v,
                compiler::ScopedInstruction::Jump(v) => vm::Instruction::Jump(v),
                compiler::ScopedInstruction::JumpIfTrue(v) => vm::Instruction::JumpIfTrue(v),
                compiler::ScopedInstruction::JumpIfFalse(v) => vm::Instruction::JumpIfFalse(v),
                compiler::ScopedInstruction::JumpAndLink(to) => {
                    let index = function_to_index
                        .get(to.as_str())
                        .ok_or(anyhow!("link: jump to non existant function {to}"))?;
                    vm::Instruction::JumpAndLink(*index + static_instructions_len)
                }
                compiler::ScopedInstruction::PushClosure(vars_count, offset) => {
                    vm::Instruction::PushClosure(vars_count, offset)
                }
            })
        })
        .collect::<Result<Vec<_>, anyhow::Error>>()?;

    let mut instructions = folded
        .into_iter()
        .map(|v| {
            Ok(match v {
                compiler::ScopedInstruction::Real(v) => v,
                compiler::ScopedInstruction::Jump(v) => vm::Instruction::Jump(v),
                compiler::ScopedInstruction::JumpIfTrue(v) => vm::Instruction::JumpIfTrue(v),
                compiler::ScopedInstruction::JumpIfFalse(v) => vm::Instruction::JumpIfFalse(v),
                compiler::ScopedInstruction::JumpAndLink(to) => {
                    let index = function_to_index
                        .get(to.as_str())
                        .ok_or(anyhow!("link: jump to non existant function {to}"))?;
                    vm::Instruction::JumpAndLink(*index)
                }
                compiler::ScopedInstruction::PushClosure(vars_count, offset) => {
                    vm::Instruction::PushClosure(vars_count, offset)
                }
            })
        })
        .collect::<Result<Vec<_>, anyhow::Error>>()?;

    instructions
        .iter_mut()
        .for_each(|v| v.add_jump_offset(static_instructions.len()));

    Ok([static_instructions, instructions].concat())
}
