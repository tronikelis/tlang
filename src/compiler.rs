use std::collections::HashMap;

use super::{ast, lexer, vm};

struct EnvironmentVariable {
    offset: usize,
    size: usize,
}

struct Environment {
    variables: HashMap<String, EnvironmentVariable>,
    sp: usize,
}

impl Environment {
    fn new() -> Self {
        Self {
            variables: HashMap::new(),
            sp: 0,
        }
    }

    fn add<T>(&mut self, identifier: String) {
        let size = size_of::<T>();
        self.sp += size;
        self.variables.insert(
            identifier,
            EnvironmentVariable {
                offset: self.sp,
                size,
            },
        );
    }

    fn reset_offset(self) -> usize {
        self.sp
    }
}

fn compile_addition(
    environment: &mut Environment,
    addition: &ast::Addition,
) -> (Vec<vm::Instruction>, usize) {
    let mut instructions = Vec::new();

    let (mut l_instructions, l_size) = compile_expression(environment, &addition.left);
    instructions.append(&mut l_instructions);

    let (mut r_instructions, _) = compile_expression(environment, &addition.right);
    instructions.append(&mut r_instructions);

    instructions.push(vm::Instruction::AddI(0, l_size));
    instructions.push(vm::Instruction::Reset(l_size));

    (instructions, l_size)
}

fn compile_literal(literal: &lexer::Literal) -> (vm::Instruction, usize) {
    match literal {
        lexer::Literal::Int(int) => (vm::Instruction::PushI(*int), size_of::<isize>()),
    }
}

fn compile_identifier(
    environment: &Environment,
    identifier: &String,
) -> (Vec<vm::Instruction>, usize) {
    let variable = environment.variables.get(identifier).unwrap();

    let mut instructions = Vec::new();

    instructions.push(vm::Instruction::Increment(variable.size));
    instructions.push(vm::Instruction::Copy(0, variable.offset, variable.size));

    (instructions, variable.size)
}

// calling convention
// fn foo(a int, b int) int {}
//
//
// let a int = 20
// let b int = 30
//
// let c int = foo(a, b)
//
// before calling "foo"
// push a
// push b
// we have enough space for 1 return value, so stop pushing
// jump and link foo
// foo wrote return into arguments
// reset 8 <- we are now at "a" -> return argument
fn compile_function(function: &ast::Function) -> Vec<vm::Instruction> {
    todo!();
}

// pushes the result onto the stack,
// returns size of pushed item
fn compile_expression(
    environment: &mut Environment,
    expression: &ast::Expression,
) -> (Vec<vm::Instruction>, usize) {
    let instructions: Vec<vm::Instruction>;
    let size: usize;

    match expression {
        ast::Expression::Literal(v) => {
            let (instruction, s) = compile_literal(v);
            instructions = Vec::from([instruction]);
            size = s;
        }
        ast::Expression::Addition(v) => {
            let (addition_instructions, s) = compile_addition(environment, v);
            instructions = addition_instructions;
            size = s;
        }
        ast::Expression::Identifier(v) => {
            let (identifier_instructions, s) = compile_identifier(environment, v);
            instructions = identifier_instructions;
            size = s;
        }
        ast::Expression::FunctionCall(v) => {
            todo!();
        }
    }

    (instructions, size)
}

fn compile_local_variable_declaration(
    environment: &mut Environment,
    variable: &ast::VariableDeclaration,
) -> Vec<vm::Instruction> {
    todo!();
}

pub fn compile(ast: ast::Ast) -> HashMap<String, Vec<vm::Instruction>> {
    let mut instruction_map = HashMap::new();

    for node in &ast.nodes {
        match node {
            ast::Node::Function(v) => {
                instruction_map.insert(v.identifier.clone(), compile_function(v));
            }
            _ => {}
        }
    }

    instruction_map
}
