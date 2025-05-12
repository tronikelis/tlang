use std::collections::HashMap;

use super::{ast, lexer, vm};

struct EnvironmentVariable {
    offset: usize,
    size: usize,
    _type: lexer::Type,
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

    fn add<T>(&mut self, identifier: String, _type: lexer::Type) {
        let size = size_of::<T>();
        self.sp += size;
        self.variables.insert(
            identifier,
            EnvironmentVariable {
                offset: self.sp,
                size,
                _type,
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
    expected_type: lexer::Type,
) -> (Vec<vm::Instruction>, usize) {
    let mut instructions = Vec::new();

    let (mut l_instructions, l_size) =
        compile_expression(environment, &addition.left, expected_type.clone());
    instructions.append(&mut l_instructions);

    let (mut r_instructions, r_size) =
        compile_expression(environment, &addition.right, expected_type);
    instructions.append(&mut r_instructions);

    instructions.push(vm::Instruction::AddI(0, r_size));
    instructions.push(vm::Instruction::Reset(r_size));

    (instructions, l_size)
}

fn compile_literal(
    literal: &lexer::Literal,
    expected_type: lexer::Type,
) -> (vm::Instruction, usize) {
    match literal {
        lexer::Literal::Int(int) => {
            if expected_type != lexer::Type::Int {
                panic!("expected type dont match");
            }

            (vm::Instruction::PushI(*int), size_of::<isize>())
        }
    }
}

fn compile_identifier(
    environment: &Environment,
    identifier: &String,
    expected_type: lexer::Type,
) -> (Vec<vm::Instruction>, usize) {
    let variable = environment.variables.get(identifier).unwrap();
    if variable._type != expected_type {
        panic!("variable does not match expected type");
    }

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
    expected_type: lexer::Type,
) -> (Vec<vm::Instruction>, usize) {
    match expression {
        ast::Expression::Literal(v) => {
            let (instruction, size) = compile_literal(v, expected_type);
            (Vec::from([instruction]), size)
        }
        ast::Expression::Addition(v) => {
            let (instructions, size) = compile_addition(environment, v, expected_type);
            (instructions, size)
        }
        ast::Expression::Identifier(v) => {
            let (instructions, size) = compile_identifier(environment, v, expected_type);
            (instructions, size)
        }
        ast::Expression::FunctionCall(v) => {
            todo!();
        }
    }
}

// pushes variable into the stack, and adds it into environment
fn compile_local_variable_declaration(
    environment: &mut Environment,
    variable: &ast::VariableDeclaration,
) -> Vec<vm::Instruction> {
    match variable._type {
        lexer::Type::Int => {
            environment.add::<isize>(variable.identifier.clone(), lexer::Type::Int);
            let (instructions, _) =
                compile_expression(environment, &variable.expression, variable._type.clone());
            instructions
        }
        lexer::Type::Void => {
            panic!("cant declare void variable");
        }
    }
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
