use anyhow::{anyhow, Error, Result};

use crate::{ast, lexer, vm};

#[derive(Debug, Clone)]
pub enum Instruction {
    Real(vm::Instruction),
    JumpAndLink(String),
    Jump((usize, usize)),
    JumpIfTrue((usize, usize)),
}

#[derive(Debug, Clone)]
enum VarStackItem {
    Increment(usize),
    Reset(usize),
    Var(VarStackItemVar),
}

#[derive(Debug, Clone)]
struct VarStackItemVar {
    identifier: String,
    _type: ast::Type,
}

struct VarStack {
    stack: Vec<Vec<VarStackItem>>,
    arg_size: Option<usize>,
}

impl VarStack {
    fn new() -> Self {
        let mut stack = Vec::new();
        stack.push(Vec::new());
        Self {
            stack,
            arg_size: None,
        }
    }

    fn set_arg_size(&mut self) {
        self.arg_size = Some(self.total_size());
    }

    fn push_frame(&mut self) {
        self.stack.push(Vec::new());
    }

    fn pop_frame(&mut self) -> usize {
        let last = self.stack.last().unwrap();
        let size = Self::size_for(last.iter());
        self.stack.pop();
        size
    }

    fn push(&mut self, item: VarStackItem) {
        self.stack.last_mut().unwrap().push(item);
    }

    fn size_for<'a>(items: impl Iterator<Item = &'a VarStackItem>) -> usize {
        items.fold(0, |acc, curr| match curr {
            VarStackItem::Var(var) => acc + var._type.size,
            VarStackItem::Increment(size) => acc + size,
            VarStackItem::Reset(size) => acc - size,
        })
    }

    fn total_size(&self) -> usize {
        Self::size_for(self.stack.iter().flatten())
    }

    fn get_var(&self, identifier: &str) -> Option<(VarStackItemVar, usize)> {
        let mut offset: isize = 0;

        for item in self.stack.iter().flatten().rev() {
            match item {
                VarStackItem::Var(var) => {
                    if var.identifier == identifier {
                        return Some((var.clone(), offset as usize));
                    }
                    offset += var._type.size as isize;
                }
                VarStackItem::Increment(size) => offset += *size as isize,
                VarStackItem::Reset(size) => offset -= *size as isize,
            };
        }

        None
    }
}

struct LabelInstructions {
    instructions: Vec<Vec<Instruction>>,
    index: Vec<usize>,
}

impl LabelInstructions {
    fn new() -> Self {
        let mut instructions = Vec::new();
        instructions.push(Vec::new());
        Self {
            index: Vec::from([0]),
            instructions,
        }
    }

    fn push(&mut self, instruction: Instruction) {
        self.instructions[*self.index.last().unwrap()].push(instruction);
    }

    fn jump(&mut self) {
        let index = self.instructions.len();
        self.push(Instruction::Jump((index, 0)));
        self.instructions.push(Vec::new());
        self.index.push(index);
    }

    fn jump_if_true(&mut self) {
        let index = self.instructions.len();
        self.push(Instruction::JumpIfTrue((index, 0)));
        self.instructions.push(Vec::new());
        self.index.push(index);
    }

    fn back_if_true(&mut self, offset: usize) {
        let target = self.index[self.index.len() - 1 - offset];
        let target_last = self.instructions[target].len();
        self.push(Instruction::JumpIfTrue((target, target_last)));
    }

    fn again(&mut self) {
        self.push(Instruction::Jump((*self.index.last().unwrap(), 0)));
    }

    fn back(&mut self, offset: usize) {
        let target = self.index[self.index.len() - 1 - offset];
        let target_last = self.instructions[target].len();
        self.push(Instruction::Jump((target, target_last)));
    }

    fn pop_index(&mut self) {
        self.index.pop();
    }
}

pub struct FunctionCompiler<'a> {
    var_stack: VarStack,
    instructions: LabelInstructions,
    function: &'a ast::Function,
}

impl<'a> FunctionCompiler<'a> {
    pub fn new(function: &'a ast::Function) -> Self {
        Self {
            function,
            var_stack: VarStack::new(),
            instructions: LabelInstructions::new(),
        }
    }

    fn compile_function_call(&mut self, call: &ast::FunctionCall) -> Result<ast::Type> {
        let argument_size = call.arguments.iter().enumerate().try_fold(0, |acc, curr| {
            let expected_type = &call
                .function
                .arguments
                .get(curr.0)
                .ok_or(anyhow!("compile_function_call: expected_argument"))?
                ._type;

            let _type = self.compile_expression(curr.1)?;
            if *expected_type != _type {
                Err(anyhow!("compile_function_call: unexpected type"))
            } else {
                Ok::<usize, Error>(acc + _type.size)
            }
        })?;

        let return_size = call.function.return_type.size;

        let reset_size: usize;
        if argument_size < return_size {
            // the whole argument section will be used for return value
            // do not need to reset
            reset_size = 0;
            self.instructions
                .push(Instruction::Real(vm::Instruction::Increment(
                    return_size - argument_size,
                )));
            self.var_stack
                .push(VarStackItem::Increment(return_size - argument_size));
        } else {
            // reset the argument section to the return size
            reset_size = argument_size - return_size;
        }

        self.instructions
            .push(Instruction::JumpAndLink(call.function.identifier.clone()));

        if reset_size != 0 {
            self.instructions
                .push(Instruction::Real(vm::Instruction::Reset(reset_size)));
            self.var_stack.push(VarStackItem::Reset(reset_size));
        }

        Ok(call.function.return_type.clone())
    }

    fn compile_literal(&mut self, literal: &ast::Literal) -> Result<ast::Type> {
        match literal.literal {
            lexer::Literal::Int(int) => {
                self.instructions
                    .push(Instruction::Real(vm::Instruction::PushI(int as isize)));
            }
            lexer::Literal::Bool(bool) => {
                self.instructions
                    .push(Instruction::Real(vm::Instruction::PushI({
                        if bool {
                            1
                        } else {
                            0
                        }
                    })));
            }
        }

        self.var_stack
            .push(VarStackItem::Increment(literal._type.size));

        Ok(literal._type.clone())
    }

    fn compile_identifier(&mut self, identifier: &str) -> Result<ast::Type> {
        let (variable, offset) = self
            .var_stack
            .get_var(identifier)
            .ok_or(anyhow!("compile_identifier: unknown identifier"))?;

        self.instructions
            .push(Instruction::Real(vm::Instruction::Increment(
                variable._type.size,
            )));
        self.instructions
            .push(Instruction::Real(vm::Instruction::Copy(
                0,
                offset + variable._type.size, // we incremented by this above
                variable._type.size,
            )));

        self.var_stack
            .push(VarStackItem::Increment(variable._type.size));

        Ok(variable._type)
    }

    fn compile_arithmetic(&mut self, arithmetic: &ast::Arithmetic) -> Result<ast::Type> {
        let a = self.compile_expression(&arithmetic.left)?;
        let b = self.compile_expression(&arithmetic.right)?;

        if a != b {
            return Err(anyhow!("can't add different types"));
        }
        if a == ast::VOID {
            return Err(anyhow!("can't add void type"));
        }
        if a != ast::INT {
            return Err(anyhow!("can only arithmetic on int"));
        }

        match arithmetic._type {
            ast::ArithmeticType::Minus => {
                self.instructions
                    .push(Instruction::Real(vm::Instruction::MinusInt));
                self.instructions
                    .push(Instruction::Real(vm::Instruction::AddI));
            }
            ast::ArithmeticType::Plus => {
                self.instructions
                    .push(Instruction::Real(vm::Instruction::AddI));
            }
            ast::ArithmeticType::Multiply => {
                self.instructions
                    .push(Instruction::Real(vm::Instruction::MultiplyI));
            }
            ast::ArithmeticType::Divide => {
                self.instructions
                    .push(Instruction::Real(vm::Instruction::DivideI));
            }
        }

        self.var_stack.push(VarStackItem::Reset(a.size));

        Ok(a)
    }

    fn compile_compare(&mut self, compare: &ast::Compare) -> Result<ast::Type> {
        self.instructions.jump();

        let a: ast::Type;
        let b: ast::Type;

        match compare.compare_type {
            // last item on the stack is smaller
            ast::CompareType::Gt => {
                b = self.compile_expression(&compare.left)?;
                a = self.compile_expression(&compare.right)?;
            }
            // last item on the stack is bigger
            ast::CompareType::Lt => {
                b = self.compile_expression(&compare.right)?;
                a = self.compile_expression(&compare.left)?;
            }
            // dont matter
            ast::CompareType::Equals => {
                a = self.compile_expression(&compare.right)?;
                b = self.compile_expression(&compare.left)?;
            }
        };

        if a._type != b._type {
            return Err(anyhow!("can't compare different types"));
        }

        match a {
            ast::BOOL | ast::INT => {}
            _type => return Err(anyhow!("can only compare int/bool")),
        }

        match compare.compare_type {
            ast::CompareType::Gt | ast::CompareType::Lt => {
                // a = -a
                self.instructions
                    .push(Instruction::Real(vm::Instruction::MinusInt));

                // a + b
                self.instructions
                    .push(Instruction::Real(vm::Instruction::AddI));

                // >0:1 <0:0
                self.instructions
                    .push(Instruction::Real(vm::Instruction::ToBool));
            }
            ast::CompareType::Equals => self
                .instructions
                .push(Instruction::Real(vm::Instruction::CompareInt)),
        }

        self.var_stack.push(VarStackItem::Reset(ast::BOOL.size));

        if let Some(andor) = &compare.andor {
            match andor {
                ast::AndOr::And(exp) => {
                    // jump to after if false
                    self.instructions
                        .push(Instruction::Real(vm::Instruction::NegateBool));
                    self.instructions.back_if_true(1); // actual code will jump to after if
                    self.instructions
                        .push(Instruction::Real(vm::Instruction::NegateBool));

                    let exp = self.compile_expression(exp)?;
                    if exp != ast::BOOL {
                        return Err(anyhow!("incorrect type"));
                    }

                    self.instructions
                        .push(Instruction::Real(vm::Instruction::And));
                    self.var_stack.push(VarStackItem::Reset(ast::BOOL.size));
                }
                ast::AndOr::Or(exp) => {
                    // jump to after if true
                    self.instructions.back_if_true(1);

                    let exp = self.compile_expression(exp)?;
                    if exp != ast::BOOL {
                        return Err(anyhow!("incorrect type"));
                    }

                    self.instructions
                        .push(Instruction::Real(vm::Instruction::Or));
                    self.var_stack.push(VarStackItem::Reset(ast::BOOL.size));
                }
            }
        }

        self.instructions.back(1);
        self.instructions.pop_index();

        Ok(ast::BOOL)
    }

    fn compile_infix(&mut self, infix: &ast::Infix) -> Result<ast::Type> {
        let exp = self.compile_expression(&infix.expression)?;
        match infix._type {
            ast::InfixType::Plus => {}
            ast::InfixType::Minus => self
                .instructions
                .push(Instruction::Real(vm::Instruction::MinusInt)),
        }
        Ok(exp)
    }

    fn compile_expression(&mut self, expression: &ast::Expression) -> Result<ast::Type> {
        let _type = match expression {
            ast::Expression::Literal(v) => self.compile_literal(v),
            ast::Expression::Arithmetic(v) => self.compile_arithmetic(v),
            ast::Expression::Identifier(v) => self.compile_identifier(v),
            ast::Expression::FunctionCall(v) => self.compile_function_call(v),
            ast::Expression::Compare(v) => self.compile_compare(v),
            ast::Expression::Infix(v) => self.compile_infix(v),
            ast::Expression::List(v) => todo!(),
        }?;

        Ok(_type)
    }

    fn compile_variable_declaration(&mut self, variable: &ast::VariableDeclaration) -> Result<()> {
        match &variable._type._type {
            ast::TypeType::Scalar(v) => match v {
                lexer::Type::Int => {
                    let exp = self.compile_expression(&variable.expression)?;
                    if exp != ast::INT {
                        return Err(anyhow!("expected int expression"));
                    }

                    self.var_stack
                        .push(VarStackItem::Reset(variable._type.size));
                    self.var_stack.push(VarStackItem::Var(VarStackItemVar {
                        identifier: variable.identifier.clone(),
                        _type: variable._type.clone(),
                    }));
                }
                lexer::Type::Bool => {
                    let exp = self.compile_expression(&variable.expression)?;
                    if exp != ast::BOOL {
                        return Err(anyhow!("expected bool expression"));
                    }

                    self.var_stack
                        .push(VarStackItem::Reset(variable._type.size));
                    self.var_stack.push(VarStackItem::Var(VarStackItemVar {
                        identifier: variable.identifier.clone(),
                        _type: variable._type.clone(),
                    }));
                }
                lexer::Type::Void => return Err(anyhow!("cant declare void variable")),
            },
            ast::TypeType::Slice(v) => todo!(),
        };

        Ok(())
    }

    fn compile_variable_assignment(&mut self, assignment: &ast::VariableAssignment) -> Result<()> {
        // push expression
        // copy into assignment
        // reset expression
        let exp = self.compile_expression(&assignment.expression)?;
        if exp != assignment.variable._type {
            return Err(anyhow!(
                "compile_variable_assignment: assignment does match expression"
            ));
        }

        let (variable, offset) = self
            .var_stack
            .get_var(&assignment.variable.identifier)
            .ok_or(anyhow!("compile_variable_assignment: variable not found"))?;
        if variable._type != assignment.variable._type {
            return Err(anyhow!("compile_variable_assignment: types don't match"));
        }

        self.instructions
            .push(Instruction::Real(vm::Instruction::Copy(
                offset,
                0,
                variable._type.size,
            )));
        self.instructions
            .push(Instruction::Real(vm::Instruction::Reset(
                variable._type.size,
            )));
        self.var_stack
            .push(VarStackItem::Reset(variable._type.size));

        Ok(())
    }

    fn compile_if_block(&mut self, expression: &ast::Expression, body: &[ast::Node]) -> Result<()> {
        self.compile_expression(expression)?;

        self.instructions.jump_if_true();

        self.compile_body(body)?;
        self.instructions.back(2);
        self.instructions.pop_index();

        Ok(())
    }

    fn compile_if(&mut self, _if: &ast::If) -> Result<()> {
        self.var_stack.push_frame();

        self.instructions.jump();

        self.compile_if_block(&_if.expression, &_if.body)?;

        for v in &_if.elseif {
            self.compile_if_block(&v.expression, &v.body)?;
        }

        if let Some(v) = &_if._else {
            self.compile_body(&v.body)?;
        }

        self.instructions.back(1);
        self.instructions.pop_index();

        self.instructions
            .push(Instruction::Real(vm::Instruction::Reset(
                self.var_stack.pop_frame(),
            )));

        Ok(())
    }

    fn compile_for(&mut self, _for: &ast::For) -> Result<()> {
        self.var_stack.push_frame();
        self.compile_node(&_for.initializer)?;

        self.instructions.jump();
        self.var_stack.push_frame();

        let exp = self.compile_expression(&_for.expression)?;
        if exp != ast::BOOL {
            return Err(anyhow!("compile_for: expected expression to return bool"));
        }
        self.instructions
            .push(Instruction::Real(vm::Instruction::NegateBool));
        self.instructions.back_if_true(1);

        self.compile_body(&_for.body)?;
        self.compile_node(&_for.after_each)?;

        self.instructions
            .push(Instruction::Real(vm::Instruction::Reset(
                self.var_stack.pop_frame(),
            )));
        self.instructions.again();
        self.instructions.pop_index();

        self.instructions
            .push(Instruction::Real(vm::Instruction::Reset(
                ast::BOOL.size + self.var_stack.pop_frame(),
            )));

        Ok(())
    }

    fn compile_node(&mut self, node: &ast::Node) -> Result<()> {
        match node {
            ast::Node::VariableDeclaration(var) => self.compile_variable_declaration(var)?,
            ast::Node::Return(exp) => self.compile_return(exp.as_ref())?,
            ast::Node::FunctionCall(fn_call) => {
                let exp = self.compile_function_call(fn_call)?;
                if exp != fn_call.function.return_type.clone() {
                    return Err(anyhow!("incorrect type"));
                }

                // no variable to store result in, reset instantly
                self.instructions
                    .push(Instruction::Real(vm::Instruction::Reset(
                        fn_call.function.return_type.size,
                    )));
                self.var_stack
                    .push(VarStackItem::Reset(fn_call.function.return_type.size));
            }
            ast::Node::VariableAssignment(assignment) => {
                self.compile_variable_assignment(assignment)?;
            }
            ast::Node::If(v) => {
                self.compile_if(v)?;
            }
            ast::Node::Debug => {
                self.instructions
                    .push(Instruction::Real(vm::Instruction::Debug));
            }
            ast::Node::For(v) => self.compile_for(v)?,
        };

        Ok(())
    }

    fn compile_body(&mut self, body: &[ast::Node]) -> Result<()> {
        self.var_stack.push_frame();

        for node in body {
            self.compile_node(node)?;
        }

        self.instructions
            .push(Instruction::Real(vm::Instruction::Reset(
                self.var_stack.pop_frame(),
            )));

        Ok(())
    }

    fn compile_return(&mut self, exp: Option<&ast::Expression>) -> Result<()> {
        if let Some(exp) = exp {
            let exp = self.compile_expression(exp)?;
            if exp != self.function.return_type {
                return Err(anyhow!("incorrect type"));
            }

            self.instructions
                .push(Instruction::Real(vm::Instruction::Copy(
                    // -size because .total_size() gets you total stack size,
                    // while we want to index into first item
                    self.var_stack.total_size() - exp.size,
                    0,
                    exp.size,
                )));
        }

        self.instructions
            .push(Instruction::Real(vm::Instruction::Reset(
                self.var_stack.total_size() - self.var_stack.arg_size.unwrap(),
            )));

        if self.function.identifier == "main" {
            self.instructions
                .push(Instruction::Real(vm::Instruction::Exit));
        } else {
            self.instructions
                .push(Instruction::Real(vm::Instruction::Return));
        }

        Ok(())
    }

    pub fn compile(mut self) -> Result<Vec<Vec<Instruction>>> {
        for arg in &self.function.arguments {
            self.var_stack.push(VarStackItem::Var(VarStackItemVar {
                _type: arg._type.clone(),
                identifier: arg.identifier.clone(),
            }));
        }

        if self.function.identifier != "main" {
            // return address
            self.var_stack
                .push(VarStackItem::Increment(size_of::<usize>()));
        }

        self.var_stack.set_arg_size();

        self.compile_body(
            self.function
                .body
                .as_ref()
                .ok_or(anyhow!("compile: function body empty"))?,
        )?;

        self.compile_return(None)?;

        Ok(self.instructions.instructions)
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use crate::ast::Ast;
//
//     #[test]
//     fn simple() {
//         let code = String::from(
//             "
//                 fn add(a int, b int) int {
//                     return a + b
//                 }
//                 fn main() void {
//                     let a int = 0
//                     let b int = 1
//                     let c int = a + b + 37 + 200
//                     let d int = b + add(a, b)
//                 }
//                 fn add3(a int, b int, c int) int {
//                     let abc int = a + b + c
//                     return abc
//                 }
//             ",
//         );
//
//         let tokens = lexer::Lexer::new(&code).run().unwrap();
//         let ast = Ast::new(&tokens).unwrap();
//
//         for v in &ast.functions {
//             println!("{}", v.identifier);
//             println!("{:#?}", FunctionCompiler::compile_fn(v).unwrap());
//         }
//     }
// }
