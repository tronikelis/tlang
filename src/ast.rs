use anyhow::{anyhow, Result};
use lazy_static::lazy_static;
use std::collections::HashMap;

use crate::lexer;

#[derive(Debug, Clone, PartialEq)]
pub enum TypeType {
    Slice(Box<Type>),
    Scalar(lexer::Type),
    CompilerType,
    Variadic(Box<Type>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Type {
    pub size: usize,
    pub _type: TypeType,
}

impl Type {
    pub fn can_assign(&self, other: &Self) -> bool {
        match &self._type {
            TypeType::Slice(_) => self.can_assign_slice(other),
            _ => other == self,
        }
    }

    fn can_assign_slice(&self, other: &Self) -> bool {
        let (me, me_depth) = self.extract_slice_type();
        let (other, other_depth) = other.extract_slice_type();
        // some safety checks
        if other_depth == 0 && *other == VOID {
            panic!("impossible other_depth=0 VOID type in slice");
        }

        if other_depth > me_depth {
            return false;
        }

        if other_depth < me_depth {
            return *other == VOID;
        }

        return *other == VOID || other == me;
    }

    fn extract_slice_type(&self) -> (&Type, usize) {
        let mut _type = self;
        let mut i = 0;
        loop {
            if let TypeType::Slice(v) = &_type._type {
                _type = v;
                i += 1;
            } else {
                break;
            }
        }

        (_type, i)
    }

    pub fn extract_variadic(&self) -> Option<Self> {
        match &self._type {
            TypeType::Variadic(item) => Some(*item.clone()),
            _ => None,
        }
    }
}

pub const COMPILER_TYPE: Type = Type {
    // dont look at this value, this type itself will be replaced by the compiler
    size: 0,
    _type: TypeType::CompilerType,
};
pub const VOID: Type = Type {
    size: 0,
    _type: TypeType::Scalar(lexer::Type::Void),
};
pub const INT: Type = Type {
    size: size_of::<isize>(),
    _type: TypeType::Scalar(lexer::Type::Int),
};
pub const BOOL: Type = Type {
    size: size_of::<isize>(),
    _type: TypeType::Scalar(lexer::Type::Bool),
};
pub const UINT8: Type = Type {
    size: size_of::<u8>(),
    _type: TypeType::Scalar(lexer::Type::Uint8),
};
pub const PTR_SIZE: usize = size_of::<usize>();
pub const SLICE_SIZE: usize = size_of::<usize>();
pub const STRING: Type = Type {
    size: SLICE_SIZE,
    _type: TypeType::Scalar(lexer::Type::String),
};
pub const PTR: Type = Type {
    size: size_of::<usize>(),
    _type: TypeType::Scalar(lexer::Type::Ptr),
};
pub const UINT: Type = Type {
    size: size_of::<usize>(),
    _type: TypeType::Scalar(lexer::Type::Uint),
};

lazy_static! {
    pub static ref SLICE_UINT8: Type = Type {
        size: SLICE_SIZE,
        _type: TypeType::Slice(Box::new(UINT8)),
    };
}

#[derive(Debug, Clone)]
pub struct VariableDeclaration {
    pub variable: Variable,
    pub expression: Expression,
}

#[derive(Debug, Clone)]
pub struct Function {
    pub identifier: String,
    pub arguments: Vec<Variable>,
    pub return_type: Type,
    pub body: Option<Vec<Node>>,
}

#[derive(Debug, Clone)]
pub struct Arithmetic {
    pub left: Expression,
    pub right: Expression,
    pub _type: ArithmeticType,
}

#[derive(Debug, Clone)]
pub enum ArithmeticType {
    Plus,
    Minus,
    Divide,
    Multiply,
    Modulo,
}

#[derive(Debug, Clone)]
pub struct FunctionCall {
    pub function: Function,
    pub arguments: Vec<Expression>,
}

#[derive(Debug, Clone)]
pub enum InfixType {
    Plus,
    Minus,
}

#[derive(Debug, Clone)]
pub struct Infix {
    pub expression: Box<Expression>,
    pub _type: InfixType,
}

#[derive(Debug, Clone)]
pub struct Literal {
    pub literal: lexer::Literal,
    pub _type: Type,
}

#[derive(Debug, Clone)]
pub struct Index {
    pub var: Box<Expression>,
    pub expression: Box<Expression>,
}

#[derive(Debug, Clone)]
pub struct AndOr {
    pub left: Expression,
    pub right: Expression,
    pub _type: AndOrType,
}

#[derive(Debug, Clone)]
pub enum AndOrType {
    And,
    Or,
}

#[derive(Debug, Clone)]
pub struct TypeCast {
    pub expression: Expression,
    pub _type: Type,
}

#[derive(Debug, Clone)]
pub enum Expression {
    TypeCast(Box<TypeCast>),
    AndOr(Box<AndOr>),
    Infix(Infix),
    Negate(Box<Expression>),
    Literal(Literal),
    Variable(Variable),
    Arithmetic(Box<Arithmetic>),
    Compare(Box<Compare>),
    FunctionCall(FunctionCall),
    List(Vec<Expression>),
    Index(Index),
    Spread(Box<Expression>),
    Type(Type),
}

#[derive(Debug, Clone)]
pub struct VariableAssignment {
    pub var: Expression,
    pub expression: Expression,
}

#[derive(Debug, Clone)]
pub enum CompareType {
    Gt,
    Lt,
    Equals,
    NotEquals,
}

#[derive(Debug, Clone)]
pub struct Compare {
    pub left: Expression,
    pub right: Expression,
    pub compare_type: CompareType,
}

#[derive(Debug, Clone)]
pub struct If {
    pub expression: Expression,
    pub body: Vec<Node>,
    pub elseif: Vec<ElseIf>,
    pub _else: Option<Else>,
}

#[derive(Debug, Clone)]
pub struct ElseIf {
    pub expression: Expression,
    pub body: Vec<Node>,
}

#[derive(Debug, Clone)]
pub struct Else {
    pub body: Vec<Node>,
}

#[derive(Debug, Clone)]
pub struct For {
    pub initializer: Option<Box<Node>>,
    pub expression: Option<Expression>,
    pub after_each: Option<Box<Node>>,
    pub body: Vec<Node>,
}

#[derive(Debug, Clone)]
pub enum Node {
    Expression(Expression),
    VariableDeclaration(VariableDeclaration),
    VariableAssignment(VariableAssignment),
    Return(Option<Expression>),
    If(If),
    For(For),
    Debug,
    Break,
    Continue,
}

#[derive(Debug)]
pub struct Ast {
    pub functions: Vec<Function>,
}

impl Ast {
    pub fn new(tokens: &Vec<lexer::Token>) -> Result<Self> {
        Ok(Self {
            functions: TokenParser::new(tokens).parse_functions()?,
        })
    }
}

struct AstVariables {
    stack: Vec<Vec<Variable>>,
}

impl AstVariables {
    fn new() -> Self {
        let mut stack = Vec::new();
        stack.push(Vec::new());
        Self { stack }
    }

    fn push_frame(&mut self) {
        self.stack.push(Vec::new());
    }

    fn pop_frame(&mut self) {
        self.stack.pop();
    }

    fn push_variable(&mut self, variable: Variable) {
        self.stack.last_mut().unwrap().push(variable);
    }

    fn get_variable(&self, identifier: &str) -> Option<&Variable> {
        self.stack
            .iter()
            .flatten()
            .rev()
            .find(|v| v.identifier == identifier)
    }
}

struct TokenParser<'a> {
    tokens: &'a Vec<lexer::Token>,
    i: usize,
    functions: HashMap<String, Function>,
    variables: AstVariables,
}

#[derive(Debug, Clone)]
pub struct Variable {
    pub _type: Type,
    pub identifier: String,
}

impl<'a> TokenParser<'a> {
    fn new(tokens: &'a Vec<lexer::Token>) -> Self {
        Self {
            tokens,
            i: 0,
            functions: HashMap::new(),
            variables: AstVariables::new(),
        }
    }

    fn parse_context_function_declarations(&self) -> Result<HashMap<String, Function>> {
        let mut declarations = HashMap::<String, Function>::new();

        for (i, token) in self.tokens.iter().enumerate() {
            if let lexer::Token::Function = token {
                let mut temp_parser = TokenParser::new(self.tokens);
                temp_parser.i = i;

                let function = temp_parser.parse_function_declaration()?;
                declarations.insert(function.identifier.clone(), function);
            }
        }

        Ok(declarations)
    }

    fn parse_functions(mut self) -> Result<Vec<Function>> {
        self.functions = self.parse_context_function_declarations()?;

        let mut functions = Vec::<Function>::new();

        while let Some(token) = self.peek_token(0) {
            match token {
                lexer::Token::Function => functions.push(self.parse_function()?),
                v => return Err(anyhow!("parse_functions: {v:#?} token not supported")),
            };
        }

        Ok(functions)
    }

    fn parse_for(&mut self) -> Result<For> {
        self.expect_next_token(lexer::Token::For)?;
        self.next();

        self.variables.push_frame();

        // for {}
        if let lexer::Token::COpen = self.peek_token_err(0)? {
            return Ok(For {
                body: self.parse_body()?,
                initializer: None,
                expression: None,
                after_each: None,
            });
        }

        // for false {}
        if let Ok(expression) = self.parse_expression() {
            return Ok(For {
                body: self.parse_body()?,
                expression: Some(expression),
                initializer: None,
                after_each: None,
            });
        }

        let initializer = self.parse_token()?;

        self.expect_next_token(lexer::Token::Semicolon)?;
        self.next();

        let expression = self.parse_expression()?;

        self.expect_next_token(lexer::Token::Semicolon)?;
        self.next();

        let after_each = self.parse_token()?;
        let body = self.parse_body()?;

        self.variables.pop_frame();

        Ok(For {
            initializer: Some(Box::new(initializer)),
            expression: Some(expression),
            after_each: Some(Box::new(after_each)),
            body,
        })
    }

    fn parse_token_else(&mut self) -> Result<Node> {
        let exp = self.parse_expression()?;

        match self.peek_token_err(0)? {
            lexer::Token::Equals => {
                self.next();
                Ok(Node::VariableAssignment(VariableAssignment {
                    var: exp,
                    expression: self.parse_expression()?,
                }))
            }
            lexer::Token::PlusPlus | lexer::Token::MinusMinus => {
                let token = self.peek_token_err(0)?.clone();
                self.next();

                Ok(Node::VariableAssignment(VariableAssignment {
                    var: exp.clone(),
                    expression: Expression::Arithmetic(Box::new(Arithmetic {
                        left: exp,
                        right: Expression::Literal(Literal {
                            _type: INT,
                            literal: lexer::Literal::Int(1),
                        }),
                        _type: {
                            if let lexer::Token::PlusPlus = token {
                                ArithmeticType::Plus
                            } else {
                                ArithmeticType::Minus
                            }
                        },
                    })),
                }))
            }
            _ => Ok(Node::Expression(exp)),
        }
    }

    fn parse_token(&mut self) -> Result<Node> {
        match self.peek_token_err(0)? {
            lexer::Token::Debug => {
                self.next();
                Ok(Node::Debug)
            }
            lexer::Token::Let => Ok(Node::VariableDeclaration(
                self.parse_variable_declaration()?,
            )),
            lexer::Token::Return => {
                self.next();
                Ok(Node::Return(self.parse_expression().ok()))
            }
            lexer::Token::If => Ok(Node::If(self.parse_if()?)),
            lexer::Token::For => Ok(Node::For(self.parse_for()?)),
            lexer::Token::Break => {
                self.next();
                Ok(Node::Break)
            }
            lexer::Token::Continue => {
                self.next();
                Ok(Node::Continue)
            }
            _ => self.parse_token_else(),
        }
    }

    fn parse_if(&mut self) -> Result<If> {
        match self.peek_token_err(0)? {
            lexer::Token::If | lexer::Token::ElseIf => {}
            _ => return Err(anyhow!("parse_if: unknown token")),
        }
        self.next();

        let expression = self.parse_expression()?;
        let body = self.parse_body()?;

        let mut elseif = Vec::<ElseIf>::new();
        while let lexer::Token::ElseIf = self.peek_token_err(0)? {
            self.next();
            elseif.push(ElseIf {
                expression: self.parse_expression()?,
                body: self.parse_body()?,
            });
        }

        let mut _else = None;
        if let lexer::Token::Else = self.peek_token_err(0)? {
            self.next();
            _else = Some(Else {
                body: self.parse_body()?,
            });
        }

        Ok(If {
            expression,
            body,
            elseif,
            _else,
        })
    }

    fn parse_body(&mut self) -> Result<Vec<Node>> {
        self.variables.push_frame();

        let mut nodes = Vec::new();

        self.expect_next_token(lexer::Token::COpen)?;
        self.next();

        while let Some(token) = self.peek_token(0) {
            if let lexer::Token::CClose = token {
                self.next();
                break;
            }

            nodes.push(self.parse_token()?);
        }

        self.variables.pop_frame();

        Ok(nodes)
    }

    fn parse_literal(&mut self) -> Result<Literal> {
        match self.peek_token_err(0)?.clone() {
            lexer::Token::Literal(v) => {
                self.next();
                Ok(Literal {
                    literal: v.clone(),
                    _type: match v {
                        lexer::Literal::Int(_) => INT,
                        lexer::Literal::Bool(_) => BOOL,
                        lexer::Literal::String(_) => STRING.clone(),
                    },
                })
            }
            _ => Err(anyhow!("parse_literal: expected Literal")),
        }
    }

    fn parse_identifier(&mut self) -> Result<String> {
        match self.peek_token_err(0)?.clone() {
            lexer::Token::Identifier(v) => {
                self.next();
                Ok(v.clone())
            }
            _ => Err(anyhow!("parse_identifier: expected Identifier")),
        }
    }

    fn parse_type_slice(&mut self, parent: Type) -> Result<Type> {
        self.expect_next_token(lexer::Token::BOpen)?;
        self.next();
        self.expect_next_token(lexer::Token::BClose)?;
        self.next();

        let _type = Type {
            _type: TypeType::Slice(Box::new(parent)),
            size: SLICE_SIZE,
        };

        if let Ok(v) = self.parse_type_slice(_type.clone()) {
            Ok(v)
        } else {
            Ok(_type)
        }
    }

    fn parse_type(&mut self) -> Result<Type> {
        let _type = match self.peek_token_err(0)?.clone() {
            lexer::Token::Type(v) => {
                self.next();
                match v {
                    lexer::Type::Void => VOID,
                    lexer::Type::Int => INT,
                    lexer::Type::Bool => BOOL,
                    lexer::Type::CompilerType => COMPILER_TYPE,
                    lexer::Type::Uint8 => UINT8,
                    lexer::Type::String => STRING,
                    lexer::Type::Ptr => PTR,
                    lexer::Type::Uint => UINT,
                }
            }
            _ => return Err(anyhow!("parse_type: expected type")),
        };

        match self.peek_token_err(0)? {
            lexer::Token::BOpen => return self.parse_type_slice(_type),
            _ => {}
        };

        Ok(_type)
    }

    fn parse_function_declaration(&mut self) -> Result<Function> {
        self.expect_next_token(lexer::Token::Function)?;
        self.next();

        let identifier = self.parse_identifier()?;

        self.expect_next_token(lexer::Token::POpen)?;
        self.next();

        let mut function_arguments: Vec<Variable> = Vec::new();

        while let Some(token) = self.peek_token(0) {
            match token {
                lexer::Token::PClose => {
                    self.next();
                    break;
                }
                lexer::Token::Comma => {
                    self.next();
                }
                _ => {}
            }

            let identifier = self.parse_identifier()?;
            let mut _type = self.parse_type()?;

            if let lexer::Token::Dot3 = self.peek_token_err(0)? {
                _type = Type {
                    size: SLICE_SIZE,
                    _type: TypeType::Variadic(Box::new(_type)),
                };
                self.next();
                // expect variadic to be last one
                self.expect_next_token(lexer::Token::PClose).map_err(|_| {
                    anyhow!("parse_function_declaration: variadic argument expected PCLOSE")
                })?;
            }

            function_arguments.push(Variable { identifier, _type });
        }

        let return_type = self.parse_type()?;

        for v in &function_arguments {
            self.variables.push_variable(v.clone());
        }

        Ok(Function {
            identifier,
            arguments: function_arguments,
            return_type,
            body: None,
        })
    }

    fn parse_function(&mut self) -> Result<Function> {
        self.variables = AstVariables::new();

        let function = self.parse_function_declaration()?;
        let body = self.parse_body()?;

        Ok(Function {
            body: Some(body),
            ..function
        })
    }

    fn parse_function_call(&mut self) -> Result<FunctionCall> {
        let identifier = self.parse_identifier()?;

        self.expect_next_token(lexer::Token::POpen)?;
        self.next();

        let mut arguments = Vec::new();

        while let Some(token) = self.peek_token(0) {
            match token {
                lexer::Token::PClose => {
                    self.next();
                    break;
                }
                lexer::Token::Comma => {
                    self.next();
                }
                _ => {}
            }

            arguments.push(self.parse_expression()?);
        }

        let function = self
            .functions
            .get(&identifier)
            .ok_or(anyhow!(
                "parse_function_call: context function declaration does not exist"
            ))?
            .clone();

        Ok(FunctionCall {
            arguments,
            function,
        })
    }

    fn parse_expression_identifier(&mut self) -> Result<Expression> {
        match self.peek_token_err(1)? {
            lexer::Token::POpen => Ok(Expression::FunctionCall(self.parse_function_call()?)),
            _ => {
                let identifier = self.parse_identifier()?;
                Ok(Expression::Variable(
                    self
                        .variables
                        .get_variable(&identifier)
                        .ok_or(anyhow!(
                            "parse_expression_identifier: identifier variable {identifier} not found"
                        ))?
                        .clone(),
                ))
            }
        }
    }

    fn parse_expression_literal(&mut self) -> Result<Expression> {
        Ok(Expression::Literal(self.parse_literal()?))
    }

    fn parse_expression_list(&mut self) -> Result<Expression> {
        self.expect_next_token(lexer::Token::COpen)?;
        self.next();

        let mut expressions = Vec::new();

        while let Some(v) = self.peek_token(0) {
            if let lexer::Token::CClose = v {
                self.next();
                break;
            }

            expressions.push(self.parse_expression()?);

            if *self.peek_token_err(0)? != lexer::Token::CClose {
                self.expect_next_token(lexer::Token::Comma)?;
                self.next();
            }
        }

        Ok(Expression::List(expressions))
    }

    fn pratt_binding_power(token: &lexer::Token) -> Option<(usize, usize)> {
        match token {
            lexer::Token::Percent => Some((11, 12)),

            lexer::Token::Star | lexer::Token::Slash => Some((9, 10)),
            lexer::Token::Plus | lexer::Token::Minus => Some((7, 8)),

            lexer::Token::Lt
            | lexer::Token::Gt
            | lexer::Token::EqualsEquals
            | lexer::Token::BangEquals => Some((5, 6)),

            lexer::Token::AmperAmper => Some((3, 4)),
            lexer::Token::PipePipe => Some((1, 2)),

            _ => None,
        }
    }

    fn parse_expression(&mut self) -> Result<Expression> {
        self.parse_expression_pratt(0)
    }

    fn parse_expression_type(&mut self) -> Result<Expression> {
        let _type = self.parse_type()?;

        if *self.peek_token_err(0)? != lexer::Token::POpen {
            return Ok(Expression::Type(_type));
        }
        self.next();

        let expression = self.parse_expression()?;

        self.expect_next_token(lexer::Token::PClose)?;
        self.next();

        Ok(Expression::TypeCast(Box::new(TypeCast {
            expression,
            _type,
        })))
    }

    fn parse_expression_pratt(&mut self, min_bp: usize) -> Result<Expression> {
        let mut left: Expression = {
            let token = self.peek_token_err(0)?.clone();
            match token {
                lexer::Token::POpen => {
                    self.next();
                    let exp = self.parse_expression()?;
                    self.expect_next_token(lexer::Token::PClose)?;
                    self.next();
                    exp
                }
                lexer::Token::Plus | lexer::Token::Minus => {
                    self.next();
                    Expression::Infix(Infix {
                        expression: Box::new(self.parse_expression_pratt(100)?),
                        _type: match token {
                            lexer::Token::Plus => InfixType::Plus,
                            lexer::Token::Minus => InfixType::Minus,
                            _ => unreachable!(),
                        },
                    })
                }
                lexer::Token::Bang => {
                    self.next();
                    Expression::Negate(Box::new(self.parse_expression_pratt(100)?))
                }
                lexer::Token::COpen => self.parse_expression_list()?,
                lexer::Token::Identifier(_) => self.parse_expression_identifier()?,
                lexer::Token::Literal(_) => self.parse_expression_literal()?,
                lexer::Token::Type(_) => self.parse_expression_type()?,
                token => return Err(anyhow!("parse_expression: incorrect token {token:#?}")),
            }
        };

        loop {
            let token = self.peek_token_err(0)?.clone();
            match token {
                lexer::Token::BOpen => {
                    self.next();
                    left = Expression::Index(Index {
                        var: Box::new(left),
                        expression: Box::new(self.parse_expression()?),
                    });
                    self.expect_next_token(lexer::Token::BClose)?;
                    self.next();
                    continue;
                }
                lexer::Token::Dot3 => {
                    left = Expression::Spread(Box::new(left));
                    self.next();
                    break;
                }
                _ => {}
            }

            let token = self.peek_token_err(0)?.clone();
            let (l_bp, r_bp) = match Self::pratt_binding_power(&token) {
                Some(v) => v,
                None => break,
            };

            if l_bp < min_bp {
                break;
            }
            self.next();
            let right = self.parse_expression_pratt(r_bp)?;

            match token {
                lexer::Token::Plus
                | lexer::Token::Minus
                | lexer::Token::Star
                | lexer::Token::Slash
                | lexer::Token::Percent => {
                    left = Expression::Arithmetic(Box::new(Arithmetic {
                        left,
                        right,
                        _type: match token {
                            lexer::Token::Plus => ArithmeticType::Plus,
                            lexer::Token::Minus => ArithmeticType::Minus,
                            lexer::Token::Star => ArithmeticType::Multiply,
                            lexer::Token::Slash => ArithmeticType::Divide,
                            lexer::Token::Percent => ArithmeticType::Modulo,
                            _ => unreachable!(),
                        },
                    }));
                }
                lexer::Token::AmperAmper | lexer::Token::PipePipe => {
                    left = Expression::AndOr(Box::new(AndOr {
                        left,
                        right,
                        _type: match token {
                            lexer::Token::AmperAmper => AndOrType::And,
                            lexer::Token::PipePipe => AndOrType::Or,
                            _ => unreachable!(),
                        },
                    }))
                }
                lexer::Token::Lt
                | lexer::Token::Gt
                | lexer::Token::EqualsEquals
                | lexer::Token::BangEquals => {
                    left = Expression::Compare(Box::new(Compare {
                        left,
                        right,
                        compare_type: match token {
                            lexer::Token::Lt => CompareType::Lt,
                            lexer::Token::Gt => CompareType::Gt,
                            lexer::Token::EqualsEquals => CompareType::Equals,
                            lexer::Token::BangEquals => CompareType::NotEquals,
                            _ => unreachable!(),
                        },
                    }))
                }
                token => return Err(anyhow!("parse_expression: incorrect token {token:#?}")),
            }
        }

        Ok(left)
    }

    fn parse_variable_declaration(&mut self) -> Result<VariableDeclaration> {
        self.expect_next_token(lexer::Token::Let)?;
        self.next();

        let identifier = self.parse_identifier()?;

        let _type = self.parse_type()?;

        self.expect_next_token(lexer::Token::Equals)?;
        self.next();

        let expression = self.parse_expression()?;

        self.variables.push_variable(Variable {
            _type: _type.clone(),
            identifier: identifier.clone(),
        });

        Ok(VariableDeclaration {
            variable: Variable { identifier, _type },
            expression,
        })
    }

    fn expect_next_token(&mut self, token: lexer::Token) -> Result<()> {
        if token == *self.peek_token_err(0)? {
            Ok(())
        } else {
            Err(anyhow!(
                "expect_next_token: assertion failed, want: {:#?}, got: {:#?}",
                token,
                self.peek_token_err(0)
            ))
        }
    }

    fn next(&mut self) {
        self.i += 1;
    }

    fn peek_token(&self, n: usize) -> Option<&lexer::Token> {
        self.tokens.get(self.i + n)
    }

    fn peek_token_err(&self, n: usize) -> Result<&lexer::Token> {
        self.peek_token(n)
            .ok_or(anyhow!("peek_token_err: expected Some"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple() {
        let code = String::from(
            "
                fn main() void {
                    let a = 20
                    if a > 20 {
                        a = 220
                    }
                }
            ",
        );

        let tokens = lexer::Lexer::new(&code).run().unwrap();
        let ast = Ast::new(&tokens).unwrap();
        println!("{ast:#?}");
    }
}
