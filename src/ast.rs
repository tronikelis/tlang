use anyhow::{anyhow, Result};
use std::collections::HashMap;

use crate::lexer;

#[derive(Debug, Clone, PartialEq)]
pub enum TypeType {
    Slice(Box<Type>),
    Scalar(lexer::Type),
    CompilerType,
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
pub enum Expression {
    Infix(Infix),
    Literal(Literal),
    Variable(Variable),
    Arithmetic(Box<Arithmetic>),
    Compare(Box<Compare>),
    FunctionCall(FunctionCall),
    List(Vec<Expression>),
    Index(Index),
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
}

#[derive(Debug, Clone)]
pub enum AndOr {
    And(Expression),
    Or(Expression),
}

#[derive(Debug, Clone)]
pub struct Compare {
    pub left: Expression,
    pub right: Expression,
    pub compare_type: CompareType,
    pub andor: Option<AndOr>,
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
    pub initializer: Box<Node>,
    pub expression: Expression,
    pub after_each: Box<Node>,
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

struct TokenParser<'a> {
    tokens: &'a Vec<lexer::Token>,
    i: usize,
    context: TokenParserContext,
}

#[derive(Debug, Clone)]
pub struct Variable {
    pub _type: Type,
    pub identifier: String,
}

struct TokenParserContext {
    functions: HashMap<String, Function>,
    variables: HashMap<String, Variable>,
}

impl<'a> TokenParser<'a> {
    fn new(tokens: &'a Vec<lexer::Token>) -> Self {
        Self {
            tokens,
            i: 0,
            context: TokenParserContext {
                functions: HashMap::new(),
                variables: HashMap::new(),
            },
        }
    }

    fn parse_context_function_declarations(&self) -> Result<HashMap<String, Function>> {
        let mut declarations = HashMap::<String, Function>::new();
        let mut temp_parser = TokenParser::new(self.tokens);

        for (i, token) in self.tokens.iter().enumerate() {
            temp_parser.i = i;
            if let lexer::Token::Function = token {
                let function = temp_parser.parse_function_declaration()?;
                declarations.insert(function.identifier.clone(), function);
            }
        }

        Ok(declarations)
    }

    fn parse_context(&self) -> Result<TokenParserContext> {
        Ok(TokenParserContext {
            functions: self.parse_context_function_declarations()?,
            variables: HashMap::new(),
        })
    }

    fn parse_functions(mut self) -> Result<Vec<Function>> {
        self.context = self.parse_context()?;

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

        let initializer = self.parse_token()?;

        self.expect_next_token(lexer::Token::Semicolon)?;
        self.next();

        let expression = self.parse_expression(None)?;

        self.expect_next_token(lexer::Token::Semicolon)?;
        self.next();

        let after_each = self.parse_token()?;
        let body = self.parse_body()?;

        Ok(For {
            initializer: Box::new(initializer),
            expression,
            after_each: Box::new(after_each),
            body,
        })
    }

    fn parse_token_else(&mut self) -> Result<Node> {
        let exp = self.parse_expression(None)?;

        match self.peek_token_err(0)? {
            lexer::Token::Equals => {
                self.next();
                Ok(Node::VariableAssignment(VariableAssignment {
                    var: exp,
                    expression: self.parse_expression(None)?,
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
                Ok(Node::Return(self.parse_expression(None).ok()))
            }
            lexer::Token::If => Ok(Node::If(self.parse_if()?)),
            lexer::Token::For => Ok(Node::For(self.parse_for()?)),
            _ => self.parse_token_else(),
        }
    }

    fn parse_if(&mut self) -> Result<If> {
        match self.peek_token_err(0)? {
            lexer::Token::If | lexer::Token::ElseIf => {}
            _ => return Err(anyhow!("parse_if: unknown token")),
        }
        self.next();

        let expression = self.parse_expression(None)?;
        let body = self.parse_body()?;

        let mut elseif = Vec::<ElseIf>::new();
        while let lexer::Token::ElseIf = self.peek_token_err(0)? {
            self.next();
            elseif.push(ElseIf {
                expression: self.parse_expression(None)?,
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

        Ok(nodes)
    }

    fn parse_literal(&mut self, prev_type: Option<&Type>) -> Result<Literal> {
        match self.peek_token_err(0)?.clone() {
            lexer::Token::Literal(v) => {
                self.next();
                Ok(Literal {
                    literal: v.clone(),
                    _type: prev_type.map(|v| v.clone()).unwrap_or(match v {
                        lexer::Literal::Int(_) => INT,
                        lexer::Literal::Bool(_) => BOOL,
                    }),
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
            size: size_of::<usize>(),
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

            function_arguments.push(Variable {
                identifier: self.parse_identifier()?,
                _type: self.parse_type()?,
            });
        }

        let return_type = self.parse_type()?;

        for v in &function_arguments {
            self.context.variables.insert(
                v.identifier.clone(),
                Variable {
                    identifier: v.identifier.clone(),
                    _type: v._type.clone(),
                },
            );
        }

        Ok(Function {
            identifier,
            arguments: function_arguments,
            return_type,
            body: None,
        })
    }

    fn parse_function(&mut self) -> Result<Function> {
        self.context.variables = HashMap::new();

        let function = self.parse_function_declaration()?;
        let body = self.parse_body()?;

        Ok(Function {
            body: Some(body),
            ..function
        })
    }

    fn parse_function_call(&mut self, prev_type: Option<&Type>) -> Result<FunctionCall> {
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

            arguments.push(self.parse_expression(prev_type)?);
        }

        let function = self
            .context
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

    // this does not have "prev_type" because indexing only supports ints
    // and we don't need to change that into something else
    fn parse_expression_index(&mut self, var: Expression) -> Result<Index> {
        self.expect_next_token(lexer::Token::BOpen)?;
        self.next();

        let expression = self.parse_expression(None)?;

        self.expect_next_token(lexer::Token::BClose)?;
        self.next();

        let index = Index {
            var: Box::new(var),
            expression: Box::new(expression),
        };

        if let lexer::Token::BOpen = self.peek_token_err(0)? {
            Ok(self.parse_expression_index(Expression::Index(index))?)
        } else {
            Ok(index)
        }
    }

    fn parse_expression_identifier(&mut self, prev_type: Option<&Type>) -> Result<Expression> {
        match self.peek_token_err(1)? {
            lexer::Token::POpen => Ok(Expression::FunctionCall(
                self.parse_function_call(prev_type)?,
            )),
            _ => {
                let identifier = self.parse_identifier()?;
                Ok(Expression::Variable(
                    self.context
                        .variables
                        .get(&identifier)
                        .ok_or(anyhow!(
                            "parse_expression_identifier: identifier variable not found"
                        ))?
                        .clone(),
                ))
            }
        }
    }

    fn parse_expression_literal(&mut self, prev_type: Option<&Type>) -> Result<Expression> {
        Ok(Expression::Literal(self.parse_literal(prev_type)?))
    }

    fn parse_pratt(
        &mut self,
        min_bp: usize,
        mut left: Expression,
        prev_type: Option<&Type>,
    ) -> Result<Expression> {
        loop {
            let token = self.peek_token_err(0)?.clone();
            let (l_bp, r_bp) = match pratt_binding_power(&token) {
                Some(v) => v,
                None => break,
            };

            if l_bp < min_bp {
                break;
            }

            self.next();
            let right = self.parse_expression_in_pratt(prev_type)?;
            let right = self.parse_pratt(r_bp, right, prev_type)?;

            left = Expression::Arithmetic(Box::new(Arithmetic {
                left,
                right,
                _type: match token {
                    lexer::Token::Plus => ArithmeticType::Plus,
                    lexer::Token::Minus => ArithmeticType::Minus,
                    lexer::Token::Slash => ArithmeticType::Divide,
                    lexer::Token::Star => ArithmeticType::Multiply,
                    token => return Err(anyhow!("parse_pratt: unknown token {token:#?}")),
                },
            }));
        }

        Ok(left)
    }

    fn parse_expression_list(&mut self, prev_type: Option<&Type>) -> Result<Vec<Expression>> {
        self.expect_next_token(lexer::Token::COpen)?;
        self.next();

        let mut expressions = Vec::new();

        while let Some(v) = self.peek_token(0) {
            if let lexer::Token::CClose = v {
                self.next();
                break;
            }

            let prev_type = prev_type
                .map(|v| match &v._type {
                    TypeType::Slice(slice_item) => Ok(slice_item.as_ref()),
                    _type => Err(anyhow!(
                        "parse_expression_list: trying to cast to non slice {_type:#?}"
                    )),
                })
                .transpose()?;
            expressions.push(self.parse_expression(prev_type)?);

            if *self.peek_token_err(0)? != lexer::Token::CClose {
                self.expect_next_token(lexer::Token::Comma)?;
                self.next();
            }
        }

        Ok(expressions)
    }

    fn parse_expression_in_pratt(&mut self, prev_type: Option<&Type>) -> Result<Expression> {
        match self.peek_token_err(0)? {
            lexer::Token::COpen => {
                return Ok(Expression::List(self.parse_expression_list(prev_type)?))
            }
            _ => {}
        };

        let infix_type;
        match self.peek_token_err(0)? {
            lexer::Token::Minus => {
                infix_type = Some(InfixType::Minus);
                self.next();
            }
            lexer::Token::Plus => {
                infix_type = Some(InfixType::Plus);
                self.next();
            }
            _ => infix_type = None,
        };

        let mut expression = match self.peek_token_err(0)?.clone() {
            lexer::Token::Identifier(_) => self.parse_expression_identifier(prev_type)?,
            lexer::Token::Literal(_) => self.parse_expression_literal(prev_type)?,
            token => return Err(anyhow!("parse_expression: wrong token {token:#?}")),
        };

        if let Some(_type) = infix_type {
            expression = Expression::Infix(Infix {
                expression: Box::new(expression),
                _type,
            });
        }

        let token = self.peek_token_err(0)?.clone();
        match token {
            lexer::Token::Gt | lexer::Token::Lt | lexer::Token::EqualsEquals => {
                self.next();
                let right = self.parse_expression(prev_type)?;

                let mut andor = None;

                match self.peek_token_err(0)? {
                    lexer::Token::AmperAmper => {
                        self.next();
                        andor = Some(AndOr::And(self.parse_expression(prev_type)?));
                    }
                    lexer::Token::PipePipe => {
                        self.next();
                        andor = Some(AndOr::Or(self.parse_expression(prev_type)?));
                    }
                    _ => {}
                }

                return Ok(Expression::Compare(Box::new(Compare {
                    compare_type: match token {
                        lexer::Token::Gt => CompareType::Gt,
                        lexer::Token::Lt => CompareType::Lt,
                        lexer::Token::EqualsEquals => CompareType::Equals,
                        token => return Err(anyhow!("token not supported {token:#?}")),
                    },
                    left: expression,
                    right,
                    andor,
                })));
            }
            lexer::Token::BOpen => {
                return Ok(Expression::Index(self.parse_expression_index(expression)?))
            }
            _ => {}
        }

        Ok(expression)
    }

    fn parse_expression(&mut self, prev_type: Option<&Type>) -> Result<Expression> {
        let expression = self.parse_expression_in_pratt(prev_type)?;
        if let lexer::Token::Plus | lexer::Token::Minus | lexer::Token::Star | lexer::Token::Slash =
            self.peek_token_err(0)?
        {
            Ok(self.parse_pratt(0, expression, prev_type)?)
        } else {
            Ok(expression)
        }
    }

    fn parse_variable_declaration(&mut self) -> Result<VariableDeclaration> {
        self.expect_next_token(lexer::Token::Let)?;
        self.next();

        let identifier = self.parse_identifier()?;

        let _type = self.parse_type()?;

        self.expect_next_token(lexer::Token::Equals)?;
        self.next();

        let expression = self.parse_expression(Some(&_type))?;

        self.context.variables.insert(
            identifier.clone(),
            Variable {
                _type: _type.clone(),
                identifier: identifier.clone(),
            },
        );

        Ok(VariableDeclaration {
            variable: Variable { identifier, _type },
            expression,
        })
    }

    fn expect_next_token(&mut self, token: lexer::Token) -> Result<()> {
        if token == *self.peek_token_err(0)? {
            Ok(())
        } else {
            Err(anyhow!("expect_next_token: assertion failed"))
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

fn pratt_binding_power(token: &lexer::Token) -> Option<(usize, usize)> {
    match token {
        lexer::Token::Plus | lexer::Token::Minus => Some((1, 2)),
        lexer::Token::Star | lexer::Token::Slash => Some((3, 4)),
        _ => None,
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
