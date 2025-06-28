use anyhow::{anyhow, Result};
use std::collections::HashMap;

use crate::lexer;

const BOOL_ALIAS: &'static str = "bool";
const INT_ALIAS: &'static str = "int";
const STRING_ALIAS: &'static str = "string";

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

#[derive(Debug, Clone)]
struct LexerNavigator<'a> {
    i: usize,
    tokens: &'a [lexer::Token],
}

impl<'a> LexerNavigator<'a> {
    fn new(tokens: &'a [lexer::Token]) -> Self {
        Self { i: 0, tokens }
    }

    fn next(&mut self) {
        self.i += 1;
    }

    fn expect_identifier(&mut self) -> Result<String> {
        match self.peek_token_err(0)?.clone() {
            lexer::Token::Identifier(string) => {
                self.next();
                Ok(string.clone())
            }
            token => Err(anyhow!("expect_identifier: got {token:#?}")),
        }
    }

    fn expect_next_token(&self, token: lexer::Token) -> Result<()> {
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

    fn peek_token(&self, n: usize) -> Option<&lexer::Token> {
        self.tokens.get(self.i + n)
    }

    fn peek_token_err(&self, n: usize) -> Result<&lexer::Token> {
        self.peek_token(n)
            .ok_or(anyhow!("peek_token_err: expected Some"))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypeStruct {
    pub fields: Vec<(String, Type)>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Alias(String),
    Struct(TypeStruct),
    Slice(Box<Type>),
    Variadic(Box<Type>),
}

struct TypeDeclarationParser<'a, 'b> {
    lexer_navigator: &'b mut LexerNavigator<'a>,
}

#[derive(Debug, Clone)]
pub struct TypeDeclarations(pub HashMap<String, Type>);

impl<'a, 'b> TypeDeclarationParser<'a, 'b> {
    fn new(lexer_navigator: &'b mut LexerNavigator<'a>) -> Self {
        Self { lexer_navigator }
    }

    fn parse_type_struct(&mut self) -> Result<TypeStruct> {
        self.lexer_navigator
            .expect_next_token(lexer::Token::Struct)?;
        self.lexer_navigator.next();

        self.lexer_navigator
            .expect_next_token(lexer::Token::COpen)?;
        self.lexer_navigator.next();

        let mut fields: Vec<(String, Type)> = Vec::new();

        while *self.lexer_navigator.peek_token_err(0)? != lexer::Token::CClose {
            let field_identifier = self.lexer_navigator.expect_identifier()?;
            let field_type = self.parse_type()?;
            fields.push((field_identifier, field_type));
        }

        self.lexer_navigator.next();

        Ok(TypeStruct { fields })
    }

    fn parse_type(&mut self) -> Result<Type> {
        match self.lexer_navigator.peek_token_err(0)? {
            lexer::Token::Struct => Ok(Type::Struct(self.parse_type_struct()?)),
            lexer::Token::Identifier(alias) => {
                let mut _type = Type::Alias(alias.clone());

                while let Some(token) = self.lexer_navigator.peek_token(0) {
                    if *token != lexer::Token::BOpen {
                        break;
                    }
                    self.lexer_navigator.next();

                    self.lexer_navigator
                        .expect_next_token(lexer::Token::BClose)?;
                    self.lexer_navigator.next();

                    _type = Type::Slice(Box::new(_type));
                }

                Ok(_type)
            }
            token => return Err(anyhow!("type_declaration: parse unknown {token:#?}")),
        }
    }

    fn all(mut self) -> Result<TypeDeclarations> {
        let mut map = HashMap::new();

        while let Some(token) = self.lexer_navigator.peek_token(0) {
            if *token != lexer::Token::Type {
                self.lexer_navigator.next();
                continue;
            }

            self.lexer_navigator.next();
            let identifier = self.lexer_navigator.expect_identifier()?;

            let _type = self.parse_type()?;
            map.insert(identifier.clone(), _type);
        }

        Ok(TypeDeclarations(map))
    }
}

struct FunctionDeclarationParser<'a, 'b> {
    lexer_navigator: &'b mut LexerNavigator<'a>,
}

#[derive(Debug, Clone)]
struct FunctionDeclaration {
    i: usize,
    function: Function,
}

struct FunctionDeclarations(HashMap<String, FunctionDeclaration>);

impl<'a, 'b> FunctionDeclarationParser<'a, 'b> {
    fn new(lexer_navigator: &'b mut LexerNavigator<'a>) -> Self {
        Self { lexer_navigator }
    }

    fn parse_function_declaration(&mut self) -> Result<Function> {
        self.lexer_navigator
            .expect_next_token(lexer::Token::Function)?;
        self.lexer_navigator.next();

        let identifier = self.lexer_navigator.expect_identifier()?;

        self.lexer_navigator
            .expect_next_token(lexer::Token::POpen)?;
        self.lexer_navigator.next();

        let mut function_arguments: Vec<Variable> = Vec::new();

        while let Some(token) = self.lexer_navigator.peek_token(0) {
            match token {
                lexer::Token::PClose => {
                    self.lexer_navigator.next();
                    break;
                }
                lexer::Token::Comma => {
                    self.lexer_navigator.next();
                }
                _ => {}
            }

            let identifier = self.lexer_navigator.expect_identifier()?;
            let mut _type = TypeDeclarationParser::new(self.lexer_navigator).parse_type()?;

            if let lexer::Token::Dot3 = self.lexer_navigator.peek_token_err(0)? {
                self.lexer_navigator.next();
                _type = Type::Variadic(Box::new(_type));
                self.lexer_navigator
                    .expect_next_token(lexer::Token::PClose)?;
            }

            function_arguments.push(Variable { _type, identifier })
        }

        let return_type = TypeDeclarationParser::new(self.lexer_navigator).parse_type()?;

        Ok(Function {
            body: None,
            return_type,
            identifier,
            arguments: function_arguments,
        })
    }

    fn all(mut self) -> Result<FunctionDeclarations> {
        let mut map = HashMap::new();

        while let Some(token) = self.lexer_navigator.peek_token(0) {
            if *token != lexer::Token::Function {
                self.lexer_navigator.next();
                continue;
            }

            let function = self.parse_function_declaration()?;
            map.insert(
                function.identifier.clone(),
                FunctionDeclaration {
                    function,
                    i: self.lexer_navigator.i,
                },
            );
        }

        Ok(FunctionDeclarations(map))
    }
}

#[derive(Debug)]
pub struct Ast {
    pub type_declarations: TypeDeclarations,
    pub functions: HashMap<String, Function>,
}

impl Ast {
    pub fn new(tokens: &[lexer::Token]) -> Result<Self> {
        let parser = TokenParser::new(tokens)?;
        Ok(Self {
            type_declarations: parser.type_declarations.clone(),
            functions: parser.parse_functions()?,
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
    lexer_navigator: LexerNavigator<'a>,
    variables: AstVariables,
    function_declarations: FunctionDeclarations,
    type_declarations: TypeDeclarations,
}

#[derive(Debug, Clone)]
pub struct Variable {
    pub _type: Type,
    pub identifier: String,
}

impl<'a> TokenParser<'a> {
    fn new(tokens: &'a [lexer::Token]) -> Result<Self> {
        Ok(Self {
            lexer_navigator: LexerNavigator::new(tokens),
            variables: AstVariables::new(),
            type_declarations: TypeDeclarationParser::new(&mut LexerNavigator::new(tokens))
                .all()?,
            function_declarations: FunctionDeclarationParser::new(&mut LexerNavigator::new(tokens))
                .all()?,
        })
    }

    fn parse_functions(mut self) -> Result<HashMap<String, Function>> {
        let mut functions = HashMap::new();

        for (k, v) in self.function_declarations.0.clone() {
            self.lexer_navigator.i = v.i;
            functions.insert(
                k.clone(),
                Function {
                    body: Some(self.parse_body()?),
                    ..v.function
                },
            );
        }

        Ok(functions)
    }

    fn parse_for(&mut self) -> Result<For> {
        self.lexer_navigator.expect_next_token(lexer::Token::For)?;
        self.lexer_navigator.next();

        self.variables.push_frame();

        // for {}
        if let lexer::Token::COpen = self.lexer_navigator.peek_token_err(0)? {
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

        self.lexer_navigator
            .expect_next_token(lexer::Token::Semicolon)?;
        self.lexer_navigator.next();

        let expression = self.parse_expression()?;

        self.lexer_navigator
            .expect_next_token(lexer::Token::Semicolon)?;
        self.lexer_navigator.next();

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

        match self.lexer_navigator.peek_token_err(0)? {
            lexer::Token::Equals => {
                self.lexer_navigator.next();
                Ok(Node::VariableAssignment(VariableAssignment {
                    var: exp,
                    expression: self.parse_expression()?,
                }))
            }
            lexer::Token::PlusPlus | lexer::Token::MinusMinus => {
                let token = self.lexer_navigator.peek_token_err(0)?.clone();
                self.lexer_navigator.next();

                Ok(Node::VariableAssignment(VariableAssignment {
                    var: exp.clone(),
                    expression: Expression::Arithmetic(Box::new(Arithmetic {
                        left: exp,
                        right: Expression::Literal(Literal {
                            _type: Type::Alias(INT_ALIAS.to_string()),
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
        match self.lexer_navigator.peek_token_err(0)? {
            lexer::Token::Debug => {
                self.lexer_navigator.next();
                Ok(Node::Debug)
            }
            lexer::Token::Let => Ok(Node::VariableDeclaration(
                self.parse_variable_declaration()?,
            )),
            lexer::Token::Return => {
                self.lexer_navigator.next();
                Ok(Node::Return(self.parse_expression().ok()))
            }
            lexer::Token::If => Ok(Node::If(self.parse_if()?)),
            lexer::Token::For => Ok(Node::For(self.parse_for()?)),
            lexer::Token::Break => {
                self.lexer_navigator.next();
                Ok(Node::Break)
            }
            lexer::Token::Continue => {
                self.lexer_navigator.next();
                Ok(Node::Continue)
            }
            _ => self.parse_token_else(),
        }
    }

    fn parse_if(&mut self) -> Result<If> {
        match self.lexer_navigator.peek_token_err(0)? {
            lexer::Token::If | lexer::Token::ElseIf => {}
            _ => return Err(anyhow!("parse_if: unknown token")),
        }
        self.lexer_navigator.next();

        let expression = self.parse_expression()?;
        let body = self.parse_body()?;

        let mut elseif = Vec::<ElseIf>::new();
        while let lexer::Token::ElseIf = self.lexer_navigator.peek_token_err(0)? {
            self.lexer_navigator.next();
            elseif.push(ElseIf {
                expression: self.parse_expression()?,
                body: self.parse_body()?,
            });
        }

        let mut _else = None;
        if let lexer::Token::Else = self.lexer_navigator.peek_token_err(0)? {
            self.lexer_navigator.next();
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

        self.lexer_navigator
            .expect_next_token(lexer::Token::COpen)?;
        self.lexer_navigator.next();

        while let Some(token) = self.lexer_navigator.peek_token(0) {
            if let lexer::Token::CClose = token {
                self.lexer_navigator.next();
                break;
            }

            nodes.push(self.parse_token()?);
        }

        self.variables.pop_frame();

        Ok(nodes)
    }

    fn parse_literal(&mut self) -> Result<Literal> {
        match self.lexer_navigator.peek_token_err(0)?.clone() {
            lexer::Token::Literal(v) => {
                self.lexer_navigator.next();
                Ok(Literal {
                    literal: v.clone(),
                    _type: match v {
                        lexer::Literal::Int(_) => Type::Alias(INT_ALIAS.to_string()),
                        lexer::Literal::Bool(_) => Type::Alias(BOOL_ALIAS.to_string()),
                        lexer::Literal::String(_) => Type::Alias(STRING_ALIAS.to_string()),
                    },
                })
            }
            _ => Err(anyhow!("parse_literal: expected Literal")),
        }
    }

    fn parse_identifier(&mut self) -> Result<String> {
        match self.lexer_navigator.peek_token_err(0)?.clone() {
            lexer::Token::Identifier(v) => {
                self.lexer_navigator.next();
                Ok(v.clone())
            }
            _ => Err(anyhow!("parse_identifier: expected Identifier")),
        }
    }

    fn parse_type(&mut self) -> Result<Type> {
        TypeDeclarationParser::new(&mut self.lexer_navigator).parse_type()
    }

    fn parse_function_call(&mut self) -> Result<FunctionCall> {
        let identifier = self.parse_identifier()?;

        self.lexer_navigator
            .expect_next_token(lexer::Token::POpen)?;
        self.lexer_navigator.next();

        let mut arguments = Vec::new();

        while let Some(token) = self.lexer_navigator.peek_token(0) {
            match token {
                lexer::Token::PClose => {
                    self.lexer_navigator.next();
                    break;
                }
                lexer::Token::Comma => {
                    self.lexer_navigator.next();
                }
                _ => {}
            }

            arguments.push(self.parse_expression()?);
        }

        let function = self
            .function_declarations
            .0
            .get(&identifier)
            .ok_or(anyhow!(
                "parse_function_call: context function declaration does not exist"
            ))?
            .clone();

        Ok(FunctionCall {
            arguments,
            function: function.function,
        })
    }

    fn parse_expression_type(&mut self, _type: Type) -> Result<Expression> {
        if *self.lexer_navigator.peek_token_err(0)? == lexer::Token::POpen {
            let exp = self.parse_expression()?;
            self.lexer_navigator
                .expect_next_token(lexer::Token::PClose)?;
            self.lexer_navigator.next();
            return Ok(Expression::TypeCast(Box::new(TypeCast {
                _type,
                expression: exp,
            })));
        }

        return Ok(Expression::Type(_type));
    }

    fn parse_expression_identifier(&mut self) -> Result<Expression> {
        let maybe_type = self.parse_type()?;
        let Type::Alias(identifier) = &maybe_type else {
            return self.parse_expression_type(maybe_type);
        };

        if let Some(_) = self.type_declarations.0.get(identifier) {
            return self.parse_expression_type(maybe_type);
        }

        if *self.lexer_navigator.peek_token_err(0)? == lexer::Token::POpen {
            return Ok(Expression::FunctionCall(self.parse_function_call()?));
        }

        Ok(Expression::Variable(
            self.variables
                .get_variable(&identifier)
                .ok_or(anyhow!(
                    "parse_expression_identifier: identifier variable {identifier} not found"
                ))?
                .clone(),
        ))
    }

    fn parse_expression_literal(&mut self) -> Result<Expression> {
        Ok(Expression::Literal(self.parse_literal()?))
    }

    fn parse_expression_list(&mut self) -> Result<Expression> {
        self.lexer_navigator
            .expect_next_token(lexer::Token::COpen)?;
        self.lexer_navigator.next();

        let mut expressions = Vec::new();

        while let Some(v) = self.lexer_navigator.peek_token(0) {
            if let lexer::Token::CClose = v {
                self.lexer_navigator.next();
                break;
            }

            expressions.push(self.parse_expression()?);

            if *self.lexer_navigator.peek_token_err(0)? != lexer::Token::CClose {
                self.lexer_navigator
                    .expect_next_token(lexer::Token::Comma)?;
                self.lexer_navigator.next();
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

    fn parse_expression_pratt(&mut self, min_bp: usize) -> Result<Expression> {
        let mut left: Expression = {
            let token = self.lexer_navigator.peek_token_err(0)?.clone();
            match token {
                lexer::Token::POpen => {
                    self.lexer_navigator.next();
                    let exp = self.parse_expression()?;
                    self.lexer_navigator
                        .expect_next_token(lexer::Token::PClose)?;
                    self.lexer_navigator.next();
                    exp
                }
                lexer::Token::Plus | lexer::Token::Minus => {
                    self.lexer_navigator.next();
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
                    self.lexer_navigator.next();
                    Expression::Negate(Box::new(self.parse_expression_pratt(100)?))
                }
                lexer::Token::COpen => self.parse_expression_list()?,
                lexer::Token::Identifier(_) => self.parse_expression_identifier()?,
                lexer::Token::Literal(_) => self.parse_expression_literal()?,
                lexer::Token::Struct => Expression::Type(self.parse_type()?),
                token => return Err(anyhow!("parse_expression: incorrect token {token:#?}")),
            }
        };

        loop {
            let token = self.lexer_navigator.peek_token_err(0)?.clone();
            match token {
                lexer::Token::BOpen => {
                    self.lexer_navigator.next();
                    left = Expression::Index(Index {
                        var: Box::new(left),
                        expression: Box::new(self.parse_expression()?),
                    });
                    self.lexer_navigator
                        .expect_next_token(lexer::Token::BClose)?;
                    self.lexer_navigator.next();
                    continue;
                }
                lexer::Token::Dot3 => {
                    left = Expression::Spread(Box::new(left));
                    self.lexer_navigator.next();
                    break;
                }
                _ => {}
            }

            let token = self.lexer_navigator.peek_token_err(0)?.clone();
            let (l_bp, r_bp) = match Self::pratt_binding_power(&token) {
                Some(v) => v,
                None => break,
            };

            if l_bp < min_bp {
                break;
            }
            self.lexer_navigator.next();
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
        self.lexer_navigator.expect_next_token(lexer::Token::Let)?;
        self.lexer_navigator.next();

        let identifier = self.parse_identifier()?;

        let _type = self.parse_type()?;

        self.lexer_navigator
            .expect_next_token(lexer::Token::Equals)?;
        self.lexer_navigator.next();

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
}
