use anyhow::{anyhow, Result};
use std::collections::HashMap;

use crate::lexer;

pub enum BfsRet {
    Return,
    Found,
    Continue,
}

macro_rules! return_if_some_true {
    ($v:expr) => {
        let result = $v;
        match result {
            $crate::ast::BfsRet::Found | $crate::ast::BfsRet::Return => {
                return result;
            }
            $crate::ast::BfsRet::Continue => {}
        }
    };
}
pub(crate) use return_if_some_true;

pub trait Bfs<'a> {
    fn search_body(&self, body: impl Iterator<Item = &'a Node>) -> BfsRet {
        for node in body {
            return_if_some_true!(self.search_node(node));
        }

        BfsRet::Continue
    }

    fn search_node(&self, node: &'a Node) -> BfsRet {
        match node {
            Node::Continue | Node::Break | Node::Debug => {}
            Node::Return(exp) => {
                if let Some(exp) = exp {
                    return self.search_expression(exp);
                }
            }
            Node::Expression(v) => return self.search_expression(v),
            Node::If(v) => return self.search_node_if(v),
            Node::For(v) => return self.search_node_for(v),
            Node::VariableDeclaration(v) => return self.search_node_variable_declaration(v),
            Node::VariableAssignment(v) => return self.search_node_variable_assignment(v),
        }

        BfsRet::Continue
    }

    fn search_node_if(&self, _if: &'a If) -> BfsRet {
        return_if_some_true!(self.search_expression(&_if.expression));
        return_if_some_true!(self.search_body(_if.body.iter()));

        for else_if in &_if.elseif {
            return_if_some_true!(self.search_expression(&else_if.expression));
            return_if_some_true!(self.search_body(else_if.body.iter()));
        }

        if let Some(_else) = &_if._else {
            return_if_some_true!(self.search_body(_else.body.iter()));
        }

        BfsRet::Continue
    }

    fn search_node_for(&self, _for: &'a For) -> BfsRet {
        if let Some(node) = &_for.initializer {
            return_if_some_true!(self.search_node(node));
        }
        if let Some(exp) = &_for.expression {
            return_if_some_true!(self.search_expression(exp));
        }
        if let Some(node) = &_for.after_each {
            return_if_some_true!(self.search_node(node));
        }

        return_if_some_true!(self.search_body(_for.body.iter()));

        BfsRet::Continue
    }

    fn search_node_variable_declaration(&self, declaration: &VariableDeclaration) -> BfsRet {
        return_if_some_true!(self.search_expression(&declaration.expression));
        BfsRet::Continue
    }

    fn search_node_variable_assignment(&self, assignment: &VariableAssignment) -> BfsRet {
        return_if_some_true!(self.search_expression(&assignment.var));
        return_if_some_true!(self.search_expression(&assignment.expression));
        BfsRet::Continue
    }

    fn search_expression(&self, exp: &Expression) -> BfsRet {
        match exp {
            Expression::Call(v) => self.search_expression_call(v),
            Expression::TypeInit(v) => self.search_expression_type_init(v),
            Expression::Address(v) => self.search_expression_address(v),
            Expression::AndOr(v) => self.search_expression_andor(v),
            Expression::Arithmetic(v) => self.search_expression_arithmetic(v),
            Expression::Compare(v) => self.search_expression_compare(v),
            Expression::Deref(v) => self.search_expression_deref(v),
            Expression::DotAccess(v) => self.search_expression_dot_access(v),
            Expression::Index(v) => self.search_expression_index(v),
            Expression::Infix(v) => self.search_expression_infix(v),
            Expression::Literal(v) => self.search_expression_literal(v),
            Expression::Negate(v) => self.search_expression_negate(v),
            Expression::SliceInit(v) => self.search_expression_slice_init(v),
            Expression::Spread(v) => self.search_expression_spread(v),
            Expression::StructInit(v) => self.search_expression_struct_init(v),
            Expression::Type(v) => self.search_expression_type(v),
        }
    }

    fn search_expression_type_init(&self, _type_init: &TypeInit) -> BfsRet {
        BfsRet::Continue
    }

    fn search_expression_call(&self, call: &Call) -> BfsRet {
        for exp in &call.arguments {
            return_if_some_true!(self.search_expression(exp));
        }
        BfsRet::Continue
    }

    fn search_expression_address(&self, exp: &Expression) -> BfsRet {
        return_if_some_true!(self.search_expression(exp));
        BfsRet::Continue
    }

    fn search_expression_andor(&self, andor: &AndOr) -> BfsRet {
        return_if_some_true!(self.search_expression(&andor.left));
        return_if_some_true!(self.search_expression(&andor.right));
        BfsRet::Continue
    }

    fn search_expression_deref(&self, exp: &Expression) -> BfsRet {
        return_if_some_true!(self.search_expression(exp));
        BfsRet::Continue
    }

    fn search_expression_type(&self, _type: &Type) -> BfsRet {
        BfsRet::Continue
    }

    fn search_expression_infix(&self, infix: &Infix) -> BfsRet {
        return_if_some_true!(self.search_expression(&infix.expression));
        BfsRet::Continue
    }

    fn search_expression_index(&self, index: &Index) -> BfsRet {
        return_if_some_true!(self.search_expression(&index.expression));
        return_if_some_true!(self.search_expression(&index.var));
        BfsRet::Continue
    }

    fn search_expression_negate(&self, exp: &Expression) -> BfsRet {
        return_if_some_true!(self.search_expression(exp));
        BfsRet::Continue
    }

    fn search_expression_spread(&self, exp: &Expression) -> BfsRet {
        return_if_some_true!(self.search_expression(exp));
        BfsRet::Continue
    }

    fn search_expression_literal(&self, _literal: &Literal) -> BfsRet {
        BfsRet::Continue
    }

    fn search_expression_compare(&self, compare: &Compare) -> BfsRet {
        return_if_some_true!(self.search_expression(&compare.left));
        return_if_some_true!(self.search_expression(&compare.right));
        BfsRet::Continue
    }

    fn search_expression_arithmetic(&self, arithmetic: &Arithmetic) -> BfsRet {
        return_if_some_true!(self.search_expression(&arithmetic.left));
        return_if_some_true!(self.search_expression(&arithmetic.right));
        BfsRet::Continue
    }

    fn search_expression_dot_access(&self, dot_access: &DotAccess) -> BfsRet {
        return_if_some_true!(self.search_expression(&dot_access.expression));
        BfsRet::Continue
    }

    fn search_expression_slice_init(&self, slice_init: &SliceInit) -> BfsRet {
        for exp in &slice_init.expressions {
            return_if_some_true!(self.search_expression(exp));
        }
        BfsRet::Continue
    }

    fn search_expression_struct_init(&self, struct_init: &StructInit) -> BfsRet {
        for exp in struct_init.fields.values() {
            return_if_some_true!(self.search_expression(exp));
        }
        BfsRet::Continue
    }
}

#[derive(Debug, Clone)]
pub struct VariableDeclaration {
    pub variable: Variable,
    pub expression: Expression,
}

#[derive(Debug, Clone)]
pub struct TypeDeclaration {
    pub _type: Type,
    pub identifier: String,
}

#[derive(Debug, Clone)]
pub struct FunctionDeclaration {
    pub identifier: String,
    pub arguments: Vec<Variable>,
    pub return_type: Type,
    pub body: Vec<Node>,
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
pub struct Call {
    pub _type: Type,
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
pub struct StructInit {
    pub fields: HashMap<String, Expression>,
    pub _type: Type,
}

#[derive(Debug, Clone)]
pub struct DotAccess {
    pub expression: Expression,
    pub identifier: String,
}

#[derive(Debug, Clone)]
pub struct SliceInit {
    pub expressions: Vec<Expression>,
    pub _type: Type,
}

#[derive(Debug, Clone)]
pub struct TypeInit {
    pub _type: Type,
}

#[derive(Debug, Clone)]
pub enum Expression {
    AndOr(Box<AndOr>),
    Infix(Infix),
    Negate(Box<Expression>),
    Literal(Literal),
    Arithmetic(Box<Arithmetic>),
    Compare(Box<Compare>),
    Call(Call),
    Index(Index),
    Spread(Box<Expression>),
    Type(Type),
    DotAccess(Box<DotAccess>),
    SliceInit(SliceInit),
    StructInit(StructInit),
    TypeInit(TypeInit),
    Address(Box<Expression>),
    Deref(Box<Expression>),
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
pub struct TypeStruct {
    pub fields: Vec<(String, Type)>,
}

#[derive(Debug, Clone)]
pub enum Type {
    Alias(String),
    Struct(TypeStruct),
    Slice(Box<Type>),
    Variadic(Box<Type>),
    Address(Box<Type>),
}

#[derive(Debug)]
enum Declaration {
    Function(FunctionDeclaration),
    Type(TypeDeclaration),
}

#[derive(Debug)]
pub struct Ast {
    pub type_declarations: HashMap<String, Type>,
    pub function_declarations: HashMap<String, FunctionDeclaration>,
}

impl Ast {
    pub fn new(tokens: &[lexer::Token]) -> Result<Self> {
        let mut type_declarations = HashMap::new();
        let mut function_declarations = HashMap::new();

        let declarations = TokenParser::new(tokens).parse()?;
        for v in declarations {
            match v {
                Declaration::Type(type_declaration) => {
                    type_declarations.insert(type_declaration.identifier, type_declaration._type);
                }
                Declaration::Function(function_declaration) => {
                    function_declarations.insert(
                        function_declaration.identifier.clone(),
                        function_declaration,
                    );
                }
            }
        }

        Ok(Self {
            type_declarations,
            function_declarations,
        })
    }
}

struct TokenParser<'a> {
    tokens: &'a [lexer::Token],
    i: usize,
}

#[derive(Debug, Clone)]
pub struct Variable {
    pub _type: Type,
    pub identifier: String,
}

impl<'a> TokenParser<'a> {
    fn new(tokens: &'a [lexer::Token]) -> Self {
        Self { tokens, i: 0 }
    }

    fn parse(mut self) -> Result<Vec<Declaration>> {
        let mut declarations = Vec::new();
        while let Some(token) = self.peek_token(0) {
            declarations.push(match token {
                lexer::Token::Type => Declaration::Type(self.parse_type_declaration()?),
                lexer::Token::Function => Declaration::Function(self.parse_function_declaration()?),
                token => return Err(anyhow!("parse: unknown token {token:#?}")),
            });
        }

        Ok(declarations)
    }

    fn next(&mut self) {
        self.i += 1;
    }

    fn expect_next_token(&self, token: lexer::Token) -> Result<()> {
        if token == *self.peek_token_err(0)? {
            Ok(())
        } else {
            Err(anyhow!(
                "expect_next_token: assertion failed, want: {:#?}, got: {:#?}, i: {}",
                token,
                self.peek_token_err(0),
                self.i,
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

    fn parse_type_struct(&mut self) -> Result<TypeStruct> {
        self.expect_next_token(lexer::Token::Struct)?;
        self.next();

        self.expect_next_token(lexer::Token::COpen)?;
        self.next();

        let mut fields: Vec<(String, Type)> = Vec::new();

        while *self.peek_token_err(0)? != lexer::Token::CClose {
            let field_identifier = self.parse_identifier()?;
            let field_type = self.parse_type()?;
            fields.push((field_identifier, field_type));
        }

        self.next();

        Ok(TypeStruct { fields })
    }

    fn parse_type(&mut self) -> Result<Type> {
        match self.peek_token_err(0)?.clone() {
            lexer::Token::Star => {
                self.next();
                Ok(Type::Address(Box::new(self.parse_type()?)))
            }
            lexer::Token::Struct => Ok(Type::Struct(self.parse_type_struct()?)),
            lexer::Token::Identifier(alias) => {
                self.next();

                let mut _type = Type::Alias(alias.clone());

                while let Some(token) = self.peek_token(0) {
                    if *token != lexer::Token::BOpen {
                        break;
                    }
                    if *self.peek_token_err(1)? != lexer::Token::BClose {
                        break;
                    }

                    self.next();
                    self.next();

                    _type = Type::Slice(Box::new(_type));
                }

                Ok(_type)
            }
            token => return Err(anyhow!("type_declaration: parse unknown {token:#?}")),
        }
    }

    fn parse_type_declaration(&mut self) -> Result<TypeDeclaration> {
        self.expect_next_token(lexer::Token::Type)?;
        self.next();

        let identifier = self.parse_identifier()?;
        let _type = self.parse_type()?;

        Ok(TypeDeclaration { _type, identifier })
    }

    fn parse_function_declaration(&mut self) -> Result<FunctionDeclaration> {
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
                self.next();
                _type = Type::Variadic(Box::new(_type));
                self.expect_next_token(lexer::Token::PClose)?;
            }

            function_arguments.push(Variable { _type, identifier })
        }

        let return_type = self.parse_type()?;
        let body = self.parse_body()?;

        Ok(FunctionDeclaration {
            body,
            return_type,
            identifier,
            arguments: function_arguments,
        })
    }

    fn parse_for(&mut self) -> Result<For> {
        self.expect_next_token(lexer::Token::For)?;
        self.next();

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

    fn parse_literal(&mut self) -> Result<Literal> {
        match self.peek_token_err(0)?.clone() {
            lexer::Token::Literal(v) => {
                self.next();
                Ok(Literal { literal: v.clone() })
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

    fn parse_call(&mut self, _type: Type) -> Result<Call> {
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

        Ok(Call { _type, arguments })
    }

    fn parse_expression_literal(&mut self) -> Result<Expression> {
        Ok(Expression::Literal(self.parse_literal()?))
    }

    fn parse_slice_init(&mut self, _type: Type) -> Result<SliceInit> {
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

        Ok(SliceInit { _type, expressions })
    }

    fn parse_struct_init(&mut self, _type: Type) -> Result<StructInit> {
        self.expect_next_token(lexer::Token::COpen)?;
        self.next();

        let mut fields = HashMap::new();

        while let Some(token) = self.peek_token(0) {
            if let lexer::Token::CClose = token {
                self.next();
                break;
            }

            let identifier = self.parse_identifier()?;
            self.expect_next_token(lexer::Token::Colon)?;
            self.next();

            let exp = self.parse_expression()?;

            self.expect_next_token(lexer::Token::Comma)?;
            self.next();

            fields.insert(identifier, exp);
        }

        Ok(StructInit { fields, _type })
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
                        expression: Box::new(self.parse_expression()?),
                        _type: match token {
                            lexer::Token::Plus => InfixType::Plus,
                            lexer::Token::Minus => InfixType::Minus,
                            _ => unreachable!(),
                        },
                    })
                }
                lexer::Token::Bang => {
                    self.next();
                    Expression::Negate(Box::new(self.parse_expression()?))
                }
                lexer::Token::Star => {
                    self.next();
                    Expression::Deref(Box::new(self.parse_expression()?))
                }
                lexer::Token::Amper => {
                    self.next();
                    Expression::Address(Box::new(self.parse_expression()?))
                }
                lexer::Token::Literal(_) => self.parse_expression_literal()?,
                lexer::Token::Struct | lexer::Token::Identifier(_) => {
                    Expression::Type(self.parse_type()?)
                }
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
                    continue;
                }
                lexer::Token::POpen => {
                    let Expression::Type(_type) = left else {
                        return Err(anyhow!("parse_expression: wrong token on POpen"));
                    };

                    left = Expression::Call(self.parse_call(_type)?);
                    continue;
                }
                lexer::Token::COpen => {
                    if let Expression::Type(_type) = left {
                        // variants:
                        // struct init
                        // slice init
                        // type init
                        match self.peek_token_err(1)? {
                            lexer::Token::CClose => {
                                left = Expression::TypeInit(TypeInit { _type });
                                self.next();
                                self.next();
                            }
                            lexer::Token::Identifier(_) | lexer::Token::Literal(_) => match self
                                .peek_token_err(2)?
                            {
                                lexer::Token::Colon => {
                                    left = Expression::StructInit(self.parse_struct_init(_type)?);
                                }
                                _ => {
                                    left = Expression::SliceInit(self.parse_slice_init(_type)?);
                                }
                            },
                            token => {
                                return Err(anyhow!("parse_expression: COpen incorrect {token:#?}"))
                            }
                        }
                        continue;
                    };
                }
                lexer::Token::Dot => {
                    self.next();
                    let lexer::Token::Identifier(identifier) = self.peek_token_err(0)?.clone()
                    else {
                        return Err(anyhow!("dot access expected identifier"));
                    };
                    self.next();
                    left = Expression::DotAccess(Box::new(DotAccess {
                        expression: left,
                        identifier: identifier.clone(),
                    }));
                    continue;
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

        Ok(VariableDeclaration {
            variable: Variable { identifier, _type },
            expression,
        })
    }
}
