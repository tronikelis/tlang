use std::collections::HashMap;

use anyhow::{anyhow, Result};

use crate::lexer;

pub enum DfsRet {
    Found,
    Continue,
    Break,
}

macro_rules! dfsret_return_if {
    ($v:expr) => {
        match $v {
            $crate::ast::DfsRet::Found => {
                return $crate::ast::DfsRet::Found;
            }
            $crate::ast::DfsRet::Break => {
                return $crate::ast::DfsRet::Break;
            }
            $crate::ast::DfsRet::Continue => {}
        }
    };
}
pub(crate) use dfsret_return_if;

macro_rules! dfsret_search_body {
    ($self:ident, $body:expr) => {{
        for node in $body {
            match $self.search_node(node) {
                $crate::ast::DfsRet::Found => {
                    return $crate::ast::DfsRet::Found;
                }
                $crate::ast::DfsRet::Break => {
                    break;
                }
                $crate::ast::DfsRet::Continue => {}
            }
        }

        $crate::ast::DfsRet::Continue
    }};
}
pub(crate) use dfsret_search_body;

pub trait Dfs<'a> {
    fn search_body(&self, body: impl Iterator<Item = &'a Node>) -> DfsRet {
        dfsret_search_body!(self, body)
    }

    fn search_node(&self, node: &'a Node) -> DfsRet {
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

        DfsRet::Continue
    }

    fn search_node_if(&self, _if: &'a If) -> DfsRet {
        dfsret_return_if!(self.search_expression(&_if.expression));
        dfsret_return_if!(self.search_body(_if.body.iter()));

        for else_if in &_if.elseif {
            dfsret_return_if!(self.search_expression(&else_if.expression));
            dfsret_return_if!(self.search_body(else_if.body.iter()));
        }

        if let Some(_else) = &_if._else {
            dfsret_return_if!(self.search_body(_else.body.iter()));
        }

        DfsRet::Continue
    }

    fn search_node_for(&self, _for: &'a For) -> DfsRet {
        if let Some(node) = &_for.initializer {
            dfsret_return_if!(self.search_node(node));
        }
        if let Some(exp) = &_for.expression {
            dfsret_return_if!(self.search_expression(exp));
        }
        if let Some(node) = &_for.after_each {
            dfsret_return_if!(self.search_node(node));
        }

        self.search_body(_for.body.iter())
    }

    fn search_node_variable_declaration(&self, declaration: &'a VariableDeclaration) -> DfsRet {
        dfsret_return_if!(self.search_expression(&declaration.expression));
        DfsRet::Continue
    }

    fn search_node_variable_assignment(&self, assignment: &'a VariableAssignment) -> DfsRet {
        dfsret_return_if!(self.search_expression(&assignment.var));
        dfsret_return_if!(self.search_expression(&assignment.expression));
        DfsRet::Continue
    }

    fn search_expression(&self, exp: &'a Expression) -> DfsRet {
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
            Expression::Closure(v) => self.search_expression_closure(v),
            Expression::Nil => DfsRet::Continue,
        }
    }

    fn search_expression_closure(&self, closure: &'a Closure) -> DfsRet {
        self.search_body(closure.body.iter())
    }

    fn search_expression_type_init(&self, _type_init: &TypeInit) -> DfsRet {
        DfsRet::Continue
    }

    fn search_expression_call(&self, call: &'a Call) -> DfsRet {
        for exp in &call.arguments {
            dfsret_return_if!(self.search_expression(exp));
        }
        DfsRet::Continue
    }

    fn search_expression_address(&self, exp: &'a Expression) -> DfsRet {
        dfsret_return_if!(self.search_expression(exp));
        DfsRet::Continue
    }

    fn search_expression_andor(&self, andor: &'a AndOr) -> DfsRet {
        dfsret_return_if!(self.search_expression(&andor.left));
        dfsret_return_if!(self.search_expression(&andor.right));
        DfsRet::Continue
    }

    fn search_expression_deref(&self, exp: &'a Expression) -> DfsRet {
        dfsret_return_if!(self.search_expression(exp));
        DfsRet::Continue
    }

    fn search_expression_type(&self, _type: &Type) -> DfsRet {
        DfsRet::Continue
    }

    fn search_expression_infix(&self, infix: &'a Infix) -> DfsRet {
        dfsret_return_if!(self.search_expression(&infix.expression));
        DfsRet::Continue
    }

    fn search_expression_index(&self, index: &'a Index) -> DfsRet {
        dfsret_return_if!(self.search_expression(&index.expression));
        dfsret_return_if!(self.search_expression(&index.var));
        DfsRet::Continue
    }

    fn search_expression_negate(&self, exp: &'a Expression) -> DfsRet {
        dfsret_return_if!(self.search_expression(exp));
        DfsRet::Continue
    }

    fn search_expression_spread(&self, exp: &'a Expression) -> DfsRet {
        dfsret_return_if!(self.search_expression(exp));
        DfsRet::Continue
    }

    fn search_expression_literal(&self, _literal: &Literal) -> DfsRet {
        DfsRet::Continue
    }

    fn search_expression_compare(&self, compare: &'a Compare) -> DfsRet {
        dfsret_return_if!(self.search_expression(&compare.left));
        dfsret_return_if!(self.search_expression(&compare.right));
        DfsRet::Continue
    }

    fn search_expression_arithmetic(&self, arithmetic: &'a Arithmetic) -> DfsRet {
        dfsret_return_if!(self.search_expression(&arithmetic.left));
        dfsret_return_if!(self.search_expression(&arithmetic.right));
        DfsRet::Continue
    }

    fn search_expression_dot_access(&self, dot_access: &'a DotAccess) -> DfsRet {
        dfsret_return_if!(self.search_expression(&dot_access.expression));
        DfsRet::Continue
    }

    fn search_expression_slice_init(&self, slice_init: &'a SliceInit) -> DfsRet {
        for exp in &slice_init.expressions {
            dfsret_return_if!(self.search_expression(exp));
        }
        DfsRet::Continue
    }

    fn search_expression_struct_init(&self, struct_init: &'a StructInit) -> DfsRet {
        for exp in struct_init.fields.values() {
            dfsret_return_if!(self.search_expression(exp));
        }
        DfsRet::Continue
    }
}

pub trait DfsMut<'a> {
    fn search_body(&mut self, body: impl Iterator<Item = &'a Node>) -> DfsRet {
        dfsret_search_body!(self, body)
    }

    fn search_node(&mut self, node: &'a Node) -> DfsRet {
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

        DfsRet::Continue
    }

    fn search_node_if(&mut self, _if: &'a If) -> DfsRet {
        dfsret_return_if!(self.search_expression(&_if.expression));
        dfsret_return_if!(self.search_body(_if.body.iter()));

        for else_if in &_if.elseif {
            dfsret_return_if!(self.search_expression(&else_if.expression));
            dfsret_return_if!(self.search_body(else_if.body.iter()));
        }

        if let Some(_else) = &_if._else {
            dfsret_return_if!(self.search_body(_else.body.iter()));
        }

        DfsRet::Continue
    }

    fn search_node_for(&mut self, _for: &'a For) -> DfsRet {
        if let Some(node) = &_for.initializer {
            dfsret_return_if!(self.search_node(node));
        }
        if let Some(exp) = &_for.expression {
            dfsret_return_if!(self.search_expression(exp));
        }
        if let Some(node) = &_for.after_each {
            dfsret_return_if!(self.search_node(node));
        }

        self.search_body(_for.body.iter())
    }

    fn search_node_variable_declaration(&mut self, declaration: &'a VariableDeclaration) -> DfsRet {
        dfsret_return_if!(self.search_expression(&declaration.expression));
        DfsRet::Continue
    }

    fn search_node_variable_assignment(&mut self, assignment: &'a VariableAssignment) -> DfsRet {
        dfsret_return_if!(self.search_expression(&assignment.var));
        dfsret_return_if!(self.search_expression(&assignment.expression));
        DfsRet::Continue
    }

    fn search_expression(&mut self, exp: &'a Expression) -> DfsRet {
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
            Expression::Closure(v) => self.search_expression_closure(v),
            Expression::Nil => DfsRet::Continue,
        }
    }

    fn search_expression_closure(&mut self, closure: &'a Closure) -> DfsRet {
        self.search_body(closure.body.iter())
    }

    fn search_expression_type_init(&mut self, _type_init: &TypeInit) -> DfsRet {
        DfsRet::Continue
    }

    fn search_expression_call(&mut self, call: &'a Call) -> DfsRet {
        for exp in &call.arguments {
            dfsret_return_if!(self.search_expression(exp));
        }
        DfsRet::Continue
    }

    fn search_expression_address(&mut self, exp: &'a Expression) -> DfsRet {
        dfsret_return_if!(self.search_expression(exp));
        DfsRet::Continue
    }

    fn search_expression_andor(&mut self, andor: &'a AndOr) -> DfsRet {
        dfsret_return_if!(self.search_expression(&andor.left));
        dfsret_return_if!(self.search_expression(&andor.right));
        DfsRet::Continue
    }

    fn search_expression_deref(&mut self, exp: &'a Expression) -> DfsRet {
        dfsret_return_if!(self.search_expression(exp));
        DfsRet::Continue
    }

    fn search_expression_type(&mut self, _type: &Type) -> DfsRet {
        DfsRet::Continue
    }

    fn search_expression_infix(&mut self, infix: &'a Infix) -> DfsRet {
        dfsret_return_if!(self.search_expression(&infix.expression));
        DfsRet::Continue
    }

    fn search_expression_index(&mut self, index: &'a Index) -> DfsRet {
        dfsret_return_if!(self.search_expression(&index.expression));
        dfsret_return_if!(self.search_expression(&index.var));
        DfsRet::Continue
    }

    fn search_expression_negate(&mut self, exp: &'a Expression) -> DfsRet {
        dfsret_return_if!(self.search_expression(exp));
        DfsRet::Continue
    }

    fn search_expression_spread(&mut self, exp: &'a Expression) -> DfsRet {
        dfsret_return_if!(self.search_expression(exp));
        DfsRet::Continue
    }

    fn search_expression_literal(&mut self, _literal: &Literal) -> DfsRet {
        DfsRet::Continue
    }

    fn search_expression_compare(&mut self, compare: &'a Compare) -> DfsRet {
        dfsret_return_if!(self.search_expression(&compare.left));
        dfsret_return_if!(self.search_expression(&compare.right));
        DfsRet::Continue
    }

    fn search_expression_arithmetic(&mut self, arithmetic: &'a Arithmetic) -> DfsRet {
        dfsret_return_if!(self.search_expression(&arithmetic.left));
        dfsret_return_if!(self.search_expression(&arithmetic.right));
        DfsRet::Continue
    }

    fn search_expression_dot_access(&mut self, dot_access: &'a DotAccess) -> DfsRet {
        dfsret_return_if!(self.search_expression(&dot_access.expression));
        DfsRet::Continue
    }

    fn search_expression_slice_init(&mut self, slice_init: &'a SliceInit) -> DfsRet {
        for exp in &slice_init.expressions {
            dfsret_return_if!(self.search_expression(exp));
        }
        DfsRet::Continue
    }

    fn search_expression_struct_init(&mut self, struct_init: &'a StructInit) -> DfsRet {
        for exp in struct_init.fields.values() {
            dfsret_return_if!(self.search_expression(exp));
        }
        DfsRet::Continue
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

impl DotAccess {
    pub fn deepest(&self) -> &Self {
        let mut curr = self;
        while let Expression::DotAccess(dot_access) = &curr.expression {
            curr = dot_access;
        }
        curr
    }
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
pub struct Closure {
    pub _type: Type,
    pub body: Vec<Node>,
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
    Closure(Closure),
    Nil,
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
pub struct TypeClosure {
    arguments: Vec<(String, Type)>,
    return_type: Type,
}

#[derive(Debug, Clone)]
pub enum Type {
    Alias(String),
    Struct(TypeStruct),
    Slice(Box<Type>),
    Variadic(Box<Type>),
    Address(Box<Type>),
    Closure(Box<TypeClosure>),
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

#[derive(Debug, Clone)]
pub struct Variable {
    pub _type: Type,
    pub identifier: String,
}

struct TokenIter<'a> {
    tokens: &'a [lexer::Token],
    i: usize,
}

impl<'a> TokenIter<'a> {
    fn new(tokens: &'a [lexer::Token]) -> Self {
        Self { tokens, i: 0 }
    }

    fn is_newline_i(&self, i: usize) -> bool {
        if let Some(token) = self.tokens.get(i) {
            return *token == lexer::Token::NL;
        } else {
            false
        }
    }

    fn is_newline(&self) -> bool {
        self.is_newline_i(self.i)
    }

    fn expect(&self, token: lexer::Token) -> Result<()> {
        if *self.peek_err(0)? == token {
            Ok(())
        } else {
            Err(anyhow!(
                "expect: expected {token:#?}, got: {:#?}",
                self.peek_err(0)?
            ))
        }
    }

    fn peek_err(&self, n: usize) -> Result<&lexer::Token> {
        self.peek(n).ok_or(anyhow!("peek_err: expected Some"))
    }

    fn skip_nl(&self, i: &mut usize) {
        while self.is_newline_i(*i) {
            *i += 1;
        }
    }

    fn next_skip_nl(&mut self) {
        let mut i = self.i;
        self.skip_nl(&mut i);
        self.i = i;
    }

    fn peek(&self, n: usize) -> Option<&lexer::Token> {
        let mut i = self.i;
        self.skip_nl(&mut i);

        for _ in 0..n {
            i += 1;
            self.skip_nl(&mut i);
        }

        self.tokens.get(i)
    }

    fn next(&mut self) {
        self.next_skip_nl();
        self.i += 1;
    }
}

struct TokenParser<'a> {
    iter: TokenIter<'a>,
}

impl<'a> TokenParser<'a> {
    fn new(tokens: &'a [lexer::Token]) -> Self {
        Self {
            iter: TokenIter::new(tokens),
        }
    }

    fn parse(mut self) -> Result<Vec<Declaration>> {
        let mut declarations = Vec::new();
        while let Some(token) = self.iter.peek(0) {
            declarations.push(match token {
                lexer::Token::Type => Declaration::Type(self.parse_type_declaration()?),
                lexer::Token::Function => Declaration::Function(self.parse_function_declaration()?),
                token => return Err(anyhow!("parse: unknown token {token:#?}")),
            });
        }

        Ok(declarations)
    }

    fn parse_type_struct(&mut self) -> Result<TypeStruct> {
        self.iter.expect(lexer::Token::Struct)?;
        self.iter.next();

        self.iter.expect(lexer::Token::COpen)?;
        self.iter.next();

        let mut fields: Vec<(String, Type)> = Vec::new();

        while *self.iter.peek_err(0)? != lexer::Token::CClose {
            let field_identifier = self.parse_identifier()?;
            let field_type = self.parse_type()?;
            fields.push((field_identifier, field_type));
        }

        self.iter.next();

        Ok(TypeStruct { fields })
    }

    fn parse_type(&mut self) -> Result<Type> {
        match self.iter.peek_err(0)?.clone() {
            lexer::Token::Star => {
                self.iter.next();
                Ok(Type::Address(Box::new(self.parse_type()?)))
            }
            lexer::Token::Struct => Ok(Type::Struct(self.parse_type_struct()?)),
            lexer::Token::Identifier(alias) => {
                self.iter.next();

                let mut _type = Type::Alias(alias.clone());

                while let Some(token) = self.iter.peek(0) {
                    if *token != lexer::Token::BOpen {
                        break;
                    }
                    if *self.iter.peek_err(1)? != lexer::Token::BClose {
                        break;
                    }

                    self.iter.next();
                    self.iter.next();

                    _type = Type::Slice(Box::new(_type));
                }

                Ok(_type)
            }
            lexer::Token::Function => {
                self.iter.next();

                self.iter.expect(lexer::Token::POpen)?;
                self.iter.next();

                let mut arguments = Vec::new();

                while let Some(token) = self.iter.peek(0) {
                    match token {
                        lexer::Token::PClose => {
                            self.iter.next();
                            break;
                        }
                        lexer::Token::Comma => {
                            self.iter.next();
                        }
                        _ => {}
                    }

                    let iden = self.parse_identifier()?;
                    let _type = self.parse_type()?;

                    arguments.push((iden, _type));
                }

                let return_type = self.parse_type()?;

                Ok(Type::Closure(Box::new(TypeClosure {
                    return_type,
                    arguments,
                })))
            }
            token => return Err(anyhow!("parse_type: parse unknown {token:#?}")),
        }
    }

    fn parse_type_declaration(&mut self) -> Result<TypeDeclaration> {
        self.iter.expect(lexer::Token::Type)?;
        self.iter.next();

        let identifier = self.parse_identifier()?;
        let _type = self.parse_type()?;

        Ok(TypeDeclaration { _type, identifier })
    }

    fn parse_function_declaration(&mut self) -> Result<FunctionDeclaration> {
        self.iter.expect(lexer::Token::Function)?;
        self.iter.next();

        let identifier = self.parse_identifier()?;

        self.iter.expect(lexer::Token::POpen)?;
        self.iter.next();

        let mut function_arguments: Vec<Variable> = Vec::new();

        while let Some(token) = self.iter.peek(0) {
            match token {
                lexer::Token::PClose => {
                    self.iter.next();
                    break;
                }
                lexer::Token::Comma => {
                    self.iter.next();
                }
                _ => {}
            }

            let identifier = self.parse_identifier()?;
            let mut _type = self.parse_type()?;

            if let lexer::Token::Dot3 = self.iter.peek_err(0)? {
                self.iter.next();
                _type = Type::Variadic(Box::new(_type));
                self.iter.expect(lexer::Token::PClose)?;
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
        self.iter.expect(lexer::Token::For)?;
        self.iter.next();

        // for {}
        if let lexer::Token::COpen = self.iter.peek_err(0)? {
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

        self.iter.expect(lexer::Token::Semicolon)?;
        self.iter.next();

        let expression = self.parse_expression()?;

        self.iter.expect(lexer::Token::Semicolon)?;
        self.iter.next();

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

        match self.iter.peek_err(0)? {
            lexer::Token::Equals => {
                self.iter.next();
                Ok(Node::VariableAssignment(VariableAssignment {
                    var: exp,
                    expression: self.parse_expression()?,
                }))
            }
            lexer::Token::PlusPlus | lexer::Token::MinusMinus => {
                let token = self.iter.peek_err(0)?.clone();
                self.iter.next();

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
        match self.iter.peek_err(0)? {
            lexer::Token::Debug => {
                self.iter.next();
                Ok(Node::Debug)
            }
            lexer::Token::Let => Ok(Node::VariableDeclaration(
                self.parse_variable_declaration()?,
            )),
            lexer::Token::Return => {
                self.iter.next();
                Ok(Node::Return(self.parse_expression().ok()))
            }
            lexer::Token::If => Ok(Node::If(self.parse_if()?)),
            lexer::Token::For => Ok(Node::For(self.parse_for()?)),
            lexer::Token::Break => {
                self.iter.next();
                Ok(Node::Break)
            }
            lexer::Token::Continue => {
                self.iter.next();
                Ok(Node::Continue)
            }
            _ => self.parse_token_else(),
        }
    }

    fn parse_if(&mut self) -> Result<If> {
        match self.iter.peek_err(0)? {
            lexer::Token::If | lexer::Token::ElseIf => {}
            _ => return Err(anyhow!("parse_if: unknown token")),
        }
        self.iter.next();

        let expression = self.parse_expression()?;
        let body = self.parse_body()?;

        let mut elseif = Vec::<ElseIf>::new();
        while let lexer::Token::ElseIf = self.iter.peek_err(0)? {
            self.iter.next();
            elseif.push(ElseIf {
                expression: self.parse_expression()?,
                body: self.parse_body()?,
            });
        }

        let mut _else = None;
        if let lexer::Token::Else = self.iter.peek_err(0)? {
            self.iter.next();
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

        self.iter.expect(lexer::Token::COpen)?;
        self.iter.next();

        while let Some(token) = self.iter.peek(0) {
            if let lexer::Token::CClose = token {
                self.iter.next();
                break;
            }

            nodes.push(self.parse_token()?);
        }

        Ok(nodes)
    }

    fn parse_literal(&mut self) -> Result<Literal> {
        match self.iter.peek_err(0)?.clone() {
            lexer::Token::Literal(v) => {
                self.iter.next();
                Ok(Literal { literal: v.clone() })
            }
            _ => Err(anyhow!("parse_literal: expected Literal")),
        }
    }

    fn parse_identifier(&mut self) -> Result<String> {
        match self.iter.peek_err(0)?.clone() {
            lexer::Token::Identifier(v) => {
                self.iter.next();
                Ok(v.clone())
            }
            _ => Err(anyhow!("parse_identifier: expected Identifier")),
        }
    }

    fn parse_call(&mut self, _type: Type) -> Result<Call> {
        self.iter.expect(lexer::Token::POpen)?;
        self.iter.next();

        let mut arguments = Vec::new();

        while let Some(token) = self.iter.peek(0) {
            match token {
                lexer::Token::PClose => {
                    self.iter.next();
                    break;
                }
                lexer::Token::Comma => {
                    self.iter.next();
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
        self.iter.expect(lexer::Token::COpen)?;
        self.iter.next();

        let mut expressions = Vec::new();

        while let Some(v) = self.iter.peek(0) {
            if let lexer::Token::CClose = v {
                self.iter.next();
                break;
            }

            expressions.push(self.parse_expression()?);

            if *self.iter.peek_err(0)? != lexer::Token::CClose {
                self.iter.expect(lexer::Token::Comma)?;
                self.iter.next();
            }
        }

        Ok(SliceInit { _type, expressions })
    }

    fn parse_struct_init(&mut self, _type: Type) -> Result<StructInit> {
        self.iter.expect(lexer::Token::COpen)?;
        self.iter.next();

        let mut fields = HashMap::new();

        while let Some(token) = self.iter.peek(0) {
            if let lexer::Token::CClose = token {
                self.iter.next();
                break;
            }

            let identifier = self.parse_identifier()?;
            self.iter.expect(lexer::Token::Colon)?;
            self.iter.next();

            let exp = self.parse_expression()?;

            self.iter.expect(lexer::Token::Comma)?;
            self.iter.next();

            fields.insert(identifier, exp);
        }

        Ok(StructInit { fields, _type })
    }

    fn parse_closure(&mut self) -> Result<Closure> {
        let _type = self.parse_type()?;
        let body = self.parse_body()?;

        Ok(Closure { _type, body })
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
            let token = self.iter.peek_err(0)?.clone();
            match token {
                lexer::Token::POpen => {
                    self.iter.next();
                    let exp = self.parse_expression()?;
                    self.iter.expect(lexer::Token::PClose)?;
                    self.iter.next();
                    exp
                }
                lexer::Token::Plus | lexer::Token::Minus => {
                    self.iter.next();
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
                    self.iter.next();
                    Expression::Negate(Box::new(self.parse_expression()?))
                }
                lexer::Token::Star => {
                    self.iter.next();
                    Expression::Deref(Box::new(self.parse_expression()?))
                }
                lexer::Token::Amper => {
                    self.iter.next();
                    Expression::Address(Box::new(self.parse_expression()?))
                }
                lexer::Token::Literal(_) => self.parse_expression_literal()?,
                lexer::Token::Struct | lexer::Token::Identifier(_) => {
                    Expression::Type(self.parse_type()?)
                }
                lexer::Token::Nil => {
                    self.iter.next();
                    Expression::Nil
                }
                lexer::Token::Function => Expression::Closure(self.parse_closure()?),
                token => return Err(anyhow!("parse_expression: incorrect token {token:#?}")),
            }
        };

        loop {
            if self.iter.is_newline() {
                break;
            }

            let token = self.iter.peek_err(0)?.clone();
            match token {
                lexer::Token::BOpen => {
                    self.iter.next();
                    left = Expression::Index(Index {
                        var: Box::new(left),
                        expression: Box::new(self.parse_expression()?),
                    });
                    self.iter.expect(lexer::Token::BClose)?;
                    self.iter.next();
                    continue;
                }
                lexer::Token::Dot3 => {
                    left = Expression::Spread(Box::new(left));
                    self.iter.next();
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
                        match self.iter.peek_err(1)? {
                            lexer::Token::CClose => {
                                left = Expression::TypeInit(TypeInit { _type });
                                self.iter.next();
                                self.iter.next();
                            }
                            lexer::Token::Identifier(_) | lexer::Token::Literal(_) => {
                                match self.iter.peek_err(2)? {
                                    lexer::Token::Colon => {
                                        left =
                                            Expression::StructInit(self.parse_struct_init(_type)?);
                                    }
                                    _ => {
                                        left = Expression::SliceInit(self.parse_slice_init(_type)?);
                                    }
                                }
                            }
                            token => {
                                return Err(anyhow!("parse_expression: COpen incorrect {token:#?}"))
                            }
                        }
                        continue;
                    };
                }
                lexer::Token::Dot => {
                    self.iter.next();
                    let lexer::Token::Identifier(identifier) = self.iter.peek_err(0)?.clone()
                    else {
                        return Err(anyhow!("dot access expected identifier"));
                    };
                    self.iter.next();
                    left = Expression::DotAccess(Box::new(DotAccess {
                        expression: left,
                        identifier: identifier.clone(),
                    }));
                    continue;
                }
                _ => {}
            }

            let token = self.iter.peek_err(0)?.clone();
            let (l_bp, r_bp) = match Self::pratt_binding_power(&token) {
                Some(v) => v,
                None => break,
            };

            if l_bp < min_bp {
                break;
            }
            self.iter.next();
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
        self.iter.expect(lexer::Token::Let)?;
        self.iter.next();

        let identifier = self.parse_identifier()?;

        let _type = self.parse_type()?;

        self.iter.expect(lexer::Token::Equals)?;
        self.iter.next();

        let expression = self.parse_expression()?;

        Ok(VariableDeclaration {
            variable: Variable { identifier, _type },
            expression,
        })
    }
}
