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
            Expression::Address(v) => self.search_expression_address(v),
            Expression::AndOr(v) => self.search_expression_andor(v),
            Expression::Arithmetic(v) => self.search_expression_arithmetic(v),
            Expression::Compare(v) => self.search_expression_compare(v),
            Expression::Deref(v) => self.search_expression_deref(v),
            Expression::DotAccess(v) => self.search_expression_dot_access(v),
            Expression::FunctionCall(v) => self.search_expression_function_call(v),
            Expression::Index(v) => self.search_expression_index(v),
            Expression::Infix(v) => self.search_expression_infix(v),
            Expression::Literal(v) => self.search_expression_literal(v),
            Expression::Negate(v) => self.search_expression_negate(v),
            Expression::SliceInit(v) => self.search_expression_slice_init(v),
            Expression::Spread(v) => self.search_expression_spread(v),
            Expression::StructInit(v) => self.search_expression_struct_init(v),
            Expression::Type(v) => self.search_expression_type(v),
            Expression::TypeCast(v) => self.search_expression_type_cast(v),
            Expression::Variable(v) => self.search_expression_variable(v),
        }
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

    fn search_expression_function_call(&self, function_call: &FunctionCall) -> BfsRet {
        for arg in &function_call.arguments {
            return_if_some_true!(self.search_expression(arg));
        }
        BfsRet::Continue
    }

    fn search_expression_variable(&self, _variable: &Variable) -> BfsRet {
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

    fn search_expression_type_cast(&self, type_cast: &TypeCast) -> BfsRet {
        return_if_some_true!(self.search_expression(&type_cast.expression));
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
    Index(Index),
    Spread(Box<Expression>),
    Type(Type),
    DotAccess(Box<DotAccess>),
    SliceInit(SliceInit),
    StructInit(StructInit),
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
struct LexerNavigator<'a> {
    i: usize,
    tokens: &'a [lexer::Token],
}

impl<'a> LexerNavigator<'a> {
    fn new(tokens: &'a [lexer::Token], i: usize) -> Self {
        Self { i, tokens }
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

struct TypeDeclarationParser<'a, 'b> {
    lexer_navigator: &'b mut LexerNavigator<'a>,
}

#[derive(Debug, Clone)]
pub struct TypeDeclarations(pub HashMap<String, Type>);

impl TypeDeclarations {
    fn resolve_alias<'a>(&'a self, mut _type: &'a Type) -> Result<&'a Type> {
        loop {
            match _type {
                Type::Alias(alias) => {
                    _type = self
                        .0
                        .get(alias)
                        .ok_or(anyhow!("defer_alias: type not found"))?;

                    // this is for builtin types (int / uint)
                    if let Type::Alias(alias2) = _type {
                        if alias2 == alias {
                            return Err(anyhow!("defer_alias: reached end"));
                        }
                    }
                }
                _type => return Ok(_type),
            }
        }
    }
}

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
        match self.lexer_navigator.peek_token_err(0)?.clone() {
            lexer::Token::Star => {
                self.lexer_navigator.next();
                Ok(Type::Address(Box::new(self.parse_type()?)))
            }
            lexer::Token::Struct => Ok(Type::Struct(self.parse_type_struct()?)),
            lexer::Token::Identifier(alias) => {
                self.lexer_navigator.next();

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

#[derive(Debug)]
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
    pub functions: Vec<Function>,
}

impl Ast {
    pub fn new(tokens: &[lexer::Token]) -> Result<Self> {
        let type_declarations =
            TypeDeclarationParser::new(&mut LexerNavigator::new(tokens, 0)).all()?;
        let function_declarations =
            FunctionDeclarationParser::new(&mut LexerNavigator::new(tokens, 0)).all()?;

        let mut function_declarations_sorted = function_declarations.0.iter().collect::<Vec<_>>();
        function_declarations_sorted.sort_by(|a, b| a.0.cmp(&b.0));

        let mut functions = Vec::new();

        for (_, function) in function_declarations_sorted {
            let mut variables = AstVariables::new();

            for arg in &function.function.arguments {
                variables.push_variable(arg.clone());
            }

            let mut token_parser = TokenParser::new(
                LexerNavigator::new(tokens, function.i),
                &function_declarations,
                &type_declarations,
                variables,
            );
            functions.push(Function {
                body: Some(token_parser.parse_body()?),
                ..function.function.clone()
            });
        }

        Ok(Self {
            type_declarations,
            functions,
        })
    }
}

#[derive(Debug)]
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

struct TokenParser<'a, 'b, 'c> {
    lexer_navigator: LexerNavigator<'a>,
    variables: AstVariables,
    function_declarations: &'b FunctionDeclarations,
    type_declarations: &'c TypeDeclarations,
}

#[derive(Debug, Clone)]
pub struct Variable {
    pub _type: Type,
    pub identifier: String,
}

impl<'a, 'b, 'c> TokenParser<'a, 'b, 'c> {
    fn new(
        lexer_navigator: LexerNavigator<'a>,
        function_declarations: &'b FunctionDeclarations,
        type_declarations: &'c TypeDeclarations,
        variables: AstVariables,
    ) -> Self {
        Self {
            variables,
            lexer_navigator,
            type_declarations,
            function_declarations,
        }
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

    fn parse_function_call(&mut self, identifier: String) -> Result<FunctionCall> {
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

    fn parse_expression_type_cast(&mut self, _type: Type) -> Result<Expression> {
        self.lexer_navigator
            .expect_next_token(lexer::Token::POpen)?;
        self.lexer_navigator.next();

        let exp = self.parse_expression()?;
        self.lexer_navigator
            .expect_next_token(lexer::Token::PClose)?;
        self.lexer_navigator.next();

        Ok(Expression::TypeCast(Box::new(TypeCast {
            _type,
            expression: exp,
        })))
    }

    fn parse_expression_type(&mut self) -> Result<Expression> {
        let _type = self.parse_type()?;
        if let lexer::Token::POpen = self.lexer_navigator.peek_token_err(0)? {
            return self.parse_expression_type_cast(_type);
        }

        Ok(Expression::Type(_type))
    }

    fn parse_expression_identifier(&mut self) -> Result<Expression> {
        let lexer::Token::Identifier(identifier) = self.lexer_navigator.peek_token_err(0)?.clone()
        else {
            panic!("incorrect usage of parse_expression_identifier");
        };

        let is_type = self.type_declarations.0.get(&identifier).is_some();
        match is_type {
            true => self.parse_expression_type(),
            false => {
                self.lexer_navigator.next();

                if let lexer::Token::POpen = self.lexer_navigator.peek_token_err(0)? {
                    return Ok(Expression::FunctionCall(
                        self.parse_function_call(identifier)?,
                    ));
                }

                Ok(Expression::Variable(
                    self.variables
                        .get_variable(&identifier)
                        .ok_or(anyhow!(
                            "parse_expression_identifier: identifier variable {} not found",
                            identifier,
                        ))?
                        .clone(),
                ))
            }
        }
    }

    fn parse_expression_literal(&mut self) -> Result<Expression> {
        Ok(Expression::Literal(self.parse_literal()?))
    }

    fn parse_slice_init(&mut self, _type: Type) -> Result<SliceInit> {
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

        Ok(SliceInit { _type, expressions })
    }

    fn parse_struct_init(&mut self, _type: Type) -> Result<StructInit> {
        self.lexer_navigator
            .expect_next_token(lexer::Token::COpen)?;
        self.lexer_navigator.next();

        let mut fields = HashMap::new();

        while let Some(token) = self.lexer_navigator.peek_token(0) {
            if let lexer::Token::CClose = token {
                self.lexer_navigator.next();
                break;
            }

            let identifier = self.parse_identifier()?;
            self.lexer_navigator
                .expect_next_token(lexer::Token::Colon)?;
            self.lexer_navigator.next();

            let exp = self.parse_expression()?;

            self.lexer_navigator
                .expect_next_token(lexer::Token::Comma)?;
            self.lexer_navigator.next();

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
                        expression: Box::new(self.parse_expression()?),
                        _type: match token {
                            lexer::Token::Plus => InfixType::Plus,
                            lexer::Token::Minus => InfixType::Minus,
                            _ => unreachable!(),
                        },
                    })
                }
                lexer::Token::Bang => {
                    self.lexer_navigator.next();
                    Expression::Negate(Box::new(self.parse_expression()?))
                }
                lexer::Token::Star => {
                    self.lexer_navigator.next();
                    Expression::Deref(Box::new(self.parse_expression()?))
                }
                lexer::Token::Amper => {
                    self.lexer_navigator.next();
                    Expression::Address(Box::new(self.parse_expression()?))
                }
                lexer::Token::Identifier(_) => self.parse_expression_identifier()?,
                lexer::Token::Literal(_) => self.parse_expression_literal()?,
                lexer::Token::Struct => self.parse_expression_type()?,
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
                    continue;
                }
                lexer::Token::COpen => {
                    if let Expression::Type(_type) = &left {
                        match self
                            .type_declarations
                            .resolve_alias(&_type)
                            .map_err(|_| anyhow!("cant initialize this type {_type:#?}"))?
                        {
                            Type::Slice(_) => {
                                left = Expression::SliceInit(self.parse_slice_init(_type.clone())?)
                            }
                            Type::Struct(_) => {
                                left =
                                    Expression::StructInit(self.parse_struct_init(_type.clone())?)
                            }
                            _ => unreachable!(),
                        }
                        continue;
                    };
                }
                lexer::Token::Dot => {
                    self.lexer_navigator.next();
                    let lexer::Token::Identifier(identifier) =
                        self.lexer_navigator.peek_token_err(0)?.clone()
                    else {
                        return Err(anyhow!("dot access expected identifier"));
                    };
                    self.lexer_navigator.next();
                    left = Expression::DotAccess(Box::new(DotAccess {
                        expression: left,
                        identifier: identifier.clone(),
                    }));
                    continue;
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
