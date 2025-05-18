use anyhow::{anyhow, Result};
use std::collections::HashMap;

use crate::lexer;

#[derive(Debug, Clone, PartialEq)]
pub struct Type {
    pub size: usize,
    pub _type: lexer::Type,
}

pub const INT: Type = Type {
    size: size_of::<isize>(),
    _type: lexer::Type::Int,
};

pub const VOID: Type = Type {
    size: 0,
    _type: lexer::Type::Void,
};

#[derive(Debug, Clone)]
pub struct VariableDeclaration {
    pub identifier: String,
    pub expression: Expression,
    pub _type: Type,
}

#[derive(Debug, Clone)]
pub struct FunctionArgument {
    pub identifier: String,
    pub _type: Type,
}

#[derive(Debug, Clone)]
pub struct Function {
    pub identifier: String,
    pub arguments: Vec<FunctionArgument>,
    pub return_type: Type,
    pub body: Option<Vec<Node>>,
}

#[derive(Debug, Clone)]
pub struct Addition {
    pub left: Expression,
    pub right: Expression,
}

#[derive(Debug, Clone)]
pub struct FunctionCall {
    pub function: Function,
    pub arguments: Vec<Expression>,
}

#[derive(Debug, Clone)]
pub enum Expression {
    Literal(lexer::Literal),
    Identifier(String),
    Addition(Box<Addition>),
    FunctionCall(FunctionCall),
}

#[derive(Debug, Clone)]
pub struct VariableAssignment {
    pub variable: ContextVariable,
    pub expression: Expression,
}

#[derive(Debug, Clone)]
pub enum Node {
    VariableDeclaration(VariableDeclaration),
    VariableAssignment(VariableAssignment),
    Return(Option<Expression>),
    FunctionCall(FunctionCall),
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
pub struct ContextVariable {
    pub _type: Type,
    pub identifier: String,
}

struct TokenParserContext {
    functions: HashMap<String, Function>,
    variables: HashMap<String, ContextVariable>,
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
            match token {
                lexer::Token::Function => {
                    let function = temp_parser.parse_function_declaration()?;
                    declarations.insert(function.identifier.clone(), function);
                }
                _ => {}
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

    fn parse_token(&mut self) -> Result<Node> {
        match self.peek_token_err(0)? {
            lexer::Token::Let => Ok(Node::VariableDeclaration(
                self.parse_variable_declaration()?,
            )),
            lexer::Token::Return => {
                self.next();
                Ok(Node::Return(self.parse_expression().ok()))
            }
            lexer::Token::Identifier(_) => match self.peek_token_err(1)? {
                lexer::Token::Equals => {
                    Ok(Node::VariableAssignment(self.parse_variable_assignment()?))
                }
                lexer::Token::POpen => Ok(Node::FunctionCall(self.parse_function_call()?)),
                _ => return Err(anyhow!("parse_token: token not supported")),
            },
            token => return Err(anyhow!("parse_token: token not supported {token:#?}")),
        }
    }

    fn parse_variable_assignment(&mut self) -> Result<VariableAssignment> {
        let identifier = self.parse_identifier()?;

        self.expect_next_token(lexer::Token::Equals)?;
        self.next();

        let expression = self.parse_expression()?;

        Ok(VariableAssignment {
            variable: self
                .context
                .variables
                .get(&identifier)
                .ok_or(anyhow!("parse_variable_assignment: variable not found"))?
                .clone(),
            expression,
        })
    }

    fn parse_block(&mut self) -> Result<Vec<Node>> {
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

    fn parse_literal(&mut self) -> Result<lexer::Literal> {
        match self.peek_token_err(0)?.clone() {
            lexer::Token::Literal(v) => {
                self.next();
                Ok(v.clone())
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

    fn parse_type(&mut self) -> Result<Type> {
        match self.peek_token_err(0)?.clone() {
            lexer::Token::Type(v) => {
                self.next();

                let size = match &v {
                    lexer::Type::Void => 0,
                    lexer::Type::Int => size_of::<isize>(),
                };

                Ok(Type {
                    size,
                    _type: v.clone(),
                })
            }
            _ => Err(anyhow!("parse_type: expected Type")),
        }
    }

    fn parse_function_declaration(&mut self) -> Result<Function> {
        self.expect_next_token(lexer::Token::Function)?;
        self.next();

        let identifier = self.parse_identifier()?;

        self.expect_next_token(lexer::Token::POpen)?;
        self.next();

        let mut function_arguments: Vec<FunctionArgument> = Vec::new();

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

            function_arguments.push(FunctionArgument {
                identifier: self.parse_identifier()?,
                _type: self.parse_type()?,
            });
        }

        let return_type = self.parse_type()?;

        for v in &function_arguments {
            self.context.variables.insert(
                v.identifier.clone(),
                ContextVariable {
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
        let body = self.parse_block()?;

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

    fn parse_expression_identifier(&mut self) -> Result<Expression> {
        // follow a function call
        if let lexer::Token::POpen = self.peek_token_err(1)? {
            return Ok(Expression::FunctionCall(self.parse_function_call()?));
        }

        return Ok(Expression::Identifier(self.parse_identifier()?));
    }

    // for now this is a useless abstraction, but kept for consistency,
    // check below why
    fn parse_expression_literal(&mut self) -> Result<Expression> {
        Ok(Expression::Literal(self.parse_literal()?))
    }

    fn parse_expression(&mut self) -> Result<Expression> {
        let expression = match self.peek_token_err(0)?.clone() {
            lexer::Token::Identifier(_) => self.parse_expression_identifier()?,
            lexer::Token::Literal(_) => self.parse_expression_literal()?,
            _ => return Err(anyhow!("parse_expression: wrong token")),
        };

        match self.peek_token_err(0)? {
            lexer::Token::Plus => {
                self.next();
                return Ok(Expression::Addition(Box::new(Addition {
                    left: expression,
                    right: self.parse_expression()?,
                })));
            }
            _ => {}
        }

        Ok(expression)
    }

    fn parse_variable_declaration(&mut self) -> Result<VariableDeclaration> {
        self.expect_next_token(lexer::Token::Let)?;
        self.next();

        let identifier = self.parse_identifier()?;

        let _type = self.parse_type()?;

        self.expect_next_token(lexer::Token::Equals)?;
        self.next();

        let expression = self.parse_expression()?;

        self.context.variables.insert(
            identifier.clone(),
            ContextVariable {
                _type: _type.clone(),
                identifier: identifier.clone(),
            },
        );

        Ok(VariableDeclaration {
            identifier,
            expression,
            _type,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple() {
        let code = String::from(
            "
                fn add(a int, b int) int {
                    return a + b
                }
                fn main() void {
                    let a int = 0
                    let b int = 1
                    let c int = a + b + 37 + 200
                    let d int = b + add(a, b)
                }
            ",
        );

        let tokens = lexer::Lexer::new(&code).run().unwrap();
        let ast = Ast::new(&tokens).unwrap();
        println!("{ast:#?}");
    }
}
