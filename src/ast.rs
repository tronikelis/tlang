use anyhow::{anyhow, Result};

use super::lexer;

struct VariableDeclaration {
    identifier: String,
    expression: Expression,
    _type: lexer::Type,
}

struct FunctionArgument {
    identifier: String,
    _type: lexer::Type,
}

struct Function {
    identifier: String,
    arguments: Vec<FunctionArgument>,
    return_type: lexer::Type,
    body: Vec<Node>,
}

struct Addition {
    left: Expression,
    right: Expression,
}

struct FunctionCall {
    identifier: String,
    arguments: Vec<Expression>,
}

enum Expression {
    Literal(lexer::Literal),
    Identifier(String),
    Addition(Box<Addition>),
    FunctionCall(FunctionCall),
}

enum Node {
    VariableDeclaration(VariableDeclaration),
    Function(Function),
    Return(Expression),
}

struct Ast {
    pub nodes: Vec<Node>,
}

struct AstCreator<'a> {
    tokens: &'a Vec<lexer::Token>,
    i: usize,
}

impl<'a> AstCreator<'a> {
    fn new(tokens: &'a Vec<lexer::Token>) -> Self {
        Self { tokens, i: 0 }
    }

    fn parse(&mut self) -> Result<Ast> {
        let mut nodes: Vec<Node> = Vec::new();

        while let Ok(token) = self.parse_token() {
            nodes.push(token);
        }

        Ok(Ast { nodes })
    }

    fn parse_token(&mut self) -> Result<Node> {
        match self.peek_token_err(0)? {
            lexer::Token::Let => Ok(Node::VariableDeclaration(
                self.parse_variable_declaration()?,
            )),
            lexer::Token::Function => Ok(Node::Function(self.parse_function()?)),
            lexer::Token::Return => {
                self.next();
                Ok(Node::Return(self.parse_expression()?))
            }
            _ => return Err(anyhow!("parse_token: token not supported")),
        }
    }

    fn parse_block(&mut self) -> Result<Vec<Node>> {
        let mut nodes = Vec::new();

        if *self.peek_token_err(0)? != lexer::Token::COpen {
            return Err(anyhow!("parse_block: expected COpen"));
        }
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

    fn parse_type(&mut self) -> Result<lexer::Type> {
        match self.peek_token_err(0)?.clone() {
            lexer::Token::Type(v) => {
                self.next();
                Ok(v.clone())
            }
            _ => Err(anyhow!("parse_type: expected Type")),
        }
    }

    fn parse_function(&mut self) -> Result<Function> {
        if *self.peek_token_err(0)? != lexer::Token::Function {
            return Err(anyhow!("parse_function: called on non Function token"));
        }
        self.next();

        let identifier = self.parse_identifier()?;

        if *self.peek_token_err(0)? != lexer::Token::POpen {
            return Err(anyhow!("parse_function: expected POpen"));
        }
        self.next();

        let mut function_arguments: Vec<FunctionArgument> = Vec::new();

        while let Some(token) = self.peek_token(0) {
            if let lexer::Token::PClose = token {
                self.next();
                break;
            }

            function_arguments.push(FunctionArgument {
                identifier: self.parse_identifier()?,
                _type: self.parse_type()?,
            });
        }

        let return_type = self.parse_type()?;
        let body = self.parse_block()?;

        Ok(Function {
            identifier,
            arguments: function_arguments,
            return_type,
            body,
        })
    }

    fn parse_function_call(&mut self) -> Result<FunctionCall> {
        let identifier = self.parse_identifier()?;

        self.expect_next_token(lexer::Token::POpen)?;
        self.next();

        let mut arguments = Vec::new();

        while let Some(token) = self.peek_token(0) {
            if let lexer::Token::PClose = token {
                self.next();
                break;
            }

            arguments.push(self.parse_expression()?);

            match self.peek_token_err(0)? {
                lexer::Token::PClose => {
                    self.next();
                    break;
                }
                lexer::Token::Comma => {
                    self.next();
                    continue;
                }
                _ => return Err(anyhow!("parse_function_call: unexpected token")),
            }
        }

        Ok(FunctionCall {
            arguments,
            identifier,
        })
    }

    fn parse_expression_identifier(&mut self, identifier: String) -> Result<Expression> {
        // follow a function call
        if let lexer::Token::POpen = self.peek_token_err(1)? {
            return Ok(Expression::FunctionCall(self.parse_function_call()?));
        }

        return Ok(Expression::Identifier(identifier));
    }

    // for now this is a useless abstraction, but kept for consistency,
    // check below why
    fn parse_expression_literal(&mut self) -> Result<Expression> {
        Ok(Expression::Literal(self.parse_literal()?))
    }

    fn parse_expression(&mut self) -> Result<Expression> {
        let expression = match self.peek_token_err(0)?.clone() {
            lexer::Token::Identifier(v) => self.parse_expression_identifier(v)?,
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

        let identifier = match self.peek_token_err(0)? {
            lexer::Token::Identifier(v) => v.clone(),
            _ => return Err(anyhow!("parse_variable_declaration: expected Identifier")),
        };
        self.next();

        let _type = self.parse_type()?;

        self.expect_next_token(lexer::Token::Equals)?;
        self.next();

        let expression = self.parse_expression()?;

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

impl Ast {
    pub fn new(tokens: &Vec<lexer::Token>) -> Result<Self> {
        AstCreator::new(tokens).parse()
    }
}
