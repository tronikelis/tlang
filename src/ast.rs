use anyhow::{anyhow, Result};

use super::lexer;

struct VariableDeclaration {
    identifier: String,
    expression: Expression,
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

enum Expression {
    Literal(lexer::Literal),
    Identifier(String),
}

enum Node {
    VariableDeclaration(VariableDeclaration),
    Function(Function),
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

        while let Some(_) = self.peek_token() {
            nodes.push(self.parse_token()?);
        }

        Ok(Ast { nodes })
    }

    fn parse_token(&mut self) -> Result<Node> {
        match self.peek_token_err()? {
            lexer::Token::Let => Ok(Node::VariableDeclaration(
                self.parse_variable_declaration()?,
            )),
            lexer::Token::Function => Ok(Node::Function(self.parse_function()?)),
            _ => return Err(anyhow!("parse_token: token not supported")),
        }
    }

    fn parse_block(&mut self) -> Result<Vec<Node>> {
        let mut nodes = Vec::new();

        if *self.peek_token_err()? != lexer::Token::COpen {
            return Err(anyhow!("parse_block: expected COpen"));
        }
        self.next();

        while let Some(token) = self.peek_token() {
            if let lexer::Token::CClose = token {
                self.next();
                break;
            }

            nodes.push(self.parse_token()?);
        }

        Ok(nodes)
    }

    fn parse_identifier(&mut self) -> Result<String> {
        match self.peek_token_err()?.clone() {
            lexer::Token::Identifier(v) => {
                self.next();
                Ok(v.clone())
            }
            _ => Err(anyhow!("parse_identifier: expected Identifier")),
        }
    }

    fn parse_type(&mut self) -> Result<lexer::Type> {
        match self.peek_token_err()?.clone() {
            lexer::Token::Type(v) => {
                self.next();
                Ok(v.clone())
            }
            _ => Err(anyhow!("parse_type: expected Type")),
        }
    }

    fn parse_function(&mut self) -> Result<Function> {
        if *self.peek_token_err()? != lexer::Token::Function {
            return Err(anyhow!("parse_function: called on non Function token"));
        }
        self.next();

        let identifier = self.parse_identifier()?;

        if *self.peek_token_err()? != lexer::Token::POpen {
            return Err(anyhow!("parse_function: expected POpen"));
        }
        self.next();

        let mut function_arguments: Vec<FunctionArgument> = Vec::new();

        while let Some(token) = self.peek_token() {
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

    fn parse_expression(&mut self) -> Result<Expression> {
        match self.peek_token_err()?.clone() {
            lexer::Token::Identifier(v) => {
                self.next();
                Ok(Expression::Identifier(v.clone()))
            }
            lexer::Token::Literal(v) => {
                self.next();
                Ok(Expression::Literal(v.clone()))
            }
            _ => Err(anyhow!("parse_expression: wrong token")),
        }
    }

    fn parse_variable_declaration(&mut self) -> Result<VariableDeclaration> {
        if *self.peek_token_err()? != lexer::Token::Let {
            return Err(anyhow!("parse_variable_declaration: expected Let token"));
        }

        let identifier: String;
        let expression: Expression;

        if let lexer::Token::Identifier(v) = self.peek_token_err()? {
            identifier = v.clone();
        } else {
            return Err(anyhow!("parse_variable_declaration: expected Identifier"));
        }

        self.next();
        expression = self.parse_expression()?;

        self.next();

        Ok(VariableDeclaration {
            identifier,
            expression,
        })
    }

    fn next(&mut self) {
        self.i += 1;
    }

    fn peek_token(&self) -> Option<&lexer::Token> {
        self.tokens.get(self.i)
    }

    fn peek_token_err(&self) -> Result<&lexer::Token> {
        self.peek_token()
            .ok_or(anyhow!("peek_token_err: expected Some"))
    }
}

impl Ast {
    pub fn new(tokens: &Vec<lexer::Token>) -> Result<Self> {
        AstCreator::new(tokens).parse()
    }
}
