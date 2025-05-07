use anyhow::{anyhow, Result};

use super::lexer;

struct VariableDeclaration {
    identifier: String,
    expression: Expression,
}

impl VariableDeclaration {
    fn new(identifier: String, expression: Expression) -> Self {
        Self {
            identifier,
            expression,
        }
    }
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
            self.next();
        }

        Ok(Ast { nodes })
    }

    fn parse_token(&mut self) -> Result<Node> {
        if let Some(token) = self.peek_token() {
            match token {
                lexer::Token::Let => Ok(Node::VariableDeclaration(
                    self.parse_variable_declaration()?,
                )),
                lexer::Token::Function => Ok(Node::Function(self.parse_function()?)),
                _ => Err(anyhow!("todo")),
            }
        } else {
            Err(anyhow!("todo"))
        }
    }

    fn parse_function(&mut self) -> Result<Function> {
        todo!()
    }

    fn parse_expression(&mut self) -> Result<Expression> {
        if let Some(token) = self.peek_token() {
            match token {
                lexer::Token::Identifier(v) => Ok(Expression::Identifier(v.clone())),
                lexer::Token::Literal(v) => Ok(Expression::Literal(v.clone())),
                _ => Err(anyhow!("todo")),
            }
        } else {
            Err(anyhow!("todo"))
        }
    }

    fn parse_variable_declaration(&mut self) -> Result<VariableDeclaration> {
        self.next();

        let identifier: String;
        let expression: Expression;

        if let Some(token) = self.peek_token() {
            match token {
                lexer::Token::Identifier(v) => identifier = v.clone(),
                _ => return Err(anyhow!("variable declaration, expected Identifier")),
            }
        } else {
            return Err(anyhow!("variable declaration expected next token to exist"));
        }

        self.next();
        expression = self.parse_expression()?;

        Ok(VariableDeclaration::new(identifier, expression))
    }

    fn next(&mut self) {
        self.i += 1;
    }

    fn peek_token(&self) -> Option<&lexer::Token> {
        self.tokens.get(self.i)
    }
}

impl Ast {
    pub fn new(tokens: &Vec<lexer::Token>) -> Result<Self> {
        AstCreator::new(tokens).parse()
    }
}
