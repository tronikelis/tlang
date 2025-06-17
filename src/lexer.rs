use anyhow::{anyhow, Result};

#[derive(Debug, PartialEq, Clone)]
pub enum Type {
    Int,
    Void,
    Bool,
    CompilerType,
    Uint8,
    String,
}

#[derive(Debug, PartialEq, Clone)]
pub enum Literal {
    Int(usize), // - will be with infix expression
    String(String),
    Bool(bool),
}

#[derive(Debug, PartialEq, Clone)]
pub enum Token {
    Return,
    BOpen,
    BClose,
    CClose,
    COpen,
    Comma,
    Equals,
    Function,
    Identifier(String),
    Let,
    PClose,
    POpen,
    Plus,
    Minus,
    Slash,
    Star,
    Type(Type),
    Literal(Literal),
    Lt,
    Gt,
    If,
    ElseIf,
    Else,
    AmperAmper,
    PipePipe,
    EqualsEquals,
    Debug,
    PlusPlus,
    MinusMinus,
    For,
    Semicolon,
    Percent,
    Break,
    Continue,
}

const CONTROL_CHAR: [char; 19] = [
    ')', '(', '}', '{', ',', '>', '<', '&', '|', '=', '+', '-', ';', '*', '/', '[', ']', '"', '%',
];

pub struct Lexer {
    code: Vec<char>,
    i: usize,
}

impl Lexer {
    pub fn new(code: &str) -> Self {
        Self {
            code: code.chars().collect(),
            i: 0,
        }
    }

    pub fn run(mut self) -> Result<Vec<Token>> {
        let mut tokens: Vec<Token> = Vec::new();

        while let Some(_) = self.peek_char(0) {
            match self.peek_next_word().as_str() {
                "break" => {
                    tokens.push(Token::Break);
                    self.read_next_word();
                    continue;
                }
                "continue" => {
                    tokens.push(Token::Continue);
                    self.read_next_word();
                    continue;
                }
                "string" => {
                    tokens.push(Token::Type(Type::String));
                    self.read_next_word();
                    continue;
                }
                "uint8" => {
                    tokens.push(Token::Type(Type::Uint8));
                    self.read_next_word();
                    continue;
                }
                "Type" => {
                    tokens.push(Token::Type(Type::CompilerType));
                    self.read_next_word();
                    continue;
                }
                "true" | "false" => {
                    tokens.push(Token::Literal(Literal::Bool(
                        self.read_next_word() == "true",
                    )));
                    continue;
                }
                "for" => {
                    tokens.push(Token::For);
                    self.read_next_word();
                    continue;
                }
                "__debug__" => {
                    tokens.push(Token::Debug);
                    self.read_next_word();
                    continue;
                }
                "elseif" => {
                    tokens.push(Token::ElseIf);
                    self.read_next_word();
                    continue;
                }
                "else" => {
                    tokens.push(Token::Else);
                    self.read_next_word();
                    continue;
                }
                "if" => {
                    tokens.push(Token::If);
                    self.read_next_word();
                    continue;
                }
                "fn" => {
                    tokens.push(Token::Function);
                    self.read_next_word();
                    continue;
                }
                "let" => {
                    tokens.push(Token::Let);
                    self.read_next_word();
                    continue;
                }
                "int" => {
                    tokens.push(Token::Type(Type::Int));
                    self.read_next_word();
                    continue;
                }
                "void" => {
                    tokens.push(Token::Type(Type::Void));
                    self.read_next_word();
                    continue;
                }
                "bool" => {
                    tokens.push(Token::Type(Type::Bool));
                    self.read_next_word();
                    continue;
                }
                "return" => {
                    tokens.push(Token::Return);
                    self.read_next_word();
                    continue;
                }
                _ => {}
            }

            if let Some(ch) = self.peek_char(0) {
                match ch {
                    '%' => {
                        tokens.push(Token::Percent);
                        self.next();
                        continue;
                    }
                    '[' => {
                        tokens.push(Token::BOpen);
                        self.next();
                        continue;
                    }
                    ']' => {
                        tokens.push(Token::BClose);
                        self.next();
                        continue;
                    }
                    '*' => {
                        tokens.push(Token::Star);
                        self.next();
                        continue;
                    }
                    '/' => {
                        if let Some(ch) = self.peek_char(1) {
                            if ch == '/' {
                                self.skip_comment();
                                continue;
                            }
                        }

                        tokens.push(Token::Slash);
                        self.next();
                        continue;
                    }
                    ';' => {
                        tokens.push(Token::Semicolon);
                        self.next();
                        continue;
                    }
                    '&' => {
                        match self.peek_char(1).ok_or(anyhow!("todo: amper"))? {
                            '&' => {
                                tokens.push(Token::AmperAmper);
                                self.next();
                            }
                            _ => return Err(anyhow!("todo: unexpected token")),
                        };
                        self.next();
                        continue;
                    }
                    '|' => {
                        match self.peek_char(1).ok_or(anyhow!("todo: pipe"))? {
                            '|' => {
                                tokens.push(Token::PipePipe);
                                self.next();
                            }
                            _ => return Err(anyhow!("todo: unexpected token")),
                        };
                        self.next();
                        continue;
                    }
                    '>' => {
                        tokens.push(Token::Gt);
                        self.next();
                        continue;
                    }
                    '<' => {
                        tokens.push(Token::Lt);
                        self.next();
                        continue;
                    }
                    '(' => {
                        tokens.push(Token::POpen);
                        self.next();
                        continue;
                    }
                    ')' => {
                        tokens.push(Token::PClose);
                        self.next();
                        continue;
                    }
                    '=' => {
                        match self.peek_char(1).ok_or(anyhow!("todo: equals"))? {
                            '=' => {
                                tokens.push(Token::EqualsEquals);
                                self.next();
                            }
                            _ => tokens.push(Token::Equals),
                        };
                        self.next();
                        continue;
                    }
                    '+' => {
                        tokens.push(Token::Plus);
                        self.next();
                        continue;
                    }
                    '-' => {
                        tokens.push(Token::Minus);
                        self.next();
                        continue;
                    }
                    ',' => {
                        tokens.push(Token::Comma);
                        self.next();
                        continue;
                    }
                    '{' => {
                        tokens.push(Token::COpen);
                        self.next();
                        continue;
                    }
                    '}' => {
                        tokens.push(Token::CClose);
                        self.next();
                        continue;
                    }
                    '"' => {
                        tokens.push(Token::Literal(Literal::String(
                            self.parse_string_literal()?,
                        )));
                        continue;
                    }
                    _ => {}
                }
            }

            let identifier = self.read_next_word();
            if identifier.len() > 0 {
                if let Some(ch) = identifier.chars().next() {
                    if ch >= '0' && ch <= '9' {
                        tokens.push(Token::Literal(Literal::Int(identifier.parse()?)));
                        continue;
                    }
                }

                tokens.push(Token::Identifier(identifier));

                if let Some(ch) = self.peek_char(0) {
                    match ch {
                        '+' => {
                            if let Some(ch) = self.peek_char(1) {
                                if ch == '+' {
                                    self.next();
                                    self.next();
                                    tokens.push(Token::PlusPlus);
                                }
                            }
                        }
                        '-' => {
                            if let Some(ch) = self.peek_char(1) {
                                if ch == '-' {
                                    self.next();
                                    self.next();
                                    tokens.push(Token::MinusMinus);
                                }
                            }
                        }
                        _ => {}
                    }
                }

                continue;
            }

            self.next();
        }

        Ok(tokens)
    }

    fn parse_string_literal(&mut self) -> Result<String> {
        let mut string = String::new();
        self.next();

        while let Some(ch) = self.peek_char(0) {
            match ch {
                '"' => {
                    self.next();
                    return Ok(string);
                }
                '\\' => {
                    match self
                        .peek_char(1)
                        .ok_or(anyhow!("parse_string_literal: expected char after escape"))?
                    {
                        '"' => {
                            string.push('"');
                        }
                        'n' => {
                            string.push('\n');
                        }
                        '\\' => {
                            string.push('\\');
                        }
                        ch => {
                            return Err(anyhow!(
                                "parse_string_literal: unknown escape char {ch:#?}"
                            ))
                        }
                    }

                    self.next();
                }
                ch => string.push(ch),
            }

            self.next();
        }

        Err(anyhow!("parse_string_literal: undeterminate string"))
    }

    fn next(&mut self) {
        self.i += 1;
    }

    fn peek_char(&mut self, offset: usize) -> Option<char> {
        self.code.get(self.i + offset).map(|v| *v)
    }

    fn skip_whitespace(&mut self) {
        while let Some(ch) = self.peek_char(0) {
            if !ch.is_whitespace() {
                break;
            }

            self.next();
        }
    }

    fn peek_next_word(&mut self) -> String {
        let w = self.read_next_word();
        self.i -= w.len();
        w
    }

    fn skip_comment(&mut self) {
        while let Some(ch) = self.peek_char(0) {
            self.next();
            if ch == '\n' {
                break;
            }
        }
    }

    fn read_next_word(&mut self) -> String {
        let mut word = String::new();
        self.skip_whitespace();

        while let Some(ch) = self.peek_char(0) {
            if ch.is_whitespace() || CONTROL_CHAR.contains(&ch) {
                break;
            }

            self.next();
            word.push(ch);
        }

        word
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
                    let c int = a + b
                }
            ",
        );

        let tokens = Vec::from([
            Token::Function,
            Token::Identifier(String::from("add")),
            Token::POpen,
            Token::Identifier(String::from("a")),
            Token::Type(Type::Int),
            Token::Comma,
            Token::Identifier(String::from("b")),
            Token::Type(Type::Int),
            Token::PClose,
            Token::Type(Type::Int),
            Token::COpen,
            Token::Return,
            Token::Identifier(String::from("a")),
            Token::Plus,
            Token::Identifier(String::from("b")),
            Token::CClose,
            Token::Function,
            Token::Identifier(String::from("main")),
            Token::POpen,
            Token::PClose,
            Token::Type(Type::Void),
            Token::COpen,
            Token::Let,
            Token::Identifier(String::from("a")),
            Token::Type(Type::Int),
            Token::Equals,
            Token::Literal(Literal::Int(0)),
            Token::Let,
            Token::Identifier(String::from("b")),
            Token::Type(Type::Int),
            Token::Equals,
            Token::Literal(Literal::Int(1)),
            Token::Let,
            Token::Identifier(String::from("c")),
            Token::Type(Type::Int),
            Token::Equals,
            Token::Identifier(String::from("a")),
            Token::Plus,
            Token::Identifier(String::from("b")),
            Token::CClose,
        ]);

        assert_eq!(Lexer::new(&code).run().unwrap(), tokens);
    }
}
