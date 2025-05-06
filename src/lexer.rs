use std::num::ParseIntError;

use anyhow::Result;

#[derive(Debug, PartialEq)]
enum Type {
    Int(Option<isize>),
}

#[derive(Debug, PartialEq)]
enum Token {
    Return,
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
    Type(Type),
    Immediate(Type),
}

const CONTROL_CHAR: [char; 5] = [')', '(', '}', '{', ','];

pub struct Lexer {
    code: Vec<char>,
    i: usize,
}

impl Lexer {
    pub fn new(code: &String) -> Self {
        Self {
            code: code.chars().collect(),
            i: 0,
        }
    }

    pub fn run(&mut self) -> Result<Vec<Token>> {
        let mut tokens: Vec<Token> = Vec::new();

        while self.i < self.code.len() {
            dbg!(self.i, self.code[self.i]);

            match self.peek_next_word().as_str() {
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
                    let int: Result<isize, ParseIntError> = self.read_next_word().parse();
                    tokens.push(Token::Type(Type::Int(int.ok())));
                    continue;
                }
                "return" => {
                    tokens.push(Token::Return);
                    self.read_next_word();
                    continue;
                }
                _ => {}
            }

            if let Some(ch) = self.peek_char() {
                match ch {
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
                        tokens.push(Token::Equals);
                        self.next();
                        continue;
                    }
                    '+' => {
                        tokens.push(Token::Plus);
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
                    _ => {}
                }
            }

            let identifier = self.read_next_word();
            if identifier.len() > 0 {
                tokens.push(Token::Identifier(identifier));
                continue;
            }

            self.next();
        }

        Ok(tokens)
    }

    fn next(&mut self) {
        self.i += 1;
    }

    fn peek_char(&mut self) -> Option<char> {
        if self.i >= self.code.len() {
            return None;
        }

        Some(self.code[self.i])
    }

    fn skip_whitespace(&mut self) {
        while let Some(ch) = self.peek_char() {
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

    fn read_next_word(&mut self) -> String {
        let mut word = String::new();
        self.skip_whitespace();

        loop {
            if let Some(ch) = self.peek_char() {
                if ch.is_whitespace() || CONTROL_CHAR.contains(&ch) {
                    break;
                }

                self.next();
                word.push(ch);
            } else {
                break;
            }
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

                fn main() {
                    let a int = 0
                    let b int = 1

                    let c int = add(a, b) + b
                }
            ",
        );

        let tokens = Vec::from([
            Token::Function,
            Token::Identifier(String::from("add")),
            Token::POpen,
            Token::Identifier(String::from("a")),
            Token::Type(Type::Int(None)),
            Token::Comma,
            Token::Identifier(String::from("b")),
            Token::PClose,
            Token::Type(Type::Int(None)),
            Token::COpen,
            Token::Return,
            Token::Identifier(String::from("a")),
            Token::Plus,
            Token::Identifier(String::from("b")),
        ]);

        assert_eq!(Lexer::new(&code).run().unwrap(), tokens);
    }
}
