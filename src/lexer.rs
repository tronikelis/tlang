use anyhow::Result;

enum Type {
    Int(Option<isize>),
}

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
            self.skip_whitespace();

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
                    let int: isize = self.read_next_word().parse()?;
                    tokens.push(Token::Type(Type::Int(Some(int))));
                    continue;
                }
                _ => {}
            }

            match self.peek_char() {
                '(' => {
                    tokens.push(Token::POpen);
                    self.read_char();
                    continue;
                }
                ')' => {
                    tokens.push(Token::PClose);
                    self.read_char();
                    continue;
                }
                '=' => {
                    tokens.push(Token::Equals);
                    self.read_char();
                    continue;
                }
                '+' => {
                    tokens.push(Token::Plus);
                    self.read_char();
                    continue;
                }
                _ => {}
            }

            tokens.push(Token::Identifier(self.read_next_word()));
        }

        Ok(tokens)
    }

    fn read_char(&mut self) -> char {
        let ch = self.code[self.i];
        self.i += 1;
        ch
    }

    fn peek_char(&mut self) -> char {
        let ch = self.read_char();
        self.i -= 1;
        ch
    }

    fn skip_whitespace(&mut self) {
        while self.read_char().is_whitespace() {}
        self.i -= 1;
    }

    fn peek_next_word(&mut self) -> String {
        let w = self.read_next_word();
        self.i -= w.len() + 1;
        w
    }

    fn read_next_word(&mut self) -> String {
        let mut word = String::new();
        self.skip_whitespace();

        loop {
            let ch = self.read_char();

            if ch.is_whitespace() {
                break;
            }

            word.push(ch);
        }

        word
    }
}
