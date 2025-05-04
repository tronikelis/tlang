enum Type {
    Int(Option<isize>),
}

enum Token {
    Return,
    CClose,
    COpen,
    Comma,
    Equal,
    Function,
    Identifier(String),
    Let,
    PClose,
    POpen,
    Plus,
    Type(Type),
    Immediate(Type),
}
