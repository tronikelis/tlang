use anyhow::{anyhow, Result};
use std::{cell::RefCell, collections::HashMap, rc::Rc};

use crate::{ast, lexer};

pub fn align(alignment: usize, stack_size: usize) -> usize {
    if alignment == 0 || stack_size == 0 {
        0
    } else {
        let modulo = stack_size % alignment;
        if modulo == 0 {
            0
        } else {
            alignment - modulo
        }
    }
}

pub struct TypeResolver<'a> {
    type_declarations: &'a HashMap<String, ast::Type>,
}

struct ResolvedTypeFunction {
    type_function: TypeFunction,
    size: usize,
    alignment: usize,
    id: String,
}

impl ResolvedTypeFunction {
    fn to_function(self) -> Type {
        Type {
            id: Some(self.id),
            size: self.size,
            alignment: self.alignment,
            _type: TypeType::Function(Box::new(TypeFunction {
                arguments: self.type_function.arguments,
                return_type: self.type_function.return_type,
            })),
        }
    }

    fn to_closure(self) -> Type {
        Type {
            id: Some(self.id),
            size: self.size,
            alignment: self.alignment,
            _type: TypeType::Closure(Box::new(TypeFunction {
                arguments: self.type_function.arguments,
                return_type: self.type_function.return_type,
            })),
        }
    }
}

impl<'a> TypeResolver<'a> {
    pub fn new(type_declarations: &'a HashMap<String, ast::Type>) -> Self {
        Self { type_declarations }
    }

    fn resolve(&self, _type: &ast::Type) -> Result<Type> {
        self.resolve_with_alias(_type, None, None)
    }

    fn resolve_type_function(
        &self,
        args: &[ast::Variable],
        return_type: &ast::Type,
    ) -> Result<ResolvedTypeFunction> {
        let mut arguments = Vec::new();

        let mut id = String::from("fn (");
        for var in args {
            let resolved = self.resolve(&var._type)?;
            id.push_str(
                resolved
                    .id
                    .as_ref()
                    .ok_or(anyhow!("resolve_closure: type without id"))?,
            );
            id.push(',');
            arguments.push(Variable {
                identifier: var.identifier.clone(),
                _type: resolved,
            });
        }
        id.push(')');

        let resolved_return_type = self.resolve(return_type)?;

        id.push_str(
            resolved_return_type
                .id
                .as_ref()
                .ok_or(anyhow!("resolve_closure: return type without id"))?,
        );

        Ok(ResolvedTypeFunction {
            id,
            size: PTR_SIZE,
            alignment: PTR_SIZE,
            type_function: TypeFunction {
                arguments,
                return_type: resolved_return_type,
            },
        })
    }

    fn resolve_with_alias(
        &self,
        _type: &ast::Type,
        initial_alias: Option<&str>,
        alias: Option<&str>,
    ) -> Result<Type> {
        match _type {
            ast::Type::Alias(inner_alias) => {
                let builtin = match inner_alias.as_str() {
                    "uint" => Some(UINT.clone()),
                    "uint8" => Some(UINT8.clone()),
                    "int" => Some(INT.clone()),
                    "bool" => Some(BOOL.clone()),
                    "string" => Some(STRING.clone()),
                    "Type" => Some(COMPILER_TYPE.clone()),
                    "void" => Some(VOID.clone()),
                    "ptr" => Some(PTR.clone()),
                    _ => None,
                };
                if let Some(mut builtin) = builtin {
                    builtin.id = initial_alias.map(|v| v.to_string()).or(builtin.id);
                    return Ok(builtin);
                }

                let inner = self
                    .type_declarations
                    .get(inner_alias)
                    .ok_or(anyhow!("can't resolve {inner_alias:#?}"))?;

                if let Some(alias) = alias {
                    if alias == inner_alias {
                        match inner {
                            ast::Type::Struct(_) => {}
                            _ => panic!("recursive non struct?"),
                        }

                        return Ok(Type {
                            id: Some(alias.to_string() + "{}"),
                            size: 0,
                            alignment: 0,
                            _type: TypeType::Lazy(alias.to_string()),
                        });
                    }
                }

                self.resolve_with_alias(
                    &inner,
                    initial_alias.or(Some(inner_alias)),
                    Some(inner_alias),
                )
            }
            ast::Type::Slice(_type) => {
                let nested = self.resolve_with_alias(_type, initial_alias, alias)?;
                Ok(Type {
                    id: nested.id.as_ref().map(|id| id.clone() + "[]"),
                    size: SLICE_SIZE,
                    alignment: SLICE_SIZE,
                    _type: TypeType::Slice(Box::new(nested)),
                })
            }
            ast::Type::Variadic(_type) => {
                let nested = self.resolve_with_alias(_type, initial_alias, alias)?;
                Ok(Type {
                    id: nested.id.as_ref().map(|id| id.clone() + "..."),
                    size: size_of::<usize>(),
                    alignment: size_of::<usize>(),
                    _type: TypeType::Variadic(Box::new(nested)),
                })
            }
            ast::Type::Struct(type_struct) => {
                let mut fields: Vec<TypeStructField> = Vec::new();
                let mut size: usize = 0;
                let mut highest_alignment: usize = 0;

                for var in &type_struct.fields {
                    let resolved = self.resolve_with_alias(&var._type, initial_alias, alias)?;
                    if resolved.alignment > highest_alignment {
                        highest_alignment = resolved.alignment;
                    }

                    let alignment = align(resolved.alignment, size);
                    size += resolved.size;
                    size += alignment;
                    fields.push(TypeStructField::Padding(alignment));
                    fields.push(TypeStructField::Type(var.identifier.clone(), resolved));
                }

                let end_padding = align(highest_alignment, size);
                size += end_padding;
                fields.push(TypeStructField::Padding(end_padding));

                Ok(Type {
                    id: alias.map(|id| id.to_string() + "{}"),
                    size,
                    alignment: highest_alignment,
                    _type: TypeType::Struct(TypeStruct { fields }),
                })
            }
            ast::Type::Address(_type) => {
                let nested = self.resolve_with_alias(_type, initial_alias, alias)?;
                Ok(Type {
                    id: nested.id.as_ref().map(|id| id.clone() + "&"),
                    size: PTR_SIZE,
                    alignment: PTR_SIZE,
                    _type: TypeType::Address(Box::new(nested)),
                })
            }
            ast::Type::Closure(type_closure) => Ok(self
                .resolve_type_function(&type_closure.arguments, &type_closure.return_type)?
                .to_closure()),
        }
    }
}

lazy_static::lazy_static! {
    pub static ref NIL: Type = Type {
        id: Some("nil".to_string()),
        size: PTR_SIZE,
        alignment: PTR_SIZE,
        _type: TypeType::Builtin(TypeBuiltin::Nil),
    };
    pub static ref UINT: Type = Type {
        id: Some("uint".to_string()),
        size: size_of::<usize>(),
        alignment: size_of::<usize>(),
        _type: TypeType::Builtin(TypeBuiltin::Uint),
    };
    pub static ref UINT8: Type = Type {
        id: Some("uint8".to_string()),
        size: 1,
        alignment: 1,
        _type: TypeType::Builtin(TypeBuiltin::Uint8),
    };
    pub static ref INT: Type = Type {
        id: Some("int".to_string()),
        size: size_of::<isize>(),
        alignment: size_of::<isize>(),
        _type: TypeType::Builtin(TypeBuiltin::Int),
    };
    pub static ref BOOL: Type = Type {
        id: Some("bool".to_string()),
        size: size_of::<usize>(),      // for now
        alignment: size_of::<usize>(), // for now
        _type: TypeType::Builtin(TypeBuiltin::Bool),
    };
    pub static ref STRING: Type = Type {
        id: Some("string".to_string()),
        size: size_of::<usize>(),
        alignment: size_of::<usize>(),
        _type: TypeType::Builtin(TypeBuiltin::String),
    };
    pub static ref COMPILER_TYPE: Type = Type {
        id: Some("Type".to_string()),
        size: 0,
        alignment: 0,
        _type: TypeType::Builtin(TypeBuiltin::CompilerType),
    };
    pub static ref VOID: Type = Type {
        id: Some("void".to_string()),
        size: 0,
        alignment: 0,
        _type: TypeType::Builtin(TypeBuiltin::Void),
    };
    pub static ref PTR: Type = Type {
        id: Some("ptr".to_string()),
        size: PTR_SIZE,
        alignment: PTR_SIZE,
        _type: TypeType::Builtin(TypeBuiltin::Ptr),
    };
}

pub const SLICE_SIZE: usize = size_of::<usize>();
pub const PTR_SIZE: usize = size_of::<usize>();

#[derive(Debug, Clone)]
struct Stack<T> {
    items: Vec<Vec<T>>,
}

impl<T> Stack<T> {
    fn new() -> Self {
        let mut items = Vec::new();
        items.push(Vec::new());
        Self { items }
    }

    fn push(&mut self, item: T) {
        self.items.last_mut().unwrap().push(item);
    }

    fn push_frame(&mut self) {
        self.items.push(Vec::new());
    }

    fn pop_frame(&mut self) -> Option<Vec<T>> {
        self.items.pop()
    }
}

pub trait VariableStack {
    fn get_type(&self, identifier: &str) -> Option<Type>;

    fn get_type_err(&self, identifier: &str) -> Result<Type> {
        self.get_type(identifier)
            .ok_or(anyhow!("VariableStack.get_type_err({identifier})"))
    }
}

#[derive(Debug, Clone)]
enum VariableItem {
    Variable(Rc<RefCell<Variable>>),
    SearchEnd,
}

#[derive(Debug, Clone)]
struct Variables {
    stack: Stack<VariableItem>,
}

impl VariableStack for Variables {
    fn get_type(&self, identifier: &str) -> Option<Type> {
        self.get(identifier)
            .as_ref()
            .map(|v| v.variable.borrow()._type.clone())
    }
}

struct VariableGetResult {
    variable: Rc<RefCell<Variable>>,
    escaped: bool,
}

impl Variables {
    fn new() -> Self {
        Self {
            stack: Stack::new(),
        }
    }

    fn get(&self, identifier: &str) -> Option<VariableGetResult> {
        let mut search_ended = false;
        for item in self.stack.items.iter().flatten().rev() {
            match item {
                VariableItem::SearchEnd => search_ended = true,
                VariableItem::Variable(var) => {
                    if var.borrow().identifier == identifier {
                        if search_ended {
                            var.borrow_mut().escape();
                            return Some(VariableGetResult {
                                variable: var.clone(),
                                escaped: true,
                            });
                        }

                        return Some(VariableGetResult {
                            variable: var.clone(),
                            escaped: false,
                        });
                    }
                }
            }
        }

        None
    }

    fn get_err(&self, identifier: &str) -> Result<VariableGetResult> {
        self.get(identifier)
            .ok_or(anyhow!("VariableStack.get({identifier}) not found"))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypeStruct {
    pub fields: Vec<TypeStructField>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypeStructField {
    Type(String, Type),
    Padding(usize),
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypeFunction {
    pub arguments: Vec<Variable>,
    pub return_type: Type,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Variable {
    pub identifier: String,
    pub _type: Type,
}

impl Variable {
    fn escape(&mut self) {
        match &self._type._type {
            TypeType::Escaped(_) => {}
            _ => self._type = Type::create_escaped(self._type.clone()),
        }
    }

    fn from_ast(var: &ast::Variable, type_resolver: &TypeResolver) -> Result<Self> {
        Ok(Self {
            identifier: var.identifier.clone(),
            _type: type_resolver.resolve(&var._type)?,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypeBuiltin {
    Uint,
    Uint8,
    Int,
    String,
    Bool,
    Void,
    CompilerType,
    Ptr,
    Nil,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypeType {
    Struct(TypeStruct),
    Variadic(Box<Type>),
    Slice(Box<Type>),
    Builtin(TypeBuiltin),
    Address(Box<Type>),
    Lazy(String),
    Closure(Box<TypeFunction>),
    Function(Box<TypeFunction>),
    Escaped(Box<Type>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Type {
    // None for inline types
    pub id: Option<String>,
    pub size: usize,
    pub alignment: usize,
    pub _type: TypeType,
}

impl Type {
    fn method_id(&self, identifier: &str) -> Option<String> {
        self.id.as_ref().map(|v| format!("{}.{}", v, identifier))
    }

    fn method_id_err(&self, identifier: &str) -> Result<String> {
        self.method_id(identifier).ok_or(anyhow!("Type.method_id"))
    }

    fn skip_escaped(&self) -> &Self {
        match &self._type {
            TypeType::Escaped(_type) => _type,
            _ => self,
        }
    }

    fn create_variadic(item: Self) -> Self {
        Self {
            id: item.id.as_ref().map(|id| id.clone() + "..."),
            size: SLICE_SIZE,
            alignment: SLICE_SIZE,
            _type: TypeType::Variadic(Box::new(item)),
        }
    }

    fn create_address(item: Self) -> Self {
        Self {
            id: item.id.as_ref().map(|id| id.clone() + "&"),
            size: PTR_SIZE,
            alignment: PTR_SIZE,
            _type: TypeType::Address(Box::new(item)),
        }
    }

    fn create_escaped(item: Self) -> Self {
        Self {
            id: item.id.clone(),
            size: PTR_SIZE,
            alignment: PTR_SIZE,
            _type: TypeType::Escaped(Box::new(item)),
        }
    }
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
pub enum LiteralType {
    Int(usize),
    String(String),
    Bool(bool),
}

#[derive(Debug, Clone)]
pub struct Literal {
    pub literal_type: LiteralType,
    pub _type: Type,
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
pub enum CallType {
    Function(ExpFunction),
    Closure(Expression),
}

#[derive(Debug, Clone)]
pub struct Call {
    pub arguments: Vec<Expression>,
    pub call_type: CallType,
}

#[derive(Debug, Clone)]
pub struct Index {
    pub var: Expression,
    pub expression: Expression,
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
pub struct StructInit {
    pub fields: HashMap<String, Expression>,
    pub _type: Type,
}

#[derive(Debug, Clone)]
pub struct ExpFunction {
    pub identifier: String,
    pub _type: Type,
}

#[derive(Debug, Clone)]
pub struct Closure {
    pub _type: Type,
    pub escaped_variables: Vec<String>,
    pub actions: Vec<Action>,
}

#[derive(Debug, Clone)]
pub struct Method {
    pub type_id: String,
    pub _self: Expression,
    pub function: ExpFunction,
}

#[derive(Debug, Clone)]
pub enum Expression {
    Spread(Box<Expression>),
    Index(Box<Index>),
    Compare(Box<Compare>),
    Literal(Literal),
    Variable(String),
    Type(Type),
    Deref(Box<Expression>),
    Address(Box<Expression>),
    Arithmetic(Box<Arithmetic>),
    AndOr(Box<AndOr>),
    Infix(Infix),
    Negate(Box<Expression>),
    ToClosure(Box<Expression>),
    Function(ExpFunction),
    Method(Box<Method>),
    Closure(Closure),
    Call(Box<Call>),
    DotAccess(Box<DotAccess>),
    SliceInit(SliceInit),
    StructInit(StructInit),
    Nil,
}

impl Expression {
    pub fn _type<T: VariableStack>(&self, variables: &T) -> Result<Type> {
        match self {
            Self::Literal(literal) => Ok(literal._type.clone()),
            Self::Spread(expression) => Ok(Type::create_variadic(expression._type(variables)?)),
            Self::Index(index) => match index.var._type(variables)?._type {
                TypeType::Slice(v) => Ok(*v),
                _ => Err(anyhow!("index non slice type")),
            },
            Self::Compare(_compare) => Ok(BOOL.clone()),
            Self::Variable(identifier) => Ok(variables.get_type_err(identifier)?),
            Self::Type(v) => Ok(v.clone()),
            Self::Deref(expression) => match expression._type(variables)?._type {
                TypeType::Address(_type) => Ok(*_type),
                _ => Err(anyhow!("deref non address type")),
            },
            Self::Address(expression) => Ok(Type::create_address(expression._type(variables)?)),
            // only INT can add for now
            Self::Arithmetic(_arithmetic) => Ok(INT.clone()),
            Self::AndOr(_and_or) => Ok(BOOL.clone()),
            Self::Infix(_infix) => Ok(INT.clone()),
            Self::Negate(_expression) => Ok(BOOL.clone()),
            Self::Function(exp_function) => Ok(exp_function._type.clone()),
            Self::ToClosure(expression) => {
                let _type = expression._type(variables)?;
                match _type._type {
                    TypeType::Function(type_function) => Ok(Type {
                        size: PTR_SIZE,
                        alignment: PTR_SIZE,
                        id: _type.id,
                        _type: TypeType::Closure(type_function),
                    }),
                    _ => Err(anyhow!("to closure from non function")),
                }
            }
            Self::Call(call) => {
                let _type = match &call.call_type {
                    CallType::Closure(exp) => exp._type(variables)?,
                    CallType::Function(exp_function) => exp_function._type.clone(),
                };

                let type_function = match _type._type {
                    TypeType::Closure(type_function) | TypeType::Function(type_function) => {
                        type_function
                    }
                    _ => return Err(anyhow!("cant call non function type")),
                };

                Ok(type_function.return_type)
            }
            Self::DotAccess(dot_access) => match dot_access.expression._type(variables)?._type {
                TypeType::Struct(type_struct) => {
                    let mut _type: Option<Type> = None;

                    for field in type_struct.fields {
                        match field {
                            TypeStructField::Padding(_) => {}
                            TypeStructField::Type(identifier, v) => {
                                if identifier == dot_access.identifier {
                                    _type = Some(v);
                                    break;
                                }
                            }
                        }
                    }

                    _type.ok_or(anyhow!("dot_access unknown identifier"))
                }
                _ => Err(anyhow!("dot accessing non struct")),
            },
            Self::SliceInit(slice_init) => Ok(slice_init._type.clone()),
            Self::StructInit(struct_init) => Ok(struct_init._type.clone()),
            Self::Nil => Ok(NIL.clone()),
            Self::Closure(closure) => Ok(closure._type.clone()),
            Self::Method(method) => Ok(method.function._type.clone()),
        }
    }

    fn ensure_address<T: VariableStack>(self, variables: &T) -> Result<Self> {
        let _type = self._type(variables)?;
        match _type._type {
            TypeType::Address(_) | TypeType::Escaped(_) => Ok(self),
            _ => Ok(Expression::Address(Box::new(self))),
        }
    }
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
pub struct VariableDeclaration {
    pub variable: Rc<RefCell<Variable>>,
    pub expression: Expression,
}

#[derive(Debug, Clone)]
pub struct ElseIf {
    pub expression: Expression,
    pub actions: Vec<Action>,
}

#[derive(Debug, Clone)]
pub struct Else {
    pub actions: Vec<Action>,
}

#[derive(Debug, Clone)]
pub struct If {
    pub expression: Expression,
    pub actions: Vec<Action>,
    pub elseif: Vec<ElseIf>,
    pub _else: Option<Else>,
}

#[derive(Debug, Clone)]
pub struct For {
    pub initializer: Option<Action>,
    pub expression: Option<Expression>,
    pub after_each: Option<Action>,
    pub actions: Vec<Action>,
}

#[derive(Debug, Clone)]
pub struct VariableAssignment {
    pub var: Expression,
    pub expression: Expression,
}

#[derive(Debug, Clone)]
pub enum Action {
    VariableDeclaration(VariableDeclaration),
    VariableAssignment(VariableAssignment),
    Expression(Expression),
    Return(Option<Expression>),
    Debug,
    Break,
    If(If),
    For(Box<For>),
    Continue,
}

#[derive(Debug)]
pub struct Function {
    pub arguments: Vec<Rc<RefCell<Variable>>>,
    pub return_type: Type,
    pub actions: Vec<Action>,
}

pub struct IrParser<'a> {
    variables: Variables,
    type_resolver: TypeResolver<'a>,
    escaped_variables: Vec<String>,
    impl_resolved: HashMap<String, &'a ast::FunctionDeclaration>,
    ast: &'a ast::Ast,
}

impl<'a> IrParser<'a> {
    pub fn new(ast: &'a ast::Ast) -> Result<Self> {
        let type_resolver = TypeResolver::new(&ast.type_declarations);

        let mut impl_resolved = HashMap::new();

        for block in &ast.impl_block_declarations {
            let _type = type_resolver.resolve(&block._type)?;
            for (k, v) in &block.functions {
                impl_resolved.insert(_type.method_id_err(k)?, v);
            }
        }

        Ok(Self {
            variables: Variables::new(),
            type_resolver,
            escaped_variables: Vec::new(),
            impl_resolved,
            ast,
        })
    }

    fn get_expression(
        &mut self,
        expression: &ast::Expression,
        expected_type: Option<&Type>,
    ) -> Result<Expression> {
        match expression {
            ast::Expression::Address(exp) => {
                let inner = self.get_expression(&exp, expected_type)?;
                if let Expression::Variable(var) = &inner {
                    self.variables.get_err(var)?.variable.borrow_mut().escape();
                }

                return Ok(Expression::Address(Box::new(inner)));
            }
            ast::Expression::Deref(exp) => {
                let exp = self.get_expression(exp, expected_type)?;
                let TypeType::Address(_type) = exp._type(&self.variables)?._type else {
                    return Err(anyhow!("get_expression: deref non address {exp:#?}"));
                };
                return Ok(Expression::Deref(Box::new(exp)));
            }
            ast::Expression::Type(_type) => {
                if let Ok(_type) = self.type_resolver.resolve(_type) {
                    return Ok(Expression::Type(_type.clone()));
                }

                let ast::Type::Alias(identifier) = _type else {
                    return Err(anyhow!("unresolved type non alias {_type:#?}"));
                };

                if let Some(var) = self.variables.get(&identifier) {
                    let identifier = var.variable.borrow().identifier.clone();
                    if var.escaped {
                        self.escaped_variables.push(identifier.clone());
                    }
                    return Ok(Expression::Variable(identifier));
                }

                let function_declaration = self
                    .ast
                    .function_declarations
                    .get(identifier)
                    .ok_or(anyhow!("function not found"))?;

                let resolved_type_function = self.type_resolver.resolve_type_function(
                    &function_declaration.arguments,
                    &function_declaration.return_type,
                )?;

                let result_expression = Expression::Function(ExpFunction {
                    identifier: identifier.clone(),
                    _type: resolved_type_function.to_function(),
                });

                return match expected_type {
                    None => Ok(result_expression),
                    Some(_) => Ok(Expression::ToClosure(Box::new(result_expression))),
                };
            }
            ast::Expression::Arithmetic(arithmetic) => {
                let left_exp = self.get_expression(&arithmetic.left, expected_type)?;
                let right_exp = self
                    .get_expression(&arithmetic.right, Some(&left_exp._type(&self.variables)?))?;

                return Ok(Expression::Arithmetic(Box::new(Arithmetic {
                    left: left_exp,
                    right: right_exp,
                    _type: match arithmetic._type {
                        ast::ArithmeticType::Plus => ArithmeticType::Plus,
                        ast::ArithmeticType::Minus => ArithmeticType::Minus,
                        ast::ArithmeticType::Divide => ArithmeticType::Divide,
                        ast::ArithmeticType::Multiply => ArithmeticType::Multiply,
                        ast::ArithmeticType::Modulo => ArithmeticType::Modulo,
                    },
                })));
            }
            ast::Expression::Nil => {
                return Ok(Expression::Nil);
            }
            ast::Expression::AndOr(and_or) => {
                let left_exp = self.get_expression(&and_or.left, expected_type)?;
                let right_exp =
                    self.get_expression(&and_or.right, Some(&left_exp._type(&self.variables)?))?;

                return Ok(Expression::AndOr(Box::new(AndOr {
                    left: left_exp,
                    right: right_exp,
                    _type: match and_or._type {
                        ast::AndOrType::And => AndOrType::And,
                        ast::AndOrType::Or => AndOrType::Or,
                    },
                })));
            }
            ast::Expression::Infix(infix) => {
                let exp = self.get_expression(&infix.expression, expected_type)?;

                return Ok(Expression::Infix(Infix {
                    _type: match infix._type {
                        ast::InfixType::Plus => InfixType::Plus,
                        ast::InfixType::Minus => InfixType::Minus,
                    },
                    expression: Box::new(exp),
                }));
            }
            ast::Expression::Negate(expression) => {
                let exp = self.get_expression(expression, expected_type)?;

                return Ok(Expression::Negate(Box::new(exp)));
            }
            ast::Expression::Literal(literal) => match &literal.literal {
                lexer::Literal::String(string) => {
                    return Ok(Expression::Literal(Literal {
                        _type: STRING.clone(),
                        literal_type: LiteralType::String(string.clone()),
                    }));
                }
                lexer::Literal::Bool(bool) => {
                    return Ok(Expression::Literal(Literal {
                        _type: BOOL.clone(),
                        literal_type: LiteralType::Bool(*bool),
                    }));
                }
                lexer::Literal::Int(int) => {
                    let Some(_type) = expected_type else {
                        return Ok(Expression::Literal(Literal {
                            _type: INT.clone(),
                            literal_type: LiteralType::Int(*int),
                        }));
                    };

                    let _type = match &_type.skip_escaped()._type {
                        TypeType::Builtin(builtin) => match builtin {
                            TypeBuiltin::Int => INT.clone(),
                            TypeBuiltin::Uint => UINT.clone(),
                            TypeBuiltin::Uint8 => UINT8.clone(),
                            _ => return Err(anyhow!("literal wrong type")),
                        },
                        _ => return Err(anyhow!("literal wrong type")),
                    };

                    return Ok(Expression::Literal(Literal {
                        _type: _type.clone(),
                        literal_type: LiteralType::Int(*int),
                    }));
                }
            },
            ast::Expression::Compare(compare) => {
                let left_exp = self.get_expression(&compare.left, expected_type)?;
                let right_exp =
                    self.get_expression(&compare.right, Some(&left_exp._type(&self.variables)?))?;

                return Ok(Expression::Compare(Box::new(Compare {
                    left: left_exp,
                    right: right_exp,
                    compare_type: match &compare.compare_type {
                        ast::CompareType::Gt => CompareType::Gt,
                        ast::CompareType::Lt => CompareType::Lt,
                        ast::CompareType::Equals => CompareType::Equals,
                        ast::CompareType::NotEquals => CompareType::NotEquals,
                    },
                })));
            }
            ast::Expression::Call(call) => {
                let exp = self.get_expression(&call.expression, expected_type)?;

                let type_function = match &exp._type(&self.variables)?._type {
                    TypeType::Closure(type_function) | TypeType::Function(type_function) => {
                        *type_function.clone()
                    }
                    _ => return Err(anyhow!("cant call non function type")),
                };

                let mut arguments: Vec<Expression> = Vec::new();

                if let Expression::Method(method) = &exp {
                    arguments.push(method._self.clone());
                }

                for (i, arg) in call.arguments.iter().enumerate() {
                    let fn_arg = type_function
                        .arguments
                        .get(i)
                        .ok_or(anyhow!("type_function argument not found"))?;

                    arguments.push(self.get_expression(&arg, Some(&fn_arg._type))?);
                }

                return Ok(Expression::Call(Box::new(Call {
                    arguments,
                    call_type: match exp {
                        Expression::Function(exp_function) => CallType::Function(exp_function),
                        Expression::Method(method) => CallType::Function(method.function),
                        exp => CallType::Closure(exp),
                    },
                })));
            }
            ast::Expression::Index(index) => {
                let var_exp = self.get_expression(&index.var, expected_type)?;
                let expression_exp = self.get_expression(&index.expression, expected_type)?;

                return Ok(Expression::Index(Box::new(Index {
                    var: var_exp,
                    expression: expression_exp,
                })));
            }
            ast::Expression::Spread(expression) => {
                return Ok(Expression::Spread(Box::new(
                    self.get_expression(expression, expected_type)?,
                )));
            }
            ast::Expression::DotAccess(dot_access) => {
                let exp = self.get_expression(&dot_access.expression, expected_type)?;
                let _type = exp._type(&self.variables)?;

                if let Some(identifier) = _type.method_id(&dot_access.identifier) {
                    if let Some(function_declaration) = self.impl_resolved.get(&identifier) {
                        if let Expression::Variable(var) = &exp {
                            self.variables.get_err(var)?.variable.borrow_mut().escape();
                        }

                        let resolved_type_function = self.type_resolver.resolve_type_function(
                            &function_declaration.arguments,
                            &function_declaration.return_type,
                        )?;

                        let result_expression = Expression::Method(Box::new(Method {
                            type_id: _type.id.expect("method id returned"),
                            _self: exp.ensure_address(&self.variables)?,
                            function: ExpFunction {
                                identifier,
                                _type: resolved_type_function.to_function(),
                            },
                        }));

                        return match expected_type {
                            None => Ok(result_expression),
                            Some(_) => Ok(Expression::ToClosure(Box::new(result_expression))),
                        };
                    }
                }

                return Ok(Expression::DotAccess(Box::new(DotAccess {
                    expression: exp,
                    identifier: dot_access.identifier.clone(),
                })));
            }
            ast::Expression::SliceInit(slice_init) => {
                let _type = self.type_resolver.resolve(&slice_init._type)?;

                let expressions = slice_init
                    .expressions
                    .iter()
                    .map(|v| Ok(self.get_expression(v, expected_type)?))
                    .collect::<Result<Vec<_>>>()?;

                return Ok(Expression::SliceInit(SliceInit {
                    expressions,
                    _type: _type.clone(),
                }));
            }
            ast::Expression::StructInit(struct_init) => {
                let fields = struct_init
                    .fields
                    .iter()
                    .map(|v| Ok((v.0.clone(), self.get_expression(v.1, expected_type)?)))
                    .collect::<Result<HashMap<_, _>>>()?;

                let _type = self.type_resolver.resolve(&struct_init._type)?;

                return Ok(Expression::StructInit(StructInit { fields, _type }));
            }
            ast::Expression::TypeInit(type_init) => {
                let _type = self.type_resolver.resolve(&type_init._type)?;

                return match &_type._type {
                    TypeType::Slice(_) => Ok(Expression::SliceInit(SliceInit {
                        _type,
                        expressions: Vec::new(),
                    })),
                    TypeType::Struct(_) => Ok(Expression::StructInit(StructInit {
                        _type,
                        fields: HashMap::new(),
                    })),
                    _ => Err(anyhow!("type init non slice/struct")),
                };
            }
            ast::Expression::Closure(closure) => {
                let _type = self.type_resolver.resolve(&closure._type)?;

                let mut closure_ir = Self::new(self.ast)?;

                closure_ir.variables = self.variables.clone();
                closure_ir.variables.stack.push(VariableItem::SearchEnd);

                let actions = closure_ir.get_actions(&closure.body)?;

                for identifier in &closure_ir.escaped_variables {
                    if !self.escaped_variables.contains(identifier) {
                        self.escaped_variables.push(identifier.clone());
                    }
                }

                return Ok(Expression::Closure(Closure {
                    _type,
                    actions,
                    escaped_variables: self.escaped_variables.clone(),
                }));
            }
        }
    }

    fn get_variable_declaration(
        &mut self,
        declaration: &ast::VariableDeclaration,
    ) -> Result<VariableDeclaration> {
        let variable = Rc::new(RefCell::new(Variable::from_ast(
            &declaration.variable,
            &self.type_resolver,
        )?));
        self.variables
            .stack
            .push(VariableItem::Variable(variable.clone()));

        let expected_type = variable.borrow()._type.clone();

        Ok(VariableDeclaration {
            variable,
            expression: self.get_expression(&declaration.expression, Some(&expected_type))?,
        })
    }

    fn get_if(&mut self, _if: &ast::If) -> Result<If> {
        let expression = self.get_expression(&_if.expression, None)?;
        let actions = self.get_actions(&_if.body)?;

        let elseif = _if
            .elseif
            .iter()
            .map(|v| {
                Ok(ElseIf {
                    expression: self.get_expression(&v.expression, None)?,
                    actions: self.get_actions(&v.body)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let _else = _if
            ._else
            .as_ref()
            .map(|v| -> Result<Else> {
                Ok(Else {
                    actions: self.get_actions(&v.body)?,
                })
            })
            .transpose()?;

        Ok(If {
            expression,
            actions,
            elseif,
            _else,
        })
    }

    fn get_for(&mut self, _for: &ast::For) -> Result<For> {
        let initializer = _for
            .initializer
            .as_ref()
            .map(|v| -> Result<Action> { Ok(self.get_action(v)?) })
            .transpose()?;

        let expression = _for
            .expression
            .as_ref()
            .map(|v| -> Result<Expression> { Ok(self.get_expression(v, None)?) })
            .transpose()?;

        let after_each = _for
            .after_each
            .as_ref()
            .map(|v| -> Result<Action> { Ok(self.get_action(v)?) })
            .transpose()?;

        let actions = self.get_actions(&_for.body)?;

        Ok(For {
            initializer,
            expression,
            after_each,
            actions,
        })
    }

    fn get_variable_assignment(
        &mut self,
        assignment: &ast::VariableAssignment,
    ) -> Result<VariableAssignment> {
        let var = self.get_expression(&assignment.var, None)?;
        let var_type = var._type(&self.variables)?;
        Ok(VariableAssignment {
            var,
            expression: self.get_expression(&assignment.expression, Some(&var_type))?,
        })
    }

    fn get_action(&mut self, node: &ast::Node) -> Result<Action> {
        Ok(match node {
            ast::Node::VariableDeclaration(v) => {
                Action::VariableDeclaration(self.get_variable_declaration(v)?)
            }
            ast::Node::Debug => Action::Debug,
            ast::Node::Break => Action::Break,
            ast::Node::Continue => Action::Continue,
            ast::Node::Expression(exp) => Action::Expression(self.get_expression(exp, None)?),
            ast::Node::Return(exp) => Action::Return(
                exp.as_ref()
                    .map(|v| self.get_expression(v, None))
                    .transpose()?,
            ),
            ast::Node::If(v) => Action::If(self.get_if(v)?),
            ast::Node::For(v) => Action::For(Box::new(self.get_for(v)?)),
            ast::Node::VariableAssignment(v) => {
                Action::VariableAssignment(self.get_variable_assignment(v)?)
            }
        })
    }

    fn get_actions(&mut self, body: &[ast::Node]) -> Result<Vec<Action>> {
        self.variables.stack.push_frame();

        let mut actions: Vec<Action> = Vec::new();
        for node in body {
            actions.push(self.get_action(node)?);
        }

        self.variables.stack.pop_frame();

        Ok(actions)
    }

    fn create_from_function_declaration(
        mut self,
        declaration: &ast::FunctionDeclaration,
    ) -> Result<Function> {
        let arguments = declaration
            .arguments
            .iter()
            .map(|var| {
                Ok(Rc::new(RefCell::new(Variable::from_ast(
                    var,
                    &self.type_resolver,
                )?)))
            })
            .collect::<Result<Vec<_>>>()?;

        let return_type = self.type_resolver.resolve(&declaration.return_type)?;

        for arg in &arguments {
            self.variables
                .stack
                .push(VariableItem::Variable(arg.clone()));
        }

        let actions = self.get_actions(&declaration.body)?;

        Ok(Function {
            arguments,
            return_type,
            actions,
        })
    }
}

#[derive(Debug)]
pub struct Ir {
    pub functions: HashMap<String, Function>,
}

impl Ir {
    pub fn new(ast: &ast::Ast) -> Result<Self> {
        let mut functions = HashMap::new();
        let type_resolver = TypeResolver::new(&ast.type_declarations);

        for (identifier, declaration) in &ast.function_declarations {
            let function = IrParser::new(ast)?.create_from_function_declaration(declaration)?;
            functions.insert(identifier.clone(), function);
        }

        for block in &ast.impl_block_declarations {
            let _type = type_resolver.resolve(&block._type)?;

            for (identifier, declaration) in &block.functions {
                let function = IrParser::new(ast)?.create_from_function_declaration(declaration)?;
                functions.insert(_type.method_id_err(identifier)?, function);
            }
        }

        Ok(Self { functions })
    }
}
