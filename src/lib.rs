use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet, VecDeque};
use std::error::Error;
use std::fmt::{Debug, Display, Formatter};
use std::num::ParseIntError;
use std::ops::{Deref, DerefMut};
use std::rc::Rc;

#[derive(Clone, Eq, PartialEq, Hash)]
pub struct StateName(String, Vec<Expr>);

impl StateName {
    fn new(name: impl AsRef<str>) -> Self {
        Self(name.as_ref().into(), Vec::new())
    }

    fn new_with_args(name: impl AsRef<str>, args: Vec<Expr>) -> Self {
        Self(name.as_ref().into(), args)
    }

    fn evaluate(&self, tracer: &mut dyn Tracer, state: &WorldState) -> Result<StateName, PlanningErr> {
        let args = self.1.iter()
            .map(|a| match a.evaluate(tracer, state) {
                Ok(v) => Ok(Expr::Val(v)),
                Err(e) => Err(e),
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self(self.0.clone(), args))
    }
}

impl From<&str> for StateName {
    fn from(value: &str) -> Self {
        StateName::new(value)
    }
}

impl From<String> for StateName {
    fn from(value: String) -> Self {
        StateName::new(value)
    }
}

impl PartialOrd for StateName {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl Ord for StateName {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0)
    }
}

impl Debug for StateName {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self.1.len() < 1 {
            write!(f, "{:?}", self.0)
        } else {
            let mut t = f.debug_tuple(&self.0);
            for value in self.1.iter() {
                t.field(&value);
            }
            t.finish()
        }
    }
}

#[derive(Clone, Eq, PartialEq, PartialOrd, Hash)]
pub enum StateValue {
    Str(String),
    Int(i32),
    Bool(bool),
}

impl From<&str> for StateValue {
    fn from(value: &str) -> Self {
        Self::Str(value.into())
    }
}

impl From<String> for StateValue {
    fn from(value: String) -> Self {
        Self::Str(value)
    }
}

impl From<i32> for StateValue {
    fn from(value: i32) -> Self {
        Self::Int(value)
    }
}

impl From<bool> for StateValue {
    fn from(value: bool) -> Self {
        Self::Bool(value)
    }
}

impl Debug for StateValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Str(x) => write!(f, "{:?}", x),
            Self::Int(x) => write!(f, "{:?}", x),
            Self::Bool(x) => write!(f, "{:?}", x),
        }
    }
}

impl Display for StateValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Str(x) => write!(f, "{}", x),
            Self::Int(x) => write!(f, "{}", x),
            Self::Bool(x) => write!(f, "{}", x),
        }
    }
}

impl From<StateValue> for bool {
    fn from(value: StateValue) -> Self {
        match value {
            StateValue::Str(x) => !x.is_empty(),
            StateValue::Int(x) => x != 0,
            StateValue::Bool(x) => x,
        }
    }
}

type ExternalCall = Rc<dyn Fn(&WorldState, &[StateValue]) -> Result<StateValue, PlanningErr>>;

pub struct WorldState {
    parent: Option<Rc<RefCell<WorldState>>>,
    values: HashMap<StateName, Option<StateValue>>,
    calls: HashMap<String, ExternalCall>,
}

impl Eq for WorldState {}

impl PartialEq for WorldState {
    fn eq(&self, other: &Self) -> bool {
        self.parent.eq(&other.parent)
            && self.values.eq(&other.values)
    }
}

impl Debug for WorldState {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut t = f.debug_map();

        let mut names = self.names().collect::<Vec<_>>();
        names.sort();
        for name in names {
            if let Some(v) = self.get(&name) {
                t.entry(&name, &v);
            }
        }

        t.finish()
    }
}

impl WorldState {
    fn new() -> Self {
        Self {
            parent: None,
            values: HashMap::new(),
            calls: HashMap::new(),
        }
    }

    fn with_parent(parent: Rc<RefCell<WorldState>>) -> Self {
        Self {
            parent: Some(parent),
            values: HashMap::new(),
            calls: HashMap::new(),
        }
    }

    fn names(&self) -> impl Iterator<Item=StateName> {
        let mut result = HashSet::new();
        self.names_into(&mut result);
        result.into_iter()
    }

    fn names_into(&self, set: &mut HashSet<StateName>) {
        set.extend(self.values.keys().cloned());
        if let Some(p) = &self.parent {
            p.borrow().names_into(set);
        }
    }

    fn set(&mut self, tracer: &mut dyn Tracer, name: StateName, value: StateValue) {
        tracer.set_state(&name, &value);
        self.values.insert(name, Some(value));
    }

    fn unset(&mut self, tracer: &mut dyn Tracer, name: StateName) {
        tracer.unset_state(&name);
        self.values.insert(name, None);
    }

    fn get(&self, name: &StateName) -> Option<StateValue> {
        if let Some(value) = self.values.get(name) {
            return value.clone();
        }

        // Don't check parent scopes for parameters, they're local only
        if name.0.starts_with('?') {
            return None;
        }

        if let Some(p) = &self.parent {
            return p.borrow().get(name);
        }

        None
    }

    fn call(&self, name: &str, args: &[StateValue]) -> Result<StateValue, PlanningErr> {
        if let Some(x) = self.calls.get(name) {
            return x(self, args);
        }

        let state_name = StateName::new_with_args(name, args.iter()
            .map(|a| Expr::Val(a.clone()))
            .collect());
        if let Some(state) = self.get(&state_name) {
            return Ok(state);
        }

        if let Some(p) = &self.parent {
            return p.borrow().call(name, args);
        }

        Err(PlanningErr::UnsetState(name.into()))
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Operator {
    name: String,
    params: Vec<String>,
    additions: HashMap<StateName, Expr>,
    removals: HashSet<StateName>,
}

impl Operator {
    fn apply_to(&self, tracer: &mut dyn Tracer, state: &mut WorldState) -> Result<(), PlanningErr> {
        for removal in self.removals.iter() {
            let name = removal.evaluate(tracer, state)?;
            tracer.unset_state(&name);
            state.unset(tracer, name);
        }

        for (name, param) in self.additions.iter() {
            let name = name.evaluate(tracer, state)?;
            let value = param.evaluate(tracer, state)?;
            state.set(tracer, name, value);
        }

        Ok(())
    }
}

#[derive(Clone, Eq, PartialEq, Hash)]
pub enum Expr {
    Val(StateValue),
    State(Box<Expr>),
    BinOp(Box<Expr>, BinOp, Box<Expr>),
    Call(String, Vec<Expr>),
}

impl Debug for Expr {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Val(x) => write!(f, "{:?}", x),
            Self::State(x) => write!(f, "state[{:?}]", x),
            Self::BinOp(l, o, r) => write!(f, "{:?} {} {:?}", l, o, r),
            Self::Call(n, a) => {
                write!(f, "call:")?;
                let mut t = f.debug_tuple(n);
                for arg in a.iter() {
                    t.field(arg);
                }
                t.finish()
            }
        }
    }
}

impl Expr {
    fn evaluate(&self, tracer: &mut dyn Tracer, state: &WorldState) -> Result<StateValue, PlanningErr> {
        let result = match self {
            Self::Val(x) => Ok(x.clone()),
            Self::State(x) => {
                let index = x.evaluate(tracer, &state)?;
                let name = index.to_string().into();
                state.get(&name)
                    .ok_or_else(|| PlanningErr::UnsetState(name))
            }
            Self::BinOp(left, op, right) => {
                let left = left.evaluate(tracer, &state)?;
                let right = right.evaluate(tracer, &state)?;
                match (left, *op, right) {
                    (l, BinOp::Is, r) => Ok(StateValue::Bool(l == r)),
                    (l, BinOp::IsNot, r) => Ok(StateValue::Bool(l != r)),
                    (l, BinOp::GreaterThan, r) => Ok(StateValue::Bool(l > r)),
                    (l, BinOp::GreaterOrEqualTo, r) => Ok(StateValue::Bool(l >= r)),
                    (l, BinOp::LessThan, r) => Ok(StateValue::Bool(l < r)),
                    (l, BinOp::LessThanOrEqualTo, r) => Ok(StateValue::Bool(l <= r)),
                    (StateValue::Int(l), BinOp::Sub, StateValue::Int(r)) => Ok(StateValue::Int(l - r)),
                    (l, BinOp::Sub, r) => Err(PlanningErr::InvalidBinOpOperands(l, *op, r)),
                    (StateValue::Int(l), BinOp::Add, StateValue::Int(r)) => Ok(StateValue::Int(l + r)),
                    (l, BinOp::Add, r) => Err(PlanningErr::InvalidBinOpOperands(l, *op, r)),
                    (StateValue::Int(l), BinOp::Mul, StateValue::Int(r)) => Ok(StateValue::Int(l * r)),
                    (l, BinOp::Mul, r) => Err(PlanningErr::InvalidBinOpOperands(l, *op, r)),
                    (StateValue::Int(l), BinOp::Div, StateValue::Int(r)) => Ok(StateValue::Int(l / r)),
                    (l, BinOp::Div, r) => Err(PlanningErr::InvalidBinOpOperands(l, *op, r)),
                    (StateValue::Int(l), BinOp::Mod, StateValue::Int(r)) => Ok(StateValue::Int(l % r)),
                    (l, BinOp::Mod, r) => Err(PlanningErr::InvalidBinOpOperands(l, *op, r)),
                    (l, BinOp::And, r) => Ok(StateValue::Bool(l.into() && r.into())),
                    (l, BinOp::Or, r) => Ok(StateValue::Bool(l.into() || r.into())),
                }
            }
            Self::Call(n, args) => {
                let args = args.iter()
                    .map(|a| a.evaluate(tracer, &state))
                    .collect::<Result<Vec<_>, _>>()?;
                state.call(n, &args)
            }
        };

        tracer.expr_evaluated(self, &result);

        result
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum BinOp {
    GreaterThan,
    LessThan,
    Is,
    IsNot,
    Sub,
    Add,
    Mod,
    GreaterOrEqualTo,
    LessThanOrEqualTo,
    Or,
    And,
    Div,
    Mul,
}

impl Display for BinOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GreaterThan => write!(f, ">"),
            Self::GreaterOrEqualTo => write!(f, ">="),
            Self::LessThan => write!(f, "<"),
            Self::LessThanOrEqualTo => write!(f, "<="),
            Self::Is => write!(f, "is"),
            Self::IsNot => write!(f, "is not"),
            Self::Sub => write!(f, "-"),
            Self::Add => write!(f, "+"),
            Self::Mul => write!(f, "*"),
            Self::Mod => write!(f, "%"),
            Self::Div => write!(f, "/"),
            Self::And => write!(f, "and"),
            Self::Or => write!(f, "or"),
        }
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct Task {
    name: String,
    parameters: Vec<Expr>,
}

impl Debug for Task {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut t = f.debug_tuple(&self.name);
        for p in self.parameters.iter() {
            t.field(&p);
        }
        t.finish()
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Method {
    name: String,
    params: Vec<String>,
    conditions: Vec<Expr>,
    sub_tasks: Vec<Task>,
}

impl Method {
    fn applicable_in(&self, tracer: &mut dyn Tracer, state: &WorldState) -> bool {
        self.conditions.iter()
            .all(|c| c.evaluate(tracer, state)
                .map(|v| v.into())
                .unwrap_or(false))
    }
}

#[derive(Debug)]
pub enum PlanningErr {
    UnsetState(StateName),
    UnknownCall(String),
    InvalidBinOpOperands(StateValue, BinOp, StateValue),
}

impl Display for PlanningErr {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsetState(x) =>
                write!(f, "tried to read state variable '{}' which is not currently set", x.0),
            Self::UnknownCall(x) =>
                write!(f, "tried to call unknown fn '{}'", x),
            Self::InvalidBinOpOperands(l, op, r) =>
                write!(f, "invalid operands to binary operation: {} {} {}", l, op, r),
        }
    }
}

impl Error for PlanningErr {}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Domain {
    operators: Vec<Operator>,
    methods: Vec<Method>,
}

pub trait Tracer {
    fn plan_into(&mut self, tasks: &mut Vec<Task>, problem: &Problem);
    fn pop_task(&mut self, task: &Task);
    fn task_planned(&mut self, task: &Task, state: &Rc<RefCell<WorldState>>);
    fn has_operator(&mut self, operator: &Operator);
    fn has_method(&mut self, method: &Method, state: &Rc<RefCell<WorldState>>);
    fn has_applicable_method(&mut self, method: &Method);
    fn expr_evaluated(&mut self, expr: &Expr, result: &Result<StateValue, PlanningErr>);
    fn unset_state(&mut self, name: &StateName);
    fn set_state(&mut self, name: &StateName, value: &StateValue);
}

pub struct NullTracer;

impl Tracer for NullTracer {
    fn plan_into(&mut self, _tasks: &mut Vec<Task>, _problem: &Problem) {}
    fn pop_task(&mut self, _task: &Task) {}
    fn task_planned(&mut self, _task: &Task, _state: &Rc<RefCell<WorldState>>) {}
    fn has_operator(&mut self, _operator: &Operator) {}
    fn has_method(&mut self, _method: &Method, _state: &Rc<RefCell<WorldState>>) {}
    fn has_applicable_method(&mut self, _method: &Method) {}
    fn expr_evaluated(&mut self, _expr: &Expr, _result: &Result<StateValue, PlanningErr>) {}
    fn unset_state(&mut self, _name: &StateName) {}
    fn set_state(&mut self, _name: &StateName, _value: &StateValue) {}
}

#[derive(Clone)]
pub struct Problem {
    state: Rc<RefCell<WorldState>>,
    pub tasks: VecDeque<Task>,
    pub tracer: Rc<RefCell<dyn Tracer>>,
}

impl Eq for Problem {}

impl PartialEq for Problem {
    fn eq(&self, other: &Self) -> bool {
        self.state.eq(&other.state) &&
            self.tasks.eq(&other.tasks)
    }
}

impl Debug for Problem {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Problem")
            .field("state", self.state.borrow().deref())
            .field("tasks", &self.tasks)
            .finish_non_exhaustive()
    }
}

impl Domain {
    pub fn plan(&self, problem: Problem) -> Result<(Rc<RefCell<WorldState>>, Vec<Task>), PlanningErr> {
        let mut plan = Vec::new();

        let state = self.plan_into(&mut plan, problem)?;

        Ok((state, plan))
    }

    pub fn plan_into(&self, plan: &mut Vec<Task>, mut problem: Problem) -> Result<Rc<RefCell<WorldState>>, PlanningErr> {
        problem.tracer.borrow_mut().plan_into(plan, &problem);

        while let Some(task) = problem.tasks.pop_front() {
            problem.tracer.borrow_mut().pop_task(&task);

            if let Some(o) = self.has_operator_for(&task) {
                problem.tracer.borrow_mut().has_operator(o);

                // Set parameters in the nested state
                for (idx, param) in o.params.iter().enumerate() {
                    let value = task.parameters[idx].evaluate(problem.tracer.borrow_mut().deref_mut(), &problem.state.borrow())?;
                    problem.state.borrow_mut().set(problem.tracer.borrow_mut().deref_mut(), param.clone().into(), value);
                }

                plan.push(task.clone());
                o.apply_to(problem.tracer.borrow_mut().deref_mut(), &mut problem.state.borrow_mut())?;
                problem.tracer.borrow_mut().task_planned(&task, &problem.state);

                for param in o.params.iter() {
                    problem.state.borrow_mut().unset(problem.tracer.borrow_mut().deref_mut(), param.clone().into());
                }

                continue;
            }

            for method in self.has_methods_for(&task) {
                let nested_state = Rc::new(RefCell::new(WorldState::with_parent(problem.state.clone())));
                // Set parameters in the nested state
                for (idx, param) in method.params.iter().enumerate() {
                    let value = task.parameters[idx].evaluate(problem.tracer.borrow_mut().deref_mut(), &problem.state.borrow())?;
                    nested_state.borrow_mut().set(problem.tracer.borrow_mut().deref_mut(), param.clone().into(), value);
                }

                problem.tracer.borrow_mut().has_method(method, &nested_state);

                if !method.applicable_in(problem.tracer.borrow_mut().deref_mut(), &nested_state.borrow()) {
                    continue;
                }

                problem.tracer.borrow_mut().has_applicable_method(method);

                let nested_tasks = method.sub_tasks.iter()
                    .map(|t| -> Result<Task, PlanningErr> {
                        Ok(Task {
                            name: t.name.clone(),
                            parameters: t.parameters.iter()
                                .map(|p| p.evaluate(problem.tracer.borrow_mut().deref_mut(), &nested_state.borrow())
                                    .map(|v| Expr::Val(v)))
                                .collect::<Result<Vec<_>, _>>()?,
                        })
                    })
                    .collect::<Result<VecDeque<_>, _>>()?;

                let mut nested_plan = Vec::new();

                let nested_problem = Problem {
                    tasks: nested_tasks,
                    state: nested_state,
                    tracer: problem.tracer.clone(),
                };

                if let Ok(new_state) = self.plan_into(&mut nested_plan, nested_problem) {
                    plan.extend(nested_plan);

                    for param in method.params.iter() {
                        new_state.borrow_mut().unset(problem.tracer.borrow_mut().deref_mut(), param.clone().into());
                    }

                    problem.state = new_state;

                    break;
                }
            }
        }

        Ok(problem.state)
    }

    fn has_operator_for(&self, t: &Task) -> Option<&Operator> {
        self.operators.iter()
            .filter(|o| {
                o.name == t.name && o.params.len() == t.parameters.len()
            })
            .next()
    }

    fn has_methods_for<'a>(&'a self, t: &'a Task) -> impl Iterator<Item=&'a Method> {
        self.methods.iter()
            .filter(move |m| {
                m.name == t.name && m.params.len() == t.parameters.len()
            })
    }
}

pub const EXAMPLE_PROBLEM: &'static str = r#"
state {
    Location is Downtown,
    Weather is Good,
    Cash is 12,
    TaxiFare is 2,
    DistanceFrom(Downtown,Park) is 2000,
    TaxiLocation(Taxi1) is TaxiBase,
}

goals TravelTo(Park);
"#;

pub const EXAMPLE_DOMAIN: &'static str = r#"
op Walk(?To) {
    sets Location to ?To;
}

op Ride(?To) {
    sets Location to ?To;
}

op HailTaxi(?Location) {
    sets TaxiLocation(Taxi1) to ?Location;
}

op SetCash(?Amount) {
    sets Cash to ?Amount;
}

method TravelTo(?X) {
    requires state[Weather] is Good and
        DistanceFrom(state[Location], ?X) < 3000;

    becomes Walk(?X);
}

method TravelTo(?X) {
    requires state[Cash] >= state[TaxiFare] and
        TaxiLocation(Taxi1) is TaxiBase;

    becomes HailTaxi(state[Location]),
        Ride(?X),
        PayDriver(state[TaxiFare]);
}

method PayDriver(?Amount) {
    requires state[Cash] >= ?Amount;

    becomes SetCash(state[Cash] - ?Amount);
}
"#;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Keyword {
    Op,
    Sets,
    Unsets,
    To,
    Method,
    Requires,
    Becomes,
    State,
    Goals,
    And,
    Is,
    Or,
    Not,
    True,
    False,
}

impl Display for Keyword {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Op => write!(f, "op"),
            Self::Sets => write!(f, "sets"),
            Self::Unsets => write!(f, "unsets"),
            Self::To => write!(f, "to"),
            Self::Method => write!(f, "method"),
            Self::Requires => write!(f, "requires"),
            Self::Becomes => write!(f, "becomes"),
            Self::State => write!(f, "state"),
            Self::Goals => write!(f, "goals"),
            Self::And => write!(f, "and"),
            Self::Is => write!(f, "is"),
            Self::Or => write!(f, "or"),
            Self::Not => write!(f, "not"),
            Self::True => write!(f, "true"),
            Self::False => write!(f, "false"),
        }
    }
}

#[derive(Clone, Eq, PartialEq)]
pub enum TokenType {
    Sym(String),
    Var(String),
    Keyword(Keyword),
    OpSym(String),
    OBrace,
    CBrace,
    OParen,
    CParen,
    OSquare,
    CSquare,
    Term,
    Comma,
    LitInt(i32),
}

impl Debug for TokenType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Sym(x) => write!(f, "Sym({})", x),
            Self::Var(x) => write!(f, "Var({})", x),
            Self::Keyword(x) => write!(f, "Keyword({})", x),
            Self::OpSym(x) => write!(f, "OpSym({})", x),
            Self::OBrace => write!(f, "'{{'"),
            Self::CBrace => write!(f, "'}}'"),
            Self::OParen => write!(f, "'('"),
            Self::CParen => write!(f, "')'"),
            Self::OSquare => write!(f, "'['"),
            Self::CSquare => write!(f, "']'"),
            Self::Comma => write!(f, "','"),
            Self::Term => write!(f, "';'"),
            Self::LitInt(x) => write!(f, "LitInt({})", x),
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd)]
pub struct SourcePos {
    line: usize,
    col: usize,
}

impl Debug for SourcePos {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.line, self.col)
    }
}

impl SourcePos {
    const START_LINE: usize = 1;
    const START_COL: usize = 1;
}

impl Default for SourcePos {
    fn default() -> Self {
        Self {
            line: Self::START_LINE,
            col: Self::START_COL,
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd)]
pub struct Span {
    start: SourcePos,
    end: SourcePos,
}

impl Debug for Span {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}-{:?}", self.start, self.end)
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct Token {
    token_type: TokenType,
    span: Span,
}

impl Debug for Token {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.token_type)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TokenizeErrKind {
    IdentTooShort,
    UnexpectedChar(char),
    InvalidLitInt(ParseIntError),
}

impl Display for TokenizeErrKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IdentTooShort =>
                write!(f, "identifier too short"),
            Self::UnexpectedChar(c) =>
                write!(f, "unexpected character '{}'", c),
            Self::InvalidLitInt(e) =>
                write!(f, "invalid literal integer: {}", e),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TokenizeErr {
    pos: SourcePos,
    kind: TokenizeErrKind,
}

impl Display for TokenizeErr {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}: {}", self.pos, self.kind)
    }
}

impl Error for TokenizeErr {}

struct Tokenizer<'a> {
    pos: SourcePos,
    cursor: &'a str,
}

impl<'a> Tokenizer<'a> {
    fn new(src: &'a str) -> Self {
        Self {
            pos: SourcePos::default(),
            cursor: src,
        }
    }

    fn peek(&self) -> Option<char> {
        self.cursor.chars().next()
    }

    fn consume(&mut self) -> Option<char> {
        match self.peek() {
            Some(c) => {
                if c == '\n' {
                    self.pos.col = SourcePos::START_COL;
                    self.pos.line += 1;
                } else {
                    self.pos.col += 1;
                }

                (_, self.cursor) = self.cursor.split_at(c.len_utf8());

                Some(c)
            }
            None => None,
        }
    }

    fn take_while(&mut self, mut f: impl FnMut(char) -> bool) -> String {
        let mut result = String::new();

        while let Some(c) = self.peek() {
            if !f(c) {
                break;
            }

            self.consume();

            result.push(c);
        }

        result
    }

    fn skip_while(&mut self, mut f: impl FnMut(char) -> bool) {
        while let Some(c) = self.peek() {
            if !f(c) {
                break;
            }

            self.consume();
        }
    }

    fn skip_whitespace(&mut self) {
        self.skip_while(|c| c.is_whitespace())
    }

    fn identifier(&mut self) -> Result<String, TokenizeErr> {
        let mut first = true;

        let ident = self.take_while(|c| {
            if first {
                first = false;

                return c.is_alphabetic() || c == '?' || c == '_';
            }

            return c.is_alphanumeric() || c == '_';
        });

        let min_len = if ident.starts_with('?') { 2 } else { 1 };
        if ident.len() < min_len {
            return Err(TokenizeErr {
                pos: self.pos,
                kind: TokenizeErrKind::IdentTooShort,
            });
        }

        Ok(ident)
    }
}

pub fn tokenize(src: &str) -> Result<Vec<Token>, TokenizeErr> {
    const OP_SYMBOLS: &'static str = "<>=-+*/%";

    let mut t = Tokenizer::new(src);

    let mut tokens = Vec::new();

    loop {
        t.skip_whitespace();
        let Some(c) = t.peek() else { break; };

        let start = t.pos;
        let token_type = match c {
            '?' => {
                let identifier = t.identifier()?;
                TokenType::Var(identifier)
            }
            c @ ('(' | ')' | '[' | ']' | '{' | '}' | ',' | ';') => {
                t.consume();
                match c {
                    '(' => TokenType::OParen,
                    ')' => TokenType::CParen,
                    '[' => TokenType::OSquare,
                    ']' => TokenType::CSquare,
                    '{' => TokenType::OBrace,
                    '}' => TokenType::CBrace,
                    ',' => TokenType::Comma,
                    ';' => TokenType::Term,
                    _ => unreachable!(),
                }
            }
            c if OP_SYMBOLS.contains(c) => {
                let value = t.take_while(|c| OP_SYMBOLS.contains(c));
                TokenType::OpSym(value)
            }
            c if c.is_alphabetic() => {
                let identifier = t.identifier()?;
                if let Some(keyword) = match identifier.as_str() {
                    "op" => Some(Keyword::Op),
                    "method" => Some(Keyword::Method),
                    "sets" => Some(Keyword::Sets),
                    "unsets" => Some(Keyword::Unsets),
                    "to" => Some(Keyword::To),
                    "requires" => Some(Keyword::Requires),
                    "becomes" => Some(Keyword::Becomes),
                    "state" => Some(Keyword::State),
                    "and" => Some(Keyword::And),
                    "is" => Some(Keyword::Is),
                    "or" => Some(Keyword::Or),
                    "not" => Some(Keyword::Not),
                    "goals" => Some(Keyword::Goals),
                    "true" => Some(Keyword::True),
                    "false" => Some(Keyword::False),
                    _ => None,
                } {
                    TokenType::Keyword(keyword)
                } else {
                    TokenType::Sym(identifier)
                }
            }
            c if c.is_numeric() => {
                let value = t.take_while(|c| c.is_numeric());
                let value = value.parse::<i32>().map_err(|e| TokenizeErr {
                    kind: TokenizeErrKind::InvalidLitInt(e),
                    pos: start,
                })?;
                TokenType::LitInt(value)
            }
            _ => {
                return Err(TokenizeErr {
                    kind: TokenizeErrKind::UnexpectedChar(c),
                    pos: t.pos,
                });
            }
        };

        let span = Span {
            start,
            end: t.pos,
        };
        tokens.push(Token {
            token_type,
            span,
        });
    }

    Ok(tokens)
}

#[derive(Clone, Eq, PartialEq, Debug)]
pub enum ParseErrKind {
    UnexpectedEndOfInput,
    UnexpectedTokenType { got: Token, expected: String },
    UnbalancedBracket(char),
    DuplicateState { previous_pos: SourcePos },
    TokenizeErr(TokenizeErrKind),
}

impl From<TokenizeErrKind> for ParseErrKind {
    fn from(value: TokenizeErrKind) -> Self {
        Self::TokenizeErr(value)
    }
}

impl Display for ParseErrKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnexpectedTokenType { got, expected } =>
                write!(f, "unexpected token type, expected {} but got {:?}", expected, got),
            Self::UnexpectedEndOfInput =>
                write!(f, "unexpected end of input"),
            Self::UnbalancedBracket(c) =>
                write!(f, "unbalanced bracket (mismatched '{}')", c),
            Self::DuplicateState { previous_pos } =>
                write!(f, "duplicate state variable, previously declared at {:?}", previous_pos),
            Self::TokenizeErr(e) =>
                write!(f, "failed during tokenization: {}", e),
        }
    }
}

impl Display for ParseErr {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if !self.context.is_empty() {
            write!(f, "{}", self.context)?;
        }
        write!(f, "{} @ {:?}", self.kind, self.pos)
    }
}

impl Error for ParseErr {}

#[derive(Clone, Eq, PartialEq)]
pub struct ParseErr {
    pos: SourcePos,
    kind: ParseErrKind,
    context: String,
}

impl From<TokenizeErr> for ParseErr {
    fn from(value: TokenizeErr) -> Self {
        Self {
            context: String::new(),
            kind: value.kind.into(),
            pos: value.pos,
        }
    }
}

impl Debug for ParseErr {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "ParseErr(")?;
        if !self.context.is_empty() {
            write!(f, "{}", self.context)?;
        }
        write!(f, "{:?} @ {:?})", self.kind, self.pos)
    }
}

impl ParseErr {
    fn from_token(t: &Token, kind: ParseErrKind) -> Self {
        Self {
            pos: t.span.start,
            context: String::new(),
            kind,
        }
    }

    fn unexpected_token(t: &Token, expected: String) -> Self {
        Self::from_token(t, ParseErrKind::UnexpectedTokenType {
            got: t.clone(),
            expected,
        })
    }

    fn unexpected_eoi(tokens: &[Token]) -> ParseErr {
        Self {
            pos: tokens.iter()
                .last()
                .map(|t| t.span.end)
                .unwrap_or_default(),
            kind: ParseErrKind::UnexpectedEndOfInput,
            context: String::new(),
        }
    }

    fn with_context(mut self, ctx: impl AsRef<str>) -> Self {
        self.context.insert_str(0, &format!("{}: ", ctx.as_ref()));
        self
    }
}

trait ParseResultExt {
    fn with_context(self, ctx: impl AsRef<str>) -> Self;
}

impl<T> ParseResultExt for Result<T, ParseErr> {
    fn with_context(self, ctx: impl AsRef<str>) -> Self {
        self.map_err(|e| e.with_context(ctx))
    }
}

struct Parser<'a> {
    tokens: &'a [Token],
    index: usize,
}

impl<'a> Parser<'a> {
    fn new(tokens: &'a [Token]) -> Self {
        Self {
            tokens,
            index: 0,
        }
    }

    fn peek(&self) -> Option<Token> {
        self.tokens.iter().nth(self.index).cloned()
    }

    fn peek_is(&self, expected: TokenType) -> bool {
        self.peek().map(|t| t.token_type == expected).unwrap_or(false)
    }

    fn peek_or_err(&self) -> Result<Token, ParseErr> {
        self.peek().ok_or_else(|| ParseErr::unexpected_eoi(self.tokens))
    }

    fn consume(&mut self) -> Result<Token, ParseErr> {
        let result = self.peek_or_err()?;
        self.index += 1;
        Ok(result)
    }

    fn expect(&mut self, f: impl FnOnce(&Token) -> Result<(), ParseErrKind>) -> Result<Token, ParseErr> {
        let t = self.consume()?;
        match f(&t) {
            Ok(_) => Ok(t),
            Err(e) => Err(ParseErr::from_token(&t, e))
        }
    }

    fn expect_exact_type(&mut self, expected: TokenType, err: impl FnOnce() -> String) -> Result<Token, ParseErr> {
        self.expect(|t| if t.token_type == expected {
            Ok(())
        } else {
            Err(ParseErrKind::UnexpectedTokenType {
                got: t.clone(),
                expected: err(),
            })
        })
    }

    fn expect_keyword(&mut self, keyword: Keyword) -> Result<Token, ParseErr> {
        self.expect_exact_type(TokenType::Keyword(keyword), || format!("keyword {}", keyword))
    }

    fn expect_open_paren(&mut self) -> Result<Token, ParseErr> {
        self.expect_exact_type(TokenType::OParen, || "(".into())
    }

    fn expect_close_paren(&mut self) -> Result<Token, ParseErr> {
        self.expect_exact_type(TokenType::CParen, || ")".into())
    }

    fn expect_open_brace(&mut self) -> Result<Token, ParseErr> {
        self.expect_exact_type(TokenType::OBrace, || "{".into())
    }

    fn expect_close_brace(&mut self) -> Result<Token, ParseErr> {
        self.expect_exact_type(TokenType::CBrace, || "}".into())
    }

    fn expect_open_square(&mut self) -> Result<Token, ParseErr> {
        self.expect_exact_type(TokenType::OSquare, || "[".into())
    }

    fn expect_close_square(&mut self) -> Result<Token, ParseErr> {
        self.expect_exact_type(TokenType::CSquare, || "]".into())
    }

    fn expect_comma(&mut self) -> Result<Token, ParseErr> {
        self.expect_exact_type(TokenType::Comma, || ",".into())
    }

    fn expect_term(&mut self) -> Result<Token, ParseErr> {
        self.expect_exact_type(TokenType::Term, || ";".into())
    }

    fn parse_symbol(&mut self) -> Result<String, ParseErr> {
        let token = self.consume()?;
        match token.token_type {
            TokenType::Sym(n) => Ok(n),
            _ => Err(ParseErr::unexpected_token(&token, "symbol".into())),
        }
    }

    fn parse_var(&mut self) -> Result<String, ParseErr> {
        let token = self.consume()?;
        match token.token_type {
            TokenType::Var(n) => Ok(n),
            _ => Err(ParseErr::unexpected_token(&token, "var".into())),
        }
    }

    fn parse_param_decl(&mut self) -> Result<Vec<String>, ParseErr> {
        self.expect_open_paren()?;

        let mut result = Vec::new();
        loop {
            let mut t = self.peek_or_err()?;

            if t.token_type == TokenType::CParen {
                break;
            }

            if result.len() > 0 {
                self.expect_comma()?;
                t = self.peek_or_err()?;

                if t.token_type == TokenType::CParen {
                    break;
                }
            }

            let param = self.parse_var().with_context("while parsing var")?;
            result.push(param);
        }

        self.expect_close_paren()?;

        Ok(result)
    }

    fn parse_state_name(&mut self) -> Result<StateName, ParseErr> {
        let name = self.parse_symbol()?;
        if !self.peek_is(TokenType::OParen) {
            return Ok(name.into());
        }

        let mut args = Vec::new();
        self.expect_open_paren()?;
        loop {
            let mut t = self.peek_or_err()?;

            if t.token_type == TokenType::CParen {
                break;
            }

            if args.len() > 0 {
                self.expect_comma()?;
                t = self.peek_or_err()?;

                if t.token_type == TokenType::CParen {
                    break;
                }
            }

            let expr = self.parse_expr()?;
            args.push(expr);
        }
        self.expect_close_paren()?;

        Ok(StateName::new_with_args(name, args))
    }

    fn parse_lit_int(&mut self) -> Result<Expr, ParseErr> {
        let t = self.consume()?;
        match t.token_type {
            TokenType::LitInt(i) => Ok(Expr::Val(StateValue::Int(i))),
            _ => Err(ParseErr::unexpected_token(&t, "int literal".into())),
        }
    }

    fn parse_state_access(&mut self) -> Result<Expr, ParseErr> {
        self.expect_keyword(Keyword::State)?;
        self.expect_open_square()?;
        let expr = self.parse_expr()?;
        self.expect_close_square()?;
        Ok(Expr::State(Box::new(expr)))
    }

    fn parse_call(&mut self, name: String) -> Result<Expr, ParseErr> {
        self.expect_open_paren()?;
        let mut args = Vec::new();
        loop {
            let mut t = self.peek_or_err()?;

            if t.token_type == TokenType::CParen {
                break;
            }

            if args.len() > 0 {
                self.expect_comma()?;
                t = self.peek_or_err()?;

                if t.token_type == TokenType::CParen {
                    break;
                }
            }

            let value = self.parse_expr().with_context("while parsing argument")?;
            args.push(value);
        }
        self.expect_close_paren()?;

        Ok(Expr::Call(name, args))
    }

    fn parse_sym_or_call(&mut self) -> Result<Expr, ParseErr> {
        let name = self.parse_symbol()?;
        if !self.peek_is(TokenType::OParen) {
            Ok(Expr::Val(StateValue::Str(name)))
        } else {
            self.parse_call(name).with_context("while parsing call")
        }
    }

    fn parse_expr(&mut self) -> Result<Expr, ParseErr> {
        struct ShuntingYard {
            output_stack: Vec<ShuntOutput>,
            operator_stack: Vec<ShuntOp>,
        }

        impl ShuntingYard {
            fn push_op_stack(&mut self, op: ShuntOp) {
                let op_precedence = op.precedence();
                let op_associativity = op.associativity();
                while let Some(top) = self.operator_stack.last().cloned() {
                    if top == ShuntOp::OParen {
                        break;
                    }

                    let top_precedence = top.precedence();
                    if top_precedence > op_precedence ||
                        (top_precedence == op_precedence && op_associativity == Assoc::Left) {
                        self.operator_stack.pop();
                        self.output_stack.push(ShuntOutput::Operator(top));
                    }

                    break;
                }

                self.operator_stack.push(op);
            }
        }

        let mut yard = ShuntingYard {
            output_stack: Vec::new(),
            operator_stack: Vec::new(),
        };

        #[derive(Debug)]
        enum ShuntOutput {
            Value(Expr),
            Operator(ShuntOp),
        }

        #[derive(Debug, Copy, Clone, Eq, PartialEq)]
        enum Assoc {
            None,
            Left,
        }

        #[derive(Copy, Clone, Debug, Eq, PartialEq)]
        enum ShuntOp {
            OParen,
            Add,
            Sub,
            Mul,
            Div,
            And,
            Mod,
            Or,
            Is,
            IsNot,
            GreaterThan,
            LessThan,
            GreaterOrEqualTo,
            LessThanOrEqualTo,
        }

        impl ShuntOp {
            fn precedence(&self) -> i32 {
                match self {
                    Self::Mul | Self::Div | Self::Mod => 5,
                    Self::Add | Self::Sub => 4,
                    Self::Is | Self::IsNot | Self::GreaterThan | Self::GreaterOrEqualTo |
                    Self::LessThan | Self::LessThanOrEqualTo => 3,
                    Self::And => 2,
                    Self::Or => 1,
                    Self::OParen => panic!("can only get precedence of operators"),
                }
            }

            fn associativity(&self) -> Assoc {
                match self {
                    Self::OParen => Assoc::None,
                    _ => Assoc::Left,
                }
            }
        }

        'outer:
        while let Some(t) = self.peek() {
            match &t.token_type {
                TokenType::LitInt(_) => {
                    yard.output_stack.push(ShuntOutput::Value(self.parse_lit_int()?));
                },
                TokenType::Sym(_) => {
                    yard.output_stack.push(ShuntOutput::Value(self.parse_sym_or_call()?));
                },
                TokenType::Var(_) => {
                    yard.output_stack.push(ShuntOutput::Value(Expr::State(Box::new(
                        Expr::Val(StateValue::Str(self.parse_var()?))))));
                },
                TokenType::Keyword(Keyword::State) => {
                    yard.output_stack.push(ShuntOutput::Value(self.parse_state_access()?));
                },
                TokenType::OParen => {
                    self.expect_open_paren()?;
                    yard.operator_stack.push(ShuntOp::OParen);
                },
                TokenType::CParen => {
                    loop {
                        let Some(top) = yard.operator_stack.pop() else {
                            break 'outer;
                        };

                        if top == ShuntOp::OParen {
                            self.expect_close_paren()?;
                            break;
                        }

                        yard.output_stack.push(ShuntOutput::Operator(top));
                    }
                },
                TokenType::Keyword(Keyword::Is) => {
                    self.expect_keyword(Keyword::Is)?;
                    if self.peek_is(TokenType::Keyword(Keyword::Not)) {
                        self.expect_keyword(Keyword::Not)?;
                        yard.push_op_stack(ShuntOp::IsNot);
                    } else {
                        yard.push_op_stack(ShuntOp::Is);
                    }
                },
                TokenType::Keyword(Keyword::And) => {
                    self.expect_keyword(Keyword::And)?;
                    yard.push_op_stack(ShuntOp::And);
                },
                TokenType::Keyword(Keyword::Or) => {
                    self.expect_keyword(Keyword::Or)?;
                    yard.push_op_stack(ShuntOp::Or);
                },
                TokenType::Keyword(Keyword::True) => {
                    self.expect_keyword(Keyword::True)?;
                    yard.output_stack.push(ShuntOutput::Value(Expr::Val(true.into())));
                },
                TokenType::Keyword(Keyword::False) => {
                    self.expect_keyword(Keyword::False)?;
                    yard.output_stack.push(ShuntOutput::Value(Expr::Val(false.into())));
                },
                TokenType::OpSym(op) => {
                    let op = match op.as_str() {
                        ">" => ShuntOp::GreaterThan,
                        "<" => ShuntOp::LessThan,
                        ">=" => ShuntOp::GreaterOrEqualTo,
                        "<=" => ShuntOp::LessThanOrEqualTo,
                        "+" => ShuntOp::Add,
                        "-" => ShuntOp::Sub,
                        "*" => ShuntOp::Mul,
                        "/" => ShuntOp::Div,
                        "%" => ShuntOp::Mod,
                        _ => return Err(ParseErr::unexpected_token(&t, "operator".into())),
                    };
                    self.consume()?;
                    yard.push_op_stack(op);
                },
                _ => break,
            }
        }

        while let Some(op) = yard.operator_stack.pop() {
            if op == ShuntOp::OParen {
                return Err(ParseErr {
                    kind: ParseErrKind::UnbalancedBracket(')'),
                    pos: self.peek()
                        .map(|t| t.span.start)
                        .unwrap_or_else(|| self.tokens.last()
                            .map(|t| t.span.start)
                            .unwrap_or_default()),
                    context: String::new(),
                });
            }

            yard.output_stack.push(ShuntOutput::Operator(op));
        }

        let mut stack = Vec::with_capacity(2);
        for output in yard.output_stack.iter() {
            match output {
                ShuntOutput::Value(v) => stack.push(v.clone()),
                ShuntOutput::Operator(op) => {
                    let right = stack.pop().expect("malformed output stack (right)");
                    let left = stack.pop().expect("malformed output stack (left)");
                    let op = match op {
                        ShuntOp::Add => BinOp::Add,
                        ShuntOp::Sub => BinOp::Sub,
                        ShuntOp::Mul => BinOp::Mul,
                        ShuntOp::Div => BinOp::Div,
                        ShuntOp::Mod => BinOp::Mod,
                        ShuntOp::And => BinOp::And,
                        ShuntOp::Or => BinOp::Or,
                        ShuntOp::Is => BinOp::Is,
                        ShuntOp::IsNot => BinOp::IsNot,
                        ShuntOp::GreaterThan => BinOp::GreaterThan,
                        ShuntOp::LessThan => BinOp::LessThan,
                        ShuntOp::GreaterOrEqualTo => BinOp::GreaterOrEqualTo,
                        ShuntOp::LessThanOrEqualTo => BinOp::LessThanOrEqualTo,
                        ShuntOp::OParen => panic!("parens should not be in output stack"),
                    };
                    stack.push(Expr::BinOp(
                        Box::new(left),
                        op,
                        Box::new(right),
                    ));
                }
            }
        }

        if stack.len() != 1 {
            panic!("final shunting yard stack has non-1 length (stack: {:?}, output stack: {:?})", stack, yard.output_stack);
        }

        Ok(stack.pop().unwrap())
    }

    fn parse_operator(&mut self) -> Result<Operator, ParseErr> {
        self.expect_keyword(Keyword::Op)?;

        let name = self.parse_symbol().with_context("while parsing operator name")?;
        let params = self.parse_param_decl().with_context("while parsing parameter declarations")?;
        let mut additions = HashMap::new();
        let mut removals = HashSet::new();

        self.expect_open_brace()?;
        while let Some(t) = self.peek() {
            match t.token_type {
                TokenType::CBrace => break,
                TokenType::Keyword(Keyword::Sets) => {
                    self.expect_keyword(Keyword::Sets)?;
                    let state_name = self.parse_state_name().with_context("while parsing sets statement state name")?;
                    self.expect_keyword(Keyword::To)?;
                    let expr = self.parse_expr().with_context("while parsing sets statement value")?;
                    additions.insert(state_name, expr);
                }
                TokenType::Keyword(Keyword::Unsets) => {
                    self.expect_keyword(Keyword::Unsets)?;
                    let state_name = self.parse_state_name().with_context("while parsing unsets statement state name")?;
                    removals.insert(state_name);
                }
                _ => return Err(ParseErr::unexpected_token(&t, "sets or unsets statement".into()))
            }

            self.expect_term()?;
        }

        self.expect_close_brace()?;

        Ok(Operator {
            name,
            params,
            additions,
            removals,
        })
    }

    fn parse_task(&mut self) -> Result<Task, ParseErr> {
        let name = self.parse_symbol()?;
        let mut parameters = Vec::new();

        self.expect_open_paren()?;
        loop {
            let mut t = self.peek_or_err()?;

            if t.token_type == TokenType::CParen {
                break;
            }

            if parameters.len() > 0 {
                self.expect_comma()?;
                t = self.peek_or_err()?;

                if t.token_type == TokenType::CParen {
                    break;
                }
            }

            let expr = self.parse_expr().with_context("while parsing task parameter")?;
            parameters.push(expr);
        }
        self.expect_close_paren()?;

        Ok(Task {
            name,
            parameters,
        })
    }

    fn parse_method(&mut self) -> Result<Method, ParseErr> {
        self.expect_keyword(Keyword::Method)?;

        let name = self.parse_symbol()?;
        let params = self.parse_param_decl()?;
        let mut conditions = Vec::new();
        let mut sub_tasks = Vec::new();

        self.expect_open_brace()?;
        while let Some(t) = self.peek() {
            match t.token_type {
                TokenType::CBrace => break,
                TokenType::Keyword(Keyword::Requires) => {
                    self.expect_keyword(Keyword::Requires)?;
                    let condition = self.parse_expr().with_context("while parsing requires statement")?;
                    conditions.push(condition);
                }
                TokenType::Keyword(Keyword::Becomes) => {
                    if sub_tasks.len() > 0 {
                        return Err(ParseErr::unexpected_token(&t, "method can only contain one becomes statement".into()));
                    }

                    self.expect_keyword(Keyword::Becomes)?;
                    loop {
                        let mut t = self.peek_or_err()?;

                        if t.token_type == TokenType::Term {
                            break;
                        }

                        if sub_tasks.len() > 0 {
                            self.expect_comma()?;
                            t = self.peek_or_err()?;

                            if t.token_type == TokenType::Term {
                                break;
                            }
                        }

                        let task = self.parse_task().with_context("while parsing subtask")?;
                        sub_tasks.push(task);
                    }
                }
                _ => return Err(ParseErr::unexpected_token(&t, "requires or becomes statement".into()))
            }

            self.expect_term()?;
        }

        self.expect_close_brace()?;

        Ok(Method {
            name,
            params,
            conditions,
            sub_tasks,
        })
    }

    fn parse_domain(&mut self) -> Result<Domain, ParseErr> {
        let mut result = Domain {
            operators: Vec::new(),
            methods: Vec::new(),
        };

        while let Some(t) = self.peek() {
            match t.token_type {
                TokenType::Keyword(Keyword::Op) => {
                    let operator = self.parse_operator().with_context("while parsing operator")?;
                    result.operators.push(operator);
                }
                TokenType::Keyword(Keyword::Method) => {
                    let method = self.parse_method().with_context("while parsing method")?;
                    result.methods.push(method);
                }
                _ => return Err(ParseErr::unexpected_token(&t, "operator or method".into())),
            }
        }

        Ok(result)
    }

    fn parse_initial_state_value(&mut self) -> Result<(StateName, StateValue), ParseErr> {
        let name = self.parse_state_name()?;
        self.expect_keyword(Keyword::Is)?;
        let value = self.consume()?;
        let value = match value.token_type {
            TokenType::Sym(x) => StateValue::Str(x),
            TokenType::LitInt(x) => StateValue::Int(x),
            TokenType::Keyword(Keyword::True) => StateValue::Bool(true),
            TokenType::Keyword(Keyword::False) => StateValue::Bool(false),
            _ => return Err(ParseErr::unexpected_token(&value, "expected simple state value".into())),
        };

        Ok((name, value))
    }

    fn parse_world_state(&mut self) -> Result<Rc<RefCell<WorldState>>, ParseErr> {
        self.expect_keyword(Keyword::State)?;

        let mut lines = HashMap::new();
        let mut states = HashMap::new();

        self.expect_open_brace()?;
        loop {
            let mut t = self.peek_or_err()?;

            if t.token_type == TokenType::CBrace {
                break;
            }

            if states.len() > 0 {
                self.expect_comma()?;
                t = self.peek_or_err()?;

                if t.token_type == TokenType::CBrace {
                    break;
                }
            }

            let (name, value) = self.parse_initial_state_value()?;
            if let Some(pos) = lines.get(&name) {
                return Err(ParseErr::from_token(&t, ParseErrKind::DuplicateState {
                    previous_pos: *pos,
                }));
            }

            states.insert(name.clone(), Some(value));
            lines.insert(name, t.span.start);
        }
        self.expect_close_brace()?;

        Ok(Rc::new(RefCell::new(WorldState {
            parent: None,
            calls: HashMap::new(),
            values: states,
        })))
    }

    fn parse_goals(&mut self) -> Result<Vec<Task>, ParseErr> {
        self.expect_keyword(Keyword::Goals)?;
        let mut goals = Vec::new();
        loop {
            let mut t = self.peek_or_err()?;

            if t.token_type == TokenType::Term {
                break;
            }

            if goals.len() > 0 {
                self.expect_comma()?;
                t = self.peek_or_err()?;

                if t.token_type == TokenType::Term {
                    break;
                }
            }

            let task = self.parse_task()?;
            goals.push(task);
        }
        let term = self.expect_term()?;
        if goals.len() < 1 {
            return Err(ParseErr::unexpected_token(&term, "goals statement must contain at least one task".into()));
        }

        Ok(goals)
    }

    fn parse_problem(&mut self) -> Result<Problem, ParseErr> {
        let mut result = Problem {
            state: Rc::new(RefCell::new(WorldState::new())),
            tasks: VecDeque::new(),
            tracer: Rc::new(RefCell::new(NullTracer)),
        };

        let mut seen_state = false;
        let mut seen_goals = false;
        while let Some(t) = self.peek() {
            match t.token_type {
                TokenType::Keyword(Keyword::State) => {
                    if seen_state {
                        return Err(ParseErr::unexpected_token(&t, "only one state block can appear per problem".into()));
                    }

                    result.state = self.parse_world_state().with_context("while parsing world state")?;
                    seen_state = true;
                }
                TokenType::Keyword(Keyword::Goals) => {
                    if seen_goals {
                        return Err(ParseErr::unexpected_token(&t, "only one goals statement can appear per problem".into()));
                    }

                    result.tasks.extend(self.parse_goals().with_context("while parsing goals")?);
                    seen_goals = true;
                }
                _ => return Err(ParseErr::unexpected_token(&t, "world state or goal tasks".into())),
            }
        }

        Ok(result)
    }
}

pub fn parse_domain_from_tokens(tokens: &[Token]) -> Result<Domain, ParseErr> {
    Parser::new(tokens).parse_domain().with_context("while parsing domain")
}

pub fn parse_domain(src: &str) -> Result<Domain, ParseErr> {
    let tokens = tokenize(src)?;
    parse_domain_from_tokens(&tokens)
}

pub fn parse_problem_from_tokens(tokens: &[Token]) -> Result<Problem, ParseErr> {
    Parser::new(tokens).parse_problem().with_context("while parsing problem")
}

pub fn parse_problem(src: &str) -> Result<Problem, ParseErr> {
    let tokens = tokenize(src)?;
    parse_problem_from_tokens(&tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! assert_dbg_eq {
        ($expected:expr, $actual:expr) => {
            assert_eq!(format!("{:#?}", $expected), format!("{:#?}", $actual));
        }
    }

    #[test]
    fn test_tokenize_domain() {
        let result = tokenize(EXAMPLE_DOMAIN);
        assert_dbg_eq!(result.clone().err(), None::<TokenizeErr>);
        let tokens = result.unwrap();
        let types = tokens.iter()
            .map(|t| t.token_type.clone())
            .collect::<Vec<_>>();
        assert_dbg_eq!(types, vec![
            TokenType::Keyword(Keyword::Op),
            TokenType::Sym("Walk".into()),
            TokenType::OParen,
            TokenType::Var("?To".into()),
            TokenType::CParen,
            TokenType::OBrace,
            TokenType::Keyword(Keyword::Sets),
            TokenType::Sym("Location".into()),
            TokenType::Keyword(Keyword::To),
            TokenType::Var("?To".into()),
            TokenType::Term,
            TokenType::CBrace,
            TokenType::Keyword(Keyword::Op),
            TokenType::Sym("Ride".into()),
            TokenType::OParen,
            TokenType::Var("?To".into()),
            TokenType::CParen,
            TokenType::OBrace,
            TokenType::Keyword(Keyword::Sets),
            TokenType::Sym("Location".into()),
            TokenType::Keyword(Keyword::To),
            TokenType::Var("?To".into()),
            TokenType::Term,
            TokenType::CBrace,
            TokenType::Keyword(Keyword::Op),
            TokenType::Sym("HailTaxi".into()),
            TokenType::OParen,
            TokenType::Var("?Location".into()),
            TokenType::CParen,
            TokenType::OBrace,
            TokenType::Keyword(Keyword::Sets),
            TokenType::Sym("TaxiLocation".into()),
            TokenType::OParen,
            TokenType::Sym("Taxi1".into()),
            TokenType::CParen,
            TokenType::Keyword(Keyword::To),
            TokenType::Var("?Location".into()),
            TokenType::Term,
            TokenType::CBrace,
            TokenType::Keyword(Keyword::Op),
            TokenType::Sym("SetCash".into()),
            TokenType::OParen,
            TokenType::Var("?Amount".into()),
            TokenType::CParen,
            TokenType::OBrace,
            TokenType::Keyword(Keyword::Sets),
            TokenType::Sym("Cash".into()),
            TokenType::Keyword(Keyword::To),
            TokenType::Var("?Amount".into()),
            TokenType::Term,
            TokenType::CBrace,
            TokenType::Keyword(Keyword::Method),
            TokenType::Sym("TravelTo".into()),
            TokenType::OParen,
            TokenType::Var("?X".into()),
            TokenType::CParen,
            TokenType::OBrace,
            TokenType::Keyword(Keyword::Requires),
            TokenType::Keyword(Keyword::State),
            TokenType::OSquare,
            TokenType::Sym("Weather".into()),
            TokenType::CSquare,
            TokenType::Keyword(Keyword::Is),
            TokenType::Sym("Good".into()),
            TokenType::Keyword(Keyword::And),
            TokenType::Sym("DistanceFrom".into()),
            TokenType::OParen,
            TokenType::Keyword(Keyword::State),
            TokenType::OSquare,
            TokenType::Sym("Location".into()),
            TokenType::CSquare,
            TokenType::Comma,
            TokenType::Var("?X".into()),
            TokenType::CParen,
            TokenType::OpSym("<".into()),
            TokenType::LitInt(3000),
            TokenType::Term,
            TokenType::Keyword(Keyword::Becomes),
            TokenType::Sym("Walk".into()),
            TokenType::OParen,
            TokenType::Var("?X".into()),
            TokenType::CParen,
            TokenType::Term,
            TokenType::CBrace,
            TokenType::Keyword(Keyword::Method),
            TokenType::Sym("TravelTo".into()),
            TokenType::OParen,
            TokenType::Var("?X".into()),
            TokenType::CParen,
            TokenType::OBrace,
            TokenType::Keyword(Keyword::Requires),
            TokenType::Keyword(Keyword::State),
            TokenType::OSquare,
            TokenType::Sym("Cash".into()),
            TokenType::CSquare,
            TokenType::OpSym(">=".into()),
            TokenType::Keyword(Keyword::State),
            TokenType::OSquare,
            TokenType::Sym("TaxiFare".into()),
            TokenType::CSquare,
            TokenType::Keyword(Keyword::And),
            TokenType::Sym("TaxiLocation".into()),
            TokenType::OParen,
            TokenType::Sym("Taxi1".into()),
            TokenType::CParen,
            TokenType::Keyword(Keyword::Is),
            TokenType::Sym("TaxiBase".into()),
            TokenType::Term,
            TokenType::Keyword(Keyword::Becomes),
            TokenType::Sym("HailTaxi".into()),
            TokenType::OParen,
            TokenType::Keyword(Keyword::State),
            TokenType::OSquare,
            TokenType::Sym("Location".into()),
            TokenType::CSquare,
            TokenType::CParen,
            TokenType::Comma,
            TokenType::Sym("Ride".into()),
            TokenType::OParen,
            TokenType::Var("?X".into()),
            TokenType::CParen,
            TokenType::Comma,
            TokenType::Sym("PayDriver".into()),
            TokenType::OParen,
            TokenType::Keyword(Keyword::State),
            TokenType::OSquare,
            TokenType::Sym("TaxiFare".into()),
            TokenType::CSquare,
            TokenType::CParen,
            TokenType::Term,
            TokenType::CBrace,
            TokenType::Keyword(Keyword::Method),
            TokenType::Sym("PayDriver".into()),
            TokenType::OParen,
            TokenType::Var("?Amount".into()),
            TokenType::CParen,
            TokenType::OBrace,
            TokenType::Keyword(Keyword::Requires),
            TokenType::Keyword(Keyword::State),
            TokenType::OSquare,
            TokenType::Sym("Cash".into()),
            TokenType::CSquare,
            TokenType::OpSym(">=".into()),
            TokenType::Var("?Amount".into()),
            TokenType::Term,
            TokenType::Keyword(Keyword::Becomes),
            TokenType::Sym("SetCash".into()),
            TokenType::OParen,
            TokenType::Keyword(Keyword::State),
            TokenType::OSquare,
            TokenType::Sym("Cash".into()),
            TokenType::CSquare,
            TokenType::OpSym("-".into()),
            TokenType::Var("?Amount".into()),
            TokenType::CParen,
            TokenType::Term,
            TokenType::CBrace,
        ]);
    }

    #[test]
    fn test_tokenize_problem() {
        let result = tokenize(EXAMPLE_PROBLEM);
        assert_dbg_eq!(result.clone().err(), None::<TokenizeErr>);
        let tokens = result.unwrap();
        let types = tokens.iter()
            .map(|t| t.token_type.clone())
            .collect::<Vec<_>>();
        assert_dbg_eq!(types, vec![
            TokenType::Keyword(Keyword::State),
            TokenType::OBrace,
            TokenType::Sym("Location".into()),
            TokenType::Keyword(Keyword::Is),
            TokenType::Sym("Downtown".into()),
            TokenType::Comma,
            TokenType::Sym("Weather".into()),
            TokenType::Keyword(Keyword::Is),
            TokenType::Sym("Good".into()),
            TokenType::Comma,
            TokenType::Sym("Cash".into()),
            TokenType::Keyword(Keyword::Is),
            TokenType::LitInt(12),
            TokenType::Comma,
            TokenType::Sym("TaxiFare".into()),
            TokenType::Keyword(Keyword::Is),
            TokenType::LitInt(2),
            TokenType::Comma,
            TokenType::Sym("DistanceFrom".into()),
            TokenType::OParen,
            TokenType::Sym("Downtown".into()),
            TokenType::Comma,
            TokenType::Sym("Park".into()),
            TokenType::CParen,
            TokenType::Keyword(Keyword::Is),
            TokenType::LitInt(2000),
            TokenType::Comma,
            TokenType::Sym("TaxiLocation".into()),
            TokenType::OParen,
            TokenType::Sym("Taxi1".into()),
            TokenType::CParen,
            TokenType::Keyword(Keyword::Is),
            TokenType::Sym("TaxiBase".into()),
            TokenType::Comma,
            TokenType::CBrace,
            TokenType::Keyword(Keyword::Goals),
            TokenType::Sym("TravelTo".into()),
            TokenType::OParen,
            TokenType::Sym("Park".into()),
            TokenType::CParen,
            TokenType::Term,
        ]);
    }

    #[test]
    fn test_parse_problem() {
        let tokens = tokenize(EXAMPLE_PROBLEM)
            .expect("tokenizing failed");
        let result = parse_problem_from_tokens(&tokens);
        assert_dbg_eq!(result.clone().err(), None::<ParseErr>);
        let problem = result.unwrap();
        assert_dbg_eq!(problem, Problem {
            state: Rc::new(RefCell::new(WorldState {
                parent: None,
                calls: HashMap::new(),
                values: HashMap::from([
                    ("Location".into(), Some(StateValue::Str("Downtown".into()))),
                    ("Weather".into(), Some(StateValue::Str("Good".into()))),
                    ("Cash".into(), Some(StateValue::Int(12))),
                    ("TaxiFare".into(), Some(StateValue::Int(2))),
                    (StateName::new_with_args("DistanceFrom", vec![Expr::Val("downtown".into()), Expr::Val("Park".into())]), Some(2000.into())),
                    (StateName::new_with_args("TaxiLocation", vec![Expr::Val("Taxi1".into())]), Some("TaxiBase".into())),
                ]),
            })),
            tasks: VecDeque::from([
                Task {
                    name: "TravelTo".into(),
                    parameters: vec![
                        Expr::Val(StateValue::Str("Park".into())),
                    ],
                },
            ]),
            tracer: Rc::new(RefCell::new(NullTracer)),
        });
    }

    #[test]
    fn test_parse_domain() {
        let tokens = tokenize(EXAMPLE_DOMAIN)
            .expect("tokenizing failed");
        let result = parse_domain_from_tokens(&tokens);
        assert_dbg_eq!(result.clone().err(), None::<ParseErr>);
        let domain = result.unwrap();
        assert_dbg_eq!(domain, Domain {
            operators: vec![
                Operator {
                    name: "Walk".into(),
                    params: vec![
                        "?To".into(),
                    ],
                    removals: HashSet::from([]),
                    additions: HashMap::from([
                        ("Location".into(), Expr::State(Box::new(Expr::Val("?To".into())))),
                    ]),
                },
                Operator {
                    name: "Ride".into(),
                    params: vec![
                        "?To".into(),
                    ],
                    removals: HashSet::from([]),
                    additions: HashMap::from([
                        ("Location".into(), Expr::State(Box::new(Expr::Val(StateValue::Str("?To".into()))))),
                    ]),
                },
                Operator {
                    name: "HailTaxi".into(),
                    params: vec![
                        "?Location".into(),
                    ],
                    removals: HashSet::from([]),
                    additions: HashMap::from([
                        (StateName::new_with_args("TaxiLocation", vec![Expr::Val("Taxi".into())]), Expr::State(Box::new(Expr::Val("?Location".into())))),
                    ]),
                },
                Operator {
                    name: "SetCash".into(),
                    params: vec![
                        "?Amount".into(),
                    ],
                    removals: HashSet::from([]),
                    additions: HashMap::from([
                        ("Cash".into(), Expr::State(Box::new(Expr::Val("?Amount".into())))),
                    ]),
                },
            ],

            methods: vec![
                Method {
                    name: "TravelTo".into(),
                    params: vec![
                        "?X".into(),
                    ],
                    conditions: vec![
                        Expr::BinOp(
                            Box::new(
                                Expr::BinOp(
                                    Box::new(Expr::State(Box::new(Expr::Val(StateValue::Str("Weather".into()))))),
                                    BinOp::Is,
                                    Box::new(Expr::Val(StateValue::Str("Good".into()))),
                                ),
                            ),
                            BinOp::And,
                            Box::new(
                                Expr::BinOp(
                                    Box::new(Expr::Call(
                                        "DistanceFrom".into(),
                                        vec![
                                            Expr::State(Box::new(Expr::Val(StateValue::Str("Location".into())))),
                                            Expr::State(Box::new(Expr::Val(StateValue::Str("?X".into())))),
                                        ],
                                    )),
                                    BinOp::LessThan,
                                    Box::new(Expr::Val(StateValue::Int(3_000))),
                                ),
                            ),
                        ),
                    ],
                    sub_tasks: vec![
                        Task {
                            name: "Walk".into(),
                            parameters: vec![
                                Expr::State(Box::new(Expr::Val(StateValue::Str("?X".into())))),
                            ],
                        },
                    ],
                },
                Method {
                    name: "TravelTo".into(),
                    params: vec![
                        "?X".into(),
                    ],
                    conditions: vec![
                        Expr::BinOp(
                            Box::new(
                                Expr::BinOp(
                                    Box::new(Expr::State(Box::new(Expr::Val(StateValue::Str("Cash".into()))))),
                                    BinOp::GreaterOrEqualTo,
                                    Box::new(Expr::State(Box::new(Expr::Val(StateValue::Str("TaxiFare".into()))))),
                                ),
                            ),
                            BinOp::And,
                            Box::new(
                                Expr::BinOp(
                                    Box::new(Expr::Call(
                                        "TaxiLocation".into(),
                                        vec![
                                            Expr::Val(StateValue::Str("Taxi1".into())),
                                        ],
                                    )),
                                    BinOp::Is,
                                    Box::new(Expr::Val(StateValue::Str("TaxiBase".into()))),
                                ),
                            ),
                        ),
                    ],
                    sub_tasks: vec![
                        Task {
                            name: "HailTaxi".into(),
                            parameters: vec![
                                Expr::State(Box::new(Expr::Val(StateValue::Str("Location".into())))),
                            ],
                        },
                        Task {
                            name: "Ride".into(),
                            parameters: vec![
                                Expr::State(Box::new(Expr::Val(StateValue::Str("?X".into())))),
                            ],
                        },
                        Task {
                            name: "PayDriver".into(),
                            parameters: vec![
                                Expr::State(Box::new(Expr::Val(StateValue::Str("TaxiFare".into())))),
                            ],
                        },
                    ],
                },
                Method {
                    name: "PayDriver".into(),
                    params: vec![
                        "?Amount".into(),
                    ],
                    conditions: vec![
                        Expr::BinOp(
                            Box::new(Expr::State(Box::new(Expr::Val(StateValue::Str("Cash".into()))))),
                            BinOp::GreaterOrEqualTo,
                            Box::new(Expr::State(Box::new(Expr::Val(StateValue::Str("?Amount".into()))))),
                        ),
                    ],
                    sub_tasks: vec![
                        Task {
                            name: "SetCash".into(),
                            parameters: vec![
                                Expr::BinOp(
                                    Box::new(Expr::State(Box::new(Expr::Val(StateValue::Str("Cash".into()))))),
                                    BinOp::Sub,
                                    Box::new(Expr::State(Box::new(Expr::Val(StateValue::Str("?Amount".into()))))),
                                ),
                            ],
                        },
                    ],
                },
            ],
        });
    }
}
