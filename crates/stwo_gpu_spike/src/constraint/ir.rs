//! A component's constraints as a hash-consed SSA program, lowered from the expressions a
//! constraint-framework `ExprEvaluator` traces.
//!
//! Every value is an M31 (`Ty::Base`) or a QM31 (`Ty::Ext`). The lowering:
//! - resolves `intermediate<k>` parameters to their definitions, each lowered once;
//! - maps a column to `(interaction, index among the interaction's columns)` through the
//!   evaluator's `column_slots`, and a preprocessed column (`BaseExpr::Param(id)`) to its slot in
//!   interaction 0 through `preprocessed_columns` (first slot of an id): the order the SIMD
//!   evaluator reads them in;
//! - recovers base-field values from the extension embedding (`SecureCol([e, 0, 0, 0])` is `e`), so
//!   that an extension times a base value costs 4 multiplications, not 16;
//! - folds constants and applies exact field identities (`x + 0`, `x * 1`, `x * 0`, `--x`), and
//!   orders the operands of `+` and `*` so that equal expressions share one node.
//!
//! All of this is exact in the field: the values a program computes are those of the traced
//! expressions, as canonical field elements. A node whose operands are all constants or formal
//! parameters (relation elements, the claimed sum, the column size) is *uniform*: the same for
//! every row, computed once on the host ([`Program::bind`]).

use std::collections::HashMap;

use stwo::core::fields::cm31::CM31;
use stwo::core::fields::m31::{M31, P};
use stwo::core::fields::qm31::QM31;
use stwo_constraint_framework::expr::{BaseExpr, ExprEvaluator, ExtExpr};

pub type NodeId = u32;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Ty {
    Base,
    Ext,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Op {
    /// Column `index` of interaction `interaction` (interaction 0: the component's preprocessed
    /// columns, remapped), at `offset` rows of the trace domain from the row.
    Col {
        interaction: u32,
        index: u32,
        offset: i32,
    },
    /// A constant `(a, b, c, d)`, canonical; a base constant when `b = c = d = 0`.
    Const([u32; 4]),
    /// Formal parameter `Program::params[i]`.
    Param(u32),
    /// The extension value of four base values.
    Secure([NodeId; 4]),
    Add(NodeId, NodeId),
    Sub(NodeId, NodeId),
    Mul(NodeId, NodeId),
    Neg(NodeId),
    /// The inverse of a base value (0 for 0).
    Inv(NodeId),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Node {
    pub op: Op,
    pub ty: Ty,
    /// Whether the value is the same for every row (no column below it).
    pub uniform: bool,
}

/// A component's constraints: `constraints[i]` is constraint `i`'s node. Nodes come after their
/// operands.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Program {
    pub nodes: Vec<Node>,
    /// The formal parameters' names and types.
    pub params: Vec<(String, Ty)>,
    pub constraints: Vec<NodeId>,
}

pub fn qm31(words: [u32; 4]) -> QM31 {
    QM31::from_u32_unchecked(words[0], words[1], words[2], words[3])
}

pub fn words(value: QM31) -> [u32; 4] {
    value.to_m31_array().map(|m| m.0)
}

/// A field element as canonical words (`P` read as 0).
pub fn canonical(value: QM31) -> [u32; 4] {
    words(value).map(|w| w % P)
}

struct Lowering<'a> {
    eval: &'a ExprEvaluator,
    nodes: Vec<Node>,
    index: HashMap<Op, NodeId>,
    params: Vec<(String, Ty)>,
    param_ids: HashMap<String, NodeId>,
    intermediates: HashMap<String, NodeId>,
    preprocessed: HashMap<String, u32>,
}

/// Lowers the constraints `eval` traced.
pub fn lower(eval: &ExprEvaluator) -> Program {
    let mut preprocessed = HashMap::new();
    for (id, slot) in &eval.preprocessed_columns {
        preprocessed.entry(id.clone()).or_insert(*slot as u32);
    }
    let mut lowering = Lowering {
        eval,
        nodes: vec![],
        index: HashMap::new(),
        params: vec![],
        param_ids: HashMap::new(),
        intermediates: HashMap::new(),
        preprocessed,
    };
    let constraints = eval.constraints.iter().map(|c| lowering.ext(c)).collect();
    Program { nodes: lowering.nodes, params: lowering.params, constraints }
}

impl Lowering<'_> {
    fn node(&mut self, op: Op) -> NodeId {
        if let Some(&id) = self.index.get(&op) {
            return id;
        }
        let operands = op_operands(&op);
        let ty = match &op {
            Op::Col { .. } | Op::Inv(_) => Ty::Base,
            Op::Const(c) => const_ty(c),
            Op::Param(i) => self.params[*i as usize].1,
            Op::Secure(_) => Ty::Ext,
            _ => {
                if operands.iter().any(|o| self.nodes[*o as usize].ty == Ty::Ext) {
                    Ty::Ext
                } else {
                    Ty::Base
                }
            }
        };
        let uniform = match &op {
            Op::Col { .. } => false,
            _ => operands.iter().all(|o| self.nodes[*o as usize].uniform),
        };
        let id = self.nodes.len() as NodeId;
        self.nodes.push(Node { op: op.clone(), ty, uniform });
        self.index.insert(op, id);
        id
    }

    fn constant_of(&self, id: NodeId) -> Option<QM31> {
        match &self.nodes[id as usize].op {
            Op::Const(c) => Some(qm31(*c)),
            _ => None,
        }
    }

    fn is_const(&self, id: NodeId, value: u32) -> bool {
        self.constant_of(id) == Some(qm31([value, 0, 0, 0]))
    }

    fn constant(&mut self, value: QM31) -> NodeId {
        self.node(Op::Const(canonical(value)))
    }

    fn add(&mut self, a: NodeId, b: NodeId) -> NodeId {
        if let (Some(x), Some(y)) = (self.constant_of(a), self.constant_of(b)) {
            return self.constant(x + y);
        }
        if self.is_const(a, 0) {
            return b;
        }
        if self.is_const(b, 0) {
            return a;
        }
        self.node(Op::Add(a.min(b), a.max(b)))
    }

    fn sub(&mut self, a: NodeId, b: NodeId) -> NodeId {
        if let (Some(x), Some(y)) = (self.constant_of(a), self.constant_of(b)) {
            return self.constant(x - y);
        }
        if self.is_const(b, 0) {
            return a;
        }
        if self.is_const(a, 0) {
            return self.neg(b);
        }
        self.node(Op::Sub(a, b))
    }

    fn mul(&mut self, a: NodeId, b: NodeId) -> NodeId {
        if let (Some(x), Some(y)) = (self.constant_of(a), self.constant_of(b)) {
            return self.constant(x * y);
        }
        if self.is_const(a, 0) || self.is_const(b, 0) {
            return self.constant(QM31::from_u32_unchecked(0, 0, 0, 0));
        }
        if self.is_const(a, 1) {
            return b;
        }
        if self.is_const(b, 1) {
            return a;
        }
        self.node(Op::Mul(a.min(b), a.max(b)))
    }

    fn neg(&mut self, a: NodeId) -> NodeId {
        if let Some(x) = self.constant_of(a) {
            return self.constant(-x);
        }
        if let Op::Neg(inner) = self.nodes[a as usize].op {
            return inner;
        }
        self.node(Op::Neg(a))
    }

    fn inv(&mut self, a: NodeId) -> NodeId {
        assert_eq!(self.nodes[a as usize].ty, Ty::Base, "only base values are inverted");
        if let Some(x) = self.constant_of(a) {
            let x = x.to_m31_array()[0];
            let inverse = if x.0.is_multiple_of(P) { M31(0) } else { x.inverse() };
            return self.constant(inverse.into());
        }
        self.node(Op::Inv(a))
    }

    fn secure(&mut self, parts: [NodeId; 4]) -> NodeId {
        let constants = parts.map(|p| self.constant_of(p));
        if constants.iter().all(Option::is_some) {
            let [a, b, c, d] = constants.map(|c| c.unwrap().to_m31_array()[0]);
            return self.constant(QM31(CM31(a, b), CM31(c, d)));
        }
        if parts[1..].iter().all(|p| self.is_const(*p, 0)) {
            return parts[0];
        }
        self.node(Op::Secure(parts))
    }

    fn param(&mut self, name: &str, ty: Ty) -> NodeId {
        if let Some(&id) = self.param_ids.get(name) {
            assert_eq!(self.nodes[id as usize].ty, ty, "parameter {name} used as two types");
            return id;
        }
        self.params.push((name.to_owned(), ty));
        let id = self.node(Op::Param(self.params.len() as u32 - 1));
        self.param_ids.insert(name.to_owned(), id);
        id
    }

    fn base(&mut self, expr: &BaseExpr) -> NodeId {
        let eval = self.eval;
        match expr {
            BaseExpr::Col(column) => {
                let (interaction, index) = eval.column_slots[column.idx()];
                assert_eq!(interaction, column.interaction(), "a column's interaction");
                self.node(Op::Col {
                    interaction: interaction as u32,
                    index: index as u32,
                    offset: column.offset() as i32,
                })
            }
            BaseExpr::Const(value) => self.constant((*value).into()),
            BaseExpr::Param(name) => {
                if let Some(&id) = self.intermediates.get(name) {
                    return id;
                }
                if let Some(definition) = eval.intermediates.get(name) {
                    let id = self.base(definition);
                    self.intermediates.insert(name.clone(), id);
                    return id;
                }
                if let Some(&slot) = self.preprocessed.get(name) {
                    return self.node(Op::Col { interaction: 0, index: slot, offset: 0 });
                }
                self.param(name, Ty::Base)
            }
            BaseExpr::Add(a, b) => {
                let (a, b) = (self.base(a), self.base(b));
                self.add(a, b)
            }
            BaseExpr::Sub(a, b) => {
                let (a, b) = (self.base(a), self.base(b));
                self.sub(a, b)
            }
            BaseExpr::Mul(a, b) => {
                let (a, b) = (self.base(a), self.base(b));
                self.mul(a, b)
            }
            BaseExpr::Neg(a) => {
                let a = self.base(a);
                self.neg(a)
            }
            BaseExpr::Inv(a) => {
                let a = self.base(a);
                self.inv(a)
            }
        }
    }

    fn ext(&mut self, expr: &ExtExpr) -> NodeId {
        let eval = self.eval;
        match expr {
            ExtExpr::SecureCol(parts) => {
                let parts = [
                    self.base(&parts[0]),
                    self.base(&parts[1]),
                    self.base(&parts[2]),
                    self.base(&parts[3]),
                ];
                self.secure(parts)
            }
            ExtExpr::Const(value) => self.constant(*value),
            ExtExpr::Param(name) => {
                if let Some(&id) = self.intermediates.get(name) {
                    return id;
                }
                if let Some(definition) = eval.ext_intermediates.get(name) {
                    let id = self.ext(definition);
                    self.intermediates.insert(name.clone(), id);
                    return id;
                }
                self.param(name, Ty::Ext)
            }
            ExtExpr::Add(a, b) => {
                let (a, b) = (self.ext(a), self.ext(b));
                self.add(a, b)
            }
            ExtExpr::Sub(a, b) => {
                let (a, b) = (self.ext(a), self.ext(b));
                self.sub(a, b)
            }
            ExtExpr::Mul(a, b) => {
                let (a, b) = (self.ext(a), self.ext(b));
                self.mul(a, b)
            }
            ExtExpr::Neg(a) => {
                let a = self.ext(a);
                self.neg(a)
            }
        }
    }
}

fn const_ty(c: &[u32; 4]) -> Ty {
    if c[1..] == [0, 0, 0] { Ty::Base } else { Ty::Ext }
}

pub fn op_operands(op: &Op) -> Vec<NodeId> {
    match op {
        Op::Col { .. } | Op::Const(_) | Op::Param(_) => vec![],
        Op::Secure(parts) => parts.to_vec(),
        Op::Add(a, b) | Op::Sub(a, b) | Op::Mul(a, b) => vec![*a, *b],
        Op::Neg(a) | Op::Inv(a) => vec![*a],
    }
}

/// Counts of a program's row (non-uniform) operations.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OpCounts {
    pub columns: usize,
    pub base_mul: usize,
    pub ext_mul_base: usize,
    pub ext_mul: usize,
    pub base_add: usize,
    pub ext_add: usize,
    pub inv: usize,
    pub secure: usize,
}

impl Program {
    /// The formal parameters' values from `values` by name, then every uniform node's value
    /// (`None` for the others).
    pub fn bind(&self, values: &HashMap<String, QM31>) -> Result<Vec<Option<QM31>>, String> {
        let mut out: Vec<Option<QM31>> = Vec::with_capacity(self.nodes.len());
        for node in &self.nodes {
            if !node.uniform {
                out.push(None);
                continue;
            }
            let v = |id: &NodeId| out[*id as usize].unwrap();
            let value = match &node.op {
                Op::Col { .. } => unreachable!(),
                Op::Const(c) => qm31(*c),
                Op::Param(i) => {
                    let (name, ty) = &self.params[*i as usize];
                    let value = *values
                        .get(name)
                        .ok_or_else(|| format!("no value for parameter {name}"))?;
                    if *ty == Ty::Base && const_ty(&words(value)) != Ty::Base {
                        return Err(format!("parameter {name} is not a base value"));
                    }
                    value
                }
                Op::Secure(parts) => {
                    let [a, b, c, d] = parts.map(|p| v(&p).to_m31_array()[0]);
                    QM31(CM31(a, b), CM31(c, d))
                }
                Op::Add(a, b) => v(a) + v(b),
                Op::Sub(a, b) => v(a) - v(b),
                Op::Mul(a, b) => v(a) * v(b),
                Op::Neg(a) => -v(a),
                Op::Inv(a) => {
                    let x = v(a).to_m31_array()[0];
                    if x.0.is_multiple_of(P) {
                        QM31::from_u32_unchecked(0, 0, 0, 0)
                    } else {
                        x.inverse().into()
                    }
                }
            };
            out.push(Some(value));
        }
        Ok(out)
    }

    /// The non-uniform nodes the constraints `range` need, in node order.
    pub fn row_nodes(&self, range: std::ops::Range<usize>) -> Vec<NodeId> {
        let mut needed = vec![false; self.nodes.len()];
        let mut stack: Vec<NodeId> = self.constraints[range].to_vec();
        while let Some(id) = stack.pop() {
            let node = &self.nodes[id as usize];
            if node.uniform || needed[id as usize] {
                continue;
            }
            needed[id as usize] = true;
            stack.extend(op_operands(&node.op));
        }
        (0..self.nodes.len() as NodeId).filter(|id| needed[*id as usize]).collect()
    }

    pub fn op_counts(&self) -> OpCounts {
        let mut counts = OpCounts::default();
        for id in self.row_nodes(0..self.constraints.len()) {
            let node = &self.nodes[id as usize];
            let ext = |o: &NodeId| self.nodes[*o as usize].ty == Ty::Ext;
            match &node.op {
                Op::Col { .. } => counts.columns += 1,
                Op::Mul(a, b) => match (ext(a), ext(b)) {
                    (false, false) => counts.base_mul += 1,
                    (true, true) => counts.ext_mul += 1,
                    _ => counts.ext_mul_base += 1,
                },
                Op::Add(..) | Op::Sub(..) | Op::Neg(_) => {
                    if node.ty == Ty::Ext {
                        counts.ext_add += 1
                    } else {
                        counts.base_add += 1
                    }
                }
                Op::Inv(_) => counts.inv += 1,
                Op::Secure(_) => counts.secure += 1,
                Op::Const(_) | Op::Param(_) => {}
            }
        }
        counts
    }

    /// The highest column index used per interaction, plus one.
    pub fn n_columns(&self) -> Vec<usize> {
        let mut n = vec![];
        for node in &self.nodes {
            if let Op::Col { interaction, index, .. } = node.op {
                let interaction = interaction as usize;
                if n.len() <= interaction {
                    n.resize(interaction + 1, 0);
                }
                n[interaction] = n[interaction].max(index as usize + 1);
            }
        }
        n
    }

    /// The distinct nonzero row offsets the columns are read at.
    pub fn offsets(&self) -> Vec<i32> {
        let mut offsets: Vec<i32> = self
            .nodes
            .iter()
            .filter_map(|n| match n.op {
                Op::Col { offset, .. } if offset != 0 => Some(offset),
                _ => None,
            })
            .collect();
        offsets.sort();
        offsets.dedup();
        offsets
    }

    /// A line-based text form ([`Self::from_text`] reads it back).
    pub fn to_text(&self) -> String {
        let mut s = String::new();
        for (name, ty) in &self.params {
            s += &format!("param {name} {}\n", if *ty == Ty::Base { "b" } else { "e" });
        }
        for node in &self.nodes {
            s += &match &node.op {
                Op::Col { interaction, index, offset } => {
                    format!("col {interaction} {index} {offset}")
                }
                Op::Const(c) => format!("const {} {} {} {}", c[0], c[1], c[2], c[3]),
                Op::Param(i) => format!("p {i}"),
                Op::Secure(p) => format!("secure {} {} {} {}", p[0], p[1], p[2], p[3]),
                Op::Add(a, b) => format!("add {a} {b}"),
                Op::Sub(a, b) => format!("sub {a} {b}"),
                Op::Mul(a, b) => format!("mul {a} {b}"),
                Op::Neg(a) => format!("neg {a}"),
                Op::Inv(a) => format!("inv {a}"),
            };
            s.push('\n');
        }
        s += "constraints";
        for c in &self.constraints {
            s += &format!(" {c}");
        }
        s.push('\n');
        s
    }

    pub fn from_text(text: &str) -> Result<Self, String> {
        let mut params = vec![];
        let mut ops = vec![];
        let mut constraints = vec![];
        for line in text.lines() {
            let mut fields = line.split_whitespace();
            let Some(kind) = fields.next() else { continue };
            let rest: Vec<&str> = fields.collect();
            let num = |i: usize| -> Result<u32, String> {
                rest.get(i).ok_or("missing field")?.parse::<u32>().map_err(|e| e.to_string())
            };
            let op = match kind {
                "param" => {
                    params.push((
                        rest[0].to_owned(),
                        if rest[1] == "b" { Ty::Base } else { Ty::Ext },
                    ));
                    continue;
                }
                "constraints" => {
                    constraints = rest
                        .iter()
                        .map(|w| w.parse::<u32>().map_err(|e| e.to_string()))
                        .collect::<Result<_, _>>()?;
                    continue;
                }
                "col" => Op::Col {
                    interaction: num(0)?,
                    index: num(1)?,
                    offset: rest[2].parse::<i32>().map_err(|e| e.to_string())?,
                },
                "const" => Op::Const([num(0)?, num(1)?, num(2)?, num(3)?]),
                "p" => Op::Param(num(0)?),
                "secure" => Op::Secure([num(0)?, num(1)?, num(2)?, num(3)?]),
                "add" => Op::Add(num(0)?, num(1)?),
                "sub" => Op::Sub(num(0)?, num(1)?),
                "mul" => Op::Mul(num(0)?, num(1)?),
                "neg" => Op::Neg(num(0)?),
                "inv" => Op::Inv(num(0)?),
                other => return Err(format!("unknown op {other}")),
            };
            ops.push(op);
        }
        // Types and uniformity as the lowering derives them.
        let mut nodes: Vec<Node> = vec![];
        for op in ops {
            let operands = op_operands(&op);
            let ty = match &op {
                Op::Col { .. } | Op::Inv(_) => Ty::Base,
                Op::Const(c) => const_ty(c),
                Op::Param(i) => params[*i as usize].1,
                Op::Secure(_) => Ty::Ext,
                _ => {
                    if operands.iter().any(|o| nodes[*o as usize].ty == Ty::Ext) {
                        Ty::Ext
                    } else {
                        Ty::Base
                    }
                }
            };
            let uniform = !matches!(op, Op::Col { .. })
                && operands.iter().all(|o| nodes[*o as usize].uniform);
            nodes.push(Node { op, ty, uniform });
        }
        Ok(Program { nodes, params, constraints })
    }
}

/// The formal parameters' values of a component's evaluation: its relations' elements, its
/// claimed sum and its column size, by the names the lowering gives them.
pub fn param_values(
    eval: &ExprEvaluator,
    claimed_sum: QM31,
    trace_log_size: u32,
) -> HashMap<String, QM31> {
    let mut values = HashMap::new();
    for (name, z, alphas) in &eval.relation_values {
        values.insert(format!("{name}_z"), *z);
        for (k, alpha) in alphas.iter().enumerate() {
            values.insert(format!("{name}_alpha{k}"), *alpha);
        }
    }
    values.insert("claimed_sum".to_owned(), claimed_sum);
    values.insert("column_size".to_owned(), M31(1 << trace_log_size).into());
    values
}
