//! A CPU interpreter of a [`Program`], the first oracle for the lowering and the kernels: stwo's
//! own field arithmetic, row by row.

use stwo::core::fields::cm31::CM31;
use stwo::core::fields::m31::{M31, P};
use stwo::core::fields::qm31::QM31;
use stwo::core::utils::offset_bit_reversed_circle_domain_index;

use super::ir::{NodeId, Op, Program, Ty, qm31};

/// The values of a program's row nodes at one row, reusing its buffers.
pub struct Interpreter<'a> {
    program: &'a Program,
    uniforms: &'a [Option<QM31>],
    row_nodes: Vec<NodeId>,
    values: Vec<QM31>,
}

fn base(value: QM31) -> M31 {
    value.to_m31_array()[0]
}

impl<'a> Interpreter<'a> {
    /// `uniforms`: [`Program::bind`]'s values.
    pub fn new(program: &'a Program, uniforms: &'a [Option<QM31>]) -> Self {
        let row_nodes = program.row_nodes(0..program.constraints.len());
        Self { program, uniforms, row_nodes, values: vec![QM31::default(); program.nodes.len()] }
    }

    fn value(&self, id: NodeId) -> QM31 {
        match self.uniforms[id as usize] {
            Some(value) => value,
            None => self.values[id as usize],
        }
    }

    /// The constraints' values at a row, `column(interaction, index, offset)` reading the columns.
    pub fn constraints(&mut self, column: impl Fn(u32, u32, i32) -> M31) -> Vec<QM31> {
        for i in 0..self.row_nodes.len() {
            let id = self.row_nodes[i];
            let node = &self.program.nodes[id as usize];
            let ty = |o: &NodeId| self.program.nodes[*o as usize].ty;
            let value = match &node.op {
                Op::Col { interaction, index, offset } => {
                    // Words as stored may be P; the field arithmetic reduces.
                    let word = column(*interaction, *index, *offset).0 % P;
                    QM31::from(M31(word))
                }
                Op::Const(c) => qm31(*c),
                Op::Param(_) => unreachable!("parameters are uniform"),
                Op::Secure(parts) => {
                    let [a, b, c, d] = parts.map(|p| base(self.value(p)));
                    QM31(CM31(a, b), CM31(c, d))
                }
                Op::Add(a, b) => self.value(*a) + self.value(*b),
                Op::Sub(a, b) => self.value(*a) - self.value(*b),
                Op::Mul(a, b) => match (ty(a), ty(b)) {
                    (Ty::Base, Ty::Base) => (base(self.value(*a)) * base(self.value(*b))).into(),
                    (Ty::Ext, Ty::Base) => self.value(*a) * base(self.value(*b)),
                    (Ty::Base, Ty::Ext) => self.value(*b) * base(self.value(*a)),
                    (Ty::Ext, Ty::Ext) => self.value(*a) * self.value(*b),
                },
                Op::Neg(a) => -self.value(*a),
                Op::Inv(a) => {
                    let x = base(self.value(*a));
                    if x.0.is_multiple_of(P) { QM31::default() } else { x.inverse().into() }
                }
            };
            self.values[id as usize] = value;
        }
        self.program.constraints.iter().map(|c| self.value(*c)).collect()
    }

    /// `sum_i coeffs[i] * constraint_i` at a row.
    pub fn row_res(&mut self, coeffs: &[QM31], column: impl Fn(u32, u32, i32) -> M31) -> QM31 {
        let constraints = self.constraints(column);
        assert_eq!(constraints.len(), coeffs.len(), "one coefficient per constraint");
        coeffs.iter().zip(constraints).map(|(c, v)| *c * v).sum()
    }
}

/// A component's evaluation inputs as words: `columns[interaction][index]` on the whole
/// evaluation domain (bit-reversed), and the evaluation's sizes.
pub struct RowInputs<'a> {
    pub columns: &'a [Vec<&'a [u32]>],
    pub trace_log_size: u32,
    pub eval_log_size: u32,
}

impl RowInputs<'_> {
    /// Column `(interaction, index)` at `offset` trace rows from evaluation-domain row `row`.
    pub fn at(&self, row: usize, interaction: u32, index: u32, offset: i32) -> M31 {
        let column = self.columns[interaction as usize][index as usize];
        let row = if offset == 0 {
            row
        } else {
            offset_bit_reversed_circle_domain_index(
                row,
                self.trace_log_size,
                self.eval_log_size,
                offset as isize,
            )
        };
        M31(column[row])
    }
}

/// The accumulator after the evaluation, as canonical words `[coordinate][row]`: `before[r] +
/// row_res(r) * denom_inv[r >> trace_log_size]`, computed by the interpreter on rows `rows`.
pub fn accumulate_rows(
    program: &Program,
    uniforms: &[Option<QM31>],
    coeffs: &[QM31],
    denom_inv: &[u32],
    inputs: &RowInputs<'_>,
    before: &[Vec<u32>; 4],
    rows: std::ops::Range<usize>,
) -> Vec<[u32; 4]> {
    let mut interpreter = Interpreter::new(program, uniforms);
    rows.map(|row| {
        let res = interpreter.row_res(coeffs, |i, j, o| inputs.at(row, i, j, o));
        let d = M31(denom_inv[row >> inputs.trace_log_size] % P);
        let prior = QM31::from_u32_unchecked(
            before[0][row] % P,
            before[1][row] % P,
            before[2][row] % P,
            before[3][row] % P,
        );
        super::ir::canonical(prior + res * d)
    })
    .collect()
}
