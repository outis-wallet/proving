//! A WGSL kernel from a [`Program`]: one invocation per row of the evaluation domain, adding
//! `sum_i coeffs[i] * constraint_i(row) * denom_inv[row >> trace_log]` to the accumulator, for the
//! constraints of a range (a component's constraints may be split over several kernels, each
//! adding its share).
//!
//! Bindings (group 0):
//! - 0: `uniforms: array<vec4<u32>>`, the uniform nodes' values ([`Kernel::uniform_nodes`]), a base
//!   value in `.x`;
//! - 1: `coeffs: array<vec4<u32>>`, constraint `i`'s random coefficient at `i`;
//! - 2: `denom_inv: array<u32>`;
//! - 3: `accum: array<u32>`, read-write, coordinate `c` of row `r` at `c * N + r`;
//! - 4 + g: `cols_g: array<u32>`, column group `g`: its `k`-th column's row `r` at `k * N + r`.
//!
//! Every value is canonical (in `[0, P)`); the arithmetic is `shaders/m31.wgsl` and
//! `shaders/qm31.wgsl`, exact in the field.

use std::collections::HashMap;
use std::fmt::Write as _;

use super::ir::{NodeId, Op, Program, Ty};

pub const QM31_WGSL: &str = include_str!("../shaders/qm31.wgsl");

/// Which binding each column is in: `groups[g]` lists group `g`'s columns
/// `(interaction, index)`, in order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ColumnGroups {
    pub groups: Vec<Vec<(u32, u32)>>,
}

impl ColumnGroups {
    /// The columns `n_columns[interaction]` per interaction, in order, in groups of at most
    /// `max_columns` columns.
    pub fn plan(n_columns: &[usize], max_columns: usize) -> Self {
        let all: Vec<(u32, u32)> = n_columns
            .iter()
            .enumerate()
            .flat_map(|(i, n)| (0..*n as u32).map(move |j| (i as u32, j)))
            .collect();
        Self { groups: all.chunks(max_columns.max(1)).map(|c| c.to_vec()).collect() }
    }

    fn locations(&self) -> HashMap<(u32, u32), (usize, usize)> {
        let mut map = HashMap::new();
        for (g, group) in self.groups.iter().enumerate() {
            for (k, column) in group.iter().enumerate() {
                map.insert(*column, (g, k));
            }
        }
        map
    }
}

/// A generated kernel.
pub struct Kernel {
    pub source: String,
    /// The uniform nodes the kernel reads, in the order of the `uniforms` binding.
    pub uniform_nodes: Vec<NodeId>,
    /// The number of row nodes (straight-line statements) it evaluates.
    pub n_row_nodes: usize,
}

/// The uniform non-constant nodes the constraints `range` read, in node order.
pub fn uniform_nodes(program: &Program, range: std::ops::Range<usize>) -> Vec<NodeId> {
    let mut used = vec![false; program.nodes.len()];
    let mark = |id: NodeId, used: &mut Vec<bool>| {
        let node = &program.nodes[id as usize];
        if node.uniform && !matches!(node.op, Op::Const(_)) {
            used[id as usize] = true;
        }
    };
    for id in program.row_nodes(range.clone()) {
        for o in super::ir::op_operands(&program.nodes[id as usize].op) {
            mark(o, &mut used);
        }
    }
    for c in &program.constraints[range] {
        mark(*c, &mut used);
    }
    (0..program.nodes.len() as NodeId).filter(|id| used[*id as usize]).collect()
}

fn offset_name(offset: i32) -> String {
    if offset < 0 { format!("row_n{}", -offset) } else { format!("row_p{offset}") }
}

/// The kernel of the constraints `range` of `program`, for a trace of log size `trace_log` on an
/// evaluation domain of log size `eval_log`, `workgroup_size` invocations per workgroup.
/// `uniforms` assigns the uniform nodes their slots in the `uniforms` binding (e.g. from
/// [`uniform_nodes`] over all the constraints, shared by the kernels of a split).
pub fn emit(
    program: &Program,
    range: std::ops::Range<usize>,
    groups: &ColumnGroups,
    uniforms: &[NodeId],
    trace_log: u32,
    eval_log: u32,
    workgroup_size: u32,
) -> Kernel {
    let locations = groups.locations();
    let uniform_slot: HashMap<NodeId, usize> =
        uniforms.iter().enumerate().map(|(k, id)| (*id, k)).collect();
    let row_nodes = program.row_nodes(range.clone());
    let operand = |id: NodeId| -> String {
        let node = &program.nodes[id as usize];
        if let Op::Const(c) = &node.op {
            return match node.ty {
                Ty::Base => format!("{}u", c[0]),
                Ty::Ext => format!("vec4<u32>({}u, {}u, {}u, {}u)", c[0], c[1], c[2], c[3]),
            };
        }
        if node.uniform {
            let slot = uniform_slot[&id];
            return match node.ty {
                Ty::Base => format!("uniforms[{slot}].x"),
                Ty::Ext => format!("uniforms[{slot}]"),
            };
        }
        format!("v{id}")
    };

    // Constraint `i`'s term, to add once its node is evaluated.
    let mut terms: HashMap<NodeId, Vec<usize>> = HashMap::new();
    let mut uniform_terms = vec![];
    for i in range.clone() {
        let id = program.constraints[i];
        if program.nodes[id as usize].uniform {
            uniform_terms.push(i);
        } else {
            terms.entry(id).or_default().push(i);
        }
    }
    let term = |i: usize| -> String {
        let id = program.constraints[i];
        match program.nodes[id as usize].ty {
            Ty::Base => format!("    acc = m31_add4(acc, qm31_mul_m31(coeffs[{i}], {}));\n", operand(id)),
            Ty::Ext => format!("    acc = m31_add4(acc, qm31_mul(coeffs[{i}], {}));\n", operand(id)),
        }
    };

    let mut s = String::new();
    s += crate::M31_WGSL;
    s += QM31_WGSL;
    s += "\n@group(0) @binding(0) var<storage, read> uniforms: array<vec4<u32>>;\n";
    s += "@group(0) @binding(1) var<storage, read> coeffs: array<vec4<u32>>;\n";
    s += "@group(0) @binding(2) var<storage, read> denom_inv: array<u32>;\n";
    s += "@group(0) @binding(3) var<storage, read_write> accum: array<u32>;\n";
    for g in 0..groups.groups.len() {
        writeln!(s, "@group(0) @binding({}) var<storage, read> cols_{g}: array<u32>;", 4 + g).unwrap();
    }
    writeln!(s, "\nconst N: u32 = {}u;", 1u64 << eval_log).unwrap();
    writeln!(s, "const TRACE_LOG: u32 = {trace_log}u;").unwrap();
    writeln!(s, "const EVAL_LOG: u32 = {eval_log}u;").unwrap();
    writeln!(s, "\n@compute @workgroup_size({workgroup_size})").unwrap();
    s += "fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {\n";
    writeln!(s, "    let row = gid.x + gid.y * nwg.x * {workgroup_size}u;").unwrap();
    s += "    if (row >= N) {\n        return;\n    }\n";
    for offset in program.offsets() {
        writeln!(s, "    let {} = offset_row(row, TRACE_LOG, EVAL_LOG, {offset});", offset_name(offset))
            .unwrap();
    }
    s += "    var acc = vec4<u32>(0u);\n";
    for &id in &row_nodes {
        let node = &program.nodes[id as usize];
        let ty = |o: &NodeId| program.nodes[*o as usize].ty;
        let o = |o: &NodeId| operand(*o);
        let expr = match &node.op {
            Op::Col { interaction, index, offset } => {
                let (g, k) = locations[&(*interaction, *index)];
                let row = if *offset == 0 { "row".to_owned() } else { offset_name(*offset) };
                format!("cols_{g}[{k}u * N + {row}]")
            }
            Op::Const(_) | Op::Param(_) => unreachable!("uniform"),
            Op::Secure(p) => format!("vec4<u32>({}, {}, {}, {})", o(&p[0]), o(&p[1]), o(&p[2]), o(&p[3])),
            Op::Add(a, b) => match (ty(a), ty(b)) {
                (Ty::Base, Ty::Base) => format!("m31_add({}, {})", o(a), o(b)),
                (Ty::Ext, Ty::Ext) => format!("m31_add4({}, {})", o(a), o(b)),
                (Ty::Ext, Ty::Base) => format!("qm31_add_m31({}, {})", o(a), o(b)),
                (Ty::Base, Ty::Ext) => format!("qm31_add_m31({}, {})", o(b), o(a)),
            },
            Op::Sub(a, b) => match (ty(a), ty(b)) {
                (Ty::Base, Ty::Base) => format!("m31_sub({}, {})", o(a), o(b)),
                (Ty::Ext, Ty::Ext) => format!("m31_sub4({}, {})", o(a), o(b)),
                (Ty::Ext, Ty::Base) => format!("qm31_sub_m31({}, {})", o(a), o(b)),
                (Ty::Base, Ty::Ext) => format!("m31_sub_qm31({}, {})", o(a), o(b)),
            },
            Op::Mul(a, b) => match (ty(a), ty(b)) {
                (Ty::Base, Ty::Base) => format!("m31_mul({}, {})", o(a), o(b)),
                (Ty::Ext, Ty::Ext) => format!("qm31_mul({}, {})", o(a), o(b)),
                (Ty::Ext, Ty::Base) => format!("qm31_mul_m31({}, {})", o(a), o(b)),
                (Ty::Base, Ty::Ext) => format!("qm31_mul_m31({}, {})", o(b), o(a)),
            },
            Op::Neg(a) => match ty(a) {
                Ty::Base => format!("m31_neg({})", o(a)),
                Ty::Ext => format!("m31_neg4({})", o(a)),
            },
            Op::Inv(a) => format!("m31_inv({})", o(a)),
        };
        let wgsl_ty = if node.ty == Ty::Base { "u32" } else { "vec4<u32>" };
        writeln!(s, "    let v{id}: {wgsl_ty} = {expr};").unwrap();
        if let Some(constraints) = terms.get(&id) {
            for i in constraints {
                s += &term(*i);
            }
        }
    }
    for i in uniform_terms {
        s += &term(i);
    }
    s += "    let res = qm31_mul_m31(acc, denom_inv[row >> TRACE_LOG]);\n";
    s += "    accum[row] = m31_add(accum[row], res.x);\n";
    s += "    accum[N + row] = m31_add(accum[N + row], res.y);\n";
    s += "    accum[2u * N + row] = m31_add(accum[2u * N + row], res.z);\n";
    s += "    accum[3u * N + row] = m31_add(accum[3u * N + row], res.w);\n";
    s += "}\n";
    Kernel { source: s, uniform_nodes: uniforms.to_vec(), n_row_nodes: row_nodes.len() }
}
