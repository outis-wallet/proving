//! The constraint evaluation spike (`stwo_gpu_spike::constraint`):
//! - the representation question: SIMD's interpolation reads a raw `P` word as 0, so the
//!   accumulator's representation (canonical or not) does not reach the proof;
//! - the lowering of a synthetic component against its traced expressions;
//! - the kernels validate as core WGSL (naga, MSL, HLSL);
//! - on a device (`gpu_` tests; lavapipe first: `WGPU_ADAPTER_NAME=llvmpipe`): a synthetic
//!   component's kernel against the interpreter, and the captured pay fixtures
//!   (`CONSTRAINT_FIXTURES=<dir>`, see `constraint::capture`) word for word against stwo's SIMD
//!   output, with negative tests.
use std::collections::HashMap;
use std::path::PathBuf;

use naga::valid::{Capabilities, ValidationFlags, Validator};
use num_traits::One;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use stwo::core::fields::FieldExpOps;
use stwo::core::fields::m31::{M31, P};
use stwo::core::fields::qm31::QM31;
use stwo::core::poly::circle::CanonicCoset;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::simd::column::BaseColumn;
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;
use stwo_constraint_framework::expr::ExprEvaluator;
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;
use stwo_constraint_framework::{EvalAtRow, FrameworkEval, RelationEntry, relation};
use stwo_gpu_spike::GpuContext;
use stwo_gpu_spike::constraint::capture::check_random_assignment;
use stwo_gpu_spike::constraint::fixture::Fixture;
use stwo_gpu_spike::constraint::interp::{RowInputs, accumulate_rows};
use stwo_gpu_spike::constraint::ir::{Program, Ty, lower, param_values};
use stwo_gpu_spike::constraint::kernel::{ConstraintKernels, KernelOptions};
use stwo_gpu_spike::constraint::wgsl::{ColumnGroups, emit, uniform_nodes};

relation!(TestRelation, 4);

/// Exercises what the Cairo components use: preprocessed and trace columns, base and extension
/// intermediates, an inverse, constants, relations with several multiplicities, logup in pairs
/// (an odd count, so the last batch has one fraction).
struct Synthetic {
    relation: TestRelation,
    log_size: u32,
}

impl FrameworkEval for Synthetic {
    fn log_size(&self) -> u32 {
        self.log_size
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_size + 1
    }

    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let pre = eval.get_preprocessed_column(PreProcessedColumnId { id: "seq_x".to_owned() });
        let a = eval.next_trace_mask();
        let b = eval.next_trace_mask();
        let c = eval.next_trace_mask();
        let pre2 = eval.get_preprocessed_column(PreProcessedColumnId { id: "seq_y".to_owned() });
        let ab = eval.add_intermediate(a.clone() * b.clone());
        eval.add_constraint(ab.clone() * c.clone() - pre.clone());
        eval.add_constraint(
            (a.clone() + c.clone() + E::F::from(M31(5))).inverse() * b.clone() - E::F::from(M31(7)),
        );
        let ext = eval.add_extension_intermediate(E::combine_ef([
            a.clone(),
            b.clone(),
            c.clone(),
            pre2.clone(),
        ]));
        eval.add_constraint(ext.clone() * ext - E::EF::from(a.clone()));
        eval.add_to_relation(RelationEntry::new(&self.relation, E::EF::one(), &[a.clone(), b.clone()]));
        eval.add_to_relation(RelationEntry::new(
            &self.relation,
            -E::EF::from(c.clone()),
            &[pre.clone(), ab.clone(), c.clone()],
        ));
        eval.add_to_relation(RelationEntry::new(&self.relation, E::EF::from(b.clone()), &[c.clone()]));
        eval.finalize_logup_in_pairs();
        eval
    }
}

fn synthetic() -> (ExprEvaluator, Program) {
    let component = Synthetic { relation: TestRelation::dummy(), log_size: 10 };
    let eval = component.evaluate(ExprEvaluator::new());
    let program = lower(&eval);
    (eval, program)
}

fn random_qm31(rng: &mut SmallRng) -> QM31 {
    QM31::from_u32_unchecked(
        rng.random_range(0..P),
        rng.random_range(0..P),
        rng.random_range(0..P),
        rng.random_range(0..P),
    )
}

#[test]
fn simd_interpolation_reads_p_as_zero() {
    let mut rng = SmallRng::seed_from_u64(1);
    for log_size in [5, 8, 11] {
        let n = 1usize << log_size;
        let mut words: Vec<u32> = (0..n).map(|_| rng.random_range(0..P)).collect();
        let zeros: Vec<usize> = (0..8).map(|_| rng.random_range(0..n)).collect();
        for &i in &zeros {
            words[i] = 0;
        }
        let domain = CanonicCoset::new(log_size).circle_domain();
        let interpolate = |words: &[u32]| {
            let column: BaseColumn = words.iter().map(|w| M31(*w)).collect();
            assert_eq!(column.as_slice().iter().filter(|v| v.0 == P).count(), words.iter().filter(|w| **w == P).count());
            let coeffs = CircleEvaluation::<SimdBackend, M31, BitReversedOrder>::new(domain, column).interpolate();
            let extended = coeffs.evaluate(CanonicCoset::new(log_size + 2).circle_domain());
            (
                coeffs.coeffs.as_slice().iter().map(|v| v.0).collect::<Vec<u32>>(),
                extended.values.as_slice().iter().map(|v| v.0).collect::<Vec<u32>>(),
            )
        };
        let canonical = interpolate(&words);
        for &i in &zeros {
            words[i] = P;
        }
        let with_p = interpolate(&words);
        assert!(canonical.0.iter().all(|w| *w < P), "canonical coefficients");
        assert_eq!(canonical, with_p, "log size {log_size}: P read differently from 0");
    }
}

#[test]
fn lowering_matches_traced_expressions() {
    let (eval, program) = synthetic();
    assert_eq!(eval.constraints.len(), program.constraints.len());
    // Columns: preprocessed slots in call order, trace and interaction columns per interaction.
    assert_eq!(eval.preprocessed_columns, vec![("seq_x".to_owned(), 0), ("seq_y".to_owned(), 1)]);
    assert_eq!(program.n_columns(), vec![2, 3, 8]);
    assert_eq!(program.offsets(), vec![-1]);
    let mut rng = SmallRng::seed_from_u64(2);
    for _ in 0..4 {
        let coeffs: Vec<QM31> = (0..program.constraints.len()).map(|_| random_qm31(&mut rng)).collect();
        assert!(check_random_assignment(&eval, &program, &coeffs));
    }
    // The text form reads back.
    assert_eq!(Program::from_text(&program.to_text()).unwrap(), program);
    // The relation's elements, recovered from `combine`.
    let relation = TestRelation::dummy();
    let values = param_values(&eval, QM31::one(), 10);
    use stwo_constraint_framework::Relation;
    let zero: QM31 = Relation::<M31, QM31>::combine(&relation, &[M31(0)]);
    assert_eq!(values["TestRelation_z"], -zero);
    let e2: QM31 = Relation::<M31, QM31>::combine(&relation, &[M31(0), M31(0), M31(1)]);
    assert_eq!(values["TestRelation_alpha2"], e2 - zero);
}

fn validate(name: &str, source: &str) {
    let module = naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(source)));
    let info = Validator::new(ValidationFlags::all(), Capabilities::empty())
        .validate(&module)
        .unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(source)));
    naga::back::msl::write_string(
        &module,
        &info,
        &naga::back::msl::Options::default(),
        &naga::back::msl::PipelineOptions::default(),
    )
    .unwrap_or_else(|e| panic!("{name}: MSL: {e}"));
    let mut hlsl = String::new();
    naga::back::hlsl::Writer::new(
        &mut hlsl,
        &naga::back::hlsl::Options::default(),
        &naga::back::hlsl::PipelineOptions::default(),
    )
    .write(&module, &info, None)
    .unwrap_or_else(|e| panic!("{name}: HLSL: {e}"));
}

fn fixture_dirs() -> Vec<PathBuf> {
    let Ok(root) = std::env::var("CONSTRAINT_FIXTURES") else { return vec![] };
    let filter = std::env::var("CONSTRAINT_FIXTURE_FILTER").unwrap_or_default();
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(root)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.join("meta.txt").exists())
        .filter(|p| p.to_string_lossy().contains(&filter))
        .collect();
    dirs.sort();
    dirs
}

#[test]
fn kernels_validate() {
    let (_, program) = synthetic();
    let n = program.constraints.len();
    let uniforms = uniform_nodes(&program, 0..n);
    for (split, max_columns) in [(1, 64), (2, 3)] {
        let groups = ColumnGroups::plan(&program.n_columns(), max_columns);
        for k in 0..split {
            let range = k * n / split..(k + 1) * n / split;
            let kernel = emit(&program, range, &groups, &uniforms, 10, 11, 256);
            validate("synthetic", &kernel.source);
        }
    }
    for dir in fixture_dirs() {
        let text = std::fs::read_to_string(dir.join("program.txt")).unwrap();
        let program = Program::from_text(&text).unwrap();
        let n = program.constraints.len();
        let groups = ColumnGroups::plan(&program.n_columns(), 1 << 10);
        let kernel = emit(&program, 0..n, &groups, &uniform_nodes(&program, 0..n), 15, 16, 256);
        validate(&dir.to_string_lossy(), &kernel.source);
        eprintln!(
            "{}: {} lines of WGSL, {} row nodes, {} uniforms",
            dir.display(),
            kernel.source.lines().count(),
            kernel.n_row_nodes,
            kernel.uniform_nodes.len()
        );
    }
}

/// Runs the kernels on the inputs and returns the accumulator's words.
#[allow(clippy::too_many_arguments)]
fn run_on_gpu(
    gpu: &GpuContext,
    program: &Program,
    n_columns: &[usize],
    trace_log: u32,
    eval_log: u32,
    options: KernelOptions,
    columns: &[Vec<&[u32]>],
    params: &HashMap<String, QM31>,
    coeffs: &[QM31],
    denom_inv: &[u32],
    before: &[Vec<u32>; 4],
) -> [Vec<u32>; 4] {
    let kernels = ConstraintKernels::new(gpu, program, n_columns, trace_log, eval_log, options);
    let uniforms = program.bind(params).unwrap();
    let before: [&[u32]; 4] = std::array::from_fn(|c| before[c].as_slice());
    let inputs = kernels.upload(gpu, columns, &uniforms, coeffs, denom_inv, &before);
    let bind_group = kernels.bind_group(gpu, &inputs);
    kernels.run(gpu, &bind_group);
    kernels.read_accum(gpu, &inputs)
}

fn mismatches(a: &[Vec<u32>; 4], b: &[Vec<u32>; 4]) -> usize {
    (0..4).map(|c| a[c].iter().zip(&b[c]).filter(|(x, y)| x != y).count()).sum()
}

#[test]
fn gpu_synthetic_kernel_matches_interpreter() {
    let gpu = GpuContext::new();
    let limits = gpu.device.limits();
    eprintln!(
        "adapter: {:?}; max binding {} MiB, max buffer {} MiB, storage buffers per stage {}",
        gpu.adapter_info,
        limits.max_storage_buffer_binding_size as u64 >> 20,
        limits.max_buffer_size >> 20,
        limits.max_storage_buffers_per_shader_stage
    );
    let (eval, program) = synthetic();
    let mut rng = SmallRng::seed_from_u64(3);
    let (trace_log, eval_log) = (10u32, 11u32);
    let n = 1usize << eval_log;
    let n_columns = program.n_columns();
    let columns: Vec<Vec<Vec<u32>>> = n_columns
        .iter()
        .map(|k| (0..*k).map(|_| (0..n).map(|_| rng.random_range(0..P)).collect()).collect())
        .collect();
    let slices: Vec<Vec<&[u32]>> =
        columns.iter().map(|cs| cs.iter().map(|c| c.as_slice()).collect()).collect();
    let mut params = param_values(&eval, random_qm31(&mut rng), trace_log);
    for value in params.values_mut().filter(|v| **v != QM31::from(M31(1 << trace_log))) {
        *value = random_qm31(&mut rng);
    }
    let coeffs: Vec<QM31> = (0..program.constraints.len()).map(|_| random_qm31(&mut rng)).collect();
    let denom_inv: Vec<u32> = (0..2).map(|_| rng.random_range(1..P)).collect();
    let before: [Vec<u32>; 4] = std::array::from_fn(|_| (0..n).map(|_| rng.random_range(0..P)).collect());
    let uniforms = program.bind(&params).unwrap();
    let inputs = RowInputs { columns: &slices, trace_log_size: trace_log, eval_log_size: eval_log };
    let expected = accumulate_rows(&program, &uniforms, &coeffs, &denom_inv, &inputs, &before, 0..n);
    let expected: [Vec<u32>; 4] = std::array::from_fn(|c| expected.iter().map(|e| e[c]).collect());
    for options in [
        KernelOptions::default(),
        KernelOptions { split: 3, ..Default::default() },
        KernelOptions { max_binding_bytes: Some(3 * 4 * n as u64), workgroup_size: 64, ..Default::default() },
        KernelOptions { unchecked: true, ..Default::default() },
    ] {
        let got = run_on_gpu(
            &gpu, &program, &n_columns, trace_log, eval_log, options, &slices, &params, &coeffs,
            &denom_inv, &before,
        );
        assert_eq!(mismatches(&got, &expected), 0, "{options:?}");
    }
    assert!(program.nodes.iter().any(|n| n.ty == Ty::Base && matches!(n.op, stwo_gpu_spike::constraint::ir::Op::Inv(_))));
}

/// The captured fixtures: the interpreter and the kernels against stwo's SIMD output, word for
/// word, and the negative tests.
#[test]
fn gpu_fixtures_bit_identical() {
    let dirs = fixture_dirs();
    if dirs.is_empty() {
        eprintln!("no fixtures (CONSTRAINT_FIXTURES)");
        return;
    }
    let gpu = GpuContext::new();
    eprintln!("adapter: {:?}", gpu.adapter_info);
    let split: usize = std::env::var("CONSTRAINT_SPLIT").ok().map_or(1, |s| s.parse().unwrap());
    for dir in dirs {
        let fixture = Fixture::load(&dir).unwrap();
        let name = dir.file_name().unwrap().to_string_lossy().to_string();
        let (trace_log, eval_log) = (fixture.trace_log_size, fixture.eval_log_size);
        let n = 1usize << eval_log;
        let slices = fixture.column_slices();
        let words = (0..4).map(|c| fixture.after[c].len()).sum::<usize>();
        assert_eq!(fixture.program.constraints.len(), fixture.n_constraints, "{name}: constraints");
        assert!(fixture.after.iter().flatten().all(|w| *w < P), "{name}: SIMD's words are canonical");

        // The interpreter (first oracle) on every row.
        let uniforms = fixture.program.bind(&fixture.params).unwrap();
        let inputs = RowInputs { columns: &slices, trace_log_size: trace_log, eval_log_size: eval_log };
        let interpreted = accumulate_rows(
            &fixture.program, &uniforms, &fixture.coeffs, &fixture.denom_inv, &inputs, &fixture.before, 0..n,
        );
        let interpreted: [Vec<u32>; 4] = std::array::from_fn(|c| interpreted.iter().map(|e| e[c]).collect());
        assert_eq!(mismatches(&interpreted, &fixture.after), 0, "{name}: interpreter");

        let options = KernelOptions { split, ..Default::default() };
        let run = |columns: &[Vec<&[u32]>], coeffs: &[QM31]| {
            run_on_gpu(
                &gpu, &fixture.program, &fixture.n_columns, trace_log, eval_log, options, columns,
                &fixture.params, coeffs, &fixture.denom_inv, &fixture.before,
            )
        };
        let got = run(&slices, &fixture.coeffs);
        assert_eq!(mismatches(&got, &fixture.after), 0, "{name}: GPU");
        eprintln!("{name}: {words} words identical to SIMD's ({n} rows, split {split})");

        // Negative tests: each must change the output.
        let mut flipped = fixture.columns.clone();
        flipped[1][0][n / 3] ^= 1;
        let flipped_slices: Vec<Vec<&[u32]>> =
            flipped.iter().map(|cs| cs.iter().map(|c| c.as_slice()).collect()).collect();
        let bad = mismatches(&run(&flipped_slices, &fixture.coeffs), &fixture.after);
        assert!(bad > 0, "{name}: a flipped trace word went unnoticed");
        let mut swapped = fixture.coeffs.clone();
        swapped.swap(0, 1);
        let bad_coeffs = mismatches(&run(&slices, &swapped), &fixture.after);
        assert!(bad_coeffs > 0, "{name}: swapped coefficients went unnoticed");
        let mut remap = String::from("n/a (no preprocessed columns)");
        if fixture.n_columns[0] > 1 {
            // Without the remap, the component's preprocessed slot k would read another column:
            // rotate the slots.
            let mut rotated = slices.clone();
            rotated[0].rotate_left(1);
            let bad_remap = mismatches(&run(&rotated, &fixture.coeffs), &fixture.after);
            assert!(bad_remap > 0, "{name}: a wrong preprocessed remap went unnoticed");
            remap = format!("{bad_remap} words differ");
        }
        eprintln!(
            "{name}: negative tests: flipped word {bad} words differ, swapped coefficients {bad_coeffs}, \
             preprocessed slots rotated {remap}"
        );
    }
}
