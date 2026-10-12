//! A constraint evaluation hook that observes, for the spike: for every component of a proof's
//! composition it traces and lowers the constraints, checks the lowering against the traced
//! expressions on a random assignment, records the component's shape, and (optionally) checks
//! the interpreter against the SIMD evaluation on every row and writes the component's
//! [`super::fixture`]. It always declines: the SIMD evaluation runs as without a hook, and the
//! proof is unchanged.

use std::collections::HashMap;
use std::fs;
use std::io::Write as _;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use rayon::prelude::*;
use stwo::core::fields::m31::{M31, P};
use stwo::core::fields::qm31::QM31;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::secure_column::SecureColumnByCoords;
use stwo_constraint_framework::constraint_hook::{
    ConstraintEvalJob, ConstraintEvalOutcome, set_constraint_eval_hook,
};
use stwo_constraint_framework::expr::ExprEvaluator;

use super::fixture::{write_coeffs, write_params, write_words};
use super::interp::{Interpreter, RowInputs, accumulate_rows};
use super::ir::{Program, lower, param_values};

#[derive(Clone, Debug)]
pub struct CaptureConfig {
    /// Where the fixtures and `summary.txt` go.
    pub dir: PathBuf,
    /// Components (substrings of their `FrameworkEval` type names) to write fixtures of.
    pub dump: Vec<String>,
    /// Whether to check the interpreter against the SIMD evaluation on every row of every
    /// component (the dumped ones always are).
    pub check_all: bool,
}

/// Registers the capture hook.
pub fn install(config: CaptureConfig) {
    fs::create_dir_all(&config.dir).expect("capture directory");
    let log = Arc::new(Mutex::new(
        fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(config.dir.join("summary.txt"))
            .expect("summary.txt"),
    ));
    set_constraint_eval_hook(Some(Arc::new(move |job, accum| capture(&config, &log, job, accum))));
}

pub fn uninstall() {
    set_constraint_eval_hook(None);
}

fn record(log: &Mutex<fs::File>, line: String) {
    eprintln!("{line}");
    let mut file = log.lock().unwrap();
    writeln!(file, "{line}").unwrap();
}

/// A short name for a component type: `cairo_air::components::blake_g::Eval` -> `cairo.blake_g`.
pub fn short_name(component: &str) -> String {
    let crate_name = component.split("::").next().unwrap_or("");
    let module = component.rsplit("::").nth(1).unwrap_or(component);
    let prefix = if crate_name.starts_with("circuit") { "circuit" } else { "cairo" };
    format!("{prefix}.{module}")
}

/// The traced expressions against the lowered program on `eval`'s random assignment, with
/// `coeffs`: whether `sum_i coeffs[i] * constraint_i` agree.
pub fn check_random_assignment(eval: &ExprEvaluator, program: &Program, coeffs: &[QM31]) -> bool {
    let assignment = eval.random_assignment();
    let expected: QM31 =
        eval.constraints.iter().zip(coeffs).map(|(c, k)| *k * c.assign(&assignment)).sum();
    let mut values: HashMap<String, QM31> =
        assignment.1.iter().map(|(name, v)| (name.clone(), QM31::from(*v))).collect();
    values.extend(assignment.2.iter().map(|(name, v)| (name.clone(), *v)));
    let Ok(uniforms) = program.bind(&values) else { return false };
    // (interaction, index) -> the evaluator's column index, and slot -> preprocessed id.
    let global: HashMap<(u32, u32), usize> = eval
        .column_slots
        .iter()
        .enumerate()
        .map(|(g, (i, j))| ((*i as u32, *j as u32), g))
        .collect();
    let preprocessed: HashMap<u32, &String> =
        eval.preprocessed_columns.iter().map(|(id, slot)| (*slot as u32, id)).collect();
    let mut interpreter = Interpreter::new(program, &uniforms);
    let got = interpreter.row_res(coeffs, |interaction, index, offset| {
        if let Some(g) = global.get(&(interaction, index)) {
            return assignment.0[&(interaction as usize, *g, offset as isize)];
        }
        assert_eq!(interaction, 0, "a column the evaluator did not hand out");
        assignment.1[preprocessed[&index]]
    });
    got == expected
}

fn capture(
    config: &CaptureConfig,
    log: &Arc<Mutex<fs::File>>,
    job: &ConstraintEvalJob<'_>,
    accum: &mut SecureColumnByCoords<SimdBackend>,
) -> ConstraintEvalOutcome {
    let name = short_name(job.component);
    let start = Instant::now();
    let eval = job.constraints();
    let program = lower(&eval);
    let lower_time = start.elapsed();
    let counts = program.op_counts();
    let random_ok = check_random_assignment(&eval, &program, job.random_coeff_powers);
    // Per interaction, as the SIMD evaluator counts them (preprocessed reads included).
    let n_columns: Vec<usize> = {
        let mut n = vec![0usize; 4];
        for (i, _) in &eval.column_slots {
            n[*i] += 1;
        }
        n[0] += eval.preprocessed_columns.len();
        n
    };
    record(
        log,
        format!(
            "SHAPE {name} trace_log={} eval_log={} constraints={} traced={} columns={:?} \
             relations={} nodes={} row_nodes={} uniforms={} ops={:?} offsets={:?} \
             random_assignment={} lower_ms={:.1}",
            job.trace_log_size,
            job.eval_log_size,
            job.n_constraints,
            eval.constraints.len(),
            &n_columns[..3],
            eval.logup.fracs.len(),
            program.nodes.len(),
            program.row_nodes(0..program.constraints.len()).len(),
            super::wgsl::uniform_nodes(&program, 0..program.constraints.len()).len(),
            counts,
            program.offsets(),
            if random_ok { "ok" } else { "MISMATCH" },
            lower_time.as_secs_f64() * 1e3,
        ),
    );
    let dump = config.dump.iter().any(|d| job.component.contains(d.as_str()));
    if !(dump || config.check_all) {
        return ConstraintEvalOutcome::Declined(None);
    }

    // The inputs as the SIMD evaluation reads them.
    let start = Instant::now();
    let columns = job.columns_on_eval_domain();
    let extend_time = start.elapsed();
    let columns: Vec<Vec<Vec<u32>>> = columns
        .iter()
        .map(|cs| cs.iter().map(|c| bytemuck::cast_slice::<M31, u32>(c.as_slice()).to_vec()).collect())
        .collect();
    let input_p_words: usize =
        columns.iter().flatten().map(|c| c.iter().filter(|w| **w == P).count()).sum();
    let before: [Vec<u32>; 4] = std::array::from_fn(|c| {
        bytemuck::cast_slice::<M31, u32>(accum.columns[c].as_slice()).to_vec()
    });
    let params = param_values(&eval, job.claimed_sum, job.trace_log_size);
    let uniforms = program.bind(&params).expect("the component's parameters");
    let coeffs = job.random_coeff_powers.to_vec();
    let denom_inv: Vec<u32> = job.denom_inv.iter().map(|d| d.0).collect();
    let (trace_log_size, eval_log_size) = (job.trace_log_size, job.eval_log_size);
    let fixture_dir = config.dir.join(&name);
    if dump {
        fs::create_dir_all(&fixture_dir).unwrap();
        let n_columns: Vec<String> = columns.iter().map(|c| c.len().to_string()).collect();
        let indices: Vec<String> =
            job.preprocessed_column_indices.iter().map(|i| i.to_string()).collect();
        fs::write(
            fixture_dir.join("meta.txt"),
            format!(
                "component {}\ntrace_log_size {trace_log_size}\neval_log_size {eval_log_size}\n\
                 n_constraints {}\nn_columns {}\npreprocessed_column_indices {}\n",
                job.component,
                job.n_constraints,
                n_columns.join(" "),
                indices.join(" ")
            ),
        )
        .unwrap();
        fs::write(fixture_dir.join("program.txt"), program.to_text()).unwrap();
        write_params(&fixture_dir.join("params.txt"), &params).unwrap();
        write_coeffs(&fixture_dir.join("coeffs.bin"), &coeffs).unwrap();
        write_words(&fixture_dir.join("denom_inv.bin"), &[&denom_inv]).unwrap();
        let all: Vec<&[u32]> = columns.iter().flatten().map(|c| c.as_slice()).collect();
        write_words(&fixture_dir.join("columns.bin"), &all).unwrap();
        let before_slices: Vec<&[u32]> = before.iter().map(|c| c.as_slice()).collect();
        write_words(&fixture_dir.join("accum_before.bin"), &before_slices).unwrap();
    }

    let log = log.clone();
    let observer = move |after: &SecureColumnByCoords<SimdBackend>| {
        let after: [Vec<u32>; 4] = std::array::from_fn(|c| {
            bytemuck::cast_slice::<M31, u32>(after.columns[c].as_slice()).to_vec()
        });
        if dump {
            let slices: Vec<&[u32]> = after.iter().map(|c| c.as_slice()).collect();
            write_words(&fixture_dir.join("accum_after.bin"), &slices).unwrap();
        }
        let start = Instant::now();
        let slices: Vec<Vec<&[u32]>> =
            columns.iter().map(|cs| cs.iter().map(|c| c.as_slice()).collect()).collect();
        let inputs = RowInputs { columns: &slices, trace_log_size, eval_log_size };
        let n_rows = 1usize << eval_log_size;
        let chunk = 1 << 12;
        let expected: Vec<[u32; 4]> = (0..n_rows.div_ceil(chunk))
            .into_par_iter()
            .flat_map_iter(|k| {
                accumulate_rows(
                    &program,
                    &uniforms,
                    &coeffs,
                    &denom_inv,
                    &inputs,
                    &before,
                    k * chunk..((k + 1) * chunk).min(n_rows),
                )
            })
            .collect();
        let mut canonical_mismatches = 0;
        let mut raw_mismatches = 0;
        let mut p_words = 0;
        for (row, e) in expected.iter().enumerate() {
            for c in 0..4 {
                let raw = after[c][row];
                p_words += (raw == P) as usize;
                raw_mismatches += (raw != e[c]) as usize;
                canonical_mismatches += (raw % P != e[c]) as usize;
            }
        }
        record(
            &log,
            format!(
                "CHECK {name} rows={n_rows} canonical_mismatches={canonical_mismatches} \
                 raw_mismatches={raw_mismatches} accum_p_words={p_words} input_p_words={input_p_words} \
                 extend_ms={:.1} interp_ms={:.1}{}",
                extend_time.as_secs_f64() * 1e3,
                start.elapsed().as_secs_f64() * 1e3,
                if dump { " dumped" } else { "" }
            ),
        );
    };
    ConstraintEvalOutcome::Declined(Some(Box::new(observer)))
}

/// A dense `[coordinate][row]` copy of `col`'s words as stored.
pub fn raw_words(col: &SecureColumnByCoords<SimdBackend>) -> [Vec<u32>; 4] {
    std::array::from_fn(|c| bytemuck::cast_slice::<M31, u32>(col.columns[c].as_slice()).to_vec())
}
