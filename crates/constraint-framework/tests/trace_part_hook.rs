#![cfg(all(feature = "prover", feature = "std"))]
//! The `TracePartHook` on a striped commitment (a wide Fibonacci AIR, as stwo-examples', whose
//! degree-2 constraints are evaluated on two parts of each column's domain, the second
//! interpolated from the first by `trace_part`): a hook that provides the CPU's own evaluation of
//! every column, one that declines, and one that answers with the wrong number of columns all give
//! the proof without a hook, byte for byte; a hook that provides a wrong column makes `prove` fail
//! its out-of-domain check. The hook is process-wide, so this file is a test binary of its own,
//! with one test.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use itertools::Itertools;
use num_traits::{One, Zero};
use stwo::core::channel::Blake2sM31Channel;
use stwo::core::fields::FieldExpOps;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::core::fri::FriConfig;
use stwo::core::pcs::PcsConfig;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sM31MerkleChannel;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::poly::BitReversedOrder;
use stwo::prover::poly::circle::{CircleEvaluation, PolyOps, part_domain};
use stwo::prover::{CommitmentSchemeProver, Poly, ProvingError, prove};
use stwo_constraint_framework::{
    EvalAtRow, FrameworkComponent, FrameworkEval, TraceLocationAllocator, TracePartRequest,
    set_trace_part_hook,
};
const FIB_SEQUENCE_LENGTH: usize = 16;
/// Columns of 2^12 rows at blowup 2^3: domains of 2^15, striped (2^14 and up).
const LOG_N_ROWS: u32 = 12;
const LOG_BLOWUP: u32 = 3;

/// stwo-examples' wide Fibonacci AIR: `c = a^2 + b^2` along each row.
struct WideFibonacciEval<const N: usize> {
    log_n_rows: u32,
}

impl<const N: usize> FrameworkEval for WideFibonacciEval<N> {
    fn log_size(&self) -> u32 {
        self.log_n_rows
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_n_rows + 1
    }
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let mut a = eval.next_trace_mask();
        let mut b = eval.next_trace_mask();
        for _ in 2..N {
            let c = eval.next_trace_mask();
            eval.add_constraint(c.clone() - (a.square() + b.square()));
            a = b;
            b = c;
        }
        eval
    }
}

/// The trace: row `i` starts with `(1, i)`.
fn generate_trace<const N: usize>(
    log_n_rows: u32,
) -> Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
    let mut columns = (0..N).map(|_| vec![BaseField::zero(); 1 << log_n_rows]).collect_vec();
    for row in 0..1 << log_n_rows {
        let (mut a, mut b) = (BaseField::one(), BaseField::from_u32_unchecked(row as u32));
        columns[0][row] = a;
        columns[1][row] = b;
        for column in columns.iter_mut().skip(2) {
            (a, b) = (b, a.square() + b.square());
            column[row] = b;
        }
    }
    let domain = CanonicCoset::new(log_n_rows).circle_domain();
    columns
        .into_iter()
        .map(|values| CircleEvaluation::new(domain, values.into_iter().collect()))
        .collect()
}

/// The proof's `Debug` text (every commitment, sampled value and decommitment), or the error.
fn prove_striped() -> Result<String, ProvingError> {
    let config = PcsConfig {
        fri_config: FriConfig::new(10, 0, LOG_BLOWUP, 3, 1),
        trace_lifting_log_size: LOG_N_ROWS + LOG_BLOWUP,
        preprocessed_lifting_log_size: 0,
    };
    let twiddles = SimdBackend::precompute_twiddles(
        CanonicCoset::new(LOG_N_ROWS + 1 + LOG_BLOWUP).circle_domain().half_coset,
    );
    let channel = &mut Blake2sM31Channel::default();
    let mut commitment_scheme =
        CommitmentSchemeProver::<SimdBackend, Blake2sM31MerkleChannel>::new(config, &twiddles);
    commitment_scheme.set_stripe_residency();
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(vec![]);
    tree_builder.commit(channel);
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(generate_trace::<FIB_SEQUENCE_LENGTH>(LOG_N_ROWS));
    tree_builder.commit(channel);
    // The trace's columns hold their first stripe only.
    let trace = &commitment_scheme.trees[1].polynomials;
    assert!(trace.iter().all(|poly| poly.evals.values.length == 1 << LOG_N_ROWS));
    let component = FrameworkComponent::new(
        &mut TraceLocationAllocator::default(),
        WideFibonacciEval::<FIB_SEQUENCE_LENGTH> { log_n_rows: LOG_N_ROWS },
        SecureField::zero(),
    );
    prove::<SimdBackend, Blake2sM31MerkleChannel>(&[&component], channel, commitment_scheme)
        .map(|proof| format!("{proof:?}"))
}

/// The CPU's evaluation of a requested column, as `trace_part` computes it.
fn cpu_part(
    request: &TracePartRequest<'_>,
    k: usize,
) -> stwo::prover::backend::Col<SimdBackend, BaseField> {
    let column = request.columns[k].column;
    let first_part = part_domain(column.domain, request.log_parts, 0);
    let stripe_twiddles = SimdBackend::precompute_twiddles(first_part.half_coset);
    let twiddles = SimdBackend::precompute_twiddles(request.domain.half_coset);
    Poly::new(None, column.clone())
        .stripe_coefficients(&stripe_twiddles)
        .evaluate_with_twiddles(request.domain, &twiddles)
        .values
}

#[test]
fn trace_part_hook_provides_declines_or_is_caught() {
    set_trace_part_hook(None);
    let expected = prove_striped().expect("the proof without a hook");

    // The CPU's own evaluation of every column: the same proof.
    let columns = Arc::new(AtomicUsize::new(0));
    let counted = columns.clone();
    set_trace_part_hook(Some(Arc::new(move |request: &TracePartRequest<'_>| {
        assert_eq!(request.part, 1);
        assert_eq!(request.log_parts, LOG_BLOWUP);
        assert!(request.columns.iter().all(|column| column.tree == 1));
        counted.fetch_add(request.columns.len(), Ordering::SeqCst);
        (0..request.columns.len()).map(|k| Some(cpu_part(request, k))).collect()
    })));
    assert_eq!(prove_striped().expect("provided"), expected);
    assert_eq!(columns.load(Ordering::SeqCst), FIB_SEQUENCE_LENGTH);

    // Declined, or a wrong number of answers (ignored): the CPU evaluates them.
    set_trace_part_hook(Some(Arc::new(|request: &TracePartRequest<'_>| {
        request.columns.iter().map(|_| None).collect()
    })));
    assert_eq!(prove_striped().expect("declined"), expected);
    set_trace_part_hook(Some(Arc::new(|_: &TracePartRequest<'_>| vec![])));
    assert_eq!(prove_striped().expect("ignored"), expected);

    // Half provided, half declined.
    set_trace_part_hook(Some(Arc::new(|request: &TracePartRequest<'_>| {
        (0..request.columns.len()).map(|k| (k % 2 == 0).then(|| cpu_part(request, k))).collect()
    })));
    assert_eq!(prove_striped().expect("half provided"), expected);

    // A wrong column: the prover's out-of-domain check fails.
    set_trace_part_hook(Some(Arc::new(|request: &TracePartRequest<'_>| {
        (0..request.columns.len())
            .map(|k| {
                let mut values = cpu_part(request, k);
                if k == 0 {
                    let mut words = values.as_slice().to_vec();
                    words[3] += BaseField::one();
                    values = words.into_iter().collect();
                }
                Some(values)
            })
            .collect()
    })));
    assert!(matches!(prove_striped(), Err(ProvingError::ConstraintsNotSatisfied)));
    set_trace_part_hook(None);
}
