//! A2 (`recompute`): stripe 1 of committed columns recomputed on the GPU from their stripe 0,
//! against the CPU's evaluation of it in constraint-framework's `trace_part` (the column
//! interpolated from stripe 0 with `Poly::stripe_coefficients`, evaluated on part 1 with
//! `evaluate_with_twiddles`), raw word for raw word; and the coefficients the GPU interpolated
//! against `SimdBackend::interpolate`'s, as stwo stores them.
//!
//! The largest sizes run with `STWO_GPU_SPIKE_FULL=1` (slow on lavapipe).

use itertools::Itertools;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use stwo::core::fields::m31::{BaseField, P};
use stwo::core::poly::circle::CanonicCoset;
use stwo::prover::Poly;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::simd::column::BaseColumn;
use stwo::prover::poly::BitReversedOrder;
use stwo::prover::poly::circle::{CircleCoefficients, CircleEvaluation, PolyOps, part_domain};
use stwo_gpu_spike::committer::GpuCommitter;
use stwo_gpu_spike::parts::PartsMode;

fn full() -> bool {
    std::env::var("STWO_GPU_SPIKE_FULL").is_ok_and(|v| v == "1")
}

fn words(column: &BaseColumn) -> Vec<u32> {
    column.as_slice().iter().map(|value| value.0).collect()
}

/// A column as a striped tree holds it.
type Committed = CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>;

/// `n` random columns of `2^log_size` coefficients, as the tree holds them committed at
/// `log_blowup`: stripe 0's evaluations, on the whole domain.
fn committed_columns(
    log_size: u32,
    log_blowup: u32,
    n: usize,
    seed: u64,
) -> (Vec<CircleCoefficients<SimdBackend>>, Vec<Committed>) {
    let domain = CanonicCoset::new(log_size + log_blowup).circle_domain();
    let stripe0 = part_domain(domain, log_blowup, 0);
    let twiddles = SimdBackend::precompute_twiddles(stripe0.half_coset);
    (0..n)
        .map(|i| {
            let mut rng = SmallRng::seed_from_u64(seed * 1_000_003 + i as u64);
            let poly = CircleCoefficients::<SimdBackend>::new(
                (0..1 << log_size)
                    .map(|_| BaseField::from_u32_unchecked(rng.random_range(0..P)))
                    .collect::<BaseColumn>(),
            );
            let mut evaluation = poly.evaluate_with_twiddles(stripe0, &twiddles);
            evaluation.domain = domain;
            (poly, evaluation)
        })
        .unzip()
}

/// The CPU's stripe 1 and coefficients of a column, as `trace_part` computes them.
fn cpu_part1(
    column: &CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>,
    log_blowup: u32,
) -> (BaseColumn, BaseColumn) {
    let domain = column.domain;
    let first_part = part_domain(domain, log_blowup, 0);
    let stripe_twiddles = SimdBackend::precompute_twiddles(first_part.half_coset);
    let part1 = part_domain(domain, log_blowup, 1);
    let twiddles = SimdBackend::precompute_twiddles(part1.half_coset);
    let coefficients = Poly::new(None, column.clone()).stripe_coefficients(&stripe_twiddles);
    let stripe1 = coefficients.evaluate_with_twiddles(part1, &twiddles).values;
    (stripe1, coefficients.coeffs)
}

#[test]
fn recomputed_stripe1_is_the_cpus() {
    let committer = GpuCommitter::new().expect("a GPU adapter");
    println!("adapter: {:?}", committer.adapter_info());
    let mut shapes = vec![(11, 3, 3), (12, 2, 2), (13, 3, 2), (16, 3, 2), (17, 3, 3), (18, 2, 2)];
    if full() {
        shapes.extend([(20, 3, 3), (21, 2, 2)]);
    }
    for (k, &(log_size, log_blowup, n)) in shapes.iter().enumerate() {
        let (polys, columns) = committed_columns(log_size, log_blowup, n, k as u64);
        let values = columns.iter().map(|column| &column.values).collect_vec();
        let (stripes, coefficients, times) =
            committer.recompute_stripe1_with_coefficients(&values, log_size, log_blowup);
        for (i, column) in columns.iter().enumerate() {
            let (cpu_stripe1, cpu_coefficients) = cpu_part1(column, log_blowup);
            // The CPU's interpolation is the column's polynomial (a sanity check of the setup).
            assert_eq!(
                words(&cpu_coefficients),
                words(&polys[i].coeffs),
                "2^{log_size} at 2^{log_blowup}: the CPU's coefficients of column {i}"
            );
            assert_eq!(
                words(&coefficients[i]),
                words(&cpu_coefficients),
                "2^{log_size} at 2^{log_blowup}: the coefficients of column {i}"
            );
            assert_eq!(
                words(&stripes[i]),
                words(&cpu_stripe1),
                "2^{log_size} at 2^{log_blowup}: stripe 1 of column {i}"
            );
            assert!(words(&cpu_stripe1).iter().all(|&w| w < P), "the CPU's stripe 1 is canonical");
        }
        println!("2^{log_size} x{n} at 2^{log_blowup}: identical ({times:?})");
    }
}

#[test]
fn trace_part_recomputes_or_declines() {
    let committer = GpuCommitter::new().expect("a GPU adapter");
    committer.set_parts_mode(PartsMode::Recompute);
    let (log_size, log_blowup) = (12, 3);
    let (_, columns) = committed_columns(log_size, log_blowup, 3, 7);
    let request = columns.iter().map(|column| (1, column)).collect_vec();
    let answers = committer.trace_part(1, log_blowup, &request);
    for (column, answer) in columns.iter().zip(&answers) {
        let (cpu_stripe1, _) = cpu_part1(column, log_blowup);
        assert_eq!(words(answer.as_ref().expect("recomputed")), words(&cpu_stripe1));
    }
    // A flipped word of stripe 0 changes stripe 1: the comparison sees it.
    let mut flipped = columns[0].clone();
    let mut values = flipped.values.as_slice().to_vec();
    values[5] = BaseField::from_u32_unchecked((values[5].0 + 1) % P);
    flipped.values = values.into_iter().collect();
    let answer = committer.trace_part(1, log_blowup, &[(1, &flipped)]);
    assert_ne!(words(answer[0].as_ref().unwrap()), words(&cpu_part1(&columns[0], log_blowup).0));
    // Another part, or another blowup than the columns', is declined.
    assert!(committer.trace_part(2, log_blowup, &request).iter().all(Option::is_none));
    assert!(committer.trace_part(1, log_blowup - 1, &request).iter().all(Option::is_none));
    // Off: declined.
    committer.set_parts_mode(PartsMode::Off);
    assert!(committer.trace_part(1, log_blowup, &request).iter().all(Option::is_none));
}

/// A1 and A3: a striped commitment on the GPU keeps stripe 1 (read back, or on the GPU), and
/// `trace_part` answers with it: the CPU's evaluation, word for word; a column of the base trace
/// once (taken), a preprocessed one every time (copied); a column kept whole, or one whose values
/// changed, never; and the next commitment after a lookup drops the rest.
#[test]
fn kept_stripe1_is_the_cpus() {
    use stwo::prover::backend::simd::SimdBackend as B;
    let committer = GpuCommitter::new().expect("a GPU adapter");
    let (log_blowup, lifting_log_size) = (3, 17);
    // 2^10 at blowup 3 is kept whole (its domain is below 2^14).
    let log_sizes = [10u32, 11, 12, 12, 13, 14];
    let twiddles =
        B::precompute_twiddles(CanonicCoset::new(lifting_log_size).circle_domain().half_coset);
    for mode in [PartsMode::Host, PartsMode::Device] {
        committer.set_parts_mode(mode);
        committer.clear_parts();
        let polys = log_sizes
            .iter()
            .enumerate()
            .map(|(i, &log_size)| {
                let mut rng = SmallRng::seed_from_u64(99 + i as u64);
                CircleCoefficients::<B>::new(
                    (0..1 << log_size)
                        .map(|_| BaseField::from_u32_unchecked(rng.random_range(0..P)))
                        .collect::<BaseColumn>(),
                )
            })
            .collect_vec();
        let tree = committer
            .commit_striped(polys, log_blowup, &twiddles, lifting_log_size)
            .unwrap_or_else(|(_, reason)| panic!("declined: {reason:?}"));
        for (i, poly) in tree.polynomials.iter().enumerate() {
            let evals = &poly.evals;
            let whole = evals.values.length == evals.domain.size();
            assert_eq!(whole, i == 0, "{mode:?}: column {i} kept whole");
            // Tree 1 (taken) for the odd columns, tree 0 (copied) for the even ones.
            let tree_index = i % 2;
            let answer = committer.trace_part(1, log_blowup, &[(tree_index, evals)]).remove(0);
            if whole {
                assert!(answer.is_none(), "{mode:?}: a whole column is not kept");
                continue;
            }
            let (cpu_stripe1, _) = cpu_part1(evals, log_blowup);
            assert_eq!(
                words(answer.as_ref().expect("kept")),
                words(&cpu_stripe1),
                "{mode:?}: column {i}"
            );
            let again = committer.trace_part(1, log_blowup, &[(tree_index, evals)]).remove(0);
            assert_eq!(again.is_some(), tree_index == 0, "{mode:?}: column {i} asked again");
            // Other values at the same place: not found.
            let mut changed = evals.clone();
            changed.values = evals
                .values
                .as_slice()
                .iter()
                .map(|v| BaseField::from_u32_unchecked((v.0 + 1) % P))
                .collect();
            assert!(committer.trace_part(1, log_blowup, &[(0, &changed)])[0].is_none());
        }
        let stats = committer.take_part_stats();
        assert!(stats.hits > 0 && stats.held_bytes > 0, "{mode:?}: {stats:?}");
        // The next commitment after a lookup drops what was kept (here: the preprocessed columns).
        let small = vec![CircleCoefficients::<B>::new(
            (0..1 << 13).map(BaseField::from).collect::<BaseColumn>(),
        )];
        let _ = committer.commit_striped(small, log_blowup, &twiddles, lifting_log_size);
        let stats = committer.take_part_stats();
        assert_eq!(stats.clears, 1, "{mode:?}: {stats:?}");
        let answer = committer.trace_part(1, log_blowup, &[(0, &tree.polynomials[2].evals)]);
        assert!(answer[0].is_none(), "{mode:?}: dropped");
    }
    committer.set_parts_mode(PartsMode::Off);
}
