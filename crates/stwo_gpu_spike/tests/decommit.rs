//! The decommitment's evaluations on the GPU (`decommit::eval_at_domain_indices`) against
//! `SimdBackend::eval_at_domain_indices`, word for word, for columns of several sizes (below and
//! above the size whose coefficients stwo stores transposed), at the positions a decommitment
//! reads (aligned blocks around random queries); a changed coefficient changes them.
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use stwo::core::fields::m31::{BaseField, P};
use stwo::core::poly::circle::CanonicCoset;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::simd::column::BaseColumn;
use stwo::prover::poly::circle::{CircleCoefficients, PolyOps};
use stwo_gpu_spike::GpuContext;
use stwo_gpu_spike::decommit::{FoldKernels, eval_at_domain_indices};

fn positions(rng: &mut SmallRng, domain_log: u32, n_queries: usize, block_log: u32) -> Vec<usize> {
    let mut positions = vec![];
    for _ in 0..n_queries {
        let q = rng.random_range(0..1usize << domain_log) >> block_log << block_log;
        positions.extend(q..q + (1 << block_log));
    }
    positions.sort();
    positions.dedup();
    positions
}

#[test]
fn folds_are_stwos_evaluations() {
    let gpu = GpuContext::new();
    eprintln!("adapter: {}", gpu.adapter_info.name);
    let kernels = FoldKernels::new(&gpu);
    let mut rng = SmallRng::seed_from_u64(17);
    let log_blowup = 3;
    for log_size in [5u32, 11, 13, 16, 17, 20] {
        let n_columns = if log_size >= 17 { 3 } else { 5 };
        let columns: Vec<CircleCoefficients<SimdBackend>> = (0..n_columns)
            .map(|_| {
                CircleCoefficients::new(
                    (0..1u32 << log_size)
                        .map(|_| BaseField::from_u32_unchecked(rng.random_range(0..P)))
                        .collect::<BaseColumn>(),
                )
            })
            .collect();
        // The columns one after the other, past a gap.
        let mut words = vec![0u32; 1000];
        let mut offsets = vec![];
        for column in &columns {
            offsets.push(words.len() as u32);
            words.extend(column.coeffs.as_slice().iter().map(|v| v.0));
        }
        let buffer = gpu.upload("coefficients", &words);
        let domain = CanonicCoset::new(log_size + log_blowup).circle_domain();
        let indices = positions(&mut rng, domain.log_size(), 23, 3.min(log_size));
        let got =
            eval_at_domain_indices(&gpu, &kernels, &buffer, &offsets, log_size, domain, &indices);
        for (c, column) in columns.iter().enumerate() {
            let expected = SimdBackend::eval_at_domain_indices(column, domain, &indices);
            let expected: Vec<u32> = expected.iter().map(|v| v.0).collect();
            let got: Vec<u32> = got[c].iter().map(|v| v.0).collect();
            assert_eq!(got, expected, "log size {log_size}, column {c}");
        }
        // A changed coefficient changes the values.
        words[offsets[0] as usize + 7] = (words[offsets[0] as usize + 7] + 1) % P;
        let buffer = gpu.upload("coefficients", &words);
        let changed =
            eval_at_domain_indices(&gpu, &kernels, &buffer, &offsets, log_size, domain, &indices);
        assert_ne!(changed[0], got[0], "log size {log_size}: a changed coefficient unnoticed");
        assert_eq!(changed[1], got[1]);
        eprintln!(
            "log size {log_size}: {} positions x {n_columns} columns identical",
            indices.len()
        );
    }
}
