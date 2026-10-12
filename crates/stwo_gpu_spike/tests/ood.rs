//! The out-of-domain samples on the GPU (`ood::eval_jobs`) against
//! `CircleCoefficients::eval_at_point`, word for word, for columns of several sizes (below and
//! above the size whose coefficients stwo stores transposed; with coefficients equal to P), at
//! several points, in one submission over two buffers; a changed coefficient changes them.
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use stwo::core::circle::CirclePoint;
use stwo::core::fields::m31::{BaseField, P};
use stwo::core::fields::qm31::SecureField;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::simd::column::BaseColumn;
use stwo::prover::poly::circle::CircleCoefficients;
use stwo_gpu_spike::GpuContext;
use stwo_gpu_spike::ood::{OodJob, OodKernels, eval_jobs};

fn words(values: &[SecureField]) -> Vec<[u32; 4]> {
    values.iter().map(|v| v.to_m31_array().map(|m| m.0)).collect()
}

#[test]
fn samples_are_stwos() {
    let gpu = GpuContext::new();
    eprintln!("adapter: {}", gpu.adapter_info.name);
    let kernels = OodKernels::new(&gpu);
    let mut rng = SmallRng::seed_from_u64(29);
    let points: Vec<CirclePoint<SecureField>> =
        (0..3).map(|_| CirclePoint::get_point(rng.random::<u64>() as u128)).collect();
    for log_size in [5u32, 6, 8, 9, 11, 12, 13, 16, 17, 20] {
        let n_columns = if log_size >= 17 { 3 } else { 6 };
        let columns: Vec<CircleCoefficients<SimdBackend>> = (0..n_columns)
            .map(|c| {
                CircleCoefficients::new(
                    (0..1u32 << log_size)
                        .map(|i| match c == 1 && i % 5 == 0 {
                            // Some coefficients equal to P, which stwo reads as zero.
                            true => BaseField::from_u32_unchecked(P),
                            false => BaseField::from_u32_unchecked(rng.random_range(0..P)),
                        })
                        .collect::<BaseColumn>(),
                )
            })
            .collect();
        // The columns one after the other, past a gap, the first half in one buffer, the rest in
        // another.
        let split = n_columns / 2;
        let mut words_of = [vec![0u32; 1000], vec![0u32; 333]];
        let mut offsets = [vec![], vec![]];
        for (c, column) in columns.iter().enumerate() {
            let b = usize::from(c >= split);
            offsets[b].push(words_of[b].len() as u32);
            words_of[b].extend(column.coeffs.as_slice().iter().map(|v| v.0));
        }
        let buffers = [
            gpu.upload("coefficients 0", &words_of[0]),
            gpu.upload("coefficients 1", &words_of[1]),
        ];
        let jobs: Vec<OodJob<'_>> = points
            .iter()
            .flat_map(|&point| (0..2).map(move |b| (b, point)))
            .map(|(b, point)| OodJob { buffer: &buffers[b], columns: &offsets[b], log_size, point })
            .collect();
        let got = eval_jobs(&gpu, &kernels, &jobs);
        for (j, job) in jobs.iter().enumerate() {
            let first = if std::ptr::eq(job.buffer, &buffers[0]) { 0 } else { split };
            let expected: Vec<SecureField> = (0..job.columns.len())
                .map(|c| columns[first + c].eval_at_point(job.point))
                .collect();
            assert_eq!(words(&got[j]), words(&expected), "log size {log_size}, job {j}");
        }
        // A changed coefficient changes its column's samples only.
        let at = offsets[0][0] as usize + 7;
        words_of[0][at] = (words_of[0][at] + 1) % P;
        let changed_buffer = gpu.upload("coefficients 0", &words_of[0]);
        let changed = eval_jobs(
            &gpu,
            &kernels,
            &[OodJob { buffer: &changed_buffer, columns: &offsets[0], log_size, point: points[0] }],
        );
        assert_ne!(
            changed[0][0], got[0][0],
            "log size {log_size}: a changed coefficient unnoticed"
        );
        assert_eq!(changed[0][1..], got[0][1..]);
        eprintln!("log size {log_size}: {n_columns} columns x {} points identical", points.len());
    }
}
