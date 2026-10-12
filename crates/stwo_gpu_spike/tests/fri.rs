//! The FRI quotients on the GPU (`fri::fri_quotients`) against stwo's `compute_fri_quotients` on
//! `SimdBackend`, word for word: columns of several log sizes in three trees (most from their
//! coefficients on the GPU, some from their values), one or two samples each (two add stwo's
//! periodicity sample), at blowups 2 and 3; a changed coefficient changes the quotients.
use std::cell::RefCell;
use std::collections::BTreeSet;
use std::sync::Arc;

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use stwo::core::circle::{CirclePoint, SECURE_FIELD_CIRCLE_GEN};
use stwo::core::fields::m31::{BaseField, P};
use stwo::core::fields::qm31::SecureField;
use stwo::core::pcs::TreeVec;
use stwo::core::pcs::quotients::PointSample;
use stwo::core::poly::circle::CanonicCoset;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::simd::column::BaseColumn;
use stwo::prover::pcs::quotient_ops::compute_fri_quotients;
use stwo::prover::poly::BitReversedOrder;
use stwo::prover::poly::circle::{CircleCoefficients, CircleEvaluation, PolyOps};
use stwo::prover::poly::twiddles::TwiddleTree;
use stwo_gpu_spike::GpuContext;
use stwo_gpu_spike::extension::{ExtensionKernels, StripeTwiddles};
use stwo_gpu_spike::fri::{FriColumn, FriResources, FriState, Source, fri_quotients};
use stwo_gpu_spike::transfer::{Readback, ReadbackRing, Upload, UploadRing};

struct Resources<'a> {
    gpu: &'a GpuContext,
    tree: &'a TwiddleTree<SimdBackend>,
    buffers: RefCell<Vec<wgpu::Buffer>>,
    upload: RefCell<UploadRing>,
    readback: RefCell<ReadbackRing>,
}

impl FriResources for Resources<'_> {
    fn buffer(&self, _slot: usize, bytes: u64) -> wgpu::Buffer {
        let buffer = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test"),
            size: bytes.next_multiple_of(256),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.buffers.borrow_mut().push(buffer.clone());
        buffer
    }

    fn stripe_twiddles(
        &self,
        log_sizes: &BTreeSet<u32>,
        log_blowup: u32,
    ) -> Option<Arc<StripeTwiddles>> {
        Some(Arc::new(StripeTwiddles::new(self.gpu, log_sizes, log_blowup, self.tree)))
    }

    fn upload(&self, pieces: &[Upload<'_>]) -> Result<(), String> {
        self.upload.borrow_mut().upload(self.gpu, pieces);
        Ok(())
    }

    fn read(&self, pieces: Vec<Readback<'_, '_>>) -> Result<(), String> {
        self.readback.borrow_mut().read(self.gpu, pieces);
        Ok(())
    }
}

fn random_column(rng: &mut SmallRng, log_size: u32) -> CircleCoefficients<SimdBackend> {
    CircleCoefficients::new(
        (0..1u32 << log_size)
            .map(|_| BaseField::from_u32_unchecked(rng.random_range(0..P)))
            .collect::<BaseColumn>(),
    )
}

fn point(rng: &mut SmallRng) -> CirclePoint<SecureField> {
    SECURE_FIELD_CIRCLE_GEN.mul(rng.random::<u64>() as u128)
}

#[test]
fn quotients_are_stwos() {
    let gpu = GpuContext::new();
    eprintln!("adapter: {}", gpu.adapter_info.name);
    let extension = ExtensionKernels::new(&gpu);
    let mut state = FriState::new(&gpu);
    let mut rng = SmallRng::seed_from_u64(41);
    // (blowup, trees of trace log sizes)
    // The last above the size whose coefficients stwo stores transposed.
    let cases: [(u32, Vec<Vec<u32>>); 3] = [
        (3, vec![vec![13, 9, 5], vec![13, 13, 12, 9, 9, 7, 4], vec![11, 13, 6, 13]]),
        (2, vec![vec![14, 10], vec![14, 8, 8, 12, 5], vec![14, 14, 9]]),
        (3, vec![vec![18, 17, 12], vec![18, 9, 17], vec![16]]),
    ];
    for (log_blowup, trees) in cases {
        let max = *trees.iter().flatten().max().unwrap();
        let lifting_log_size = max + log_blowup;
        let tree = SimdBackend::precompute_twiddles(
            CanonicCoset::new(lifting_log_size).circle_domain().half_coset,
        );
        let polys: Vec<Vec<CircleCoefficients<SimdBackend>>> = trees
            .iter()
            .map(|sizes| sizes.iter().map(|&n| random_column(&mut rng, n)).collect())
            .collect();
        let evaluations: Vec<Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>> =
            polys
                .iter()
                .map(|tree_polys| {
                    tree_polys
                        .iter()
                        .map(|poly| {
                            let domain =
                                CanonicCoset::new(poly.log_size() + log_blowup).circle_domain();
                            poly.evaluate_with_twiddles(domain, &tree)
                        })
                        .collect()
                })
                .collect();
        // Samples: one point for most, two for some (the second at the first's shift).
        let points = [point(&mut rng), point(&mut rng), point(&mut rng)];
        let samples = TreeVec::new(
            evaluations
                .iter()
                .enumerate()
                .map(|(t, tree_evals)| {
                    tree_evals
                        .iter()
                        .enumerate()
                        .map(|(c, _)| {
                            let value = SecureField::from_m31_array(std::array::from_fn(|_| {
                                BaseField::from_u32_unchecked(rng.random_range(0..P))
                            }));
                            let main = PointSample { point: points[t % 2], value };
                            match (t + c) % 3 {
                                0 => vec![PointSample { point: points[2], value }, main],
                                _ => vec![main],
                            }
                        })
                        .collect()
                })
                .collect(),
        );
        let random_coeff = SecureField::from_m31_array(std::array::from_fn(|_| {
            BaseField::from_u32_unchecked(rng.random_range(0..P))
        }));
        let columns = TreeVec::new(evaluations.iter().map(|tree| tree.iter().collect()).collect());
        let expected = compute_fri_quotients(
            &columns,
            &samples,
            random_coeff,
            lifting_log_size,
            &tree,
            log_blowup,
        );

        // On the GPU: the coefficients in one buffer, past a gap; every fifth column from its
        // values.
        let mut words = vec![0u32; 777];
        let mut offsets = vec![];
        for poly in polys.iter().flatten() {
            offsets.push(words.len() as u32);
            words.extend(poly.coeffs.as_slice().iter().map(|v| v.0));
            words.resize(words.len().next_multiple_of(64), 0);
        }
        let mut run = |words: &[u32]| {
            let buffer = gpu.upload("coefficients", words);
            let mut k = 0;
            let fri_columns = TreeVec::new(
                evaluations
                    .iter()
                    .map(|tree_evals| {
                        tree_evals
                            .iter()
                            .map(|evaluation| {
                                let source = match k % 5 {
                                    4 => Source::Values(evaluation.values.as_slice()),
                                    _ => {
                                        Source::Coefficients { buffer: &buffer, offset: offsets[k] }
                                    }
                                };
                                k += 1;
                                FriColumn { source, log_size: evaluation.domain.log_size() }
                            })
                            .collect()
                    })
                    .collect(),
            );
            let resources = Resources {
                gpu: &gpu,
                tree: &tree,
                buffers: RefCell::new(vec![]),
                upload: RefCell::new(UploadRing::with_defaults(&gpu)),
                readback: RefCell::new(ReadbackRing::with_defaults(&gpu)),
            };
            let (got, times) = fri_quotients(
                &gpu,
                &mut state,
                &extension,
                &resources,
                &fri_columns,
                &samples,
                random_coeff,
                lifting_log_size,
                log_blowup,
            )
            .expect("fri quotients");
            eprintln!("{times:?}");
            got.map(|column| column.as_slice().iter().map(|v| v.0).collect::<Vec<u32>>())
        };
        let got = run(&words);
        for (k, column) in got.iter().enumerate() {
            let want: Vec<u32> =
                expected.values.columns[k].as_slice().iter().map(|v| v.0).collect();
            assert_eq!(column.len(), want.len());
            let first_diff = column.iter().zip(&want).position(|(a, b)| a != b);
            assert_eq!(first_diff, None, "blowup {log_blowup}, coordinate {k}");
        }
        // A changed coefficient changes the quotients.
        words[offsets[0] as usize + 3] = (words[offsets[0] as usize + 3] + 1) % P;
        let changed = run(&words);
        assert_ne!(changed, got, "blowup {log_blowup}: a changed coefficient unnoticed");
        eprintln!("blowup {log_blowup}: {} words x 4 identical", got[0].len());
    }
}
