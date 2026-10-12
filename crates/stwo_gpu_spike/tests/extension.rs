//! The GPU extension against stwo's SIMD backend, word for word: every column's evaluation on a
//! stripe of its domain, as `CommitmentTreeProver::new_striped` makes it with
//! `SimdBackend::part_twiddles` and `SimdBackend::evaluate_into`.

use std::sync::Arc;

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use stwo::core::fields::m31::{BaseField, P};
use stwo::core::poly::circle::CanonicCoset;
use stwo::prover::backend::Column;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::simd::column::BaseColumn;
use stwo::prover::poly::circle::{CircleCoefficients, PolyOps, part_domain};
use stwo::prover::poly::twiddles::TwiddleTree;
use stwo_gpu_spike::GpuContext;
use stwo_gpu_spike::extension::{
    ColumnsLayout, ExtensionKernels, GpuColumns, StripeExtension, TILE_LOG, max_buffer_words,
    passes,
};

/// The distinct log sizes of the Cairo base trace's columns (`data/cairo_trace_shapes_pay.txt`).
fn cairo_log_sizes() -> Vec<u32> {
    let text = include_str!("../data/cairo_trace_shapes_pay.txt");
    let log_sizes = text
        .lines()
        .filter(|line| !line.trim().is_empty() && !line.starts_with('#'))
        .map(|line| line.split_whitespace().nth(1).unwrap().parse::<u32>().unwrap())
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        log_sizes.iter().copied().collect::<Vec<_>>(),
        [4, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    );
    log_sizes.into_iter().collect()
}

/// stwo's evaluation of `poly` on stripe `stripe` of its domain, as `new_striped` makes it.
fn simd_stripe(
    poly: &CircleCoefficients<SimdBackend>,
    log_blowup: u32,
    stripe: usize,
    tree: &TwiddleTree<SimdBackend>,
) -> Vec<u32> {
    let domain = CanonicCoset::new(poly.log_size() + log_blowup).circle_domain();
    let twiddles = SimdBackend::part_twiddles(domain, log_blowup, stripe, tree);
    let buffer = BaseColumn::zeros(1 << poly.log_size());
    let eval = SimdBackend::evaluate_into(
        poly,
        part_domain(domain, log_blowup, stripe),
        &twiddles,
        buffer,
    );
    eval.values.as_slice().iter().map(|x| x.0).collect()
}

struct Case<'a> {
    /// The columns' log sizes, in input order.
    log_sizes: &'a [u32],
    log_blowup: u32,
    stripes: &'a [usize],
    /// The largest buffer, in words (default: the device's).
    max_buffer_words: Option<u64>,
    /// The twiddle tree's domain exceeds the largest column domain by this many bits.
    tree_extra_bits: u32,
}

impl Case<'_> {
    fn new<'a>(log_sizes: &'a [u32], log_blowup: u32, stripes: &'a [usize]) -> Case<'a> {
        Case { log_sizes, log_blowup, stripes, max_buffer_words: None, tree_extra_bits: 0 }
    }
}

/// Evaluates the columns `coeffs` on the case's stripes on the GPU, and checks every word against
/// stwo's SIMD backend.
fn check_columns(gpu: &GpuContext, kernels: &ExtensionKernels, case: &Case, coeffs: Vec<Vec<u32>>) {
    let polys = coeffs
        .iter()
        .map(|words| {
            CircleCoefficients::<SimdBackend>::new(
                words.iter().map(|&w| BaseField::from_u32_unchecked(w)).collect(),
            )
        })
        .collect::<Vec<_>>();
    for (poly, &log_size) in polys.iter().zip(case.log_sizes) {
        assert_eq!(poly.log_size(), log_size);
    }
    let max_log_size = case.log_sizes.iter().max().unwrap() + case.log_blowup;
    let tree = SimdBackend::precompute_twiddles(
        CanonicCoset::new(max_log_size + case.tree_extra_bits).circle_domain().half_coset,
    );
    let max_words = case.max_buffer_words.unwrap_or_else(|| max_buffer_words(gpu));
    let layout = Arc::new(ColumnsLayout::new(case.log_sizes, max_words));
    let gpu_coeffs = GpuColumns::alloc(gpu, layout.clone());
    let words =
        polys.iter().map(|poly| bytemuck::cast_slice(poly.coeffs.as_slice())).collect::<Vec<_>>();
    gpu_coeffs.write(gpu, &words);
    let out = GpuColumns::alloc(gpu, layout.clone());
    let extension = StripeExtension::new(gpu, kernels, &gpu_coeffs, &out, case.log_blowup, &tree);
    for &stripe in case.stripes {
        extension.submit(gpu, kernels, stripe);
        let got = out.read(gpu);
        for (i, poly) in polys.iter().enumerate() {
            let want = simd_stripe(poly, case.log_blowup, stripe, &tree);
            if let Some(j) = (0..want.len()).find(|&j| got[i][j] != want[j]) {
                panic!(
                    "column {i} (log size {}), blowup {}, stripe {stripe}: word {j} is {}, stwo's \
                     {} ({} of {} words differ)",
                    case.log_sizes[i],
                    case.log_blowup,
                    got[i][j],
                    want[j],
                    (0..want.len()).filter(|&j| got[i][j] != want[j]).count(),
                    want.len(),
                );
            }
            // Equal to stwo's, so canonical as stwo's are; checked on its own all the same.
            assert!(got[i].iter().all(|&w| w < P));
        }
    }
    eprintln!(
        "ok: {} columns, log sizes {:?}, blowup {}, stripes {:?}, {} buffer(s), {} batch(es), {} \
         dispatches per stripe",
        case.log_sizes.len(),
        case.log_sizes,
        case.log_blowup,
        case.stripes,
        layout.buffer_words.len(),
        layout.batches.len(),
        extension.n_dispatches(),
    );
}

fn random_columns(log_sizes: &[u32], seed: u64) -> Vec<Vec<u32>> {
    let mut rng = SmallRng::seed_from_u64(seed);
    log_sizes.iter().map(|&l| (0..1 << l).map(|_| rng.random_range(0..P)).collect()).collect()
}

fn check(gpu: &GpuContext, kernels: &ExtensionKernels, case: &Case, seed: u64) {
    check_columns(gpu, kernels, case, random_columns(case.log_sizes, seed));
}

/// Each log size `copies` times, a batch of several columns (an odd number: batches that end
/// within a tile).
fn repeated(log_sizes: &[u32], copies: impl Fn(u32) -> usize) -> Vec<u32> {
    log_sizes.iter().flat_map(|&l| std::iter::repeat_n(l, copies(l))).collect()
}

#[test]
fn cairo_log_sizes_blowup_3_every_stripe() {
    let gpu = GpuContext::new();
    let kernels = ExtensionKernels::new(&gpu);
    let log_sizes = repeated(&cairo_log_sizes(), |l| match l {
        ..=11 => 5,
        12..=18 => 3,
        _ => 1,
    });
    check(&gpu, &kernels, &Case::new(&log_sizes, 3, &[0, 1, 2, 3, 4, 5, 6, 7]), 3);
}

#[test]
fn cairo_log_sizes_blowup_2_every_stripe() {
    let gpu = GpuContext::new();
    let kernels = ExtensionKernels::new(&gpu);
    let log_sizes = repeated(&cairo_log_sizes(), |l| if l <= 16 { 3 } else { 1 });
    check(&gpu, &kernels, &Case::new(&log_sizes, 2, &[0, 1, 2, 3]), 2);
}

#[test]
fn every_log_size_4_to_20() {
    let gpu = GpuContext::new();
    let kernels = ExtensionKernels::new(&gpu);
    let log_sizes = (4..=20).rev().collect::<Vec<_>>();
    check(&gpu, &kernels, &Case::new(&log_sizes, 3, &[1, 6]), 4);
    check(&gpu, &kernels, &Case::new(&log_sizes, 1, &[0, 1]), 5);
}

#[test]
fn log_sizes_21_to_23() {
    let gpu = GpuContext::new();
    let kernels = ExtensionKernels::new(&gpu);
    check(&gpu, &kernels, &Case::new(&[23, 21, 22], 3, &[0, 5]), 21);
    check(&gpu, &kernels, &Case::new(&[21, 22, 23], 2, &[3]), 22);
}

/// A twiddle tree larger than the domains need, as the prover's (it covers its tallest tree).
#[test]
fn larger_twiddle_tree() {
    let gpu = GpuContext::new();
    let kernels = ExtensionKernels::new(&gpu);
    let log_sizes = [4, 9, 12, 13, 17, 18];
    let case = Case { tree_extra_bits: 3, ..Case::new(&log_sizes, 3, &[0, 2, 7]) };
    check(&gpu, &kernels, &case, 7);
}

/// Coefficients at the ends of the field's range, and constant columns.
#[test]
fn edge_coefficients() {
    let gpu = GpuContext::new();
    let kernels = ExtensionKernels::new(&gpu);
    let log_sizes = [4, 4, 4, 12, 12, 12, 17, 17, 17];
    let mut rng = SmallRng::seed_from_u64(11);
    let edges = [0, 1, 2, P - 1, P - 2, 1 << 30, (1 << 30) - 1, 1 << 16, (1 << 16) - 1];
    let columns = log_sizes
        .iter()
        .enumerate()
        .map(|(i, &l)| {
            (0..1u32 << l)
                .map(|_| match i % 3 {
                    0 => edges[rng.random_range(0..edges.len())],
                    1 => 0,
                    _ => P - 1,
                })
                .collect()
        })
        .collect();
    check_columns(&gpu, &kernels, &Case::new(&log_sizes, 3, &[0, 1, 2, 3, 4, 5, 6, 7]), columns);
}

/// Coefficients equal to P, the other representation of 0, which stwo's SIMD arithmetic accepts
/// (it keeps values in [0, P]): the GPU's butterflies are the same as the SIMD backend's, so the
/// outputs are still equal word for word. (Log size 4 is left out: there stwo evaluates on the CPU
/// backend, whose arithmetic reduces differently; stwo's coefficients are canonical anyway.)
#[test]
fn coefficients_equal_to_p() {
    let gpu = GpuContext::new();
    let kernels = ExtensionKernels::new(&gpu);
    let log_sizes = [5, 12, 13, 17, 18];
    let mut rng = SmallRng::seed_from_u64(12);
    let columns = log_sizes
        .iter()
        .map(|&l| {
            (0..1u32 << l)
                .map(|_| if rng.random_bool(0.3) { P } else { rng.random_range(0..P) })
                .collect()
        })
        .collect();
    check_columns(&gpu, &kernels, &Case::new(&log_sizes, 3, &[0, 4, 7]), columns);
}

/// Within WebGPU's default limits (a browser's, unless it asks for more), with buffers small
/// enough that the columns spread over several and batches of one log size are split.
#[test]
fn webgpu_default_limits_and_several_buffers() {
    let gpu = GpuContext::with_limits(wgpu::Limits::default());
    let kernels = ExtensionKernels::new(&gpu);
    let log_sizes = repeated(&[4, 8, 12, 13, 16, 17, 18, 19], |l| if l <= 16 { 5 } else { 3 });
    let case = Case { max_buffer_words: Some(1 << 20), ..Case::new(&log_sizes, 3, &[0, 3, 6]) };
    let layout = ColumnsLayout::new(&log_sizes, 1 << 20);
    assert!(layout.buffer_words.len() >= 3);
    assert!(layout.batches.len() > 8);
    check(&gpu, &kernels, &case, 8);
}

#[test]
fn layout_groups_by_log_size_within_buffer_limits() {
    let log_sizes = [10, 4, 12, 10, 4, 12, 12, 4];
    let layout = ColumnsLayout::new(&log_sizes, 1 << 13);
    // By log size, in input order within one; 4096-word columns two to a buffer.
    let order = layout.batches.iter().flat_map(|batch| batch.columns.clone()).collect::<Vec<_>>();
    assert_eq!(order, [1, 4, 7, 0, 3, 2, 5, 6]);
    for batch in &layout.batches {
        assert_eq!(batch.offset % 64, 0);
        for (k, &i) in batch.columns.iter().enumerate() {
            let slot = layout.slots[i];
            assert_eq!(slot.log_size, log_sizes[i]);
            assert_eq!(
                (slot.buffer, slot.offset),
                (batch.buffer, batch.offset + ((k as u64) << slot.log_size))
            );
        }
    }
    for (b, &words) in layout.buffer_words.iter().enumerate() {
        assert!(words <= 1 << 13);
        // No two columns overlap.
        let mut ranges = layout
            .slots
            .iter()
            .filter(|slot| slot.buffer == b)
            .map(|slot| (slot.offset, slot.offset + (1 << slot.log_size)))
            .collect::<Vec<_>>();
        ranges.sort();
        assert!(ranges.windows(2).all(|w| w[0].1 <= w[1].0));
    }
}

#[test]
fn passes_cover_every_layer_once() {
    for log_size in 4..=28 {
        let passes = passes(log_size);
        let mut top = log_size;
        for pass in &passes {
            assert_eq!(pass.hi, top);
            assert!(pass.lo < pass.hi);
            assert!(pass.c <= pass.lo);
            assert!(pass.hi - pass.lo + pass.c <= TILE_LOG);
            if pass.lo > 0 {
                // A tile keeps 16 consecutive words together at least, and fills 2^12 words.
                assert!(pass.c >= 4);
                assert_eq!(pass.hi - pass.lo + pass.c, TILE_LOG);
            }
            top = pass.lo;
        }
        assert_eq!(top, 0);
        let expected = if log_size <= 12 { 1 } else { 1 + (log_size - 12).div_ceil(8) as usize };
        assert_eq!(passes.len(), expected, "log size {log_size}");
    }
}

#[test]
fn print_adapter_limits() {
    let gpu = GpuContext::new();
    let limits = gpu.device.limits();
    eprintln!("adapter: {:?}", gpu.adapter_info);
    eprintln!(
        "max_buffer_size {} max_storage_buffer_binding_size {} max_compute_workgroup_storage_size \
         {} max_compute_invocations_per_workgroup {} max_compute_workgroups_per_dimension {} \
         min_uniform_buffer_offset_alignment {}",
        limits.max_buffer_size,
        limits.max_storage_buffer_binding_size,
        limits.max_compute_workgroup_storage_size,
        limits.max_compute_invocations_per_workgroup,
        limits.max_compute_workgroups_per_dimension,
        limits.min_uniform_buffer_offset_alignment,
    );
}
