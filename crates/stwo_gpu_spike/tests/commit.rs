//! The GPU striped commitment against `CommitmentTreeProver::new_striped` on the same random
//! polynomials: every layer of the tree (root included), and every column's kept evaluations
//! (stripe 0, or the whole evaluation of the small columns), with their domains.
//!
//! Every case has `new_striped`'s preconditions for its striped path, and checks that the CPU took
//! it: the columns it does not keep whole hold their stripe 0 only, and there is at least one.
//!
//! The full-size Cairo and circuit trees run with `STWO_GPU_SPIKE_FULL=1` (slow on lavapipe).

use itertools::Itertools;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use stwo::core::fields::m31::{BaseField, P};
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sM31MerkleChannel;
use stwo::prover::CommitmentTreeProver;
use stwo::prover::backend::Column;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::simd::column::BaseColumn;
use stwo::prover::mempool::BaseColumnPool;
use stwo::prover::poly::circle::{CircleCoefficients, PolyOps};
use stwo::prover::poly::twiddles::TwiddleTree;
use stwo_gpu_spike::GpuContext;
use stwo_gpu_spike::commit::{
    CommitKernels, StripedCommitment, commit_striped, kept_whole, takes_striped_path,
};
use stwo_gpu_spike::extension::max_buffer_words;
use stwo_gpu_spike::shapes::{
    CAIRO_LIFTING_LOG_SIZE, CAIRO_LOG_BLOWUP, CIRCUIT_LIFTING_LOG_SIZE, CIRCUIT_LOG_BLOWUP,
    base_trace_log_sizes, cairo_pay_components,
};

type TreeProver = CommitmentTreeProver<SimdBackend, Blake2sM31MerkleChannel>;

fn full() -> bool {
    std::env::var("STWO_GPU_SPIKE_FULL").is_ok_and(|v| v == "1")
}

/// Random polynomials of these coefficients log sizes (canonical coefficients, as stwo's).
fn random_polys(log_sizes: &[u32], seed: u64) -> Vec<CircleCoefficients<SimdBackend>> {
    log_sizes
        .par_iter()
        .enumerate()
        .map(|(i, &log_size)| {
            let mut rng = SmallRng::seed_from_u64(seed * 1_000_003 + i as u64);
            CircleCoefficients::new(
                (0..1 << log_size)
                    .map(|_| BaseField::from_u32_unchecked(rng.random_range(0..P)))
                    .collect::<BaseColumn>(),
            )
        })
        .collect()
}

fn twiddles(lifting_log_size: u32) -> TwiddleTree<SimdBackend> {
    SimdBackend::precompute_twiddles(CanonicCoset::new(lifting_log_size).circle_domain().half_coset)
}

/// stwo's striped commitment of `polys`, checked to have taken the striped path.
fn cpu_commit(
    polys: &[CircleCoefficients<SimdBackend>],
    log_blowup: u32,
    twiddles: &TwiddleTree<SimdBackend>,
    lifting_log_size: u32,
) -> TreeProver {
    let log_sizes = polys.iter().map(|poly| poly.log_size()).collect_vec();
    assert!(
        takes_striped_path(&log_sizes, log_blowup, lifting_log_size),
        "not the striped path's shapes"
    );
    let pool = BaseColumnPool::<SimdBackend>::new();
    let tree =
        TreeProver::new_striped(polys.to_vec(), log_blowup, twiddles, lifting_log_size, &pool);
    // The striped path keeps stripe 0 only of the columns it does not keep whole; `new` keeps
    // every column whole.
    let mut n_striped = 0;
    for (i, (poly, &log_size)) in tree.polynomials.iter().zip(&log_sizes).enumerate() {
        assert_eq!(poly.evals.domain.log_size(), log_size + log_blowup, "column {i}");
        if kept_whole(log_size, log_blowup) {
            assert_eq!(poly.evals.values.len(), 1 << (log_size + log_blowup), "column {i}");
        } else {
            assert_eq!(poly.evals.values.len(), 1 << log_size, "column {i}: not striped");
            n_striped += 1;
        }
    }
    assert!(n_striped > 0, "no striped column: the striped path is not told from new's");
    tree
}

/// Asserts that the GPU's tree prover is the CPU's: every layer, every column's kept evaluations
/// and their domain, no coefficients.
fn assert_trees_eq(gpu: &TreeProver, cpu: &TreeProver, what: &str) {
    let (gpu_layers, cpu_layers) = (&gpu.commitment.layers, &cpu.commitment.layers);
    assert_eq!(gpu_layers.len(), cpu_layers.len(), "{what}: number of layers");
    for (k, (g, c)) in gpu_layers.iter().zip(cpu_layers).enumerate() {
        assert_eq!(g.len(), c.len(), "{what}: layer {k} (from the root) length");
        if let Some(i) = (0..g.len()).find(|&i| g[i] != c[i]) {
            panic!("{what}: layer {k} (from the root) differs at {i} of {}", g.len());
        }
    }
    assert_eq!(gpu.commitment.root(), cpu.commitment.root(), "{what}: root");
    assert_eq!(gpu.polynomials.len(), cpu.polynomials.len(), "{what}: number of columns");
    for (i, (g, c)) in gpu.polynomials.iter().zip(&cpu.polynomials).enumerate() {
        assert!(g.coeffs.is_none() && c.coeffs.is_none(), "{what}: column {i} coefficients");
        assert_eq!(g.evals.domain, c.evals.domain, "{what}: column {i} domain");
        assert_eq!(g.evals.values.len(), c.evals.values.len(), "{what}: column {i} length");
        let (g, c) = (g.evals.values.as_slice(), c.evals.values.as_slice());
        if let Some(j) = (0..g.len()).find(|&j| g[j] != c[j]) {
            panic!(
                "{what}: column {i} value {j} is {}, stwo's {} ({} of {} differ)",
                g[j],
                c[j],
                (0..g.len()).filter(|&j| g[j] != c[j]).count(),
                g.len()
            );
        }
    }
}

/// The GPU commitment of random polynomials of `log_sizes` against `new_striped`'s.
fn check(
    gpu: &GpuContext,
    kernels: &CommitKernels,
    log_sizes: &[u32],
    log_blowup: u32,
    lifting_log_size: u32,
    max_words: Option<u64>,
    seed: u64,
) {
    let polys = random_polys(log_sizes, seed);
    let twiddles = twiddles(lifting_log_size);
    let cpu = cpu_commit(&polys, log_blowup, &twiddles, lifting_log_size);
    let commitment = StripedCommitment::with_max_buffer_words(
        gpu,
        kernels,
        log_sizes,
        log_blowup,
        lifting_log_size,
        &twiddles,
        max_words.unwrap_or_else(|| max_buffer_words(gpu)),
    );
    commitment.upload(gpu, &polys);
    let output = commitment.run_and_read(gpu, kernels);
    let gpu_tree = commitment.tree_prover(output);
    let what = format!(
        "{} columns, log sizes {:?}, blowup {log_blowup}, lifting {lifting_log_size}",
        log_sizes.len(),
        log_sizes.iter().copied().sorted().dedup().collect_vec()
    );
    assert_trees_eq(&gpu_tree, &cpu, &what);
    let bytes = commitment.gpu_bytes();
    let (ext, hash, top) = commitment.n_dispatches();
    eprintln!(
        "ok: {what}; {} buffer(s), {} batches; {} levels stored from {}; dispatches per stripe: \
         {ext} extension, {hash} hashing; {top} above; GPU {:.1} MiB + staging {:.1} MiB; root {}",
        commitment.layout().buffer_words.len(),
        commitment.layout().batches.len(),
        cpu.commitment.layers.iter().filter(|layer| !layer.is_empty()).count(),
        commitment.tree_layout().min_stored_level,
        bytes.device() as f64 / (1 << 20) as f64,
        bytes.staging as f64 / (1 << 20) as f64,
        cpu.commitment.root(),
    );
}

/// The Cairo base trace's columns, every component's rows divided by `2^scale` (columns of 2^4
/// coefficients at least), at blowup 3 in a tree of lifting log size `23 - scale`: 8 stripes as
/// tall as the largest column.
fn cairo(scale: u32) {
    let gpu = GpuContext::new();
    let kernels = CommitKernels::new(&gpu);
    let log_sizes = base_trace_log_sizes(&cairo_pay_components(), scale, 4);
    let lifting_log_size = CAIRO_LIFTING_LOG_SIZE - scale;
    assert_eq!(log_sizes.iter().max().unwrap() + CAIRO_LOG_BLOWUP, lifting_log_size);
    check(&gpu, &kernels, &log_sizes, CAIRO_LOG_BLOWUP, lifting_log_size, None, 100 + scale as u64);
}

/// Columns up to 2^17: the FFT's input transposed (past `CACHED_FFT_LOG_SIZE`, 16) and two
/// passes, columns of every size kept whole at blowup 3 (2^4 to 2^10), several stripes high.
#[test]
fn cairo_shapes_scaled_blowup_3() {
    cairo(3);
}

/// Smaller: every column at most 2^12, one FFT pass.
#[test]
fn cairo_shapes_scaled_small_blowup_3() {
    cairo(8);
}

#[test]
fn cairo_shapes_full_blowup_3() {
    if !full() {
        eprintln!("skipped: STWO_GPU_SPIKE_FULL=1 runs it");
        return;
    }
    cairo(0);
}

/// A spread of sizes like the circuit leg's trace (whose shapes are not in `data/`), every
/// column count not a multiple of 16, less `scale`, at blowup 2: 4 stripes.
fn circuit(scale: u32) {
    let gpu = GpuContext::new();
    let kernels = CommitKernels::new(&gpu);
    let mut log_sizes = vec![];
    for (log_size, n) in
        [(21, 21), (20, 9), (19, 35), (18, 3), (17, 50), (15, 7), (12, 19), (8, 5), (5, 2)]
    {
        log_sizes.extend(std::iter::repeat_n((log_size - scale as i32).max(4) as u32, n));
    }
    // Interleaved, as components' columns are.
    let mut rng = SmallRng::seed_from_u64(200 + scale as u64);
    for i in (1..log_sizes.len()).rev() {
        log_sizes.swap(i, rng.random_range(0..=i));
    }
    let lifting_log_size = CIRCUIT_LIFTING_LOG_SIZE - scale;
    check(&gpu, &kernels, &log_sizes, CIRCUIT_LOG_BLOWUP, lifting_log_size, None, 300);
}

#[test]
fn circuit_shapes_scaled_blowup_2() {
    circuit(4);
    // The largest columns 2^12: the smallest that blowup 2 stripes.
    circuit(9);
}

#[test]
fn circuit_shapes_full_blowup_2() {
    if !full() {
        eprintln!("skipped: STWO_GPU_SPIKE_FULL=1 runs it");
        return;
    }
    circuit(0);
}

/// A tree taller than its largest column (stwo keeps every level, the leaves too, and the leaves
/// lift every column).
#[test]
fn tree_taller_than_the_columns() {
    let gpu = GpuContext::new();
    let kernels = CommitKernels::new(&gpu);
    let log_sizes = base_trace_log_sizes(&cairo_pay_components(), 9, 4);
    let max = *log_sizes.iter().max().unwrap();
    check(&gpu, &kernels, &log_sizes, 3, max + 3 + 2, None, 400);
    // At blowup 2, columns of 2^12 at least are striped.
    let log_sizes = base_trace_log_sizes(&cairo_pay_components(), 8, 4);
    let max = *log_sizes.iter().max().unwrap();
    check(&gpu, &kernels, &log_sizes, 2, max + 2 + 1, None, 401);
}

/// Within WebGPU's default limits, with buffers of 2^16 words: the columns spread over many
/// buffers, a log size split across buffers, so several batches per size on both sides.
#[test]
fn webgpu_default_limits_and_several_buffers() {
    let gpu = GpuContext::with_limits(wgpu::Limits::default());
    let kernels = CommitKernels::new(&gpu);
    let log_sizes = base_trace_log_sizes(&cairo_pay_components(), 8, 4);
    let lifting_log_size = CAIRO_LIFTING_LOG_SIZE - 8;
    let max_words = 1 << 16;
    let layout = stwo_gpu_spike::extension::ColumnsLayout::new(&log_sizes, max_words);
    let distinct = log_sizes.iter().copied().sorted().dedup().count();
    assert!(layout.buffer_words.len() >= 8);
    assert!(layout.batches.len() > distinct + 4, "some size split across buffers");
    check(&gpu, &kernels, &log_sizes, CAIRO_LOG_BLOWUP, lifting_log_size, Some(max_words), 500);
}

/// The smallest stripes `new_striped` takes (height 10), 16 of them (blowup 4: the smallest
/// blowup at which columns of 2^10 are striped rather than kept whole), with columns of every
/// small size.
#[test]
fn smallest_stripes_blowup_4() {
    let gpu = GpuContext::new();
    let kernels = CommitKernels::new(&gpu);
    let log_sizes = [vec![4; 20], vec![10; 3], vec![7; 5], vec![5; 1], vec![9; 17]].concat();
    check(&gpu, &kernels, &log_sizes, 4, 14, None, 600);
    // And 2 stripes (blowup 1) of height 13, columns of 2^13 striped.
    let log_sizes = [vec![13; 2], vec![4; 3], vec![12; 18], vec![8; 1]].concat();
    check(&gpu, &kernels, &log_sizes, 1, 14, None, 601);
}

/// One commitment's buffers and plans, run on two traces of the same shapes; and the one-shot
/// [`commit_striped`], `new_striped`'s signature.
#[test]
fn reused_for_another_trace_and_one_shot() {
    let gpu = GpuContext::new();
    let kernels = CommitKernels::new(&gpu);
    let log_sizes = base_trace_log_sizes(&cairo_pay_components(), 9, 4);
    let (log_blowup, lifting_log_size) = (CAIRO_LOG_BLOWUP, CAIRO_LIFTING_LOG_SIZE - 9);
    let twiddles = twiddles(lifting_log_size);
    let commitment =
        StripedCommitment::new(&gpu, &kernels, &log_sizes, log_blowup, lifting_log_size, &twiddles);
    for seed in [700, 701] {
        let polys = random_polys(&log_sizes, seed);
        let cpu = cpu_commit(&polys, log_blowup, &twiddles, lifting_log_size);
        commitment.upload(&gpu, &polys);
        let gpu_tree = commitment.tree_prover(commitment.run_and_read(&gpu, &kernels));
        assert_trees_eq(&gpu_tree, &cpu, &format!("reused, trace {seed}"));
    }
    let polys = random_polys(&log_sizes, 702);
    let cpu = cpu_commit(&polys, log_blowup, &twiddles, lifting_log_size);
    let gpu_tree = commit_striped(&gpu, &kernels, polys, log_blowup, &twiddles, lifting_log_size);
    assert_trees_eq(&gpu_tree, &cpu, "commit_striped");
}

#[test]
fn striped_path_preconditions() {
    // new_striped's fallbacks to new.
    assert!(!takes_striped_path(&[], 3, 20));
    assert!(!takes_striped_path(&[12], 0, 20));
    assert!(!takes_striped_path(&[10], 3, 12)); // stripes lower than 10
    assert!(!takes_striped_path(&[18], 3, 20)); // a column taller than the tree
    assert!(!takes_striped_path(&[3, 12], 3, 20)); // a column of fewer than 16 coefficients
    assert!(takes_striped_path(&[4, 10], 3, 13));
    assert!(takes_striped_path(&[4, 17], 3, 20));
}
