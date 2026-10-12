//! The GPU striped commitment against `CommitmentTreeProver::new_striped` on the same random
//! polynomials: every layer of the tree (root included), and every column's kept evaluations
//! (stripe 0, or the whole evaluation of the small columns), with their domains.
//!
//! Every case has `new_striped`'s preconditions for its striped path, and checks that the CPU took
//! it: the columns it does not keep whole hold their stripe 0 only, and there is at least one.
//!
//! The full-size Cairo and circuit trees run with `STWO_GPU_SPIKE_FULL=1` (slow on lavapipe).

use std::sync::Arc;

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
    CommitBuffers, CommitKernels, CommitOptions, StripedCommitment, TwiddleCache, commit_striped,
    kept_whole, takes_striped_path,
};
use stwo_gpu_spike::committer::{Declined, GpuCommitter};
use stwo_gpu_spike::extension::{Merge, StripeTwiddles};
use stwo_gpu_spike::shapes::{
    CAIRO_LIFTING_LOG_SIZE, CAIRO_LOG_BLOWUP, CIRCUIT_LIFTING_LOG_SIZE, CIRCUIT_LOG_BLOWUP,
    base_trace_log_sizes, cairo_pay_components, interaction_trace_log_sizes,
};
use stwo_gpu_spike::stream::StreamedCommitment;
use stwo_gpu_spike::transfer::{ReadbackRing, UploadRing};

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
    options: CommitOptions,
    seed: u64,
) {
    let polys = random_polys(log_sizes, seed);
    let twiddles = twiddles(lifting_log_size);
    let cpu = cpu_commit(&polys, log_blowup, &twiddles, lifting_log_size);
    let commitment = StripedCommitment::with_options(
        gpu,
        kernels,
        log_sizes,
        log_blowup,
        lifting_log_size,
        &twiddles,
        options,
    );
    commitment.upload(gpu, &polys);
    let output = commitment.run_and_read(gpu, kernels);
    let gpu_tree = commitment.tree_prover(output);
    let what = format!(
        "{} columns, log sizes {:?}, blowup {log_blowup}, lifting {lifting_log_size}, {:?}",
        log_sizes.len(),
        log_sizes.iter().copied().sorted().dedup().collect_vec(),
        options.merge,
    );
    assert_trees_eq(&gpu_tree, &cpu, &what);
    let bytes = commitment.gpu_bytes();
    let (ext, hash, top) = commitment.n_dispatches();
    eprintln!(
        "ok: {what}; {} buffer(s), {} batches; {} levels stored from {}; dispatches per stripe: \
         {ext} extension, {hash} hashing; {top} above; GPU {:.1} MiB; root {}",
        commitment.layout().buffer_words.len(),
        commitment.layout().batches.len(),
        cpu.commitment.layers.iter().filter(|layer| !layer.is_empty()).count(),
        commitment.tree_layout().min_stored_level,
        bytes.device() as f64 / (1 << 20) as f64,
        cpu.commitment.root(),
    );
}

/// The Cairo base trace's columns, every component's rows divided by `2^scale` (columns of 2^4
/// coefficients at least), at blowup 3 in a tree of lifting log size `23 - scale`: 8 stripes as
/// tall as the largest column.
fn cairo(scale: u32, merges: &[Merge]) {
    let gpu = GpuContext::new();
    let kernels = CommitKernels::new(&gpu);
    let log_sizes = base_trace_log_sizes(&cairo_pay_components(), scale, 4);
    let lifting_log_size = CAIRO_LIFTING_LOG_SIZE - scale;
    assert_eq!(log_sizes.iter().max().unwrap() + CAIRO_LOG_BLOWUP, lifting_log_size);
    for &merge in merges {
        let options = CommitOptions { merge, ..Default::default() };
        check(
            &gpu,
            &kernels,
            &log_sizes,
            CAIRO_LOG_BLOWUP,
            lifting_log_size,
            options,
            100 + scale as u64,
        );
    }
}

/// Columns up to 2^17: the FFT's input transposed (past `CACHED_FFT_LOG_SIZE`, 16) and two
/// passes, columns of every size kept whole at blowup 3 (2^4 to 2^10), several stripes high.
#[test]
fn cairo_shapes_scaled_blowup_3() {
    cairo(3, &[Merge::Off]);
}

/// The same, with the extension's batches merged into fewer dispatches (see `Merge`).
#[test]
fn cairo_shapes_scaled_blowup_3_merged() {
    cairo(3, &[Merge::Small, Merge::All]);
}

/// Smaller: every column at most 2^12, one FFT pass.
#[test]
fn cairo_shapes_scaled_small_blowup_3() {
    cairo(8, &[Merge::Off, Merge::All]);
}

#[test]
fn cairo_shapes_full_blowup_3() {
    if !full() {
        eprintln!("skipped: STWO_GPU_SPIKE_FULL=1 runs it");
        return;
    }
    cairo(0, &[Merge::Off, Merge::Small, Merge::All]);
}

/// A spread of sizes like the circuit leg's trace (whose shapes are not in `data/`), every
/// column count not a multiple of 16, less `scale`, at blowup 2: 4 stripes.
fn circuit(scale: u32, merge: Merge) {
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
    let options = CommitOptions { merge, ..Default::default() };
    check(&gpu, &kernels, &log_sizes, CIRCUIT_LOG_BLOWUP, lifting_log_size, options, 300);
}

#[test]
fn circuit_shapes_scaled_blowup_2() {
    circuit(4, Merge::Off);
    circuit(4, Merge::All);
    // The largest columns 2^12: the smallest that blowup 2 stripes.
    circuit(9, Merge::Off);
}

#[test]
fn circuit_shapes_full_blowup_2() {
    if !full() {
        eprintln!("skipped: STWO_GPU_SPIKE_FULL=1 runs it");
        return;
    }
    circuit(0, Merge::Off);
    circuit(0, Merge::All);
}

/// A tree taller than its largest column (stwo keeps every level, the leaves too, and the leaves
/// lift every column).
#[test]
fn tree_taller_than_the_columns() {
    let gpu = GpuContext::new();
    let kernels = CommitKernels::new(&gpu);
    let log_sizes = base_trace_log_sizes(&cairo_pay_components(), 9, 4);
    let max = *log_sizes.iter().max().unwrap();
    check(&gpu, &kernels, &log_sizes, 3, max + 3 + 2, CommitOptions::default(), 400);
    // At blowup 2, columns of 2^12 at least are striped.
    let log_sizes = base_trace_log_sizes(&cairo_pay_components(), 8, 4);
    let max = *log_sizes.iter().max().unwrap();
    check(&gpu, &kernels, &log_sizes, 2, max + 2 + 1, CommitOptions::default(), 401);
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
    for merge in [Merge::Off, Merge::Small, Merge::All] {
        let options = CommitOptions { max_buffer_words: Some(max_words), merge };
        check(&gpu, &kernels, &log_sizes, CAIRO_LOG_BLOWUP, lifting_log_size, options, 500);
    }
}

/// The smallest stripes `new_striped` takes (height 10), 16 of them (blowup 4: the smallest
/// blowup at which columns of 2^10 are striped rather than kept whole), with columns of every
/// small size.
#[test]
fn smallest_stripes_blowup_4() {
    let gpu = GpuContext::new();
    let kernels = CommitKernels::new(&gpu);
    let log_sizes = [vec![4; 20], vec![10; 3], vec![7; 5], vec![5; 1], vec![9; 17]].concat();
    check(&gpu, &kernels, &log_sizes, 4, 14, CommitOptions::default(), 600);
    let merged = CommitOptions { merge: Merge::All, ..Default::default() };
    check(&gpu, &kernels, &log_sizes, 4, 14, merged, 600);
    // And 2 stripes (blowup 1) of height 13, columns of 2^13 striped.
    let log_sizes = [vec![13; 2], vec![4; 3], vec![12; 18], vec![8; 1]].concat();
    check(&gpu, &kernels, &log_sizes, 1, 14, CommitOptions::default(), 601);
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

/// The committer the prover's hook calls: `new_striped`'s tree for two traces in a row (it is
/// reused), and the polynomials given back, untouched, for shapes `new_striped` doesn't stripe.
#[test]
fn committer_matches_new_striped_and_declines() {
    let committer = GpuCommitter::new().expect("a GPU adapter");
    let scale = 3;
    let log_sizes = base_trace_log_sizes(&cairo_pay_components(), scale, 4);
    let lifting_log_size = CAIRO_LIFTING_LOG_SIZE - scale;
    let cairo_twiddles = twiddles(lifting_log_size);
    for seed in [11, 12] {
        let polys = random_polys(&log_sizes, seed);
        let cpu = cpu_commit(&polys, CAIRO_LOG_BLOWUP, &cairo_twiddles, lifting_log_size);
        let gpu = committer
            .commit_striped(polys, CAIRO_LOG_BLOWUP, &cairo_twiddles, lifting_log_size)
            .unwrap_or_else(|(_, declined)| panic!("declined: {declined:?}"));
        assert_trees_eq(&gpu, &cpu, &format!("committer, seed {seed}"));
    }

    // A column of 2^3 coefficients: below the striped path's smallest.
    let polys = random_polys(&[3, 12], 13);
    let lifting_log_size = 12 + CAIRO_LOG_BLOWUP;
    let given = polys.iter().map(|poly| poly.coeffs.to_cpu()).collect_vec();
    match committer.commit_striped(
        polys,
        CAIRO_LOG_BLOWUP,
        &twiddles(lifting_log_size),
        lifting_log_size,
    ) {
        Err((back, Declined::NotStriped)) => {
            let back = back.iter().map(|poly| poly.coeffs.to_cpu()).collect_vec();
            assert_eq!(back, given, "the polynomials come back untouched");
        }
        Err((_, declined)) => panic!("declined for another reason: {declined:?}"),
        Ok(_) => panic!("committed a shape new_striped doesn't stripe"),
    }
}

/// One committer for trees of every kind in a row, as a proof's: larger and smaller ones (its
/// buffers grow, then serve smaller trees, stale values and all), other blowups (its twiddles
/// grow, and are kept per blowup), and after its buffers are released. Every tree is
/// `new_striped`'s.
#[test]
fn committer_reuses_across_shapes() {
    let committer = GpuCommitter::new().expect("a GPU adapter");
    let cairo = |scale: u32| {
        (
            base_trace_log_sizes(&cairo_pay_components(), scale, 4),
            CAIRO_LOG_BLOWUP,
            CAIRO_LIFTING_LOG_SIZE - scale,
        )
    };
    let circuit = |scale: u32| {
        let mut log_sizes = vec![];
        for (log_size, n) in [(21, 5), (20, 9), (17, 30), (12, 19), (8, 5), (5, 2)] {
            log_sizes.extend(std::iter::repeat_n((log_size - scale as i32).max(4) as u32, n));
        }
        (log_sizes, CIRCUIT_LOG_BLOWUP, CIRCUIT_LIFTING_LOG_SIZE - scale)
    };
    let smallest =
        ([vec![4; 20], vec![10; 3], vec![7; 5], vec![5; 1], vec![9; 17]].concat(), 4, 14);
    let sequence = [cairo(6), cairo(3), circuit(4), cairo(5), smallest, circuit(5), cairo(3)];
    for (k, (log_sizes, log_blowup, lifting_log_size)) in sequence.into_iter().enumerate() {
        if k == 5 {
            committer.release_buffers();
            assert_eq!(committer.held_bytes().buffers, 0);
        }
        let polys = random_polys(&log_sizes, 800 + k as u64);
        let twiddles = twiddles(lifting_log_size);
        let cpu = cpu_commit(&polys, log_blowup, &twiddles, lifting_log_size);
        let gpu = committer
            .commit_striped(polys, log_blowup, &twiddles, lifting_log_size)
            .unwrap_or_else(|(_, declined)| panic!("declined: {declined:?}"));
        assert_trees_eq(&gpu, &cpu, &format!("tree {k} of the sequence"));
        let held = committer.held_bytes();
        let times = committer.last_times().unwrap();
        eprintln!(
            "ok: tree {k}: {} columns, blowup {log_blowup}; held: buffers {:.1} MiB, twiddles \
             {:.1} MiB; {:?}",
            log_sizes.len(),
            held.buffers as f64 / (1 << 20) as f64,
            held.twiddles as f64 / (1 << 20) as f64,
            times.total,
        );
    }
}

/// What a browser's GPU worker keeps between commitments: the twiddles got lazily (the tree only
/// on a miss) and dropped when replaced, the buffers' generation (plans built on them stay valid
/// while it holds), commitments' plans freed while the buffers are kept and reused.
#[test]
fn kept_state_for_a_browser_worker() {
    let gpu = GpuContext::new();
    let kernels = CommitKernels::new(&gpu);
    let (log_blowup, lifting_log_size) = (CAIRO_LOG_BLOWUP, 16);
    let tree = twiddles(lifting_log_size);
    let mut cache = TwiddleCache::new();
    let mut calls = 0;
    let mut get = |cache: &mut TwiddleCache, sizes: &[u32]| {
        let sizes = sizes.iter().copied().collect();
        cache
            .try_get_with(&gpu, &sizes, log_blowup, tree.root_coset, || {
                calls += 1;
                Ok::<_, ()>(TwiddleTree {
                    root_coset: tree.root_coset,
                    twiddles: tree.twiddles.clone(),
                    itwiddles: vec![],
                })
            })
            .unwrap()
    };
    let small = get(&mut cache, &[10, 12]);
    assert!(cache.holds(&small));
    let again = get(&mut cache, &[11]);
    assert!(Arc::ptr_eq(&small, &again), "covered: the same table");
    let larger = get(&mut cache, &[10, 13]);
    assert!(!Arc::ptr_eq(&small, &larger) && cache.holds(&larger) && !cache.holds(&small));
    assert_eq!(calls, 2, "the tree is got on a miss only");
    drop((small, again));

    let mut buffers = CommitBuffers::new();
    let generation = buffers.generation();
    let mut generations = vec![];
    let shapes: [&[u32]; 3] =
        [&[13, 13, 12, 10, 8, 6, 4], &[13, 13, 13, 13, 12, 12, 11, 9, 5], &[13, 12, 10, 8, 6, 4]];
    for (k, log_sizes) in shapes.into_iter().enumerate() {
        let polys = random_polys(log_sizes, 900 + k as u64);
        let cpu = cpu_commit(&polys, log_blowup, &tree, lifting_log_size);
        let table = cache.get(&gpu, &log_sizes.iter().copied().collect(), log_blowup, &tree);
        let commitment = StreamedCommitment::with_buffers(
            &gpu,
            &kernels,
            &mut buffers,
            table,
            log_sizes,
            log_blowup,
            lifting_log_size,
            1 << 14,
        );
        let mut upload = UploadRing::new(&gpu, 1 << 16, 2);
        let mut readback = ReadbackRing::new(&gpu, 1 << 16, 2);
        let output = commitment.commit(&gpu, &kernels, &mut upload, &mut readback, &polys);
        commitment.destroy_plans();
        assert_trees_eq(&commitment.tree_prover(output), &cpu, &format!("tree {k}"));
        generations.push(buffers.generation());
    }
    // The first tree's buffers are new (none replaced), the second tree's are larger (some
    // replaced), the third fits in them.
    assert_eq!(generations[0], generation);
    assert!(generations[1] > generations[0]);
    assert_eq!(generations[2], generations[1]);
    let before = buffers.generation();
    buffers.destroy();
    assert!(buffers.generation() > before && buffers.bytes() == 0);
    cache.destroy();
    assert!(cache.bytes() == 0 && !cache.holds(&larger));
}

/// The streamed commitment (groups of columns, every stripe of a group at once) of random
/// polynomials of `log_sizes` against `new_striped`'s, in groups of `group_words` words (several
/// sizes: groups of many sizes, sizes straddling groups, a group per column).
fn check_streamed(
    gpu: &GpuContext,
    kernels: &CommitKernels,
    log_sizes: &[u32],
    log_blowup: u32,
    lifting_log_size: u32,
    group_words: &[u64],
    seed: u64,
) {
    let polys = random_polys(log_sizes, seed);
    let twiddles = twiddles(lifting_log_size);
    let cpu = cpu_commit(&polys, log_blowup, &twiddles, lifting_log_size);
    let distinct = log_sizes.iter().copied().collect();
    let table = Arc::new(StripeTwiddles::new(gpu, &distinct, log_blowup, &twiddles));
    let mut buffers = CommitBuffers::new();
    // Small chunks, so that the rings go round many times.
    let mut upload = UploadRing::new(gpu, 1 << 16, 2);
    let mut readback = ReadbackRing::new(gpu, 1 << 18, 3);
    // The largest group the device binds every stripe's slot of.
    let max_words = stwo_gpu_spike::extension::max_buffer_words(gpu) >> log_blowup;
    // The largest columns that go in groups (the largest size goes stripe by stripe if it is the
    // stripes' height).
    let max = *log_sizes.iter().max().unwrap();
    let grouped = log_sizes
        .iter()
        .copied()
        .filter(|&l| l != max || max + log_blowup != lifting_log_size)
        .max()
        .unwrap_or(4);
    for &words in group_words {
        let words = words.max(1 << grouped).min(max_words / 64 * 64);
        let mut upload_big;
        let mut readback_big;
        let (upload, readback) = if words * 4 > readback.chunk_bytes() {
            upload_big = UploadRing::new(gpu, words * 4, 2);
            readback_big = ReadbackRing::new(gpu, words * 4, 3);
            (&mut upload_big, &mut readback_big)
        } else {
            (&mut upload, &mut readback)
        };
        let commitment = StreamedCommitment::with_buffers(
            gpu,
            kernels,
            &mut buffers,
            table.clone(),
            log_sizes,
            log_blowup,
            lifting_log_size,
            words,
        );
        let output = commitment.commit(gpu, kernels, upload, readback, &polys);
        let gpu_tree = commitment.tree_prover(output);
        let what = format!(
            "streamed: {} columns, log sizes {:?}, blowup {log_blowup}, lifting \
             {lifting_log_size}, groups of {words} words",
            log_sizes.len(),
            log_sizes.iter().copied().sorted().dedup().collect_vec(),
        );
        assert_trees_eq(&gpu_tree, &cpu, &what);
        eprintln!("ok: {what}: {} groups", commitment.n_groups());
    }
}

#[test]
fn streamed_cairo_shapes_scaled() {
    let gpu = GpuContext::new();
    let kernels = CommitKernels::new(&gpu);
    for scale in [3, 8] {
        let log_sizes = base_trace_log_sizes(&cairo_pay_components(), scale, 4);
        let lifting_log_size = CAIRO_LIFTING_LOG_SIZE - scale;
        check_streamed(
            &gpu,
            &kernels,
            &log_sizes,
            CAIRO_LOG_BLOWUP,
            lifting_log_size,
            &[1 << 10, 1 << 14, 1 << 18, 1 << 30],
            900 + scale as u64,
        );
    }
}

#[test]
fn streamed_circuit_and_small_shapes() {
    let gpu = GpuContext::new();
    let kernels = CommitKernels::new(&gpu);
    let mut log_sizes = vec![];
    for (log_size, n) in [(17, 21), (16, 9), (15, 35), (14, 3), (13, 50), (11, 7), (8, 19), (4, 5)]
    {
        log_sizes.extend(std::iter::repeat_n(log_size, n));
    }
    let mut rng = SmallRng::seed_from_u64(910);
    for i in (1..log_sizes.len()).rev() {
        log_sizes.swap(i, rng.random_range(0..=i));
    }
    check_streamed(&gpu, &kernels, &log_sizes, 2, 19, &[1 << 12, 1 << 17, 1 << 20], 911);
    // Taller than its columns; the smallest stripes, blowup 4; blowup 1.
    let cairo = base_trace_log_sizes(&cairo_pay_components(), 9, 4);
    let max = *cairo.iter().max().unwrap();
    check_streamed(&gpu, &kernels, &cairo, 3, max + 3 + 2, &[1 << 12, 1 << 16], 912);
    let small = [vec![4; 20], vec![10; 3], vec![7; 5], vec![5; 1], vec![9; 17]].concat();
    check_streamed(&gpu, &kernels, &small, 4, 14, &[1 << 10, 1 << 13], 913);
    let two = [vec![13; 2], vec![4; 3], vec![12; 18], vec![8; 1]].concat();
    check_streamed(&gpu, &kernels, &two, 1, 14, &[1 << 13, 1 << 16], 914);
}

#[test]
fn streamed_full_shapes() {
    if !full() {
        eprintln!("skipped: STWO_GPU_SPIKE_FULL=1 runs it");
        return;
    }
    let gpu = GpuContext::new();
    let kernels = CommitKernels::new(&gpu);
    let log_sizes = base_trace_log_sizes(&cairo_pay_components(), 0, 4);
    check_streamed(
        &gpu,
        &kernels,
        &log_sizes,
        CAIRO_LOG_BLOWUP,
        CAIRO_LIFTING_LOG_SIZE,
        &[1 << 23, 1 << 21],
        920,
    );
    let interaction = interaction_trace_log_sizes(&cairo_pay_components(), 0, 4);
    check_streamed(
        &gpu,
        &kernels,
        &interaction,
        CAIRO_LOG_BLOWUP,
        CAIRO_LIFTING_LOG_SIZE,
        &[1 << 23],
        921,
    );
}

/// The committer's keep-warm thread: dispatching for its window after the committer's creation
/// and after each commitment (the commitments still `new_striped`'s), stopping when its window
/// passes or is unset.
#[test]
fn committer_keeps_warm_after_commitments() {
    use std::time::{Duration, Instant};
    let committer = GpuCommitter::new().expect("a GPU adapter");
    let wait_for_more = |than: u64| {
        let start = Instant::now();
        while committer.keep_warm_dispatches() <= than && start.elapsed() < Duration::from_secs(5) {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(committer.keep_warm_dispatches() > than, "no dispatch");
    };
    // Warm from the committer's creation (the default window, unless the environment sets one).
    if std::env::var_os("STWO_GPU_KEEP_WARM_MS").is_none() {
        assert!(committer.keep_warm().is_some());
        wait_for_more(0);
    }
    committer.set_keep_warm(Some(Duration::from_millis(300)));
    assert_eq!(committer.keep_warm(), Some(Duration::from_millis(300)));

    let scale = 6;
    let log_sizes = base_trace_log_sizes(&cairo_pay_components(), scale, 4);
    let lifting_log_size = CAIRO_LIFTING_LOG_SIZE - scale;
    let twiddles = twiddles(lifting_log_size);
    let commit = |seed: u64| {
        let polys = random_polys(&log_sizes, seed);
        let cpu = cpu_commit(&polys, CAIRO_LOG_BLOWUP, &twiddles, lifting_log_size);
        let gpu = committer
            .commit_striped(polys, CAIRO_LOG_BLOWUP, &twiddles, lifting_log_size)
            .unwrap_or_else(|(_, declined)| panic!("declined: {declined:?}"));
        assert_trees_eq(&gpu, &cpu, &format!("keep-warm, seed {seed}"));
    };
    // Unset: no more dispatches (the last one in flight done well within 2 s).
    committer.set_keep_warm(None);
    std::thread::sleep(Duration::from_millis(2000));
    let unset = committer.keep_warm_dispatches();
    std::thread::sleep(Duration::from_millis(200));
    assert_eq!(committer.keep_warm_dispatches(), unset, "dispatches while unset");
    // Dispatches after a commitment (one in flight at most, so a few even on lavapipe).
    committer.set_keep_warm(Some(Duration::from_millis(300)));
    commit(21);
    wait_for_more(unset);
    // Commitments while warm.
    commit(22);
    // The window passes: no more dispatches (the last one in flight done well within 2 s).
    std::thread::sleep(Duration::from_millis(300 + 2000));
    let after_window = committer.keep_warm_dispatches();
    std::thread::sleep(Duration::from_millis(200));
    assert_eq!(committer.keep_warm_dispatches(), after_window, "dispatches after the window");
    // Unset: none after a commitment either.
    committer.set_keep_warm(None);
    commit(23);
    std::thread::sleep(Duration::from_millis(200));
    assert_eq!(committer.keep_warm_dispatches(), after_window, "dispatches while unset");
    // A warm-up without a commitment, once set again.
    committer.set_keep_warm(Some(Duration::from_millis(200)));
    committer.warm_up();
    wait_for_more(after_window);
}
