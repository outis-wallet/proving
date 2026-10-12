//! The GPU Merkle tree against stwo's SIMD backend, bit for bit: the node levels against
//! `build_next_layer`, a stripe's subtree against `build_layers_sparse` (every level it keeps,
//! and the same levels left out), a whole tree against `new_striped`'s assembly, and the commitment
//! of random polynomials against `CommitmentTreeProver::new_striped` itself.
//!
//! The full-size Cairo and circuit trees run with `STWO_GPU_SPIKE_FULL=1` (slow on lavapipe).
use itertools::Itertools;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::vcs::blake2_hash::Blake2sHash;
use stwo::core::vcs_lifted::blake2_merkle::{Blake2sM31MerkleChannel, Blake2sMerkleHasher};
use stwo::prover::CommitmentTreeProver;
use stwo::prover::backend::Column;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::simd::column::BaseColumn;
use stwo::prover::mempool::BaseColumnPool;
use stwo::prover::poly::circle::{CircleCoefficients, PolyOps};
use stwo::prover::vcs_lifted::ops::MerkleOpsLifted;
use stwo_gpu_spike::GpuContext;
use stwo_gpu_spike::merkle::{ColumnBatch, GpuMerkle, TreeLayout, stwo_min_stored_level};
use stwo_gpu_spike::reference::{
    UploadedStripe, random_columns, random_values, simd_columns, simd_stripe_layers,
    simd_tree_layers,
};
use stwo_gpu_spike::shapes::{
    CAIRO_LIFTING_LOG_SIZE, CAIRO_LOG_BLOWUP, CIRCUIT_LIFTING_LOG_SIZE, CIRCUIT_LOG_BLOWUP,
    base_trace_log_sizes, cairo_pay_components, tree_order,
};

fn full() -> bool {
    std::env::var("STWO_GPU_SPIKE_FULL").is_ok_and(|v| v == "1")
}

/// Asserts that the GPU's levels are stwo's: equal where stwo keeps a level, and left out where
/// stwo leaves it empty.
fn assert_levels_eq(gpu_levels: &[Vec<Blake2sHash>], stwo_levels: &[Vec<Blake2sHash>], what: &str) {
    assert_eq!(gpu_levels.len(), stwo_levels.len(), "{what}: number of levels");
    for (level, (gpu, stwo)) in gpu_levels.iter().zip(stwo_levels).enumerate() {
        assert_eq!(gpu.len(), stwo.len(), "{what}: level {level} length");
        if let Some(i) = (0..gpu.len()).find(|&i| gpu[i] != stwo[i]) {
            panic!(
                "{what}: level {level} differs at {i} of {}: {} vs {}",
                gpu.len(),
                gpu[i],
                stwo[i]
            );
        }
    }
}

/// A tree of `2^log_n_stripes` stripes of height `height`, every stripe of random columns of
/// `log_sizes` (in any order), hashed on the GPU and by stwo; the levels of the whole tree are
/// compared, and, if `check_stripes`, every stripe's subtree with `build_layers_sparse`'s.
fn check_tree(
    gpu: &GpuContext,
    merkle: &GpuMerkle,
    log_sizes: &[u32],
    height: u32,
    log_n_stripes: u32,
    seed: u64,
) {
    let sorted_sizes = tree_order(log_sizes).into_iter().map(|i| log_sizes[i]).collect_vec();
    let min_stored = stwo_min_stored_level(&sorted_sizes, height);
    let layout = TreeLayout::new(height, log_n_stripes, min_stored);
    let tree = merkle.tree_buffer(gpu, &layout);
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut subtrees = vec![];
    let mut states = None;
    for stripe in 0..1 << log_n_stripes {
        let columns = random_columns(&mut rng, &sorted_sizes);
        let uploaded = UploadedStripe::new(gpu, &columns);
        let states =
            states.get_or_insert_with(|| merkle.state_buffers(gpu, &uploaded.shapes, height));
        let plan = merkle.plan_stripe(gpu, &uploaded.batches(), states, &tree, &layout, stripe);
        gpu.run_plans(&[&plan]);
        subtrees.push(simd_stripe_layers(&simd_columns(&columns), height));
    }
    gpu.run_plans(&[&merkle.plan_top(gpu, &tree, &layout)]);
    let gpu_levels = merkle.read_levels(gpu, &tree, &layout);
    let what = format!(
        "tree of 2^{log_n_stripes} stripes of height {height}, {} columns",
        log_sizes.len()
    );
    // The stripes' subtrees, level by level, are slices of the tree's levels.
    let stripe_count = 1usize << log_n_stripes;
    for (stripe, subtree) in subtrees.iter().enumerate() {
        let gpu_subtree = (0..=height as usize)
            .map(|level| {
                let len = gpu_levels[level].len() / stripe_count;
                gpu_levels[level][stripe * len..(stripe + 1) * len].to_vec()
            })
            .collect_vec();
        assert_levels_eq(&gpu_subtree, subtree, &format!("{what}, stripe {stripe}"));
    }
    assert_levels_eq(&gpu_levels, &simd_tree_layers(&subtrees, height), &what);
}

#[test]
fn nodes_match_build_next_layer() {
    let gpu = GpuContext::new();
    let merkle = GpuMerkle::new(&gpu);
    let mut rng = SmallRng::seed_from_u64(7);
    // From one level above a single pair to 2^13 hashes: several levels per dispatch, levels
    // smaller than a workgroup, and two dispatches (9 levels, then the rest).
    for log_len in [1, 2, 5, 8, 9, 10, 13] {
        let layout = TreeLayout::new(0, log_len, 0);
        let tree = merkle.tree_buffer(&gpu, &layout);
        let words = random_values(&mut rng, 8 << log_len);
        gpu.queue.write_buffer(&tree, 0, bytemuck::cast_slice(&words));
        gpu.run_plans(&[&merkle.plan_top(&gpu, &tree, &layout)]);
        let gpu_levels = merkle.read_levels(&gpu, &tree, &layout);
        let mut stwo_levels: Vec<Vec<Blake2sHash>> = vec![
            words
                .chunks_exact(8)
                .map(|h| Blake2sHash(bytemuck::cast_slice::<u32, u8>(h).try_into().unwrap()))
                .collect(),
        ];
        for _ in 0..log_len {
            let next = <SimdBackend as MerkleOpsLifted<Blake2sMerkleHasher>>::build_next_layer(
                stwo_levels.last().unwrap(),
            );
            stwo_levels.push(next);
        }
        assert_levels_eq(&gpu_levels, &stwo_levels, &format!("2^{log_len} hashes"));
    }
}

#[test]
fn single_column() {
    let gpu = GpuContext::new();
    let merkle = GpuMerkle::new(&gpu);
    // As tall as the tree (every level kept below 9, sparse from 9), and lifted.
    for height in [1, 2, 4, 5, 8, 9, 10, 12] {
        check_tree(&gpu, &merkle, &[height], height, 0, height as u64);
    }
    check_tree(&gpu, &merkle, &[3], 10, 0, 1);
    check_tree(&gpu, &merkle, &[1], 6, 0, 2);
    check_tree(&gpu, &merkle, &[4], 11, 1, 3);
}

#[test]
fn columns_of_one_size() {
    let gpu = GpuContext::new();
    let merkle = GpuMerkle::new(&gpu);
    // Partial, exact and multiple blocks.
    for (n, seed) in [(2, 1), (15, 2), (16, 3), (17, 4), (32, 5), (47, 6), (100, 7)] {
        check_tree(&gpu, &merkle, &vec![10; n], 10, 0, seed);
    }
    check_tree(&gpu, &merkle, &[12; 33], 12, 2, 8);
    // Lifted: the tree taller than its columns.
    check_tree(&gpu, &merkle, &[9; 20], 11, 0, 9);
}

#[test]
fn mixed_sizes() {
    let gpu = GpuContext::new();
    let merkle = GpuMerkle::new(&gpu);
    let mut sizes = vec![];
    for (log_size, n) in [(4, 3), (5, 1), (6, 2), (7, 17), (9, 31), (10, 5), (11, 1)] {
        sizes.extend(std::iter::repeat_n(log_size, n));
    }
    // Shuffled: the order of the tree is the columns' sizes'.
    let mut rng = SmallRng::seed_from_u64(10);
    for i in (1..sizes.len()).rev() {
        sizes.swap(i, rng.random_range(0..=i));
    }
    check_tree(&gpu, &merkle, &sizes, 11, 0, 11);
    check_tree(&gpu, &merkle, &sizes, 11, 3, 12);
    // Lifted past the largest column (stwo keeps every level).
    check_tree(&gpu, &merkle, &sizes, 13, 1, 13);
    // A whole number of blocks in all: 16 + 16 columns across sizes.
    let sizes = [vec![4; 5], vec![6; 11], vec![8; 7], vec![9; 9]].concat();
    check_tree(&gpu, &merkle, &sizes, 9, 1, 14);
    // Every size from 1 to 12, one column each.
    check_tree(&gpu, &merkle, &(1..=12).collect_vec(), 12, 0, 15);
}

/// The kernels without wgpu's runtime checks compute the same trees.
#[test]
fn unchecked_kernels() {
    let gpu = GpuContext::new();
    // Safety: the planner sizes and checks every buffer it binds.
    let merkle = unsafe { GpuMerkle::new_unchecked(&gpu) };
    let sizes = [vec![4; 9], vec![7; 30], vec![9; 3], vec![11; 18]].concat();
    check_tree(&gpu, &merkle, &sizes, 11, 2, 16);
    check_tree(&gpu, &merkle, &sizes, 12, 0, 17);
}

/// The smallest shapes the striped commitment takes: stripes of height 10 (`MIN_STRIPE_HEIGHT`),
/// columns of 2^4 coefficients (`MIN_COEFFS_LOG_SIZE`).
#[test]
fn smallest_striped_shapes() {
    let gpu = GpuContext::new();
    let merkle = GpuMerkle::new(&gpu);
    check_tree(&gpu, &merkle, &[vec![4; 20], vec![10; 3]].concat(), 10, 3, 20);
    check_tree(&gpu, &merkle, &[vec![4; 16], vec![5; 1], vec![10; 16]].concat(), 10, 2, 21);
}

/// A batch split into several, in one shared buffer at unaligned offsets, hashes as the batch.
#[test]
fn split_batches_in_one_buffer() {
    let gpu = GpuContext::new();
    let merkle = GpuMerkle::new(&gpu);
    let mut rng = SmallRng::seed_from_u64(30);
    let log_sizes = [vec![5; 7], vec![8; 150], vec![10; 40]].concat();
    let columns = random_columns(&mut rng, &log_sizes);
    let height = 10;
    let layout = TreeLayout::new(height, 0, stwo_min_stored_level(&log_sizes, height));

    // Reference: one buffer per size.
    let uploaded = UploadedStripe::new(&gpu, &columns);
    let states = merkle.state_buffers(&gpu, &uploaded.shapes, height);
    let tree = merkle.tree_buffer(&gpu, &layout);
    gpu.run_plans(&[&merkle.plan_stripe(&gpu, &uploaded.batches(), &states, &tree, &layout, 0)]);
    let expected = merkle.read_levels(&gpu, &tree, &layout);
    assert_levels_eq(
        &expected,
        &simd_stripe_layers(&simd_columns(&columns), height),
        "one buffer per size",
    );

    // Every column after 3 words of padding, all in one buffer; batches of 1 to 9 columns.
    let mut words = vec![0u32; 3];
    let mut offsets = vec![];
    for column in &columns {
        offsets.push(words.len() as u64);
        words.extend(column);
    }
    let arena = gpu.upload("arena", &words);
    let mut batches = vec![];
    let mut i = 0;
    while i < columns.len() {
        let log_size = log_sizes[i];
        let n = rng
            .random_range(1..=9)
            .min(log_sizes[i..].iter().take_while(|&&s| s == log_size).count());
        batches.push(ColumnBatch {
            buffer: &arena,
            word_offset: offsets[i],
            log_size,
            n_columns: n as u32,
        });
        i += n;
    }
    let shapes = batches.iter().map(|b| (b.log_size, b.n_columns)).collect_vec();
    let states = merkle.state_buffers(&gpu, &shapes, height);
    let tree = merkle.tree_buffer(&gpu, &layout);
    gpu.run_plans(&[&merkle.plan_stripe(&gpu, &batches, &states, &tree, &layout, 0)]);
    assert_levels_eq(&merkle.read_levels(&gpu, &tree, &layout), &expected, "split batches");

    // One batch per size, split by the planner into dispatches of at most 128 KiB (31 columns of
    // 2^10, 127 of 2^8).
    let small = GpuMerkle::new(&gpu).with_binding_limit(128 << 10);
    assert_eq!((small.max_columns_per_dispatch(8), small.max_columns_per_dispatch(10)), (127, 31));
    let states = small.state_buffers(&gpu, &uploaded.shapes, height);
    let tree = small.tree_buffer(&gpu, &layout);
    gpu.run_plans(&[&small.plan_stripe(&gpu, &uploaded.batches(), &states, &tree, &layout, 0)]);
    assert_levels_eq(
        &small.read_levels(&gpu, &tree, &layout),
        &expected,
        "batches split at the binding limit",
    );

    // Both: one batch per size at unaligned offsets in the arena, split at the binding limit.
    let mut whole = vec![];
    let mut i = 0;
    while i < columns.len() {
        let n = log_sizes[i..].iter().take_while(|&&s| s == log_sizes[i]).count();
        whole.push(ColumnBatch {
            buffer: &arena,
            word_offset: offsets[i],
            log_size: log_sizes[i],
            n_columns: n as u32,
        });
        i += n;
    }
    let shapes = whole.iter().map(|b| (b.log_size, b.n_columns)).collect_vec();
    let states = small.state_buffers(&gpu, &shapes, height);
    let tree = small.tree_buffer(&gpu, &layout);
    gpu.run_plans(&[&small.plan_stripe(&gpu, &whole, &states, &tree, &layout, 0)]);
    assert_levels_eq(
        &small.read_levels(&gpu, &tree, &layout),
        &expected,
        "unaligned batches split at the binding limit",
    );
}

/// The Cairo base trace's shapes, every column's log size less `scale` (at least 4, the smallest
/// striped column), in a tree of 8 stripes as tall as the largest column.
fn cairo_tree(scale: u32, log_n_stripes: u32) {
    let gpu = GpuContext::new();
    let merkle = GpuMerkle::new(&gpu);
    let log_sizes = base_trace_log_sizes(&cairo_pay_components(), scale, 4);
    let height = CAIRO_LIFTING_LOG_SIZE - CAIRO_LOG_BLOWUP - scale;
    assert_eq!(*log_sizes.iter().max().unwrap(), height);
    check_tree(&gpu, &merkle, &log_sizes, height, log_n_stripes, 40 + scale as u64);
}

#[test]
fn cairo_shapes_scaled_one_stripe() {
    cairo_tree(8, 0);
}

#[test]
fn cairo_shapes_scaled_tree() {
    cairo_tree(9, CAIRO_LOG_BLOWUP);
}

#[test]
fn cairo_shapes_full_one_stripe() {
    if !full() {
        eprintln!("skipped: STWO_GPU_SPIKE_FULL=1 runs it");
        return;
    }
    cairo_tree(0, 0);
}

#[test]
fn cairo_shapes_full_tree() {
    if !full() {
        eprintln!("skipped: STWO_GPU_SPIKE_FULL=1 runs it");
        return;
    }
    cairo_tree(0, CAIRO_LOG_BLOWUP);
}

/// The circuit leg's tree: 4 stripes (blowup 2) of height 21. Its trace shapes are not in
/// `data/`; these are a spread of sizes up to the trace's, every column count not a multiple of 16.
fn circuit_tree(scale: u32) {
    let gpu = GpuContext::new();
    let merkle = GpuMerkle::new(&gpu);
    let height = CIRCUIT_LIFTING_LOG_SIZE - CIRCUIT_LOG_BLOWUP - scale;
    let mut log_sizes = vec![];
    for (log_size, n) in
        [(21, 21), (20, 9), (19, 35), (18, 3), (17, 50), (15, 7), (12, 19), (8, 5), (5, 2)]
    {
        log_sizes.extend(std::iter::repeat_n((log_size - scale as i32).max(4) as u32, n));
    }
    check_tree(&gpu, &merkle, &log_sizes, height, CIRCUIT_LOG_BLOWUP, 50 + scale as u64);
}

#[test]
fn circuit_shapes_scaled_tree() {
    circuit_tree(10);
}

#[test]
fn circuit_shapes_full_tree() {
    if !full() {
        eprintln!("skipped: STWO_GPU_SPIKE_FULL=1 runs it");
        return;
    }
    circuit_tree(0);
}

/// The commitment of random polynomials: the GPU's tree, from their evaluations split into the
/// stripes (contiguous blocks of the bit-reversed evaluation), against the tree of
/// `CommitmentTreeProver::new_striped` (and the root of `CommitmentTreeProver::new`).
#[test]
fn new_striped_commitment() {
    let gpu = GpuContext::new();
    let merkle = GpuMerkle::new(&gpu);
    let log_blowup = 3;
    let mut coeffs_log_sizes = vec![];
    for (log_size, n) in [(4, 18), (6, 2), (7, 5), (8, 21), (10, 13), (11, 20)] {
        coeffs_log_sizes.extend(std::iter::repeat_n(log_size, n));
    }
    // Interleaved, as components' columns are.
    let mut rng = SmallRng::seed_from_u64(60);
    for i in (1..coeffs_log_sizes.len()).rev() {
        coeffs_log_sizes.swap(i, rng.random_range(0..=i));
    }
    let max_log_size = *coeffs_log_sizes.iter().max().unwrap();
    let lifting_log_size = max_log_size + log_blowup;
    // `new_striped` takes the striped path for these (otherwise it falls back to `new`).
    assert!(lifting_log_size >= log_blowup + 10);
    assert!(coeffs_log_sizes.iter().all(|&log_size| log_size >= 4));

    let polys = coeffs_log_sizes
        .iter()
        .map(|&log_size| {
            CircleCoefficients::<SimdBackend>::new(
                simd_columns(&[random_values(&mut rng, 1 << log_size)]).remove(0),
            )
        })
        .collect_vec();
    let twiddles = SimdBackend::precompute_twiddles(
        CanonicCoset::new(lifting_log_size).circle_domain().half_coset,
    );
    let pool = BaseColumnPool::<SimdBackend>::new();
    let striped = CommitmentTreeProver::<SimdBackend, Blake2sM31MerkleChannel>::new_striped(
        polys.clone(),
        log_blowup,
        &twiddles,
        lifting_log_size,
        &pool,
    );
    let plain = CommitmentTreeProver::<SimdBackend, Blake2sM31MerkleChannel>::new(
        polys.clone(),
        log_blowup,
        &twiddles,
        false,
        lifting_log_size,
        &pool,
    );
    assert_eq!(striped.commitment.root(), plain.commitment.root());

    // The full evaluations, bit-reversed; stripe s of a column is its s-th block.
    let evaluations: Vec<Vec<u32>> = polys
        .iter()
        .map(|poly| {
            let domain = CanonicCoset::new(poly.log_size() + log_blowup).circle_domain();
            let values: BaseColumn = poly.evaluate(domain).values;
            values.to_cpu().iter().map(|v| v.0).collect()
        })
        .collect();
    let order = tree_order(&coeffs_log_sizes);
    let stripe_height = lifting_log_size - log_blowup;
    let sorted_sizes = order.iter().map(|&i| coeffs_log_sizes[i]).collect_vec();
    let layout = TreeLayout::new(
        stripe_height,
        log_blowup,
        stwo_min_stored_level(&sorted_sizes, stripe_height),
    );
    let tree = merkle.tree_buffer(&gpu, &layout);
    let mut states = None;
    for stripe in 0..1usize << log_blowup {
        let columns = order
            .iter()
            .map(|&i| {
                let len = 1 << coeffs_log_sizes[i];
                evaluations[i][stripe * len..(stripe + 1) * len].to_vec()
            })
            .collect_vec();
        let uploaded = UploadedStripe::new(&gpu, &columns);
        let states = states
            .get_or_insert_with(|| merkle.state_buffers(&gpu, &uploaded.shapes, stripe_height));
        let plan =
            merkle.plan_stripe(&gpu, &uploaded.batches(), states, &tree, &layout, stripe as u32);
        gpu.run_plans(&[&plan]);
    }
    gpu.run_plans(&[&merkle.plan_top(&gpu, &tree, &layout)]);
    let gpu_levels = merkle.read_levels(&gpu, &tree, &layout);
    // stwo's layers are root first.
    let stwo_levels = striped.commitment.layers.iter().rev().cloned().collect_vec();
    assert_levels_eq(&gpu_levels, &stwo_levels, "new_striped");
    assert_eq!(gpu_levels.last().unwrap()[0], striped.commitment.root());
    eprintln!("root {}", striped.commitment.root());
}
