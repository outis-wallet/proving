//! stwo's lifted Blake2s Merkle tree on the GPU: the leaves of a stripe's columns and the layers
//! above them, equal word for word to `SimdBackend`'s `build_layers_sparse` and
//! `build_next_layer` (with `Blake2sMerkleHasher`, i.e. `Blake2sHasherGeneric<false>`, the hasher
//! of `Blake2sM31MerkleChannel`).
//!
//! # What is hashed
//!
//! A leaf of a lifted tree of height `H` hashes, with one Blake2s-256 over the whole row, the
//! values of all the columns at that row, in the tree's column order (increasing size; columns of
//! one size in polynomial order), each value as its 4 little-endian bytes. A column of log size
//! `k < H` is lifted: leaf `i` reads its value at `((i >> (H - k + 1)) << 1) | (i & 1)`. A node
//! hashes its two children's 32 bytes. Blake2s keeps the last block of a message, full or not,
//! for the finalizing compression, so the counter and the flags of every block are known from the
//! number of columns alone.
//!
//! # Input layout: column batches
//!
//! The columns of a stripe come as [`ColumnBatch`]es: runs of columns of one log size, each
//! column a contiguous run of `2^log_size` `u32` words (canonical M31 values, in bit-reversed
//! order, as stwo holds evaluations), the columns of a batch one after the other:
//!
//! ```text
//! batch.buffer, words: [ word_offset .. + 2^k ) column 0 of the batch
//!                      [ + 2^k       .. + 2·2^k ) column 1
//!                      ...                       column n_columns - 1
//! ```
//!
//! The batches are given in the tree's column order: by nondecreasing log size, and within a log
//! size in polynomial order (`new_striped`'s `order`: a stable sort of the polynomials by size).
//! A log size may be split into any number of batches, at any column boundary, in one buffer or in
//! several; a batch can start at any word of its buffer (it is bound from the alignment boundary
//! below it, `min_storage_buffer_offset_alignment`), so several batches, or a whole stripe, can
//! share one buffer. A batch larger than a storage binding (`max_storage_buffer_binding_size`,
//! 128 MiB by default in WebGPU and on lavapipe) is split by the planner into several dispatches.
//! The buffers need `STORAGE` usage, nothing else. So the FFT's output can be hashed where it is:
//! one buffer per size batch (the Cairo base trace's largest, log size 16, is 109 MiB, under
//! WebGPU's default 256 MiB buffer limit), or one arena on native.
//!
//! Joining with the FFT: record the stripe's FFT dispatches, then its [`Plan`] (from
//! [`GpuMerkle::plan_stripe`]) into the same encoder; queue order makes the hashing see the FFT's
//! writes. A batch is read by its dispatch only, so its buffer can take the next stripe's values
//! (or another size's) as soon as the dispatches recorded after it run. Plans are built once (a
//! millisecond for the Cairo trace: uniform buffers and bind groups) and recorded any number of
//! times, the eight stripes' plans differing only in where their levels go.
//!
//! Scratch: two state buffers, `(8 + up to 16) × 2^k` words each for the largest domain `2^k`
//! below the tree's height (2 × 32 MiB for the Cairo base trace: `k = 19`), and the layers.
//!
//! # Kernels
//!
//! - `absorb`, one dispatch per batch, one invocation per row of the batch's domain `2^k`: it loads
//!   the row's Blake2s state left by the previous dispatch (lifted from the previous, smaller
//!   domain if the size grew), absorbs the batch's columns, 16 values per compression, and stores
//!   the state again. A state is the chain value (8 words) and the words of the last, not yet
//!   compressed block (1 to 16 words, carried over to the next dispatch: a block can span batches,
//!   and the last block of the row is compressed only at the end, with the finalization flag).
//!   States are stored word-major (`state[w * 2^k + row]`), so adjacent invocations read and write
//!   adjacent words, as they do the columns' values: every load is coalesced. Two state buffers
//!   alternate (ping-pong).
//! - `absorb_final`, for the last batch (or, if the tree is taller than the largest column, a
//!   dispatch without columns over the leaves' domain): it absorbs, finalizes the leaf hashes, and
//!   hashes the first `min(8, H)` levels of the tree within the workgroup of 256 leaves, in
//!   workgroup memory, storing the levels that are kept.
//! - `nodes`, one invocation per node: up to 9 levels per dispatch (the first from the layer in
//!   memory, the others within the workgroup). It hashes the levels of a stripe's subtree above
//!   `absorb_final`'s, and the levels of the whole tree above the stripes' roots.
//!
//! A stripe is a handful of dispatches (one per batch, two or three for the levels), recorded
//! into one compute pass; nothing is read back.
//!
//! # Output layout: the tree's layers
//!
//! All the layers of the tree of `2^log_n_stripes` stripes of height `H` live in one buffer
//! ([`TreeLayout`]): the stored levels, from the lowest kept one to the root, one after the other,
//! level `l` holding its `2^(H + log_n_stripes - l)` hashes of 8 words each (a hash's words are
//! its 32 bytes, little-endian, as `Blake2sHash` holds them). In a level up to `H`, stripe `s`'s
//! nodes are the `s`-th of `2^log_n_stripes` equal slices, as `new_striped` lays the subtrees side
//! by side. The levels below [`TreeLayout::min_stored_level`] are not stored, as
//! `build_layers_sparse` leaves them empty (see [`stwo_min_stored_level`]).

use std::num::NonZeroU64;

use bytemuck::{Pod, Zeroable};
use stwo::core::vcs::blake2_hash::Blake2sHash;

use crate::GpuContext;
use crate::blake2s::{blake2s_wgsl, block_expr};

/// Log size of the workgroups.
pub const LOG_WORKGROUP: u32 = 8;
/// Invocations per workgroup: the leaves, or nodes, whose upper levels a workgroup hashes in its
/// own memory.
pub const WORKGROUP: u32 = 1 << LOG_WORKGROUP;
/// The levels a `nodes` dispatch hashes at most: one from memory, then the workgroup's.
pub const NODE_LEVELS_PER_DISPATCH: u32 = LOG_WORKGROUP + 1;

/// stwo's `FUSED_LEVELS` (`prover/backend/simd/blake2s_lifted.rs`, private there, replicated
/// here): the levels `build_layers_sparse` leaves empty when it hashes tile by tile.
pub const STWO_FUSED_LEVELS: u32 = 5;
/// stwo's `LOG_N_LANES`: the SIMD backend hashes 16 rows at a time.
const STWO_LOG_N_LANES: u32 = 4;

/// The lowest level `SimdBackend::build_layers_sparse` keeps for columns of these log sizes (in
/// any order) and a tree of height `lifting_log_size`: [`STWO_FUSED_LEVELS`] when it hashes tile
/// by tile, 0 (it keeps every layer, the leaves too) otherwise. A replica of stwo's private
/// `fusable` (`prover/backend/simd/blake2s_lifted.rs`): every column has at least 16 values, the
/// tree is at least 9 levels high, and the largest column is as tall as the tree.
pub fn stwo_min_stored_level(column_log_sizes: &[u32], lifting_log_size: u32) -> u32 {
    let (Some(&min), Some(&max)) = (column_log_sizes.iter().min(), column_log_sizes.iter().max())
    else {
        return 0;
    };
    // stwo compares the number of packed vectors of the largest column, `ceil(2^max / 16)`,
    // which is `2^(max - 4)` as the smallest column, so the largest, has 16 values at least.
    let fusable = min >= STWO_LOG_N_LANES
        && lifting_log_size >= STWO_LOG_N_LANES + STWO_FUSED_LEVELS
        && max == lifting_log_size;
    if fusable { STWO_FUSED_LEVELS } else { 0 }
}

/// A run of `n_columns` columns of `2^log_size` values each, at `word_offset` in `buffer` (see
/// the module docs).
#[derive(Clone, Copy, Debug)]
pub struct ColumnBatch<'a> {
    pub buffer: &'a wgpu::Buffer,
    pub word_offset: u64,
    pub log_size: u32,
    pub n_columns: u32,
}

/// Where the layers of a lifted tree made of `2^log_n_stripes` stripes are, in one buffer of
/// words (see the module docs).
#[derive(Clone, Debug)]
pub struct TreeLayout {
    /// The height of a stripe's subtree, `H`: its leaves are level 0, its root level `H`.
    pub stripe_height: u32,
    pub log_n_stripes: u32,
    /// The lowest stored level; the levels below are left out.
    pub min_stored_level: u32,
    /// The word offset of every level, if stored, from the leaves to the root.
    offsets: Vec<Option<u64>>,
    /// The size of the buffer, in words.
    pub total_words: u64,
}

impl TreeLayout {
    pub fn new(stripe_height: u32, log_n_stripes: u32, min_stored_level: u32) -> Self {
        let height = stripe_height + log_n_stripes;
        assert!(
            min_stored_level <= stripe_height.min(LOG_WORKGROUP),
            "the levels a stripe's workgroups hash last must be stored"
        );
        let mut total_words = 0;
        let offsets = (0..=height)
            .map(|level| {
                (level >= min_stored_level).then(|| {
                    let offset = total_words;
                    total_words += 8 << (height - level);
                    offset
                })
            })
            .collect();
        Self { stripe_height, log_n_stripes, min_stored_level, offsets, total_words }
    }

    /// The height of the whole tree: its root is this level.
    pub fn height(&self) -> u32 {
        self.stripe_height + self.log_n_stripes
    }

    /// The number of hashes of `level` in the whole tree.
    pub fn level_len(&self, level: u32) -> usize {
        1 << (self.height() - level)
    }

    /// The word offset of `level`, if it is stored.
    pub fn level_offset(&self, level: u32) -> Option<u64> {
        self.offsets[level as usize]
    }

    /// Every level of the tree, from the words of its buffer: the hashes of the stored levels, an
    /// empty vector for the others (as stwo's layers, leaves first).
    pub fn levels(&self, words: &[u32]) -> Vec<Vec<Blake2sHash>> {
        assert!(words.len() as u64 >= self.total_words);
        (0..=self.height())
            .map(|level| match self.level_offset(level) {
                None => vec![],
                Some(offset) => {
                    let start = offset as usize;
                    words[start..start + 8 * self.level_len(level)]
                        .chunks_exact(8)
                        .map(|hash| {
                            Blake2sHash(bytemuck::cast_slice::<u32, u8>(hash).try_into().unwrap())
                        })
                        .collect()
                }
            })
            .collect()
    }

    /// The word offset of stripe `stripe`'s nodes in `level` (at most the stripe's height).
    fn stripe_level_offset(&self, stripe: u32, level: u32) -> u64 {
        assert!(level <= self.stripe_height);
        let offset = self.level_offset(level).expect("level not stored");
        offset + ((stripe as u64 * 8) << (self.stripe_height - level))
    }
}

/// The parameters of an `absorb` or `absorb_final` dispatch, a uniform buffer.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AbsorbParams {
    /// The dispatch's domain: `2^log_rows` rows (states, or leaves for `absorb_final`).
    log_rows: u32,
    /// The batch's columns, of `2^log_rows` values each.
    n_cols: u32,
    /// The word offset of the batch's first column in the bound range of its buffer.
    col_offset: u32,
    /// Whether there are states from a previous dispatch (otherwise, the initial state).
    has_prev: u32,
    /// The previous dispatch's domain (`log_rows` if none).
    prev_log_rows: u32,
    /// The words of the last block the previous dispatch left uncompressed (0 to 16).
    r_in: u32,
    /// The bytes compressed before this dispatch, low and high words.
    t_lo: u32,
    t_hi: u32,
    /// `absorb_final`: the length of a leaf's message in bytes.
    total_lo: u32,
    total_hi: u32,
    /// `absorb_final`: the levels above the leaves hashed within a workgroup.
    fused_levels: u32,
    /// `absorb_final`: the lowest level stored.
    store_from: u32,
    /// `absorb_final`: the word offset of this stripe's nodes of level `l` (`l <= fused_levels`)
    /// in the layers' buffer, at `[l / 4][l % 4]`.
    level_base: [[u32; 4]; 4],
}

/// The parameters of a `nodes` dispatch, a uniform buffer.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct NodeParams {
    /// The nodes of the first level hashed, two children each.
    n_out: u32,
    /// The word offset of the children's level (the first child of the first node).
    in_base: u32,
    /// The levels hashed: the first one, and `levels - 1` above it within the workgroup.
    levels: u32,
    _pad: u32,
    /// The word offset of the `j`-th level hashed (the first node's) at `[j / 4][j % 4]`.
    out_base: [[u32; 4]; 4],
}

/// The WGSL of the leaves' kernels, `absorb` and `absorb_final`, Blake2s included.
pub fn absorb_shader() -> String {
    format!("{}\n{}", blake2s_wgsl(), absorb_wgsl())
}

/// The WGSL of the `nodes` kernel, Blake2s included.
pub fn nodes_shader() -> String {
    format!("{}\n{}", blake2s_wgsl(), nodes_wgsl())
}

fn absorb_wgsl() -> String {
    let unrolled = |line: &dyn Fn(usize) -> String, n: usize| -> String {
        (0..n).map(line).collect::<Vec<_>>().join("\n    ")
    };
    let load_prev_chain = format!(
        "array<u32, 8>({})",
        (0..8)
            .map(|w| format!("state_in[({w}u << P.prev_log_rows) + prev_row]"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    let store_chain =
        unrolled(&|w| format!("state_out[({w}u << P.log_rows) + row] = a.h[{w}];"), 8);
    format!(
        r#"
const WG: u32 = {WORKGROUP}u;

struct AbsorbParams {{
    log_rows: u32,
    n_cols: u32,
    col_offset: u32,
    has_prev: u32,
    prev_log_rows: u32,
    r_in: u32,
    t_lo: u32,
    t_hi: u32,
    total_lo: u32,
    total_hi: u32,
    fused_levels: u32,
    store_from: u32,
    level_base: array<vec4<u32>, 4>,
}}

@group(0) @binding(0) var<uniform> P: AbsorbParams;
// The batch's columns: column c's value at row i is cols[col_offset + c * 2^log_rows + i].
@group(0) @binding(1) var<storage, read> cols: array<u32>;
// The previous dispatch's states, word w of row i at state_in[w * 2^prev_log_rows + i]: the
// chain value (w < 8), then the r_in words of the block not compressed yet.
@group(0) @binding(2) var<storage, read> state_in: array<u32>;
// This dispatch's states, in the same layout over 2^log_rows rows (absorb).
@group(0) @binding(3) var<storage, read_write> state_out: array<u32>;
// The tree's layers (absorb_final).
@group(0) @binding(4) var<storage, read_write> layers: array<u32>;

var<workgroup> sh: array<u32, {SH_WORDS}>;

// The row of a domain 2^log_ratio times smaller that row i of the lifted domain reads.
fn lift(i: u32, log_ratio: u32) -> u32 {{
    if (log_ratio == 0u) {{ return i; }}
    return ((i >> (log_ratio + 1u)) << 1u) | (i & 1u);
}}

// Word k of the dispatch's part of a row's message: the words carried over from the previous
// dispatch (its row prev_row), then the batch's values at the row.
fn stream_word(row: u32, prev_row: u32, k: u32) -> u32 {{
    if (k < P.r_in) {{
        return state_in[((8u + k) << P.prev_log_rows) + prev_row];
    }}
    return cols[P.col_offset + ((k - P.r_in) << P.log_rows) + row];
}}

// The batch's column c at the row.
fn column_word(row: u32, c: u32) -> u32 {{
    return cols[P.col_offset + (c << P.log_rows) + row];
}}

// The same, with zeros past the end: the padding of the last block.
fn tail_word(row: u32, prev_row: u32, k: u32) -> u32 {{
    if (k < P.r_in + P.n_cols) {{ return stream_word(row, prev_row, k); }}
    return 0u;
}}

struct Absorbed {{
    h: array<u32, 8>,
    prev_row: u32,
    n_blocks: u32,
}}

// Compresses every block of the row's stream but the last one, which may be the message's last.
fn absorb_blocks(row: u32) -> Absorbed {{
    let prev_row = lift(row, P.log_rows - P.prev_log_rows);
    var h = blake2s_init();
    if (P.has_prev != 0u) {{
        h = {load_prev_chain};
    }}
    let total = P.r_in + P.n_cols;
    let n_blocks = select(0u, (total - 1u) / 16u, total > 0u);
    var b = 0u;
    // The first block may start with the words carried over.
    if (P.r_in > 0u && n_blocks > 0u) {{
        let base = 0u;
        h = blake2s_compress(h, {stream_block}, P.t_lo + 64u, P.t_hi + select(0u, 1u, P.t_lo + 64u < P.t_lo), false);
        b = 1u;
    }}
    // The others are 16 of the batch's columns: no branch per word.
    for (; b < n_blocks; b++) {{
        let base = 16u * b - P.r_in;
        let t0 = P.t_lo + 64u * (b + 1u);
        let t1 = P.t_hi + select(0u, 1u, t0 < P.t_lo);
        h = blake2s_compress(h, {column_block}, t0, t1, false);
    }}
    return Absorbed(h, prev_row, n_blocks);
}}

fn level_base(level: u32) -> u32 {{
    return P.level_base[level / 4u][level % 4u];
}}

{SHARED_HELPERS}

@compute @workgroup_size({WORKGROUP})
fn absorb(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {{
    let row = (wid.x + wid.y * nwg.x) * WG + lid;
    if (row >= (1u << P.log_rows)) {{ return; }}
    let a = absorb_blocks(row);
    {store_chain}
    // The words of the last block, kept for the next dispatch.
    let base = 16u * a.n_blocks;
    let rest = P.r_in + P.n_cols - base;
    for (var w = 0u; w < rest; w++) {{
        state_out[((8u + w) << P.log_rows) + row] = stream_word(row, a.prev_row, base + w);
    }}
}}

@compute @workgroup_size({WORKGROUP})
fn absorb_final(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {{
    let wg = wid.x + wid.y * nwg.x;
    let n = 1u << P.log_rows;
    let first = wg * WG;
    // The leaves of this workgroup: WG, or all of them in a smaller tree, or none.
    let wg_rows = select(0u, min(n - first, WG), first < n);
    let row = first + lid;
    var h: array<u32, 8>;
    if (lid < wg_rows) {{
        let a = absorb_blocks(row);
        let base = 16u * a.n_blocks;
        h = blake2s_compress(a.h, {tail_block}, P.total_lo, P.total_hi, true);
        if (P.store_from == 0u) {{ store_hash(level_base(0u) + 8u * row, h); }}
    }}
    write_sh(lid, h);
    for (var level = 1u; level <= P.fused_levels; level++) {{
        workgroupBarrier();
        let is_node = lid < (wg_rows >> level);
        var node: array<u32, 8>;
        if (is_node) {{ node = blake2s_hash_pair(read_sh(2u * lid), read_sh(2u * lid + 1u)); }}
        workgroupBarrier();
        if (is_node) {{
            write_sh(lid, node);
            if (level >= P.store_from) {{
                store_hash(level_base(level) + 8u * (wg * (WG >> level) + lid), node);
            }}
        }}
    }}
}}
"#,
        SH_WORDS = 8 * WORKGROUP,
        SHARED_HELPERS = shared_helpers_wgsl(),
        stream_block = block_expr("stream_word", "row, prev_row", "base"),
        column_block = block_expr("column_word", "row", "base"),
        tail_block = block_expr("tail_word", "row, a.prev_row", "base"),
    )
}

fn nodes_wgsl() -> String {
    format!(
        r#"
const WG: u32 = {WORKGROUP}u;

struct NodeParams {{
    n_out: u32,
    in_base: u32,
    levels: u32,
    pad: u32,
    out_base: array<vec4<u32>, 4>,
}}

@group(0) @binding(0) var<uniform> Q: NodeParams;
// The tree's layers: read the children's level, write the levels above.
@group(0) @binding(1) var<storage, read_write> layers: array<u32>;

var<workgroup> sh: array<u32, {SH_WORDS}>;

fn out_base(j: u32) -> u32 {{
    return Q.out_base[j / 4u][j % 4u];
}}

fn load_hash(offset: u32) -> array<u32, 8> {{
    return array<u32, 8>({load});
}}

{SHARED_HELPERS}

@compute @workgroup_size({WORKGROUP})
fn nodes(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {{
    let wg = wid.x + wid.y * nwg.x;
    let first = wg * WG;
    let wg_nodes = select(0u, min(Q.n_out - first, WG), first < Q.n_out);
    let j = first + lid;
    var h: array<u32, 8>;
    if (lid < wg_nodes) {{
        h = blake2s_hash_pair(load_hash(Q.in_base + 16u * j), load_hash(Q.in_base + 16u * j + 8u));
        store_hash(out_base(0u) + 8u * j, h);
    }}
    write_sh(lid, h);
    for (var level = 1u; level < Q.levels; level++) {{
        workgroupBarrier();
        let is_node = lid < (wg_nodes >> level);
        var node: array<u32, 8>;
        if (is_node) {{ node = blake2s_hash_pair(read_sh(2u * lid), read_sh(2u * lid + 1u)); }}
        workgroupBarrier();
        if (is_node) {{
            write_sh(lid, node);
            store_hash(out_base(level) + 8u * (wg * (WG >> level) + lid), node);
        }}
    }}
}}
"#,
        SH_WORDS = 8 * WORKGROUP,
        SHARED_HELPERS = shared_helpers_wgsl(),
        load = (0..8).map(|w| format!("layers[offset + {w}u]")).collect::<Vec<_>>().join(", "),
    )
}

/// `store_hash` (into `layers`), `read_sh` and `write_sh` (the workgroup's hashes, `sh`).
fn shared_helpers_wgsl() -> String {
    let store = (0..8).map(|w| format!("layers[offset + {w}u] = h[{w}];")).collect::<Vec<_>>();
    let read = (0..8).map(|w| format!("sh[8u * i + {w}u]")).collect::<Vec<_>>();
    let write = (0..8).map(|w| format!("sh[8u * i + {w}u] = h[{w}];")).collect::<Vec<_>>();
    format!(
        "fn store_hash(offset: u32, h: array<u32, 8>) {{\n    {}\n}}\n\nfn read_sh(i: u32) -> \
         array<u32, 8> {{\n    return array<u32, 8>({});\n}}\n\nfn write_sh(i: u32, h: array<u32, \
         8>) {{\n    {}\n}}\n",
        store.join("\n    "),
        read.join(", "),
        write.join("\n    "),
    )
}

/// One dispatch of the leaves' hashing of a stripe.
#[derive(Clone, Debug)]
struct AbsorbStep {
    /// The batch it absorbs, or none (a final dispatch that only lifts and finalizes).
    batch: Option<usize>,
    log_rows: u32,
    n_cols: u32,
    /// The previous dispatch's domain and the words it left uncompressed.
    prev: Option<(u32, u32)>,
    bytes_before: u64,
    is_final: bool,
}

impl AbsorbStep {
    fn n_words(&self) -> u32 {
        self.prev.map_or(0, |(_, r)| r) + self.n_cols
    }
    fn n_blocks(&self) -> u32 {
        self.n_words().saturating_sub(1) / 16
    }
    /// The words of the last block, left for the next dispatch.
    fn rest(&self) -> u32 {
        self.n_words() - 16 * self.n_blocks()
    }
    /// The words of the states this dispatch stores (not for a final one).
    fn state_words(&self) -> u64 {
        (8 + self.rest() as u64) << self.log_rows
    }
}

/// The dispatches that hash the leaves of a stripe of columns of these `(log_size, n_columns)`
/// batches, into a tree of height `height`.
fn absorb_schedule(shapes: &[(u32, u32)], height: u32) -> Vec<AbsorbStep> {
    let nonempty: Vec<usize> = (0..shapes.len()).filter(|&i| shapes[i].1 > 0).collect();
    assert!(!nonempty.is_empty(), "a tree without columns");
    let mut steps: Vec<AbsorbStep> = vec![];
    let mut bytes_before = 0;
    for (k, &i) in nonempty.iter().enumerate() {
        let (log_size, n_cols) = shapes[i];
        assert!(
            (1..=height).contains(&log_size),
            "column log size {log_size}, tree height {height}"
        );
        let prev = steps.last().map(|step: &AbsorbStep| (step.log_rows, step.rest()));
        if let Some((prev_log_size, _)) = prev {
            assert!(
                prev_log_size <= log_size,
                "batches out of order: {prev_log_size} > {log_size}"
            );
        }
        if let Some(step) = steps.last() {
            bytes_before += 64 * step.n_blocks() as u64;
        }
        let is_final = k == nonempty.len() - 1 && log_size == height;
        steps.push(AbsorbStep {
            batch: Some(i),
            log_rows: log_size,
            n_cols,
            prev,
            bytes_before,
            is_final,
        });
    }
    if !steps.last().unwrap().is_final {
        let last = steps.last().unwrap();
        let prev = Some((last.log_rows, last.rest()));
        bytes_before += 64 * last.n_blocks() as u64;
        steps.push(AbsorbStep {
            batch: None,
            log_rows: height,
            n_cols: 0,
            prev,
            bytes_before,
            is_final: true,
        });
    }
    steps
}

/// The groups of a dispatch of `n` invocations (two-dimensional past 65535 groups).
fn workgroups(n: u64) -> (u32, u32, u32) {
    let groups = n.div_ceil(WORKGROUP as u64);
    if groups <= 65535 {
        (groups as u32, 1, 1)
    } else {
        let x = 32768;
        (x as u32, groups.div_ceil(x) as u32, 1)
    }
}

fn u32_offset(words: u64) -> u32 {
    u32::try_from(words).expect("word offset past 2^32")
}

/// The two state buffers of a stripe's leaves (ping-pong), sized for its batches' shapes.
pub struct StateBuffers {
    buffers: [wgpu::Buffer; 2],
    pub words: u64,
}

impl StateBuffers {
    /// The bytes of both buffers.
    pub fn bytes(&self) -> u64 {
        self.buffers.iter().map(wgpu::Buffer::size).sum()
    }

    /// Frees both buffers now (see `StripedCommitment::destroy`).
    pub fn destroy(&self) {
        for buffer in &self.buffers {
            buffer.destroy();
        }
    }
}

/// A recorded sequence of dispatches, with their parameters and bindings, ready to be recorded
/// into a compute pass any number of times.
pub struct Plan {
    steps: Vec<(wgpu::ComputePipeline, wgpu::BindGroup, (u32, u32, u32))>,
    _uniforms: Option<wgpu::Buffer>,
}

impl Plan {
    pub fn record(&self, pass: &mut wgpu::ComputePass<'_>) {
        for (pipeline, bind_group, (x, y, z)) in &self.steps {
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(*x, *y, *z);
        }
    }

    pub fn n_dispatches(&self) -> usize {
        self.steps.len()
    }

    /// Frees its parameters' buffer now (see `StripedCommitment::destroy`).
    pub fn destroy(&self) {
        if let Some(uniforms) = &self._uniforms {
            uniforms.destroy();
        }
    }
}

/// The Merkle kernels on a device.
pub struct GpuMerkle {
    absorb: wgpu::ComputePipeline,
    absorb_final: wgpu::ComputePipeline,
    nodes: wgpu::ComputePipeline,
    /// Bound where a dispatch reads nothing (no previous states, no columns).
    dummy: wgpu::Buffer,
    uniform_align: u64,
    storage_align: u64,
    max_binding: u64,
}

impl GpuMerkle {
    pub fn new(gpu: &GpuContext) -> Self {
        Self::build(gpu, true)
    }

    /// The kernels without wgpu's runtime checks (bounds checks, loop bounding), to measure what
    /// they cost; browsers always keep them.
    ///
    /// # Safety
    ///
    /// Every buffer bound to a plan must be as large as the planner assumes (it checks the batches
    /// against their buffers' sizes, and sizes the states and the layers itself); the kernels
    /// then stay in bounds: a row past the domain or a node past the level reads nothing.
    pub unsafe fn new_unchecked(gpu: &GpuContext) -> Self {
        Self::build(gpu, false)
    }

    fn build(gpu: &GpuContext, checked: bool) -> Self {
        let (absorb_source, nodes_source) = (absorb_shader(), nodes_shader());
        let limits = gpu.device.limits();
        let pipeline = |label: &str, source: &str| {
            if checked {
                gpu.pipeline(label, source, label)
            } else {
                // Safety: see `new_unchecked`.
                unsafe { gpu.pipeline_unchecked(label, source, label) }
            }
        };
        Self {
            absorb: pipeline("absorb", &absorb_source),
            absorb_final: pipeline("absorb_final", &absorb_source),
            nodes: pipeline("nodes", &nodes_source),
            dummy: gpu.upload("dummy", &[0; 64]),
            uniform_align: limits.min_uniform_buffer_offset_alignment as u64,
            storage_align: limits.min_storage_buffer_offset_alignment as u64,
            max_binding: limits.max_storage_buffer_binding_size,
        }
    }

    /// The same kernels, binding at most `bytes` of a buffer at a time (if below the device's
    /// limit): to exercise the splitting of batches on small inputs.
    pub fn with_binding_limit(mut self, bytes: u64) -> Self {
        self.max_binding = self.max_binding.min(bytes);
        self
    }

    /// The buffer of a tree's layers.
    pub fn tree_buffer(&self, gpu: &GpuContext, layout: &TreeLayout) -> wgpu::Buffer {
        let bytes = layout.total_words * 4;
        assert!(bytes <= self.max_binding, "the tree's layers ({bytes} bytes) exceed a binding");
        gpu.storage("tree layers", layout.total_words as usize)
    }

    /// The state buffers for the leaves of stripes of these `(log_size, n_columns)` batches in a
    /// tree of height `height`.
    pub fn state_buffers(
        &self,
        gpu: &GpuContext,
        shapes: &[(u32, u32)],
        height: u32,
    ) -> StateBuffers {
        let words = absorb_schedule(&self.dispatch_shapes(shapes), height)
            .iter()
            .filter(|step| !step.is_final)
            .map(AbsorbStep::state_words)
            .max()
            .unwrap_or(0)
            .max(64);
        assert!(words * 4 <= self.max_binding, "the states ({} bytes) exceed a binding", words * 4);
        StateBuffers {
            buffers: [
                gpu.storage("states a", words as usize),
                gpu.storage("states b", words as usize),
            ],
            words,
        }
    }

    /// The most columns of `2^log_size` values one dispatch binds: a batch's binding starts at
    /// the alignment boundary below it and is at most `max_storage_buffer_binding_size` long
    /// (128 MiB by default in WebGPU, and on lavapipe).
    pub fn max_columns_per_dispatch(&self, log_size: u32) -> u32 {
        let column_bytes = 4u64 << log_size;
        let n = (self.max_binding - self.storage_align) / column_bytes;
        assert!(n >= 1, "a column of 2^{log_size} values exceeds a storage binding");
        n.min(u32::MAX as u64) as u32
    }

    /// The batches as dispatched: a batch larger than a binding is split at column boundaries.
    fn dispatch_batches<'a>(&self, batches: &[ColumnBatch<'a>]) -> Vec<ColumnBatch<'a>> {
        let mut split = vec![];
        for batch in batches {
            let max = self.max_columns_per_dispatch(batch.log_size);
            let mut first = 0;
            while first < batch.n_columns {
                let n_columns = (batch.n_columns - first).min(max);
                let word_offset = batch.word_offset + ((first as u64) << batch.log_size);
                split.push(ColumnBatch { word_offset, n_columns, ..*batch });
                first += n_columns;
            }
        }
        split
    }

    /// [`Self::dispatch_batches`]'s shapes.
    fn dispatch_shapes(&self, shapes: &[(u32, u32)]) -> Vec<(u32, u32)> {
        let mut split = vec![];
        for &(log_size, n) in shapes {
            let max = self.max_columns_per_dispatch(log_size);
            let mut first = 0;
            while first < n {
                split.push((log_size, (n - first).min(max)));
                first += (n - first).min(max);
            }
        }
        split
    }

    /// The dispatches that hash stripe `stripe` of the tree: its leaves, from `batches` (see the
    /// module docs), and every level of its subtree, into `tree` (laid out as `layout`).
    pub fn plan_stripe(
        &self,
        gpu: &GpuContext,
        batches: &[ColumnBatch<'_>],
        states: &StateBuffers,
        tree: &wgpu::Buffer,
        layout: &TreeLayout,
        stripe: u32,
    ) -> Plan {
        assert!(stripe < 1 << layout.log_n_stripes);
        let batches = &self.dispatch_batches(batches);
        let height = layout.stripe_height;
        let shapes: Vec<(u32, u32)> = batches.iter().map(|b| (b.log_size, b.n_columns)).collect();
        let schedule = absorb_schedule(&shapes, height);
        let fused_levels = height.min(LOG_WORKGROUP);

        let mut params: Vec<AbsorbParams> = vec![];
        // (batch binding: buffer, byte offset, byte size), state_in, state_out.
        let mut bindings = vec![];
        let mut current = 0;
        for step in &schedule {
            let (r_in, prev_log_rows) = step.prev.map_or((0, step.log_rows), |(log, r)| (r, log));
            let total_bytes = step.bytes_before + 4 * step.n_words() as u64;
            let (cols, col_offset) = match step.batch {
                Some(i) => {
                    let batch = &batches[i];
                    let start = batch.word_offset * 4;
                    let end =
                        (batch.word_offset + ((batch.n_columns as u64) << batch.log_size)) * 4;
                    assert!(end <= batch.buffer.size(), "batch {i} past the end of its buffer");
                    let bind_start = start / self.storage_align * self.storage_align;
                    assert!(end - bind_start <= self.max_binding, "batch {i} exceeds a binding");
                    ((batch.buffer, bind_start, end - bind_start), (start - bind_start) / 4)
                }
                None => ((&self.dummy, 0, self.dummy.size()), 0),
            };
            let mut level_base = [[0; 4]; 4];
            if step.is_final {
                for level in layout.min_stored_level..=fused_levels {
                    let offset = layout.stripe_level_offset(stripe, level);
                    level_base[level as usize / 4][level as usize % 4] = u32_offset(offset);
                }
            }
            params.push(AbsorbParams {
                log_rows: step.log_rows,
                n_cols: step.n_cols,
                col_offset: u32_offset(col_offset),
                has_prev: step.prev.is_some() as u32,
                prev_log_rows,
                r_in,
                t_lo: step.bytes_before as u32,
                t_hi: (step.bytes_before >> 32) as u32,
                total_lo: total_bytes as u32,
                total_hi: (total_bytes >> 32) as u32,
                fused_levels,
                store_from: layout.min_stored_level,
                level_base,
            });
            let state_in = if step.prev.is_some() { &states.buffers[current] } else { &self.dummy };
            let state_out = &states.buffers[1 - current];
            if !step.is_final {
                assert!(step.state_words() <= states.words, "state buffers too small");
            }
            bindings.push((cols, state_in, state_out));
            current = 1 - current;
        }
        let uniforms = self.uniforms(gpu, &params);

        let mut steps = vec![];
        for (k, (step, ((cols, cols_offset, cols_size), state_in, state_out))) in
            schedule.iter().zip(bindings).enumerate()
        {
            let pipeline = if step.is_final { &self.absorb_final } else { &self.absorb };
            let last = if step.is_final {
                (4, tree.as_entire_binding())
            } else {
                (3, state_out.as_entire_binding())
            };
            let entries = [
                (0, self.uniform_binding::<AbsorbParams>(&uniforms, k)),
                (
                    1,
                    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: cols,
                        offset: cols_offset,
                        size: NonZeroU64::new(cols_size),
                    }),
                ),
                (2, state_in.as_entire_binding()),
                last,
            ];
            let bind_group = self.bind_group(gpu, pipeline, &entries);
            steps.push((pipeline.clone(), bind_group, workgroups(1 << step.log_rows)));
        }
        let mut plan = Plan { steps, _uniforms: Some(uniforms) };
        // The levels of the subtree above the workgroups'.
        let levels = self.plan_levels(gpu, tree, layout, fused_levels, height, Some(stripe));
        plan.steps.extend(levels.steps);
        plan
    }

    /// The dispatches that hash the levels of the whole tree above the stripes' roots
    /// (`build_next_layer`, from level `stripe_height` to the root).
    pub fn plan_top(&self, gpu: &GpuContext, tree: &wgpu::Buffer, layout: &TreeLayout) -> Plan {
        self.plan_levels(gpu, tree, layout, layout.stripe_height, layout.height(), None)
    }

    /// The dispatches that hash the levels `from + 1..=to` from level `from` (which must be
    /// stored): of stripe `stripe`'s subtree, or of the whole tree.
    pub fn plan_levels(
        &self,
        gpu: &GpuContext,
        tree: &wgpu::Buffer,
        layout: &TreeLayout,
        from: u32,
        to: u32,
        stripe: Option<u32>,
    ) -> Plan {
        let offset = |level: u32| match stripe {
            Some(stripe) => layout.stripe_level_offset(stripe, level),
            None => layout.level_offset(level).expect("level not stored"),
        };
        let top = if stripe.is_some() { layout.stripe_height } else { layout.height() };
        let mut params = vec![];
        let mut level = from;
        while level < to {
            let levels = (to - level).min(NODE_LEVELS_PER_DISPATCH);
            let mut out_base = [[0; 4]; 4];
            for j in 0..levels {
                out_base[j as usize / 4][j as usize % 4] = u32_offset(offset(level + 1 + j));
            }
            params.push(NodeParams {
                n_out: 1 << (top - level - 1),
                in_base: u32_offset(offset(level)),
                levels,
                _pad: 0,
                out_base,
            });
            level += levels;
        }
        if params.is_empty() {
            return Plan { steps: vec![], _uniforms: None };
        }
        let uniforms = self.uniforms(gpu, &params);
        let steps = params
            .iter()
            .enumerate()
            .map(|(k, p)| {
                let entries = [
                    (0, self.uniform_binding::<NodeParams>(&uniforms, k)),
                    (1, tree.as_entire_binding()),
                ];
                let bind_group = self.bind_group(gpu, &self.nodes, &entries);
                (self.nodes.clone(), bind_group, workgroups(p.n_out as u64))
            })
            .collect();
        Plan { steps, _uniforms: Some(uniforms) }
    }

    /// Every level of the tree, read back: the hashes of the stored levels, an empty vector for
    /// the others (as stwo's layers, leaves first).
    pub fn read_levels(
        &self,
        gpu: &GpuContext,
        tree: &wgpu::Buffer,
        layout: &TreeLayout,
    ) -> Vec<Vec<Blake2sHash>> {
        layout.levels(&gpu.read(tree, layout.total_words as usize))
    }

    /// One uniform buffer holding every dispatch's parameters, at aligned offsets.
    fn uniforms<T: Pod>(&self, gpu: &GpuContext, params: &[T]) -> wgpu::Buffer {
        let stride = (size_of::<T>() as u64).next_multiple_of(self.uniform_align) as usize;
        let mut bytes = vec![0u8; stride * params.len()];
        for (k, p) in params.iter().enumerate() {
            bytes[k * stride..k * stride + size_of::<T>()].copy_from_slice(bytemuck::bytes_of(p));
        }
        use wgpu::util::DeviceExt;
        gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("merkle params"),
            contents: &bytes,
            usage: wgpu::BufferUsages::UNIFORM,
        })
    }

    fn uniform_binding<'a, T>(
        &self,
        uniforms: &'a wgpu::Buffer,
        k: usize,
    ) -> wgpu::BindingResource<'a> {
        let stride = (size_of::<T>() as u64).next_multiple_of(self.uniform_align);
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: uniforms,
            offset: k as u64 * stride,
            size: NonZeroU64::new(size_of::<T>() as u64),
        })
    }

    fn bind_group(
        &self,
        gpu: &GpuContext,
        pipeline: &wgpu::ComputePipeline,
        entries: &[(u32, wgpu::BindingResource<'_>)],
    ) -> wgpu::BindGroup {
        let entries: Vec<wgpu::BindGroupEntry<'_>> = entries
            .iter()
            .map(|(binding, resource)| wgpu::BindGroupEntry {
                binding: *binding,
                resource: resource.clone(),
            })
            .collect();
        gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        })
    }
}

impl GpuContext {
    /// Records `plans` in order into one compute pass, submits it, and waits for it.
    pub fn run_plans(&self, plans: &[&Plan]) {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            for plan in plans {
                plan.record(&mut pass);
            }
        }
        self.queue.submit([encoder.finish()]);
        self.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    }
}
