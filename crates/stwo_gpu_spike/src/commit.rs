//! The striped commitment of a tree on the GPU, equal to stwo's
//! `CommitmentTreeProver::new_striped` (`stwo/src/prover/pcs/mod.rs`) with `SimdBackend` and
//! `Blake2sM31MerkleChannel`: the same layers, root and all, and the same evaluations kept.
//!
//! # What `new_striped` does, and what this does
//!
//! For each stripe `k` of the `2^log_blowup` stripes of the columns' domains, stripe 0 last:
//! every column is evaluated on stripe `k` (`part_twiddles`, `evaluate_into`), and the stripe's
//! columns are hashed into the stripe's subtree (`build_layers_sparse`, the columns in tree order:
//! by increasing size, polynomial order within a size). The subtrees side by side are the tree's
//! levels up to the stripes' height; `build_next_layer` hashes the levels above. Each column keeps
//! its evaluations on stripe 0, or, if its domain has fewer than `2^14` points
//! ([`MIN_STRIPED_LOG_SIZE`]), its whole evaluation (stripe 0, then 1, ..., in bit-reversed
//! order).
//!
//! On the GPU ([`StripedCommitment`]):
//! - the coefficients are uploaded once ([`StripedCommitment::upload`]), into the [`ColumnsLayout`]
//!   of the extension (per-size batches, input order within a size: the tree's order);
//! - per stripe, in `new_striped`'s order, one encoder records the stripe's extension (a compute
//!   pass, from the coefficients into one stripe's output buffers), then the stripe's Merkle plan
//!   (a compute pass, reading the output buffers where the extension left them, writing the
//!   stripe's slices of the tree's levels), then copies of the small columns' stripe into the
//!   buffer of the columns kept whole; the next stripe's extension overwrites the output buffers;
//! - then the levels above the stripes (a compute pass), and, to read the result back, copies of
//!   the output buffers (stripe 0), of the whole columns and of the tree's levels into staging
//!   buffers. All of it is one submission; nothing is read back before the end.
//!
//! GPU memory: the coefficients, one stripe's evaluations, every stripe's twiddles, the Merkle
//! states (two buffers) and the tree's stored levels, plus the small columns' every stripe; and,
//! host-visible, the staging buffers of the readback (see [`StripedCommitment::gpu_bytes`]).

use std::iter::repeat_n;
use std::sync::Arc;

use itertools::Itertools;
use rayon::prelude::*;
use stwo::core::ColumnVec;
use stwo::core::fields::m31::BaseField;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::vcs::blake2_hash::Blake2sHash;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sM31MerkleChannel;
use stwo::prover::backend::Column;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::simd::column::BaseColumn;
use stwo::prover::poly::BitReversedOrder;
use stwo::prover::poly::circle::{CircleCoefficients, CircleEvaluation, part_domain};
use stwo::prover::poly::twiddles::TwiddleTree;
use stwo::prover::vcs_lifted::prover::MerkleProverLifted;
use stwo::prover::{CommitmentTreeProver, Poly};

use crate::GpuContext;
use crate::extension::{
    ColumnsLayout, ExtensionKernels, GpuColumns, Merge, StripeExtension, max_buffer_words,
    stripe_twiddle_words,
};
use crate::merkle::{
    ColumnBatch as MerkleBatch, GpuMerkle, Plan, StateBuffers, TreeLayout, stwo_min_stored_level,
};

/// `new_striped`'s `MIN_COEFFS_LOG_SIZE`: the smallest coefficients log size it stripes.
pub const MIN_COEFFS_LOG_SIZE: u32 = 4;
/// `new_striped`'s `MIN_STRIPE_HEIGHT`: the smallest height of a stripe's subtree.
pub const MIN_STRIPE_HEIGHT: u32 = 10;
/// `new_striped`'s `MIN_STRIPED_LOG_SIZE`: columns of smaller domains are kept whole.
pub const MIN_STRIPED_LOG_SIZE: u32 = 14;

/// Whether `new_striped` takes its striped path for polynomials of these coefficients log sizes
/// (otherwise it falls back to `CommitmentTreeProver::new`): a replica of its preconditions.
pub fn takes_striped_path(log_sizes: &[u32], log_blowup: u32, lifting_log_size: u32) -> bool {
    let Some(&max_log_size) = log_sizes.iter().max() else {
        return false;
    };
    log_blowup != 0
        && lifting_log_size >= log_blowup + MIN_STRIPE_HEIGHT
        && lifting_log_size >= max_log_size + log_blowup
        && log_sizes.iter().all(|&log_size| log_size >= MIN_COEFFS_LOG_SIZE)
}

/// Whether `new_striped` keeps a column of `2^log_size` coefficients whole (its domain is smaller
/// than [`MIN_STRIPED_LOG_SIZE`]) rather than its stripe 0 only.
pub fn kept_whole(log_size: u32, log_blowup: u32) -> bool {
    log_size + log_blowup < MIN_STRIPED_LOG_SIZE
}

/// The order of the stripes, as `new_striped` evaluates and hashes them: stripe 0, the one kept,
/// last.
pub fn stripe_order(log_blowup: u32) -> impl Iterator<Item = usize> {
    (1..1usize << log_blowup).chain([0])
}

/// The compute pipelines of the commitment, for one device.
pub struct CommitKernels {
    pub extension: ExtensionKernels,
    pub merkle: GpuMerkle,
}

impl CommitKernels {
    pub fn new(gpu: &GpuContext) -> Self {
        Self { extension: ExtensionKernels::new(gpu), merkle: GpuMerkle::new(gpu) }
    }

    /// With the Merkle kernels without wgpu's runtime checks (see [`GpuMerkle::new_unchecked`]).
    ///
    /// # Safety
    ///
    /// As [`GpuMerkle::new_unchecked`]: the commitment sizes and checks every buffer it binds.
    pub unsafe fn new_unchecked(gpu: &GpuContext) -> Self {
        // Safety: the caller's.
        Self {
            extension: ExtensionKernels::new(gpu),
            merkle: unsafe { GpuMerkle::new_unchecked(gpu) },
        }
    }
}

/// A batch of columns kept whole: its words of a stripe are copied, after the stripe's hashing,
/// from `src` in output buffer `buffer` to `dst` in the stripe's slice of the whole columns'
/// buffer.
#[derive(Clone, Copy, Debug)]
struct WholeCopy {
    buffer: usize,
    src: u64,
    dst: u64,
    words: u64,
}

/// The staging buffers the result is read back through.
struct Readback {
    out: Vec<wgpu::Buffer>,
    whole: Option<wgpu::Buffer>,
    tree: wgpu::Buffer,
}

/// The Merkle layers and the kept evaluations of a commitment, as read back.
pub struct CommitOutput {
    /// The tree's layers as stwo's `MerkleProverLifted` holds them: root first, an empty layer
    /// where `build_layers_sparse` leaves one empty.
    pub layers: Vec<Vec<Blake2sHash>>,
    /// Per column (input order): its evaluations on stripe 0, or, for a column kept whole, on its
    /// whole domain; bit-reversed, as stwo's.
    pub evaluations: Vec<BaseColumn>,
}

/// The time the GPU spent on each pass of a commitment, from timestamp queries.
#[derive(Clone, Debug)]
pub struct PassTimes {
    /// Per stripe, in the order they ran ([`stripe_order`]): (stripe, extension ns, hashing ns).
    pub stripes: Vec<(usize, f64, f64)>,
    /// The levels above the stripes, ns.
    pub top: f64,
    /// From the first pass's start to the last one's end, ns.
    pub total: f64,
}

/// Timestamp queries at the beginning and the end of every pass of a commitment.
pub struct PassTimestamps {
    query_set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
    count: u32,
    log_blowup: u32,
}

impl PassTimestamps {
    /// `None` if the device has no `TIMESTAMP_QUERY` (see
    /// [`GpuContext::with_optional_features`]).
    pub fn new(gpu: &GpuContext, log_blowup: u32) -> Option<Self> {
        if !gpu.device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return None;
        }
        let count = 4 * (1u32 << log_blowup) + 2;
        let query_set = gpu.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("commit timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count,
        });
        let bytes = 8 * count as u64;
        let resolve = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("timestamps resolve"),
            size: bytes,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("timestamps readback"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Some(Self { query_set, resolve, readback, count, log_blowup })
    }

    /// The writes of pass `k` (in the order the passes are recorded: stripe `j`'s extension `2j`,
    /// its hashing `2j + 1`, the levels above `2 n_stripes`).
    fn writes(&self, k: u32) -> wgpu::ComputePassTimestampWrites<'_> {
        wgpu::ComputePassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: Some(2 * k),
            end_of_pass_write_index: Some(2 * k + 1),
        }
    }

    /// Records the resolution of the queries (after the commitment's passes).
    pub fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.resolve_query_set(&self.query_set, 0..self.count, &self.resolve, 0);
        encoder.copy_buffer_to_buffer(&self.resolve, 0, &self.readback, 0, 8 * self.count as u64);
    }

    /// The pass times, after the submission of the resolution.
    pub fn read(&self, gpu: &GpuContext) -> PassTimes {
        let slice = self.readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |result| result.expect("map"));
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        let ticks: Vec<u64> =
            bytemuck::cast_slice(&slice.get_mapped_range().expect("mapped range")[..]).to_vec();
        self.readback.unmap();
        let period = gpu.queue.get_timestamp_period() as f64;
        let span = |k: usize| ticks[2 * k + 1].wrapping_sub(ticks[2 * k]) as f64 * period;
        let stripes = stripe_order(self.log_blowup)
            .enumerate()
            .map(|(j, stripe)| (stripe, span(2 * j), span(2 * j + 1)))
            .collect_vec();
        let n_passes = 2 * stripes.len() + 1;
        PassTimes {
            top: span(n_passes - 1),
            total: ticks[2 * n_passes - 1].wrapping_sub(ticks[0]) as f64 * period,
            stripes,
        }
    }
}

/// How a [`StripedCommitment`] lays out and dispatches its columns.
#[derive(Clone, Copy, Debug, Default)]
pub struct CommitOptions {
    /// The largest buffer of columns, in words (default: the device's largest binding): smaller
    /// spreads the columns over more buffers.
    pub max_buffer_words: Option<u64>,
    /// How the extension's dispatches take the batches (default: one dispatch per batch and
    /// pass).
    pub merge: Merge,
    /// No staging buffers for [`StripedCommitment::read`]: the caller reads the result back from
    /// the device buffers itself (a browser's GPU worker, through a ring of small mappable
    /// buffers, see `web/gpu`). [`StripedCommitment::encode_readback`] and
    /// [`StripedCommitment::read`] then panic.
    pub without_staging: bool,
}

/// The striped commitment of a tree of given shapes on the GPU (see the module docs): its
/// buffers, its stripes' extension and Merkle plans, built once, run any number of times.
pub struct StripedCommitment {
    log_sizes: Vec<u32>,
    log_blowup: u32,
    lifting_log_size: u32,
    layout: Arc<ColumnsLayout>,
    coeffs: GpuColumns,
    out: GpuColumns,
    extension: StripeExtension,
    tree_layout: TreeLayout,
    tree: wgpu::Buffer,
    states: StateBuffers,
    /// Per stripe (indexed by stripe).
    stripe_plans: Vec<Plan>,
    top: Plan,
    whole_copies: Vec<WholeCopy>,
    /// The words of a stripe of the columns kept whole.
    whole_words: u64,
    /// Per column: its word offset in a stripe's slice of `whole`, if kept whole.
    whole_slots: Vec<Option<u64>>,
    /// The columns kept whole, every stripe: stripe `s` at `s * whole_words`.
    whole: Option<wgpu::Buffer>,
    /// `None` when built [`CommitOptions::without_staging`].
    readback: Option<Readback>,
}

impl StripedCommitment {
    /// For polynomials of coefficients log sizes `log_sizes` (in polynomial order), with the
    /// twiddles of every stripe computed from `twiddles` (stwo's twiddle tree, covering the
    /// largest domain) and uploaded. Panics if `new_striped` would not take its striped path
    /// ([`takes_striped_path`]).
    pub fn new(
        gpu: &GpuContext,
        kernels: &CommitKernels,
        log_sizes: &[u32],
        log_blowup: u32,
        lifting_log_size: u32,
        twiddles: &TwiddleTree<SimdBackend>,
    ) -> Self {
        Self::with_options(
            gpu,
            kernels,
            log_sizes,
            log_blowup,
            lifting_log_size,
            twiddles,
            CommitOptions::default(),
        )
    }

    /// [`Self::new`] with `options`.
    pub fn with_options(
        gpu: &GpuContext,
        kernels: &CommitKernels,
        log_sizes: &[u32],
        log_blowup: u32,
        lifting_log_size: u32,
        twiddles: &TwiddleTree<SimdBackend>,
        options: CommitOptions,
    ) -> Self {
        let max_buffer_words = options.max_buffer_words.unwrap_or_else(|| max_buffer_words(gpu));
        assert!(
            takes_striped_path(log_sizes, log_blowup, lifting_log_size),
            "new_striped falls back to new for log sizes {log_sizes:?}, log blowup {log_blowup}, \
             lifting log size {lifting_log_size}"
        );
        let n_stripes = 1usize << log_blowup;
        let layout = Arc::new(ColumnsLayout::new(log_sizes, max_buffer_words));
        let coeffs = GpuColumns::alloc(gpu, layout.clone());
        let out = GpuColumns::alloc(gpu, layout.clone());
        let distinct = layout.log_sizes();
        let twiddle_words = (0..n_stripes)
            .into_par_iter()
            .map(|stripe| stripe_twiddle_words(&distinct, log_blowup, stripe, twiddles))
            .collect::<Vec<_>>();
        let extension = StripeExtension::with_twiddle_words_merged(
            gpu,
            &kernels.extension,
            &coeffs,
            &out,
            log_blowup,
            &twiddle_words,
            options.merge,
        );

        // The extension's batches are the tree's order: by increasing size, input order within a
        // size (`new_striped`'s `order`, a stable sort by size).
        let stripe_height = lifting_log_size - log_blowup;
        let sorted_sizes = layout
            .batches
            .iter()
            .flat_map(|batch| repeat_n(batch.log_size, batch.columns.len()))
            .collect_vec();
        let tree_layout = TreeLayout::new(
            stripe_height,
            log_blowup,
            stwo_min_stored_level(&sorted_sizes, stripe_height),
        );
        let tree = kernels.merkle.tree_buffer(gpu, &tree_layout);
        let batches = layout
            .batches
            .iter()
            .map(|batch| MerkleBatch {
                buffer: &out.buffers[batch.buffer],
                word_offset: batch.offset,
                log_size: batch.log_size,
                n_columns: batch.columns.len() as u32,
            })
            .collect_vec();
        let shapes = batches.iter().map(|batch| (batch.log_size, batch.n_columns)).collect_vec();
        let states = kernels.merkle.state_buffers(gpu, &shapes, stripe_height);
        let stripe_plans = (0..n_stripes as u32)
            .map(|stripe| {
                kernels.merkle.plan_stripe(gpu, &batches, &states, &tree, &tree_layout, stripe)
            })
            .collect_vec();
        let top = kernels.merkle.plan_top(gpu, &tree, &tree_layout);

        // The columns kept whole: every stripe's, side by side.
        let mut whole_copies = vec![];
        let mut whole_slots = vec![None; log_sizes.len()];
        let mut whole_words = 0;
        for batch in layout.batches.iter().filter(|batch| kept_whole(batch.log_size, log_blowup)) {
            whole_copies.push(WholeCopy {
                buffer: batch.buffer,
                src: batch.offset,
                dst: whole_words,
                words: batch.n_words(),
            });
            for (k, &i) in batch.columns.iter().enumerate() {
                whole_slots[i] = Some(whole_words + ((k as u64) << batch.log_size));
            }
            whole_words += batch.n_words();
        }
        let whole = (whole_words > 0)
            .then(|| gpu.storage("whole columns", n_stripes * whole_words as usize));

        let staging = |label: &str, words: u64| {
            gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (words * 4).max(4),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let readback = (!options.without_staging).then(|| Readback {
            out: layout
                .buffer_words
                .iter()
                .map(|&words| staging("stripe 0 readback", words))
                .collect(),
            whole: whole.as_ref().map(|whole| staging("whole columns readback", whole.size() / 4)),
            tree: staging("layers readback", tree_layout.total_words),
        });

        Self {
            log_sizes: log_sizes.to_vec(),
            log_blowup,
            lifting_log_size,
            layout,
            coeffs,
            out,
            extension,
            tree_layout,
            tree,
            states,
            stripe_plans,
            top,
            whole_copies,
            whole_words,
            whole_slots,
            whole,
            readback,
        }
    }

    pub fn n_stripes(&self) -> usize {
        1 << self.log_blowup
    }

    pub fn log_sizes(&self) -> &[u32] {
        &self.log_sizes
    }

    pub fn lifting_log_size(&self) -> u32 {
        self.lifting_log_size
    }

    pub fn layout(&self) -> &ColumnsLayout {
        &self.layout
    }

    pub fn tree_layout(&self) -> &TreeLayout {
        &self.tree_layout
    }

    /// The dispatches of a stripe: (extension, hashing); and of the levels above the stripes.
    pub fn n_dispatches(&self) -> (usize, usize, usize) {
        (
            self.extension.n_dispatches(),
            self.stripe_plans[0].n_dispatches(),
            self.top.n_dispatches(),
        )
    }

    /// The buffers the coefficients go to, as [`Self::layout`] places them.
    pub fn coefficient_buffers(&self) -> &[wgpu::Buffer] {
        &self.coeffs.buffers
    }

    /// The buffers of a stripe's evaluations, as [`Self::layout`] places them: stripe 0's once
    /// the commitment ran.
    pub fn evaluation_buffers(&self) -> &[wgpu::Buffer] {
        &self.out.buffers
    }

    /// The tree's stored levels, as [`Self::tree_layout`] places them.
    pub fn tree_buffer(&self) -> &wgpu::Buffer {
        &self.tree
    }

    /// The columns kept whole ([`kept_whole`]), every stripe's: stripe `s` of column `i` at word
    /// `s * whole_stride + whole_slot(i)`, `2^log_size` words.
    pub fn whole_buffer(&self) -> Option<&wgpu::Buffer> {
        self.whole.as_ref()
    }

    /// See [`Self::whole_buffer`]: `None` for a column not kept whole.
    pub fn whole_slot(&self, column: usize) -> Option<u64> {
        self.whole_slots[column]
    }

    /// See [`Self::whole_buffer`]: the words of a stripe of the columns kept whole.
    pub fn whole_stride(&self) -> u64 {
        self.whole_words
    }

    /// Frees the commitment's device and staging buffers now. In a browser a buffer's memory is
    /// otherwise held until its JS object is garbage collected (wgpu's WebGPU backend doesn't
    /// destroy buffers on drop). Nothing may be recorded or read after this.
    pub fn destroy(&self) {
        self.coeffs.destroy();
        self.out.destroy();
        self.extension.destroy();
        self.tree.destroy();
        self.states.destroy();
        for plan in self.stripe_plans.iter().chain([&self.top]) {
            plan.destroy();
        }
        if let Some(whole) = &self.whole {
            whole.destroy();
        }
        if let Some(readback) = &self.readback {
            for buffer in readback.out.iter().chain(&readback.whole).chain([&readback.tree]) {
                buffer.destroy();
            }
        }
    }

    /// The bytes of the device buffers (coefficients, one stripe's evaluations, twiddles, Merkle
    /// states, the tree's levels, the whole columns), and of the staging buffers of the readback.
    pub fn gpu_bytes(&self) -> GpuBytes {
        let sum = |buffers: &[wgpu::Buffer]| buffers.iter().map(wgpu::Buffer::size).sum::<u64>();
        GpuBytes {
            coefficients: sum(&self.coeffs.buffers),
            evaluations: sum(&self.out.buffers),
            twiddles: self.extension.twiddle_bytes(),
            states: self.states.bytes(),
            tree: self.tree.size(),
            whole: self.whole.as_ref().map_or(0, wgpu::Buffer::size),
            staging: self.readback.as_ref().map_or(0, |readback| {
                sum(&readback.out)
                    + readback.whole.as_ref().map_or(0, wgpu::Buffer::size)
                    + readback.tree.size()
            }),
        }
    }

    /// Queues the writes of the coefficients (`polys` of the log sizes given, in polynomial order,
    /// stwo's storage order). They take effect at the next submission.
    pub fn upload(&self, gpu: &GpuContext, polys: &[CircleCoefficients<SimdBackend>]) {
        assert_eq!(polys.len(), self.log_sizes.len());
        let words = polys
            .iter()
            .zip(&self.log_sizes)
            .map(|(poly, &log_size)| {
                assert_eq!(poly.log_size(), log_size);
                bytemuck::cast_slice::<BaseField, u32>(poly.coeffs.as_slice())
            })
            .collect_vec();
        self.coeffs.write(gpu, &words);
    }

    /// Records the extension of stripe `stripe` into the output buffers, in one compute pass.
    pub fn encode_extension(
        &self,
        kernels: &CommitKernels,
        encoder: &mut wgpu::CommandEncoder,
        stripe: usize,
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("extension"),
            timestamp_writes,
        });
        self.extension.record(&kernels.extension, &mut pass, stripe);
    }

    /// Records the hashing of stripe `stripe` (in the output buffers) into its subtree, in one
    /// compute pass, then the copies of its columns kept whole.
    pub fn encode_hashing(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        stripe: usize,
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) {
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("merkle stripe"),
                timestamp_writes,
            });
            self.stripe_plans[stripe].record(&mut pass);
        }
        if let Some(whole) = &self.whole {
            let base = stripe as u64 * self.whole_words;
            for copy in &self.whole_copies {
                encoder.copy_buffer_to_buffer(
                    &self.out.buffers[copy.buffer],
                    copy.src * 4,
                    whole,
                    (base + copy.dst) * 4,
                    copy.words * 4,
                );
            }
        }
    }

    /// Records the hashing of the levels above the stripes, in one compute pass.
    pub fn encode_top(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("merkle top"),
            timestamp_writes,
        });
        self.top.record(&mut pass);
    }

    /// Records the whole commitment (the coefficients uploaded): every stripe in
    /// [`stripe_order`], its extension and then its hashing, then the levels above, with
    /// timestamps at every pass if `timestamps`.
    pub fn encode(
        &self,
        kernels: &CommitKernels,
        encoder: &mut wgpu::CommandEncoder,
        timestamps: Option<&PassTimestamps>,
    ) {
        for (j, stripe) in stripe_order(self.log_blowup).enumerate() {
            let j = j as u32;
            self.encode_extension(kernels, encoder, stripe, timestamps.map(|t| t.writes(2 * j)));
            self.encode_hashing(encoder, stripe, timestamps.map(|t| t.writes(2 * j + 1)));
        }
        let n_passes = 2 * self.n_stripes() as u32;
        self.encode_top(encoder, timestamps.map(|t| t.writes(n_passes)));
        if let Some(timestamps) = timestamps {
            timestamps.resolve(encoder);
        }
    }

    /// Records the copies of the result into the staging buffers (after [`Self::encode`]): the
    /// output buffers (stripe 0), the columns kept whole, the tree's levels.
    pub fn encode_readback(&self, encoder: &mut wgpu::CommandEncoder) {
        let readback = self.readback.as_ref().expect("built without staging");
        for ((buffer, staging), &words) in
            self.out.buffers.iter().zip(&readback.out).zip(&self.layout.buffer_words)
        {
            encoder.copy_buffer_to_buffer(buffer, 0, staging, 0, words * 4);
        }
        if let (Some(whole), Some(staging)) = (&self.whole, &readback.whole) {
            encoder.copy_buffer_to_buffer(whole, 0, staging, 0, whole.size());
        }
        encoder.copy_buffer_to_buffer(
            &self.tree,
            0,
            &readback.tree,
            0,
            self.tree_layout.total_words * 4,
        );
    }

    /// Runs the commitment (the coefficients uploaded) and waits for it, without reading anything
    /// back.
    pub fn run(&self, gpu: &GpuContext, kernels: &CommitKernels) {
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        self.encode(kernels, &mut encoder, None);
        gpu.queue.submit([encoder.finish()]);
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    }

    /// Runs the commitment (the coefficients uploaded) and its readback, one submission, and
    /// reads the result.
    pub fn run_and_read(&self, gpu: &GpuContext, kernels: &CommitKernels) -> CommitOutput {
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        self.encode(kernels, &mut encoder, None);
        self.encode_readback(&mut encoder);
        gpu.queue.submit([encoder.finish()]);
        self.read(gpu)
    }

    /// The result, after the submission of [`Self::encode_readback`]: maps the staging buffers
    /// and copies them into stwo's columns.
    pub fn read(&self, gpu: &GpuContext) -> CommitOutput {
        let readback = self.readback.as_ref().expect("built without staging");
        let staging =
            readback.out.iter().chain(&readback.whole).chain([&readback.tree]).collect_vec();
        for buffer in &staging {
            buffer.slice(..).map_async(wgpu::MapMode::Read, |result| result.expect("map"));
        }
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        let output = {
            let views = staging
                .iter()
                .map(|buffer| buffer.slice(..).get_mapped_range().expect("mapped range"))
                .collect_vec();
            let words =
                views.iter().map(|view| bytemuck::cast_slice::<u8, u32>(&view[..])).collect_vec();
            let n_out = readback.out.len();
            let (out_words, rest) = words.split_at(n_out);
            let whole_words = readback.whole.as_ref().map(|_| rest[0]);
            let tree_words = *rest.last().unwrap();

            let mut layers = self.tree_layout.levels(tree_words);
            layers.reverse();
            let n_stripes = self.n_stripes();
            // Only plain data in the closure: wgpu's buffers aren't `Sync` on the web backend.
            let (layout, whole_slots, whole_stride) =
                (&self.layout, &self.whole_slots, self.whole_words);
            let evaluations = (0..self.log_sizes.len())
                .into_par_iter()
                .map(|i| {
                    let slot = layout.slots[i];
                    let len = 1usize << slot.log_size;
                    match whole_slots[i] {
                        None => {
                            let start = slot.offset as usize;
                            column_from_words(&out_words[slot.buffer][start..start + len])
                        }
                        Some(offset) => {
                            let whole = whole_words.unwrap();
                            // Safety: every value is written below.
                            let mut column = unsafe { BaseColumn::uninitialized(len * n_stripes) };
                            let values =
                                bytemuck::cast_slice_mut::<BaseField, u32>(column.as_mut_slice());
                            for (stripe, chunk) in values.chunks_exact_mut(len).enumerate() {
                                let start = (stripe as u64 * whole_stride + offset) as usize;
                                chunk.copy_from_slice(&whole[start..start + len]);
                            }
                            column
                        }
                    }
                })
                .collect();
            CommitOutput { layers, evaluations }
        };
        for buffer in staging {
            buffer.unmap();
        }
        output
    }

    /// [`CommitOutput`] as `new_striped` returns it: every column's kept evaluations on its
    /// stripe 0's domain, with the column's whole domain (or its whole evaluation), no
    /// coefficients; the layers as the commitment's.
    pub fn tree_prover(
        &self,
        output: CommitOutput,
    ) -> CommitmentTreeProver<SimdBackend, Blake2sM31MerkleChannel> {
        let polynomials = output
            .evaluations
            .into_iter()
            .zip(&self.log_sizes)
            .map(|(values, &log_size)| {
                let domain = CanonicCoset::new(log_size + self.log_blowup).circle_domain();
                let mut evals = CircleEvaluation::<SimdBackend, BaseField, BitReversedOrder>::new(
                    part_domain(domain, domain.log_size() - values.len().ilog2(), 0),
                    values,
                );
                evals.domain = domain;
                Poly::new(None, evals)
            })
            .collect();
        CommitmentTreeProver {
            polynomials,
            commitment: MerkleProverLifted { layers: output.layers },
        }
    }
}

/// The bytes of a commitment's buffers (see [`StripedCommitment::gpu_bytes`]).
#[derive(Clone, Copy, Debug)]
pub struct GpuBytes {
    pub coefficients: u64,
    pub evaluations: u64,
    pub twiddles: u64,
    pub states: u64,
    pub tree: u64,
    pub whole: u64,
    /// Host-visible (`MAP_READ`): the readback's staging buffers.
    pub staging: u64,
}

impl GpuBytes {
    /// The device buffers, the staging buffers left out.
    pub fn device(&self) -> u64 {
        self.coefficients + self.evaluations + self.twiddles + self.states + self.tree + self.whole
    }
}

/// A column of stwo's from its words.
fn column_from_words(words: &[u32]) -> BaseColumn {
    // Safety: every value is written below.
    let mut column = unsafe { BaseColumn::uninitialized(words.len()) };
    bytemuck::cast_slice_mut::<BaseField, u32>(column.as_mut_slice()).copy_from_slice(words);
    column
}

/// `CommitmentTreeProver::new_striped` on the GPU: the same arguments (stwo's twiddle tree,
/// covering the largest domain; no memory pool), the same result, layer for layer and column for
/// column. Panics where `new_striped` falls back to `CommitmentTreeProver::new`.
pub fn commit_striped(
    gpu: &GpuContext,
    kernels: &CommitKernels,
    polynomials: ColumnVec<CircleCoefficients<SimdBackend>>,
    log_blowup_factor: u32,
    twiddles: &TwiddleTree<SimdBackend>,
    lifting_log_size: u32,
) -> CommitmentTreeProver<SimdBackend, Blake2sM31MerkleChannel> {
    let log_sizes = polynomials.iter().map(|poly| poly.log_size()).collect_vec();
    let commitment = StripedCommitment::new(
        gpu,
        kernels,
        &log_sizes,
        log_blowup_factor,
        lifting_log_size,
        twiddles,
    );
    commitment.upload(gpu, &polynomials);
    // The queue holds a copy of the coefficients now.
    drop(polynomials);
    let output = commitment.run_and_read(gpu, kernels);
    commitment.tree_prover(output)
}
