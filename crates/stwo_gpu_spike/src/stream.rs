//! The striped commitment of a tree streamed through the GPU in groups of columns, every stripe
//! of a group at once: equal to `CommitmentTreeProver::new_striped`'s (and to
//! [`StripedCommitment`](crate::commit::StripedCommitment)'s), its upload overlapping its
//! compute.
//!
//! [`StripedCommitment`](crate::commit::StripedCommitment) goes stripe by stripe, so every
//! coefficient must be on the GPU before the first stripe's hashing can end: the upload (54 ms
//! for the Cairo base trace, PCIe-bound, the GPU idle) and the compute (61 ms) add up. Here the
//! columns, in the tree's order (by increasing size), go in two parts.
//!
//! The columns below the largest size are cut into groups of at most a ring's chunk (the buffers
//! of a [`ColumnsLayout`] of that capacity; a size may straddle two groups), and for each group in
//! turn:
//! - its coefficients go up through the [`UploadRing`] into the group buffer (the CPU writes them
//!   while the GPU works on the previous group);
//! - one submission evaluates every stripe of the group's columns at once ([`GroupedExtension`],
//!   stripe `s` into slot `s` of the evaluations' buffer), absorbs each stripe's columns into its
//!   leaves' hashing (every stripe has its own Merkle states, carried from group to group:
//!   [`GpuMerkle::plan_stripe_split`](crate::merkle::GpuMerkle::plan_stripe_split)), and copies the
//!   columns kept whole;
//! - stripe 0's columns are read back through the [`ReadbackRing`], in a submission of their own,
//!   and copied out by the CPU a few groups later, while the GPU works on later groups.
//!
//! The columns of the largest size, when it is the stripes' height (as in every tree a prover
//! commits), are the leaves' last absorption, which finalizes the leaves: cut into groups, every
//! stripe would carry a state per leaf (up to 24 words) from one to the next (1.5 GiB of states
//! for the Cairo base trace's 24 columns of 2^20, and the time to write and read them). So they
//! go stripe by stripe, as [`StripedCommitment`](crate::commit::StripedCommitment) does them:
//! their coefficients go up after the groups' (while the GPU works on the last groups), then for
//! each stripe, stripe 0 first, their extension and their absorption, which ends the stripe's
//! leaves; stripe 0's are read back while the GPU works on the other stripes.
//!
//! Then each stripe's levels, the levels above the stripes, and the readback of the columns kept
//! whole and of the tree's levels. The GPU's work is the same as stripe by stripe (each stripe's
//! columns absorbed in the tree's order, its subtree in its own slices of the levels): only its
//! order changes, and so nothing in the result.
//!
//! On wgpu's one queue the GPU still runs its copies and its compute one after the other (the
//! readback's 42 ms and the compute's 61 ms add up); the upload overlaps because the CPU does it,
//! writing into mapped memory. With the BAR full (the chunks in host memory), the GPU copies each
//! chunk across PCIe itself, and that copy adds up too.
//!
//! GPU memory (see [`StreamedCommitment::gpu_bytes`]): a group's coefficients and every stripe's
//! evaluations of a group (a chunk, 32 MiB, each), the largest size's coefficients and one
//! stripe's evaluations of them, every stripe's Merkle states, the twiddles, the tree's levels,
//! the columns kept whole.

use std::collections::VecDeque;
use std::iter::repeat_n;
use std::sync::Arc;
use std::time::{Duration, Instant};

use itertools::Itertools;
use stwo::core::fields::m31::BaseField;
use stwo::core::vcs::blake2_hash::Blake2sHash;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sM31MerkleChannel;
use stwo::prover::CommitmentTreeProver;
use stwo::prover::backend::Column;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::simd::column::BaseColumn;
use stwo::prover::poly::circle::CircleCoefficients;

use crate::GpuContext;
use crate::commit::{
    CommitBuffers, CommitKernels, CommitOutput, GpuBytes, WholeCopy, kept_whole,
    takes_striped_path, tree_prover,
};
use crate::extension::{
    ColumnsLayout, GpuColumns, GroupedExtension, Merge, StripeExtension, StripeTwiddles,
    max_buffer_words,
};
use crate::merkle::{
    ColumnBatch as MerkleBatch, Plan, StateBuffers, TreeLayout, stwo_min_stored_level,
};
use crate::transfer::{Readback, ReadbackRing, Ticket, Upload, UploadRing};

/// The alignment of a stripe's slot in the evaluations' buffer, in words (256 bytes).
const SLOT_ALIGN_WORDS: u64 = 64;

/// The log size of the columns that go stripe by stripe: the largest, if it is the stripes'
/// height (their absorption finalizes the leaves); otherwise none.
fn top_log_size(log_sizes: &[u32], log_blowup: u32, lifting_log_size: u32) -> Option<u32> {
    let max = *log_sizes.iter().max()?;
    (max + log_blowup == lifting_log_size).then_some(max)
}

/// The capacity of a group, in words, for a tree of these coefficients log sizes on `gpu`, with
/// rings of these chunks: as much as a chunk of either ring holds, and as every stripe's slot of
/// the evaluations' buffer allows within a binding. `None` if a column that would be grouped
/// does not fit (the tree is then committed stripe by stripe). The columns of the largest size,
/// when they go stripe by stripe, may take several chunks each.
pub fn group_words(
    gpu: &GpuContext,
    log_sizes: &[u32],
    log_blowup: u32,
    lifting_log_size: u32,
    upload_chunk: u64,
    readback_chunk: u64,
) -> Option<u64> {
    let top = top_log_size(log_sizes, log_blowup, lifting_log_size);
    let n_stripes = 1u64 << log_blowup;
    let words = (upload_chunk.min(readback_chunk) / 4).min(max_buffer_words(gpu) / n_stripes);
    let words = words / SLOT_ALIGN_WORDS * SLOT_ALIGN_WORDS;
    let largest_grouped = log_sizes.iter().copied().filter(|&l| Some(l) != top).max();
    let fits = largest_grouped.is_none_or(|l| words >= 1 << l);
    fits.then_some(words.max(SLOT_ALIGN_WORDS))
}

/// A column's place on the GPU: its index (in polynomial order), a buffer, a byte offset there.
pub type Place<'a> = (usize, &'a wgpu::Buffer, u64);

/// Some of a tree's columns: their indices (in polynomial order), and their layout (its slots
/// indexed as these).
struct Part {
    indices: Vec<usize>,
    layout: Arc<ColumnsLayout>,
}

impl Part {
    fn new(indices: Vec<usize>, log_sizes: &[u32], max_buffer_words: u64) -> Self {
        let sizes = indices.iter().map(|&i| log_sizes[i]).collect_vec();
        Self { indices, layout: Arc::new(ColumnsLayout::new(&sizes, max_buffer_words)) }
    }
}

/// The columns of the largest size, stripe by stripe (see the module docs).
struct Top {
    part: Part,
    coeffs: GpuColumns,
    out: GpuColumns,
    extension: StripeExtension,
}

/// The striped commitment of a tree of given shapes, streamed in groups (see the module docs):
/// its plans, on buffers of a [`CommitBuffers`].
pub struct StreamedCommitment {
    log_sizes: Vec<u32>,
    log_blowup: u32,
    /// The grouped columns: the layout's buffers are the groups.
    lower: Part,
    /// The words of a stripe's slot in `stripes`.
    slot_words: u64,
    group: wgpu::Buffer,
    stripes: wgpu::Buffer,
    extension: GroupedExtension,
    top: Option<Top>,
    tree_layout: TreeLayout,
    tree: wgpu::Buffer,
    states: Vec<StateBuffers>,
    /// Per stripe, per batch (the lower layout's, then the top's): the dispatches that absorb it.
    absorbs: Vec<Vec<Plan>>,
    /// Per stripe: its final dispatches (a leaves-only one, if any; its subtree's levels).
    finals: Vec<Plan>,
    above: Plan,
    /// The batches kept whole: the lower ones (`buffer`: the group), the top ones (`buffer`: the
    /// top layout's).
    lower_whole: Vec<WholeCopy>,
    top_whole: Vec<WholeCopy>,
    whole_words: u64,
    /// Per column: its word offset in a stripe's slice of `whole`, if kept whole.
    whole_slots: Vec<Option<u64>>,
    whole: Option<wgpu::Buffer>,
}

impl StreamedCommitment {
    /// For polynomials of coefficients log sizes `log_sizes` (in polynomial order), in groups of
    /// at most `group_words` words (see [`group_words`]), on `buffers` (grown if needed) and
    /// `twiddles` (of this blowup, covering these log sizes). Panics if `new_striped` would not
    /// take its striped path ([`takes_striped_path`]).
    #[allow(clippy::too_many_arguments)]
    pub fn with_buffers(
        gpu: &GpuContext,
        kernels: &CommitKernels,
        buffers: &mut CommitBuffers,
        twiddles: Arc<StripeTwiddles>,
        log_sizes: &[u32],
        log_blowup: u32,
        lifting_log_size: u32,
        group_words: u64,
    ) -> Self {
        assert!(
            takes_striped_path(log_sizes, log_blowup, lifting_log_size),
            "new_striped falls back to new for log sizes {log_sizes:?}, log blowup {log_blowup}, \
             lifting log size {lifting_log_size}"
        );
        assert_eq!(twiddles.log_blowup, log_blowup, "the twiddles of another blowup");
        let n_stripes = 1usize << log_blowup;
        let top_log_size = top_log_size(log_sizes, log_blowup, lifting_log_size);
        let (top_indices, lower_indices): (Vec<usize>, Vec<usize>) =
            (0..log_sizes.len()).partition(|&i| Some(log_sizes[i]) == top_log_size);
        let lower = Part::new(lower_indices, log_sizes, group_words);
        let slot_words = lower.layout.buffer_words.iter().copied().max().unwrap_or(0);
        let slot_words = slot_words.next_multiple_of(SLOT_ALIGN_WORDS).max(SLOT_ALIGN_WORDS);
        let (group, stripes) = buffers.groups(gpu, slot_words, n_stripes as u64 * slot_words);
        let extension = GroupedExtension::new(
            gpu,
            &kernels.extension,
            &lower.layout,
            &group,
            &stripes,
            slot_words,
            &twiddles,
        );
        let top = (!top_indices.is_empty()).then(|| {
            let part = Part::new(top_indices, log_sizes, max_buffer_words(gpu));
            let (coeffs, out) = buffers.top_columns(gpu, &part.layout);
            let extension = StripeExtension::with_twiddles(
                gpu,
                &kernels.extension,
                &coeffs,
                &out,
                twiddles.clone(),
                Merge::Off,
            );
            Top { part, coeffs, out, extension }
        });

        // The tree, its leaves' columns in the tree's order: the lower layout's, then the top's.
        let stripe_height = lifting_log_size - log_blowup;
        let all_batches = lower
            .layout
            .batches
            .iter()
            .chain(top.iter().flat_map(|top| &top.part.layout.batches))
            .collect_vec();
        let sorted_sizes = all_batches
            .iter()
            .flat_map(|batch| repeat_n(batch.log_size, batch.columns.len()))
            .collect_vec();
        let tree_layout = TreeLayout::new(
            stripe_height,
            log_blowup,
            stwo_min_stored_level(&sorted_sizes, stripe_height),
        );
        kernels.merkle.check_tree_buffer(tree_layout.total_words);
        let tree = buffers.tree_of(gpu, tree_layout.total_words);
        let shapes = all_batches
            .iter()
            .map(|batch| (batch.log_size, batch.columns.len() as u32))
            .collect_vec();
        let state_words = kernels.merkle.state_words(&shapes, stripe_height);
        let states = buffers.stripe_states(gpu, n_stripes, state_words);
        let (absorbs, finals): (Vec<_>, Vec<_>) = (0..n_stripes)
            .map(|stripe| {
                let lower_batches = lower.layout.batches.iter().map(|batch| MerkleBatch {
                    buffer: &stripes,
                    word_offset: stripe as u64 * slot_words + batch.offset,
                    log_size: batch.log_size,
                    n_columns: batch.columns.len() as u32,
                });
                let top_batches = top.iter().flat_map(|top| {
                    top.part.layout.batches.iter().map(|batch| MerkleBatch {
                        buffer: &top.out.buffers[batch.buffer],
                        word_offset: batch.offset,
                        log_size: batch.log_size,
                        n_columns: batch.columns.len() as u32,
                    })
                });
                kernels.merkle.plan_stripe_split(
                    gpu,
                    &lower_batches.chain(top_batches).collect_vec(),
                    &states[stripe],
                    &tree,
                    &tree_layout,
                    stripe as u32,
                )
            })
            .unzip();
        let above = kernels.merkle.plan_top(gpu, &tree, &tree_layout);

        // The columns kept whole: every stripe's, side by side.
        let mut whole_slots = vec![None; log_sizes.len()];
        let mut whole_words = 0;
        let mut whole_copies = |part: &Part| {
            let mut copies = vec![];
            for batch in part.layout.batches.iter().filter(|b| kept_whole(b.log_size, log_blowup)) {
                copies.push(WholeCopy {
                    buffer: batch.buffer,
                    src: batch.offset,
                    dst: whole_words,
                    words: batch.n_words(),
                });
                for (k, &i) in batch.columns.iter().enumerate() {
                    whole_slots[part.indices[i]] =
                        Some(whole_words + ((k as u64) << batch.log_size));
                }
                whole_words += batch.n_words();
            }
            copies
        };
        let lower_whole = whole_copies(&lower);
        let top_whole = top.as_ref().map_or(vec![], |top| whole_copies(&top.part));
        let whole = (whole_words > 0).then(|| buffers.whole(gpu, n_stripes as u64 * whole_words));

        Self {
            log_sizes: log_sizes.to_vec(),
            log_blowup,
            lower,
            slot_words,
            group,
            stripes,
            extension,
            top,
            tree_layout,
            tree,
            states,
            absorbs,
            finals,
            above,
            lower_whole,
            top_whole,
            whole_words,
            whole_slots,
            whole,
        }
    }

    pub fn n_stripes(&self) -> usize {
        1 << self.log_blowup
    }

    /// The groups of columns (the columns of the largest size, which go stripe by stripe, apart).
    pub fn n_groups(&self) -> usize {
        self.lower.layout.buffer_words.len()
    }

    /// The bytes of the device buffers (as [`GpuBytes`], twiddles apart): the coefficients' are a
    /// group's and the largest size's, the evaluations' every stripe's of a group and one
    /// stripe's of the largest size, the states every stripe's.
    pub fn gpu_bytes(&self) -> GpuBytes {
        let sum = |buffers: &[wgpu::Buffer]| buffers.iter().map(wgpu::Buffer::size).sum::<u64>();
        GpuBytes {
            coefficients: self.group.size()
                + self.top.as_ref().map_or(0, |top| sum(&top.coeffs.buffers)),
            evaluations: self.stripes.size()
                + self.top.as_ref().map_or(0, |top| sum(&top.out.buffers)),
            twiddles: 0,
            states: self.states.iter().map(StateBuffers::bytes).sum(),
            tree: self.tree.size(),
            whole: self.whole.as_ref().map_or(0, wgpu::Buffer::size),
        }
    }

    pub fn log_sizes(&self) -> &[u32] {
        &self.log_sizes
    }

    pub fn tree_layout(&self) -> &TreeLayout {
        &self.tree_layout
    }

    /// The tree's stored levels, as [`Self::tree_layout`] places them.
    pub fn tree_buffer(&self) -> &wgpu::Buffer {
        &self.tree
    }

    /// The columns kept whole, every stripe's, if any (see [`Self::whole_reads`]).
    pub fn whole_buffer(&self) -> Option<&wgpu::Buffer> {
        self.whole.as_ref()
    }

    /// Where group `group`'s coefficients go: each of its columns' place in the group buffer, in
    /// the buffer's order.
    pub fn group_uploads(&self, group: usize) -> Vec<Place<'_>> {
        self.group_batches(group)
            .flat_map(|(_, batch)| {
                batch.columns.iter().enumerate().map(move |(k, &i)| {
                    let offset = (batch.offset + ((k as u64) << batch.log_size)) * 4;
                    (self.lower.indices[i], &self.group, offset)
                })
            })
            .collect()
    }

    /// Where stripe 0's evaluations of group `group`'s columns are (those not kept whole), in
    /// the buffer's order: there from the group's work ([`Self::encode_group`]) until the next
    /// group's, so their readback must be submitted in between.
    pub fn group_reads(&self, group: usize) -> Vec<Place<'_>> {
        self.group_batches(group)
            .filter(|(_, batch)| !kept_whole(batch.log_size, self.log_blowup))
            .flat_map(|(_, batch)| {
                batch.columns.iter().enumerate().map(move |(k, &i)| {
                    let offset = (batch.offset + ((k as u64) << batch.log_size)) * 4;
                    (self.lower.indices[i], &self.stripes, offset)
                })
            })
            .collect()
    }

    /// Where the coefficients of the columns of the largest size go (none if they are grouped).
    pub fn top_uploads(&self) -> Vec<Place<'_>> {
        self.top_places(&|top: &Top| &top.coeffs, false)
    }

    /// Where stripe 0's evaluations of the columns of the largest size are (those not kept
    /// whole): there from stripe 0's [`Self::encode_top`] until stripe 1's, so their readback
    /// must be submitted in between.
    pub fn top_reads(&self) -> Vec<Place<'_>> {
        self.top_places(&|top: &Top| &top.out, true)
    }

    fn top_places<'a>(
        &'a self,
        columns: &dyn Fn(&'a Top) -> &'a GpuColumns,
        skip_whole: bool,
    ) -> Vec<Place<'a>> {
        let Some(top) = &self.top else { return vec![] };
        let buffers = &columns(top).buffers;
        top.part
            .layout
            .batches
            .iter()
            .filter(|batch| !(skip_whole && kept_whole(batch.log_size, self.log_blowup)))
            .flat_map(|batch| {
                batch.columns.iter().enumerate().map(move |(k, &i)| {
                    let offset = (batch.offset + ((k as u64) << batch.log_size)) * 4;
                    (top.part.indices[i], &buffers[batch.buffer], offset)
                })
            })
            .collect()
    }

    /// Where the columns kept whole are in [`Self::whole_buffer`], once every stripe's work and
    /// copies ran: `(column, stripe, byte offset)`, by column and stripe. Stripe `s` of a column of
    /// log size `l` is its kept evaluation's bytes `[s * 4 << l, (s + 1) * 4 << l)`.
    pub fn whole_reads(&self) -> Vec<(usize, usize, u64)> {
        let mut reads = vec![];
        for (i, slot) in self.whole_slots.iter().enumerate() {
            if let Some(offset) = slot {
                for stripe in 0..self.n_stripes() {
                    reads.push((i, stripe, (stripe as u64 * self.whole_words + offset) * 4));
                }
            }
        }
        reads
    }

    /// The dispatches of the whole commitment: (the extension's, the hashing's), over every
    /// group and stripe.
    pub fn n_dispatches(&self) -> (usize, usize) {
        let top =
            self.top.as_ref().map_or(0, |top| top.extension.n_dispatches() * self.n_stripes());
        let hashing = self.absorbs.iter().flatten().chain(&self.finals).chain([&self.above]);
        (self.extension.n_dispatches() + top, hashing.map(Plan::n_dispatches).sum())
    }

    /// Frees the buffers of the commitment's own plans now (the extensions' and the Merkle plans'
    /// parameters), not the [`CommitBuffers`] or the twiddles it was built on (see
    /// [`crate::commit::StripedCommitment::destroy_plans`]).
    pub fn destroy_plans(&self) {
        self.extension.destroy();
        if let Some(top) = &self.top {
            top.extension.destroy();
        }
        for plan in self.absorbs.iter().flatten().chain(&self.finals).chain([&self.above]) {
            plan.destroy();
        }
    }

    /// The lower layout's batches of group `group`, with their indices.
    fn group_batches(
        &self,
        group: usize,
    ) -> impl Iterator<Item = (usize, &crate::extension::ColumnBatch)> + Clone {
        self.lower.layout.batches.iter().enumerate().filter(move |(_, batch)| batch.buffer == group)
    }

    /// Records group `group`'s work: every stripe's extension and hashing of its columns (its
    /// coefficients in the group buffer), and the copies of its columns kept whole.
    pub fn encode_group(
        &self,
        kernels: &CommitKernels,
        encoder: &mut wgpu::CommandEncoder,
        group: usize,
    ) {
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("group extension and hashing"),
                timestamp_writes: None,
            });
            self.extension.record(&kernels.extension, &mut pass, group);
            // A batch of every stripe, one after the other: they share nothing but their reading
            // of the evaluations, so they need no barrier between them.
            for (b, _) in self.group_batches(group) {
                for absorbs in &self.absorbs {
                    absorbs[b].record(&mut pass);
                }
            }
        }
        if let Some(whole) = &self.whole {
            for copy in self.lower_whole.iter().filter(|copy| copy.buffer == group) {
                for stripe in 0..self.n_stripes() as u64 {
                    encoder.copy_buffer_to_buffer(
                        &self.stripes,
                        (stripe * self.slot_words + copy.src) * 4,
                        whole,
                        (stripe * self.whole_words + copy.dst) * 4,
                        copy.words * 4,
                    );
                }
            }
        }
    }

    /// Records stripe `stripe`'s work on the columns of the largest size (their coefficients in
    /// place): their extension, their absorption, the copies of those kept whole.
    pub fn encode_top(
        &self,
        kernels: &CommitKernels,
        encoder: &mut wgpu::CommandEncoder,
        stripe: usize,
    ) {
        let Some(top) = &self.top else { return };
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("top extension and hashing"),
                timestamp_writes: None,
            });
            top.extension.record(&kernels.extension, &mut pass, stripe);
            let n_lower = self.lower.layout.batches.len();
            for plan in &self.absorbs[stripe][n_lower..] {
                plan.record(&mut pass);
            }
        }
        if let Some(whole) = &self.whole {
            for copy in &self.top_whole {
                encoder.copy_buffer_to_buffer(
                    &top.out.buffers[copy.buffer],
                    copy.src * 4,
                    whole,
                    (stripe as u64 * self.whole_words + copy.dst) * 4,
                    copy.words * 4,
                );
            }
        }
    }

    /// Records every stripe's last dispatches and levels, and the levels above the stripes.
    pub fn encode_levels(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("levels"),
            timestamp_writes: None,
        });
        for plan in &self.finals {
            plan.record(&mut pass);
        }
        self.above.record(&mut pass);
    }

    /// Records the GPU's work of the whole commitment, without its transfers (on whatever the
    /// coefficients' buffers hold): to time the compute alone.
    pub fn encode_compute(&self, kernels: &CommitKernels, encoder: &mut wgpu::CommandEncoder) {
        for group in 0..self.n_groups() {
            self.encode_group(kernels, encoder, group);
        }
        for stripe in 0..self.n_stripes() {
            self.encode_top(kernels, encoder, stripe);
        }
        self.encode_levels(encoder);
    }

    /// The commitment of `polys` (of the log sizes given, in polynomial order): group by group,
    /// then the largest size stripe by stripe, the uploads through `upload`, the readbacks through
    /// `readback` (see the module docs). Every chunk of `readback` must be free.
    pub fn commit(
        &self,
        gpu: &GpuContext,
        kernels: &CommitKernels,
        upload: &mut UploadRing,
        readback: &mut ReadbackRing,
        polys: &[CircleCoefficients<SimdBackend>],
    ) -> CommitOutput {
        self.commit_timed(gpu, kernels, upload, readback, polys).0
    }

    /// [`Self::commit`], and the time of its phases: the groups, the columns of the largest
    /// size, the copies out of the chunks still being read back (while the GPU hashes the
    /// levels), the readback of the columns kept whole and the layers.
    pub fn commit_timed<'a>(
        &'a self,
        gpu: &GpuContext,
        kernels: &CommitKernels,
        upload: &mut UploadRing,
        readback: &mut ReadbackRing,
        polys: &'a [CircleCoefficients<SimdBackend>],
    ) -> (CommitOutput, [Duration; 4]) {
        let mut phases = [Duration::ZERO; 4];
        let mut start = Instant::now();
        let mut lap = |phase: usize| {
            phases[phase] = start.elapsed();
            start = Instant::now();
        };
        assert_eq!(polys.len(), self.log_sizes.len());
        for (poly, &log_size) in polys.iter().zip(&self.log_sizes) {
            assert_eq!(poly.log_size(), log_size);
        }
        let n_stripes = self.n_stripes();
        let coefficients =
            |i: usize| bytemuck::cast_slice::<BaseField, u8>(polys[i].coeffs.as_slice());
        let mut reads = Reads::new(self.log_sizes.len());

        let uploads = |places: Vec<Place<'a>>| {
            places
                .into_iter()
                .map(|(i, dst, dst_offset)| Upload { dst, dst_offset, src: coefficients(i) })
                .collect_vec()
        };
        for group in 0..self.n_groups() {
            // The group's coefficients.
            upload.upload(gpu, &uploads(self.group_uploads(group)));

            // Every stripe's extension and hashing of the group, the columns kept whole.
            let mut encoder = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("group") });
            self.encode_group(kernels, &mut encoder, group);
            gpu.queue.submit([encoder.finish()]);

            // Stripe 0's columns of the group (not kept whole), read back.
            reads.read(gpu, readback, &self.log_sizes, &self.group_reads(group));
        }
        lap(0);

        if self.top.is_some() {
            // The largest size's coefficients, while the GPU works on the last groups.
            upload.upload(gpu, &uploads(self.top_uploads()));
            // Stripe by stripe, stripe 0 first: its columns read back while the GPU works on the
            // other stripes.
            for stripe in 0..n_stripes {
                let mut encoder = gpu
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("top") });
                self.encode_top(kernels, &mut encoder, stripe);
                gpu.queue.submit([encoder.finish()]);
                if stripe == 0 {
                    // Started before stripe 1 overwrites them (the ring's chunks are copied out
                    // to make room if needed).
                    reads.read(gpu, readback, &self.log_sizes, &self.top_reads());
                }
            }
        }

        lap(1);

        // Every stripe's last dispatches and levels, the levels above: submitted before the last
        // chunks are copied out, so that the GPU computes while the CPU copies.
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("levels") });
        self.encode_levels(&mut encoder);
        gpu.queue.submit([encoder.finish()]);
        let mut evaluations = reads.drain(gpu, readback);
        lap(2);

        // The columns kept whole and the tree's levels, read back.
        let mut whole_columns = self
            .whole_slots
            .iter()
            .enumerate()
            .filter(|(_, slot)| slot.is_some())
            .map(|(i, _)| {
                // Safety: every value is read back below.
                (i, unsafe { BaseColumn::uninitialized(n_stripes << self.log_sizes[i]) })
            })
            .collect_vec();
        let mut levels = (0..=self.tree_layout.height())
            .map(|level| match self.tree_layout.level_offset(level) {
                Some(_) => vec![Blake2sHash::default(); self.tree_layout.level_len(level)],
                None => vec![],
            })
            .collect_vec();
        let mut pieces = vec![];
        let mut whole_reads = self.whole_reads().into_iter();
        for (i, column) in &mut whole_columns {
            let whole = self.whole.as_ref().unwrap();
            let words = bytemuck::cast_slice_mut::<BaseField, u8>(column.as_mut_slice());
            let len = words.len() / n_stripes;
            for (stripe, words) in words.chunks_exact_mut(len).enumerate() {
                let (column, s, src_offset) = whole_reads.next().unwrap();
                assert_eq!((column, s), (*i, stripe));
                pieces.push((0, src_offset, whole, words));
            }
        }
        for (level, hashes) in levels.iter_mut().enumerate() {
            if let Some(offset) = self.tree_layout.level_offset(level as u32) {
                let bytes = bytemuck::cast_slice_mut::<Blake2sHash, u8>(hashes);
                pieces.push((1, offset * 4, &self.tree, bytes));
            }
        }
        pieces.sort_by_key(|&(order, offset, _, _)| (order, offset));
        let pieces = pieces
            .into_iter()
            .map(|(_, src_offset, src, dst)| Readback { src, src_offset, dst })
            .collect_vec();
        readback.read(gpu, pieces);
        for (i, column) in whole_columns {
            evaluations[i] = Some(column);
        }
        lap(3);

        levels.reverse();
        let evaluations = evaluations
            .into_iter()
            .enumerate()
            .map(|(i, column)| column.unwrap_or_else(|| panic!("column {i} not read back")))
            .collect();
        (CommitOutput { layers: levels, evaluations }, phases)
    }

    /// [`CommitOutput`] as `new_striped` returns it (see [`tree_prover`]).
    pub fn tree_prover(
        &self,
        output: CommitOutput,
    ) -> CommitmentTreeProver<SimdBackend, Blake2sM31MerkleChannel> {
        tree_prover(&self.log_sizes, self.log_blowup, output)
    }
}

/// A piece of a chunk being read back: `(column, its byte offset, bytes)`.
type Piece = (usize, usize, usize);

/// Columns being read back through a [`ReadbackRing`]: allocated when their readback starts,
/// filled when their chunk is copied out (a column larger than a chunk across several chunks).
struct Reads {
    columns: Vec<Option<BaseColumn>>,
    /// The chunks being read back, oldest first: each with its pieces in the chunk's order.
    pending: VecDeque<(Ticket, Vec<Piece>)>,
}

impl Reads {
    fn new(n_columns: usize) -> Self {
        Self { columns: (0..n_columns).map(|_| None).collect(), pending: VecDeque::new() }
    }

    /// Starts reading back the columns `(index, buffer, byte offset)` (of log sizes
    /// `log_sizes[index]`), after the work submitted before, through as many chunks as they take;
    /// the oldest chunks are copied out first if none is free.
    fn read(
        &mut self,
        gpu: &GpuContext,
        ring: &mut ReadbackRing,
        log_sizes: &[u32],
        columns: &[(usize, &wgpu::Buffer, u64)],
    ) {
        let chunk = ring.chunk_bytes() as usize;
        let mut ranges = vec![];
        let mut pieces = vec![];
        let mut fill = 0;
        for &(i, buffer, offset) in columns {
            let bytes = 4usize << log_sizes[i];
            // Safety: every value is read back (see `finish`).
            self.columns[i] = Some(unsafe { BaseColumn::uninitialized(1 << log_sizes[i]) });
            let mut done = 0;
            while done < bytes {
                if fill == chunk {
                    self.start(gpu, ring, std::mem::take(&mut ranges), std::mem::take(&mut pieces));
                    fill = 0;
                }
                let n = (bytes - done).min(chunk - fill);
                ranges.push((buffer, offset + done as u64, n as u64));
                pieces.push((i, done, n));
                fill += n;
                done += n;
            }
        }
        if !ranges.is_empty() {
            self.start(gpu, ring, ranges, pieces);
        }
    }

    fn start(
        &mut self,
        gpu: &GpuContext,
        ring: &mut ReadbackRing,
        ranges: Vec<(&wgpu::Buffer, u64, u64)>,
        pieces: Vec<Piece>,
    ) {
        while !ring.has_free_chunk() {
            self.finish(gpu, ring);
        }
        self.pending.push_back((ring.start(gpu, &ranges), pieces));
    }

    /// Copies the oldest chunk out into its columns.
    fn finish(&mut self, gpu: &GpuContext, ring: &mut ReadbackRing) {
        let (ticket, pieces) = self.pending.pop_front().expect("no chunk being read back");
        // The destinations, in the chunk's order: each piece's range of its column (distinct
        // columns, a column's pieces in distinct chunks).
        let mut order = (0..pieces.len()).collect_vec();
        order.sort_by_key(|&k| pieces[k].0);
        let mut dsts: Vec<Option<&mut [u8]>> = (0..pieces.len()).map(|_| None).collect();
        let mut columns = self.columns.iter_mut().enumerate();
        for k in order {
            let (i, offset, bytes) = pieces[k];
            let column = loop {
                let (j, column) = columns.next().expect("pieces of distinct columns");
                if j == i {
                    break column;
                }
            };
            let words = column.as_mut().expect("allocated when started").as_mut_slice();
            dsts[k] =
                Some(&mut bytemuck::cast_slice_mut::<BaseField, u8>(words)[offset..offset + bytes]);
        }
        ring.finish(gpu, ticket, dsts.into_iter().map(Option::unwrap).collect());
    }

    /// Copies every chunk still being read back out; the columns read back.
    fn drain(mut self, gpu: &GpuContext, ring: &mut ReadbackRing) -> Vec<Option<BaseColumn>> {
        while !self.pending.is_empty() {
            self.finish(gpu, ring);
        }
        self.columns
    }
}
