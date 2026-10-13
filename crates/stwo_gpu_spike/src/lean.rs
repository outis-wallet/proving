//! EXPERIMENT (lean, grouped): a striped commitment within less memory than `new_striped`'s, for
//! unified-memory devices, with the host's coefficients the only copy (read in place, see
//! [`crate::committer`]).
//!
//! Stripe by stripe, and within a stripe group by group (the buffers of a [`ColumnsLayout`] of a
//! group's size, in the tree's order): each group's evaluations go into one small scratch buffer,
//! are absorbed into the stripe's leaves (the Merkle states carried from group to group,
//! [`GpuMerkle::plan_stripe_split`]), and the next group overwrites them. Stripe 0 goes last: after
//! each of its groups the scratch is mapped and the group's stripe 0 copied into its columns'
//! own coefficient columns (no other stripe needs them any more), which become the tree's. So the
//! GPU holds a group's evaluations, not a stripe's, and the host nothing besides its columns.
//!
//! A failure gives the polynomials back to `new_striped`: before stripe 0 untouched; during it,
//! the columns already holding stripe 0 are interpolated back to their coefficients on the CPU
//! (stripe 0 is an evaluation on as many points as the column has coefficients: they are unique),
//! exactly the ones they held.

use std::sync::Arc;

use itertools::Itertools;
use rayon::prelude::*;
use stwo::core::ColumnVec;
use stwo::core::fields::m31::BaseField;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::vcs::blake2_hash::Blake2sHash;
use stwo::prover::backend::Column;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::simd::column::BaseColumn;
use stwo::prover::poly::BitReversedOrder;
use stwo::prover::poly::circle::{CircleCoefficients, CircleEvaluation, PolyOps, part_domain};

use crate::GpuContext;
use crate::commit::{CommitKernels, CommitOutput, GpuBytes, WholeCopy, kept_whole, stripe_order};
use crate::extension::{ColumnSource, ColumnsLayout, GpuColumns, StripeExtension, StripeTwiddles};
use crate::merkle::{ColumnBatch as MerkleBatch, Plan, StateBuffers, TreeLayout, stwo_min_stored_level};
use crate::transfer::{Readback, ReadbackRing};

/// See the module docs.
pub struct GroupedCommitment {
    log_sizes: Vec<u32>,
    log_blowup: u32,
    layout: Arc<ColumnsLayout>,
    scratch: wgpu::Buffer,
    extension: StripeExtension,
    tree_layout: TreeLayout,
    tree: wgpu::Buffer,
    states: StateBuffers,
    /// Per stripe, per batch of the layout: the dispatches that absorb it.
    absorbs: Vec<Vec<Plan>>,
    /// Per stripe: its final dispatches (its subtree's levels).
    finals: Vec<Plan>,
    above: Plan,
    /// Per group: the copies of its batches kept whole (`dst` in a stripe's slice of `whole`).
    whole_copies: Vec<Vec<WholeCopy>>,
    whole_words: u64,
    whole_slots: Vec<Option<u64>>,
    whole: Option<wgpu::Buffer>,
    sources: Vec<wgpu::Buffer>,
}

/// A failure, and the stripe 0 groups already copied into the coefficients (their columns).
pub struct Failure {
    pub message: String,
    /// The columns holding their stripe 0 (to interpolate back).
    pub overwritten: Vec<usize>,
}

impl GroupedCommitment {
    /// For polynomials of `log_sizes` (in polynomial order) whose coefficients are in `sources`,
    /// in groups of at most `group_words` words. `top_whole`: whether the columns of the largest
    /// size go in one group of their own however large (the other groups then up to its size
    /// too), so that no Merkle state is kept at the stripe's full height; `None`: whichever
    /// layout needs less memory (its scratch and its two state buffers). The device must have
    /// `MAPPABLE_PRIMARY_BUFFERS` (the scratch is mapped).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        gpu: &GpuContext,
        kernels: &CommitKernels,
        twiddles: Arc<StripeTwiddles>,
        log_sizes: &[u32],
        log_blowup: u32,
        lifting_log_size: u32,
        group_words: u64,
        top_whole: Option<bool>,
        sources: Vec<ColumnSource>,
    ) -> Self {
        let n_stripes = 1usize << log_blowup;
        let stripe_height = lifting_log_size - log_blowup;
        let top = log_sizes.iter().copied().max().unwrap_or(0);
        let top_words = log_sizes.iter().filter(|&&s| s == top).count() as u64 * (1u64 << top);
        let layout_of = |whole: bool| {
            let words = if whole { group_words.max(top_words) } else { group_words };
            ColumnsLayout::grouped(log_sizes, words, whole)
        };
        // The words a layout needs on the GPU besides the tree's: its scratch and its states.
        let words_of = |layout: &ColumnsLayout| {
            let shapes = layout
                .batches
                .iter()
                .map(|batch| (batch.log_size, batch.columns.len() as u32))
                .collect_vec();
            layout.buffer_words.iter().copied().max().unwrap_or(0)
                + 2 * kernels.merkle.state_words(&shapes, stripe_height)
        };
        let layout = match top_whole {
            Some(whole) => layout_of(whole),
            None => {
                let (split, whole) = (layout_of(false), layout_of(true));
                if words_of(&whole) < words_of(&split) { whole } else { split }
            }
        };
        let layout = Arc::new(layout);
        let scratch_words = layout.buffer_words.iter().copied().max().unwrap_or(1).max(1);
        let scratch = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lean scratch"),
            size: scratch_words * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::MAP_READ
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out = GpuColumns {
            buffers: vec![scratch.clone(); layout.buffer_words.len()],
            layout: layout.clone(),
        };
        let extension = StripeExtension::with_sources(gpu, &kernels.extension, &sources, &out, twiddles);

        let sorted_sizes = layout
            .batches
            .iter()
            .flat_map(|batch| std::iter::repeat_n(batch.log_size, batch.columns.len()))
            .collect_vec();
        let tree_layout = TreeLayout::new(
            stripe_height,
            log_blowup,
            stwo_min_stored_level(&sorted_sizes, stripe_height),
        );
        kernels.merkle.check_tree_buffer(tree_layout.total_words);
        let tree = gpu.storage("lean tree", tree_layout.total_words as usize);
        let batches = layout
            .batches
            .iter()
            .map(|batch| MerkleBatch {
                buffer: &scratch,
                word_offset: batch.offset,
                log_size: batch.log_size,
                n_columns: batch.columns.len() as u32,
            })
            .collect_vec();
        let shapes = batches.iter().map(|batch| (batch.log_size, batch.n_columns)).collect_vec();
        let states = StateBuffers::new(gpu, kernels.merkle.state_words(&shapes, stripe_height));
        let (absorbs, finals): (Vec<_>, Vec<_>) = (0..n_stripes as u32)
            .map(|stripe| {
                kernels.merkle.plan_stripe_split(gpu, &batches, &states, &tree, &tree_layout, stripe)
            })
            .unzip();
        let above = kernels.merkle.plan_top(gpu, &tree, &tree_layout);

        let mut whole_copies = vec![vec![]; layout.buffer_words.len()];
        let mut whole_slots = vec![None; log_sizes.len()];
        let mut whole_words = 0;
        for batch in layout.batches.iter().filter(|batch| kept_whole(batch.log_size, log_blowup)) {
            whole_copies[batch.buffer].push(WholeCopy {
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
            .then(|| gpu.storage("lean whole columns", (n_stripes as u64 * whole_words) as usize));
        Self {
            log_sizes: log_sizes.to_vec(),
            log_blowup,
            layout,
            scratch,
            extension,
            tree_layout,
            tree,
            states,
            absorbs,
            finals,
            above,
            whole_copies,
            whole_words,
            whole_slots,
            whole,
            sources: sources.into_iter().map(|source| source.buffer).dedup().collect(),
        }
    }

    pub fn n_groups(&self) -> usize {
        self.layout.buffer_words.len()
    }

    /// The groups, for a trace: per group its words (MiB) and batches (`2^log_size x columns`).
    pub fn describe(&self) -> String {
        (0..self.n_groups())
            .map(|g| {
                let batches = self
                    .layout
                    .batches
                    .iter()
                    .filter(|batch| batch.buffer == g)
                    .map(|batch| format!("2^{}x{}", batch.log_size, batch.columns.len()))
                    .join("+");
                format!("[{} MiB: {batches}]", (self.layout.buffer_words[g] * 4) >> 20)
            })
            .join(" ")
    }

    pub fn gpu_bytes(&self) -> GpuBytes {
        GpuBytes {
            coefficients: 0,
            evaluations: self.scratch.size(),
            twiddles: self.extension.twiddle_bytes(),
            states: self.states.bytes(),
            tree: self.tree.size(),
            whole: self.whole.as_ref().map_or(0, wgpu::Buffer::size),
        }
    }

    /// Records group `group` of stripe `stripe`: its extension into the scratch, its absorption,
    /// the copies of its columns kept whole.
    fn encode_group(&self, kernels: &CommitKernels, encoder: &mut wgpu::CommandEncoder, stripe: usize, group: usize) {
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("lean extension"),
                timestamp_writes: None,
            });
            self.extension.record_buffer(&kernels.extension, &mut pass, stripe, group);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("lean absorb"),
                timestamp_writes: None,
            });
            for (b, batch) in self.layout.batches.iter().enumerate() {
                if batch.buffer == group {
                    self.absorbs[stripe][b].record(&mut pass);
                }
            }
        }
        if let Some(whole) = &self.whole {
            let base = stripe as u64 * self.whole_words;
            for copy in &self.whole_copies[group] {
                encoder.copy_buffer_to_buffer(
                    &self.scratch,
                    copy.src * 4,
                    whole,
                    (base + copy.dst) * 4,
                    copy.words * 4,
                );
            }
        }
    }

    /// The commitment of `polys` (whose coefficients the sources are): see the module docs.
    /// Tests of the failure path: `fault_submit` fails right after the first stripe is submitted
    /// (the GPU still reading the host's columns); `fault_after`, after that many groups of
    /// stripe 0 have been copied into the coefficients (all of them: `n_groups`).
    pub fn run(
        &self,
        gpu: &GpuContext,
        kernels: &CommitKernels,
        ring: &mut ReadbackRing,
        polys: &mut ColumnVec<CircleCoefficients<SimdBackend>>,
        fault_submit: bool,
        fault_after: Option<usize>,
    ) -> Result<CommitOutput, Failure> {
        let mut overwritten = vec![];
        let verbose = std::env::var("STWO_GPU_LEAN_VERBOSE").is_ok_and(|v| v == "1");
        let start = std::time::Instant::now();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let poll = || gpu.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
            for stripe in stripe_order(self.log_blowup) {
                if stripe != 0 {
                    let mut encoder = gpu.device.create_command_encoder(&Default::default());
                    for group in 0..self.n_groups() {
                        self.encode_group(kernels, &mut encoder, stripe, group);
                    }
                    self.record_final(&mut encoder, stripe);
                    gpu.queue.submit([encoder.finish()]);
                    assert!(!fault_submit, "STWO_GPU_LEAN_FAULT: a fault with the GPU at work");
                    continue;
                }
                if verbose {
                    poll();
                    eprintln!("lean grouped: stripes 1.. done at {:.3} s", start.elapsed().as_secs_f64());
                }
                for group in 0..self.n_groups() {
                    let mut encoder = gpu.device.create_command_encoder(&Default::default());
                    self.encode_group(kernels, &mut encoder, 0, group);
                    gpu.queue.submit([encoder.finish()]);
                    if fault_after == Some(group) {
                        panic!("STWO_GPU_LEAN_FAULT: a fault in stripe 0, after {group} groups");
                    }
                    // The group's stripe 0, into its columns' coefficients (the GPU is done with
                    // them: stripe 0 is the last stripe).
                    let words = self.layout.buffer_words[group];
                    let slice = self.scratch.slice(..words * 4);
                    let mapped = Arc::new(std::sync::Mutex::new(None));
                    let flag = mapped.clone();
                    slice.map_async(wgpu::MapMode::Read, move |r| *flag.lock().unwrap() = Some(r.is_ok()));
                    poll();
                    assert!(*mapped.lock().unwrap() == Some(true), "the scratch failed to map");
                    let view = slice.get_mapped_range().expect("mapped");
                    let columns = self
                        .layout
                        .batches
                        .iter()
                        .filter(|batch| batch.buffer == group && !kept_whole(batch.log_size, self.log_blowup))
                        .flat_map(|batch| {
                            batch.columns.iter().enumerate().map(move |(k, &i)| {
                                (i, (batch.offset + ((k as u64) << batch.log_size)) as usize * 4)
                            })
                        })
                        .collect_vec();
                    // Disjoint columns of `polys`, in parallel.
                    let targets: Vec<(usize, &mut BaseColumn)> = {
                        let mut wanted = vec![None; polys.len()];
                        for &(i, offset) in &columns {
                            wanted[i] = Some(offset);
                        }
                        polys
                            .iter_mut()
                            .enumerate()
                            .filter_map(|(i, poly)| wanted[i].map(|offset| (offset, &mut poly.coeffs)))
                            .collect()
                    };
                    let source: &[u8] = &view;
                    targets.into_par_iter().for_each(|(offset, column)| {
                        let bytes = bytemuck::cast_slice_mut::<BaseField, u8>(column.as_mut_slice());
                        let len = bytes.len();
                        bytes.copy_from_slice(&source[offset..offset + len]);
                    });
                    drop(view);
                    self.scratch.unmap();
                    overwritten.extend(columns.iter().map(|&(i, _)| i));
                }
                if fault_after == Some(self.n_groups()) {
                    panic!("STWO_GPU_LEAN_FAULT: a fault after every group of stripe 0");
                }
                let mut encoder = gpu.device.create_command_encoder(&Default::default());
                self.record_final(&mut encoder, 0);
                self.above_record(&mut encoder);
                gpu.queue.submit([encoder.finish()]);
            }
            poll();
            if verbose {
                eprintln!("lean grouped: stripe 0 done at {:.3} s", start.elapsed().as_secs_f64());
            }
            self.read_rest(gpu, ring)
        }));
        match result {
            Ok((levels, mut whole_columns)) => {
                // Every column's evaluation: stripe 0 in its own column, or kept whole.
                let evaluations = polys
                    .iter_mut()
                    .enumerate()
                    .map(|(i, poly)| match whole_columns.iter_mut().find(|(j, _)| *j == i) {
                        Some((_, column)) => std::mem::replace(column, BaseColumn::zeros(0)),
                        None => std::mem::replace(&mut poly.coeffs, BaseColumn::zeros(0)),
                    })
                    .collect();
                Ok(CommitOutput { layers: levels, evaluations })
            }
            Err(panic) => {
                let message = panic
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| panic.downcast_ref::<&str>().map(|m| m.to_string()))
                    .unwrap_or_else(|| "a panic".to_string());
                Err(Failure { message, overwritten })
            }
        }
    }

    fn record_final(&self, encoder: &mut wgpu::CommandEncoder, stripe: usize) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("lean stripe levels"),
            timestamp_writes: None,
        });
        self.finals[stripe].record(&mut pass);
    }

    fn above_record(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("lean top levels"),
            timestamp_writes: None,
        });
        self.above.record(&mut pass);
    }

    /// The levels and the columns kept whole (new columns), through `ring`.
    fn read_rest(&self, gpu: &GpuContext, ring: &mut ReadbackRing) -> (Vec<Vec<Blake2sHash>>, Vec<(usize, BaseColumn)>) {
        let n_stripes = 1usize << self.log_blowup;
        let mut levels = (0..=self.tree_layout.height())
            .map(|level| match self.tree_layout.level_offset(level) {
                Some(_) => vec![Blake2sHash::default(); self.tree_layout.level_len(level)],
                None => vec![],
            })
            .collect_vec();
        let mut whole_columns = self
            .whole_slots
            .iter()
            .enumerate()
            .filter_map(|(i, slot)| {
                // Safety: every value is read back below.
                slot.map(|_| (i, unsafe { BaseColumn::uninitialized(n_stripes << self.log_sizes[i]) }))
            })
            .collect_vec();
        let mut pieces = vec![];
        if let Some(whole) = &self.whole {
            for (i, column) in whole_columns.iter_mut() {
                let offset = self.whole_slots[*i].unwrap();
                let words = bytemuck::cast_slice_mut::<BaseField, u8>(column.as_mut_slice());
                let len = words.len() / n_stripes;
                for (stripe, words) in words.chunks_exact_mut(len).enumerate() {
                    let src_offset = (stripe as u64 * self.whole_words + offset) * 4;
                    pieces.push((0, src_offset, whole, words));
                }
            }
        }
        for (level, hashes) in levels.iter_mut().enumerate() {
            if let Some(offset) = self.tree_layout.level_offset(level as u32) {
                pieces.push((1, offset * 4, &self.tree, bytemuck::cast_slice_mut::<Blake2sHash, u8>(hashes)));
            }
        }
        pieces.sort_by_key(|&(order, offset, _, _)| (order, offset));
        ring.read(gpu, pieces.into_iter().map(|(_, src_offset, src, dst)| Readback { src, src_offset, dst }).collect());
        levels.reverse();
        (levels, whole_columns)
    }

    /// Frees the GPU's side of the sources (before the host's columns are written or freed),
    /// and everything else.
    pub fn destroy(&self) {
        for source in &self.sources {
            source.destroy();
        }
        self.extension.destroy();
        self.extension.twiddles().destroy();
        self.scratch.destroy();
        self.tree.destroy();
        self.states.destroy();
        if let Some(whole) = &self.whole {
            whole.destroy();
        }
    }
}

/// The coefficients of the columns `overwritten` of `polys`, which hold their stripe 0 (the
/// evaluation on the first part of their committed domain, bit-reversed): interpolated back, as
/// they were (unique: as many points as coefficients).
pub fn restore_coefficients(
    polys: &mut ColumnVec<CircleCoefficients<SimdBackend>>,
    overwritten: &[usize],
    log_blowup: u32,
) {
    let stripe_domain = |log_size: u32| {
        part_domain(CanonicCoset::new(log_size + log_blowup).circle_domain(), log_blowup, 0)
    };
    // The twiddles, once per log size.
    let twiddles: std::collections::BTreeMap<u32, _> = overwritten
        .iter()
        .map(|&i| polys[i].log_size())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .map(|log_size| (log_size, SimdBackend::precompute_twiddles(stripe_domain(log_size).half_coset)))
        .collect();
    let restore = |poly: &mut CircleCoefficients<SimdBackend>| {
        let log_size = poly.log_size();
        let values = std::mem::replace(&mut poly.coeffs, BaseColumn::zeros(0));
        let evaluation = CircleEvaluation::<SimdBackend, BaseField, BitReversedOrder>::new(
            stripe_domain(log_size),
            values,
        );
        *poly = evaluation.interpolate_with_twiddles(&twiddles[&log_size]);
        assert_eq!(poly.log_size(), log_size);
    };
    let mut wanted = vec![false; polys.len()];
    for &i in overwritten {
        wanted[i] = true;
    }
    polys
        .par_iter_mut()
        .enumerate()
        .filter(|(i, _)| wanted[*i])
        .for_each(|(_, poly)| restore(poly));
}
