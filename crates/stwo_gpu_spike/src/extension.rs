//! The polynomial extension of a commitment on the GPU: the coefficients of a tree's columns,
//! resident on the GPU, evaluated on one stripe of their domains at a time, word for word as
//! stwo's `SimdBackend` evaluates them in `CommitmentTreeProver::new_striped`:
//!
//! ```text
//! domain = CanonicCoset::new(log_size + log_blowup).circle_domain()
//! twiddles = SimdBackend::part_twiddles(domain, log_blowup, stripe, tree)
//! SimdBackend::evaluate_into(poly, part_domain(domain, log_blowup, stripe), &twiddles, buffer)
//! ```
//!
//! The pieces:
//! - [`ColumnsLayout`]: where each column lives on the GPU. Columns are grouped by log size into
//!   [`ColumnBatch`]es, each contiguous in one buffer, one batch per dispatch of a pass. A stripe's
//!   evaluation has the layout of the coefficients (a column's stripe has as many values as its
//!   coefficients), so [`GpuColumns`] holds either.
//! - [`stripe_twiddle_words`]: stwo's twiddles of a stripe (from `part_twiddles`, as the CPU uses
//!   them), for every log size of a tree, laid out for the kernels.
//! - [`ExtensionKernels`]: the compute pipelines.
//! - [`StripeExtension`]: the dispatches for one tree, from a [`GpuColumns`] of coefficients into a
//!   [`GpuColumns`] of evaluations, with the twiddles of every stripe uploaded once.
//!
//! The output contract, for kernels that consume a stripe on the GPU (the Merkle hashing): the
//! values of input column `i` are the `2^log_size` words at word offset `slots[i].offset` of
//! buffer `slots[i].buffer`, in bit-reversed order, canonical (in `[0, P)`), identical to the
//! `Col<SimdBackend, BaseField>` that `evaluate_into` returns for that stripe. The buffers have
//! `STORAGE | COPY_SRC | COPY_DST` usage. Batches go by increasing log size (the order in which
//! `new_striped` hashes the columns), and keep the input order within a log size.
//!
//! The FFT (see `shaders/extension.wgsl`): a column of `2^L` coefficients takes one pass if
//! `L <= TILE_LOG` (12), and otherwise a pass for the low 12 layers plus one or two for the others
//! (up to 8 layers each, two passes up to `L = 20`, three up to 28). A pass reads and writes each
//! word once; a workgroup works on a tile of 4096 words (16 KiB of workgroup memory) through
//! rounds of up to 4 layers held in registers (radix 16). All the columns of a batch (all the
//! columns of a log size, up to a buffer's capacity) go in one dispatch per pass.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::poly::utils::domain_line_twiddles_from_tree;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::simd::fft::CACHED_FFT_LOG_SIZE;
use stwo::prover::poly::circle::{PolyOps, part_domain};
use stwo::prover::poly::twiddles::TwiddleTree;
use wgpu::util::DeviceExt;

use crate::{GpuContext, M31_WGSL};

/// The log number of words of a workgroup's tile (16 KiB, WebGPU's default workgroup storage).
pub const TILE_LOG: u32 = 12;
/// The invocations of a workgroup (WebGPU's default maximum).
pub const WORKGROUP_SIZE: u32 = 256;
/// The smallest log size the kernels take, that of `new_striped`.
pub const MIN_LOG_SIZE: u32 = 4;
/// The largest log size the kernels take (the batch indices are 32-bit words).
pub const MAX_LOG_SIZE: u32 = 28;
/// The most layers of a pass other than the lowest: its tile keeps 2^(12 - 8) = 16 consecutive
/// words (64 bytes) together, so that its loads and stores stay coalesced.
const MAX_HIGH_PASS_LAYERS: u32 = TILE_LOG - 4;
/// The alignment of a batch in its buffer, in words (256 bytes, any storage offset alignment).
const BATCH_ALIGN_WORDS: u64 = 64;
/// The alignment of a log size's twiddles in a stripe's twiddle buffer, in words.
const TWIDDLE_ALIGN_WORDS: usize = 64;

const SHADER_TEMPLATE: &str = include_str!("shaders/extension.wgsl");

/// Where a column lives: `2^log_size` words at word `offset` of buffer `buffer`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColumnSlot {
    pub buffer: usize,
    pub offset: u64,
    pub log_size: u32,
}

/// Columns of one log size, one after the other from word `offset` of buffer `buffer`: the input
/// indices `columns`, in this order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ColumnBatch {
    pub buffer: usize,
    pub offset: u64,
    pub log_size: u32,
    pub columns: Vec<usize>,
}

impl ColumnBatch {
    pub fn n_words(&self) -> u64 {
        (self.columns.len() as u64) << self.log_size
    }
}

/// The places of a tree's columns on the GPU (see the module docs).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ColumnsLayout {
    /// The size of each buffer, in words.
    pub buffer_words: Vec<u64>,
    /// Per input column.
    pub slots: Vec<ColumnSlot>,
    /// By increasing log size, then buffer.
    pub batches: Vec<ColumnBatch>,
}

impl ColumnsLayout {
    /// Columns of the given log sizes, in buffers of at most `max_buffer_words` words.
    pub fn new(log_sizes: &[u32], max_buffer_words: u64) -> Self {
        let mut order = (0..log_sizes.len()).collect::<Vec<_>>();
        order.sort_by_key(|&i| (log_sizes[i], i));
        let mut buffer_words: Vec<u64> = vec![];
        let mut slots = vec![ColumnSlot { buffer: 0, offset: 0, log_size: 0 }; log_sizes.len()];
        let mut batches: Vec<ColumnBatch> = vec![];
        for i in order {
            let log_size = log_sizes[i];
            assert!(
                (MIN_LOG_SIZE..=MAX_LOG_SIZE).contains(&log_size),
                "column {i}: log size {log_size} out of {MIN_LOG_SIZE}..={MAX_LOG_SIZE}"
            );
            let len = 1u64 << log_size;
            assert!(len <= max_buffer_words, "column {i} (log size {log_size}) exceeds a buffer");
            let continues = batches.last().is_some_and(|batch| {
                batch.log_size == log_size && buffer_words[batch.buffer] + len <= max_buffer_words
            });
            if !continues {
                let fill =
                    buffer_words.last().map(|&fill| fill.next_multiple_of(BATCH_ALIGN_WORDS));
                let buffer = match fill {
                    Some(fill) if fill + len <= max_buffer_words => buffer_words.len() - 1,
                    _ => {
                        buffer_words.push(0);
                        buffer_words.len() - 1
                    }
                };
                buffer_words[buffer] = buffer_words[buffer].next_multiple_of(BATCH_ALIGN_WORDS);
                batches.push(ColumnBatch {
                    buffer,
                    offset: buffer_words[buffer],
                    log_size,
                    columns: vec![],
                });
            }
            let batch = batches.last_mut().unwrap();
            slots[i] =
                ColumnSlot { buffer: batch.buffer, offset: buffer_words[batch.buffer], log_size };
            buffer_words[batch.buffer] += len;
            batch.columns.push(i);
        }
        Self { buffer_words, slots, batches }
    }

    /// The distinct log sizes.
    pub fn log_sizes(&self) -> BTreeSet<u32> {
        self.slots.iter().map(|slot| slot.log_size).collect()
    }

    pub fn n_words(&self) -> u64 {
        self.slots.iter().map(|slot| 1u64 << slot.log_size).sum()
    }
}

/// Columns on the GPU, coefficients or the evaluations of a stripe, placed by `layout`.
pub struct GpuColumns {
    pub buffers: Vec<wgpu::Buffer>,
    pub layout: Arc<ColumnsLayout>,
}

impl GpuColumns {
    /// Uninitialized buffers for `layout`.
    pub fn alloc(gpu: &GpuContext, layout: Arc<ColumnsLayout>) -> Self {
        let buffers = layout
            .buffer_words
            .iter()
            .enumerate()
            .map(|(i, &words)| {
                gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("columns {i}")),
                    size: (words * 4).max(4),
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();
        Self { buffers, layout }
    }

    /// Queues the writes of the columns' words (`columns[i]` for input column `i`).
    pub fn write(&self, gpu: &GpuContext, columns: &[&[u32]]) {
        assert_eq!(columns.len(), self.layout.slots.len());
        // One staging write per batch: its columns are contiguous on the GPU.
        for batch in &self.layout.batches {
            let size = wgpu::BufferSize::new(batch.n_words() * 4).unwrap();
            let mut view = gpu
                .queue
                .write_buffer_with(&self.buffers[batch.buffer], batch.offset * 4, size)
                .expect("write_buffer_with");
            let len = 4usize << batch.log_size;
            for (k, &i) in batch.columns.iter().enumerate() {
                assert_eq!(columns[i].len() * 4, len);
                view.slice(k * len..(k + 1) * len)
                    .copy_from_slice(bytemuck::cast_slice(columns[i]));
            }
        }
    }

    /// Frees the buffers now (see `StripedCommitment::destroy`).
    pub fn destroy(&self) {
        for buffer in &self.buffers {
            buffer.destroy();
        }
    }

    /// Buffers to read the columns back through (see [`Self::read_into`]).
    pub fn staging(&self, gpu: &GpuContext) -> Vec<wgpu::Buffer> {
        self.layout
            .buffer_words
            .iter()
            .map(|&words| {
                gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("readback"),
                    size: (words * 4).max(4),
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect()
    }

    /// Every column's words, after all submitted work.
    pub fn read(&self, gpu: &GpuContext) -> Vec<Vec<u32>> {
        self.read_into(gpu, &self.staging(gpu))
    }

    /// Every column's words, after all submitted work, through `staging` (from
    /// [`Self::staging`], unmapped).
    pub fn read_into(&self, gpu: &GpuContext, staging: &[wgpu::Buffer]) -> Vec<Vec<u32>> {
        use rayon::prelude::*;
        assert_eq!(staging.len(), self.buffers.len());
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        for ((buffer, staging), &words) in
            self.buffers.iter().zip(staging).zip(&self.layout.buffer_words)
        {
            encoder.copy_buffer_to_buffer(buffer, 0, staging, 0, words * 4);
        }
        gpu.queue.submit([encoder.finish()]);
        for staging in staging {
            staging.slice(..).map_async(wgpu::MapMode::Read, |result| result.expect("map"));
        }
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        let mut columns = vec![vec![]; self.layout.slots.len()];
        for (b, staging) in staging.iter().enumerate() {
            {
                let view = staging.slice(..).get_mapped_range().expect("mapped range");
                let words: &[u32] = bytemuck::cast_slice(&view[..]);
                columns.par_iter_mut().zip(&self.layout.slots).for_each(|(column, slot)| {
                    if slot.buffer == b {
                        let start = slot.offset as usize;
                        *column = words[start..start + (1 << slot.log_size)].to_vec();
                    }
                });
            }
            staging.unmap();
        }
        columns
    }
}

/// The largest buffer the kernels can bind on `gpu`, in words.
pub fn max_buffer_words(gpu: &GpuContext) -> u64 {
    let limits = gpu.device.limits();
    let bytes = limits.max_buffer_size.min(limits.max_storage_buffer_binding_size);
    (bytes / 4).min(1 << 30) / BATCH_ALIGN_WORDS * BATCH_ALIGN_WORDS
}

/// The word offset of each log size's twiddles in a stripe's twiddle words, and their length.
pub fn twiddle_offsets(log_sizes: &BTreeSet<u32>) -> (BTreeMap<u32, usize>, usize) {
    let mut len = 0usize;
    let offsets = log_sizes
        .iter()
        .map(|&log_size| {
            let offset = len;
            len = (len + (1usize << (log_size - 1))).next_multiple_of(TWIDDLE_ALIGN_WORDS);
            (log_size, offset)
        })
        .collect();
    (offsets, len)
}

/// The twiddles of stripe `stripe` (of `2^log_blowup`) of the domains of the columns of log
/// sizes `log_sizes` (domain log size `log_size + log_blowup`), from `tree`: the ones
/// `SimdBackend::evaluate_into` takes on that stripe, from `SimdBackend::part_twiddles`.
///
/// For log size `L`, at `twiddle_offsets(log_sizes).0[L]`: line layer `j` (`j` in `0..L-1`, of
/// `2^(L-2-j)` twiddles, the layer of bit `j + 1` of the FFT) at `2^(L-1) - 2^(L-1-j)`, as the
/// canonical twiddles (stwo keeps their doubles).
pub fn stripe_twiddle_words(
    log_sizes: &BTreeSet<u32>,
    log_blowup: u32,
    stripe: usize,
    tree: &TwiddleTree<SimdBackend>,
) -> Vec<u32> {
    assert!(stripe < 1 << log_blowup);
    let (offsets, len) = twiddle_offsets(log_sizes);
    let mut words = vec![0; len];
    for (&log_size, &offset) in &offsets {
        assert!(log_size >= MIN_LOG_SIZE);
        let domain = CanonicCoset::new(log_size + log_blowup).circle_domain();
        let twiddles = SimdBackend::part_twiddles(domain, log_blowup, stripe, tree);
        let stripe_domain = part_domain(domain, log_blowup, stripe);
        let layers = domain_line_twiddles_from_tree(stripe_domain, &twiddles.twiddles);
        assert_eq!(layers.len(), log_size as usize - 1);
        for (j, layer) in layers.iter().enumerate() {
            assert_eq!(layer.len(), 1 << (log_size as usize - 2 - j));
            let start = offset + (1 << (log_size - 1)) - (1 << (log_size as usize - 1 - j));
            for (word, &double) in words[start..start + layer.len()].iter_mut().zip(*layer) {
                debug_assert_eq!(double & 1, 0);
                *word = double >> 1;
            }
        }
    }
    words
}

/// The passes of the FFT of a column of log size `log_size`, in order: each applies the layers
/// of the bits `lo..hi` of the column index, over tiles that also span the bits `0..c`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pass {
    pub lo: u32,
    pub hi: u32,
    pub c: u32,
}

pub fn passes(log_size: u32) -> Vec<Pass> {
    let low = Pass { lo: 0, hi: log_size.min(TILE_LOG), c: 0 };
    if log_size <= TILE_LOG {
        return vec![low];
    }
    // The bits above the low pass's, in passes as even as possible, the higher ones first.
    let high_bits = log_size - TILE_LOG;
    let n_high = high_bits.div_ceil(MAX_HIGH_PASS_LAYERS);
    let mut result = vec![];
    let mut hi = log_size;
    for k in 0..n_high {
        let layers = high_bits / n_high + u32::from(k < high_bits % n_high);
        result.push(Pass { lo: hi - layers, hi, c: TILE_LOG - layers });
        hi -= layers;
    }
    result.push(low);
    result
}

/// The uniform parameters of one dispatch (`Params` in the shader).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Params {
    log_n: u32,
    lo: u32,
    hi: u32,
    c: u32,
    total: u32,
    src_base: u32,
    dst_base: u32,
    tw_base: u32,
    transposed: u32,
    n_wg_x: u32,
    n_wg: u32,
    pad: [u32; 5],
}

/// A radix-2^r unit of the rounds: the words `base + (j << lb)`, `j < 2^r`, through the layers of
/// the column bits `kb + r - 1` down to `kb`, `gi` the column index of word `base`. Unrolled, so
/// that the words stay in registers.
fn radix_function(r: u32) -> String {
    let n = 1u32 << r;
    let mut s = format!("fn radix{n}(base: u32, lb: u32, kb: u32, gi: u32) {{\n");
    for j in 0..n {
        writeln!(s, "    var x{j} = tile[sw(base + ({j}u << lb))];").unwrap();
    }
    for q in (0..r).rev() {
        let half = 1 << q;
        // Layer q (bit kb + q): the blocks of 2^(q+1) words, each with its twiddle, at index
        // (gi >> (kb + q + 1)) + block.
        for block in 0..n >> (q + 1) {
            let twiddle = if q == 0 {
                format!("tw_low(kb, (gi >> (kb + 1u)) + {block}u)")
            } else {
                format!("tw_line(kb + {q}u, (gi >> (kb + {}u)) + {block}u)", q + 1)
            };
            writeln!(s, "    {{\n        let t = {twiddle};").unwrap();
            for jj in 0..half {
                let (a, b) = (block * 2 * half + jj, block * 2 * half + jj + half);
                writeln!(
                    s,
                    "        {{ let p = m31_mul(x{b}, t); let a = x{a}; x{a} = m31_add(a, p); \
                     x{b} = m31_sub(a, p); }}"
                )
                .unwrap();
            }
            s.push_str("    }\n");
        }
    }
    for j in 0..n {
        writeln!(s, "    tile[sw(base + ({j}u << lb))] = x{j};").unwrap();
    }
    s.push_str("}\n");
    s
}

/// The most batches a merged dispatch takes (see [`Merge`]).
pub const MAX_MERGED: usize = 32;

/// The parameters of one dispatch of a single batch.
const SINGLE_PARAMS: &str = "@group(0) @binding(3) var<uniform> params: Params;";
const SINGLE_PROLOGUE: &str = "    let p = params;
    let w = wid.x + wid.y * p.n_wg_x;";

/// The parameters of a merged dispatch: its batches' parameters, each batch's workgroups side by
/// side in the dispatch (the first in `pad0`), found by a workgroup from its index.
const MERGED_PARAMS: &str = "struct Merged {
    n_wg_x: u32,
    n_wg: u32,
    n_batches: u32,
    pad: u32,
    batches: array<Params, MAX_MERGED>,
}
@group(0) @binding(3) var<uniform> merged: Merged;
// The workgroup's batch's parameters, for the functions `main` calls.
var<private> params: Params;";
const MERGED_PROLOGUE: &str = "    let gw = wid.x + wid.y * merged.n_wg_x;
    if (gw >= merged.n_wg) {
        return;
    }
    // The last batch whose first workgroup is at most gw.
    var batch = 0u;
    for (var b = 1u; b < merged.n_batches; b++) {
        if (merged.batches[b].pad0 <= gw) {
            batch = b;
        }
    }
    let p = merged.batches[batch];
    params = p;
    let w = gw - p.pad0;";

/// The kernel source, out of place (from the coefficients) or in place, for a dispatch of one
/// batch or a merged one.
fn shader_source(in_place: bool, merged: bool) -> String {
    let (bindings, load) = if in_place {
        ("@group(0) @binding(1) var<storage, read_write> dst: array<u32>;", "dst[p.dst_base + g]")
    } else {
        (
            "@group(0) @binding(0) var<storage, read> src: array<u32>;\n@group(0) @binding(1) \
             var<storage, read_write> dst: array<u32>;",
            "src[src_index(g)]",
        )
    };
    let radix = (1..=4).map(radix_function).collect::<String>();
    let (params, prologue) = if merged {
        (MERGED_PARAMS.replace("MAX_MERGED", &MAX_MERGED.to_string()), MERGED_PROLOGUE)
    } else {
        (SINGLE_PARAMS.to_string(), SINGLE_PROLOGUE)
    };
    let body = SHADER_TEMPLATE
        .replace("/*BINDINGS*/", bindings)
        .replace("/*PARAMS*/", &params)
        .replace("/*PROLOGUE*/", prologue)
        .replace("/*LOAD*/", load)
        .replace("/*RADIX*/", &radix);
    format!("{M31_WGSL}\n{body}")
}

/// How a stripe's dispatches take the batches. A dispatch waits for the previous one (wgpu puts
/// a barrier between dispatches writing one buffer), so the dispatches of small batches, a few
/// workgroups each, leave most of a GPU idle; merging them fills it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Merge {
    /// One dispatch per batch and pass.
    #[default]
    Off,
    /// The batches of columns of at most `2^TILE_LOG` values (one pass each) of a buffer in one
    /// dispatch; the others one dispatch per batch and pass.
    Small,
    /// The `k`-th passes of all the batches of a buffer in one dispatch, for each `k`.
    All,
}

/// The parameters of a merged dispatch (`Merged` in the shader).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MergedParams {
    n_wg_x: u32,
    n_wg: u32,
    n_batches: u32,
    pad: u32,
    batches: [Params; MAX_MERGED],
}

/// The compute pipelines of the extension, for one device: per [`Kind`], a pipeline and its bind
/// group layout.
pub struct ExtensionKernels {
    pipelines: [(wgpu::ComputePipeline, wgpu::BindGroupLayout); 4],
    /// The distance between two dispatches' parameters, in bytes, single and merged.
    params_stride: u64,
    merged_stride: u64,
    max_workgroups_per_dimension: u32,
}

/// A kernel: out of place (the first pass, from the coefficients) or in place, of one batch or
/// merged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Kind {
    in_place: bool,
    merged: bool,
}

impl Kind {
    const ALL: [Kind; 4] = [
        Kind { in_place: false, merged: false },
        Kind { in_place: true, merged: false },
        Kind { in_place: false, merged: true },
        Kind { in_place: true, merged: true },
    ];

    fn index(self) -> usize {
        usize::from(self.in_place) + 2 * usize::from(self.merged)
    }
}

impl ExtensionKernels {
    pub fn new(gpu: &GpuContext) -> Self {
        let limits = gpu.device.limits();
        assert!(limits.max_compute_workgroup_storage_size >= 4 << TILE_LOG, "workgroup storage");
        assert!(limits.max_compute_invocations_per_workgroup >= WORKGROUP_SIZE, "invocations");
        let storage = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let params = |size: usize| wgpu::BindGroupLayoutEntry {
            binding: 3,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: wgpu::BufferSize::new(size as u64),
            },
            count: None,
        };
        let make = |label: &str, kind: Kind, entries: &[wgpu::BindGroupLayoutEntry]| {
            let layout = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries,
            });
            let pipeline_layout =
                gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(label),
                    bind_group_layouts: &[Some(&layout)],
                    immediate_size: 0,
                });
            let module = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(shader_source(kind.in_place, kind.merged).into()),
            });
            let pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
            (pipeline, layout)
        };
        let pipelines = Kind::ALL.map(|kind| {
            let size = if kind.merged { size_of::<MergedParams>() } else { size_of::<Params>() };
            let label = format!(
                "extension {}{}",
                if kind.in_place { "in place" } else { "out of place" },
                if kind.merged { ", merged" } else { "" }
            );
            if kind.in_place {
                make(&label, kind, &[storage(1, false), storage(2, true), params(size)])
            } else {
                make(
                    &label,
                    kind,
                    &[storage(0, true), storage(1, false), storage(2, true), params(size)],
                )
            }
        });
        let align = u64::from(limits.min_uniform_buffer_offset_alignment);
        Self {
            pipelines,
            params_stride: (size_of::<Params>() as u64).next_multiple_of(align),
            merged_stride: (size_of::<MergedParams>() as u64).next_multiple_of(align),
            max_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
        }
    }

    /// The WGSL of the kernels (out of place, in place), for inspection.
    pub fn sources() -> (String, String) {
        (shader_source(false, false), shader_source(true, false))
    }

    /// The WGSL of the merged kernels (out of place, in place), for inspection.
    pub fn merged_sources() -> (String, String) {
        (shader_source(false, true), shader_source(true, true))
    }

    /// The workgroup grid of `n_wg` workgroups.
    fn grid(&self, n_wg: u32) -> (u32, u32) {
        let n_wg_x = n_wg.min(self.max_workgroups_per_dimension);
        let n_wg_y = n_wg.div_ceil(n_wg_x);
        assert!(n_wg_y <= self.max_workgroups_per_dimension);
        (n_wg_x, n_wg_y)
    }
}

struct Dispatch {
    kind: Kind,
    buffer: usize,
    params_offset: u32,
    workgroups: (u32, u32),
}

/// The extension of one tree: the dispatches that evaluate a stripe of every column of `coeffs`
/// into `out` (the same layout), with every stripe's twiddles on the GPU.
pub struct StripeExtension {
    log_blowup: u32,
    /// Per stripe.
    twiddles: Vec<wgpu::Buffer>,
    _params: [wgpu::Buffer; 2],
    /// Per kind, per stripe, per buffer of the layout.
    groups: [Vec<Vec<wgpu::BindGroup>>; 4],
    dispatches: Vec<Dispatch>,
}

impl StripeExtension {
    /// With the twiddles of every stripe computed from `tree` (see [`stripe_twiddle_words`]) and
    /// uploaded.
    pub fn new(
        gpu: &GpuContext,
        kernels: &ExtensionKernels,
        coeffs: &GpuColumns,
        out: &GpuColumns,
        log_blowup: u32,
        tree: &TwiddleTree<SimdBackend>,
    ) -> Self {
        let log_sizes = coeffs.layout.log_sizes();
        let twiddles = (0..1 << log_blowup)
            .map(|stripe| stripe_twiddle_words(&log_sizes, log_blowup, stripe, tree))
            .collect::<Vec<_>>();
        Self::with_twiddle_words(gpu, kernels, coeffs, out, log_blowup, &twiddles)
    }

    /// With `twiddles[stripe]` the words of [`stripe_twiddle_words`] for the layout's log sizes.
    pub fn with_twiddle_words(
        gpu: &GpuContext,
        kernels: &ExtensionKernels,
        coeffs: &GpuColumns,
        out: &GpuColumns,
        log_blowup: u32,
        twiddles: &[Vec<u32>],
    ) -> Self {
        Self::with_twiddle_words_merged(gpu, kernels, coeffs, out, log_blowup, twiddles, Merge::Off)
    }

    /// [`Self::with_twiddle_words`], the batches dispatched as `merge` says.
    pub fn with_twiddle_words_merged(
        gpu: &GpuContext,
        kernels: &ExtensionKernels,
        coeffs: &GpuColumns,
        out: &GpuColumns,
        log_blowup: u32,
        twiddles: &[Vec<u32>],
        merge: Merge,
    ) -> Self {
        assert_eq!(coeffs.layout, out.layout, "the evaluations take the coefficients' layout");
        assert_eq!(twiddles.len(), 1 << log_blowup);
        let layout = &coeffs.layout;
        let (offsets, twiddles_len) = twiddle_offsets(&layout.log_sizes());
        let twiddles = twiddles
            .iter()
            .enumerate()
            .map(|(stripe, words)| {
                assert_eq!(words.len(), twiddles_len);
                gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("twiddles of stripe {stripe}")),
                    contents: bytemuck::cast_slice(if words.is_empty() {
                        &[0u32][..]
                    } else {
                        words
                    }),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                })
            })
            .collect::<Vec<_>>();

        // Every pass of every batch: (batch, its k-th pass, its parameters).
        let mut items = vec![];
        for (b, batch) in layout.batches.iter().enumerate() {
            let log_size = batch.log_size;
            let total = batch.n_words();
            assert!(total < 1 << 32);
            for (k, pass) in passes(log_size).into_iter().enumerate() {
                let n_wg = total.div_ceil(1 << TILE_LOG) as u32;
                let (n_wg_x, _) = kernels.grid(n_wg);
                let params = Params {
                    log_n: log_size,
                    lo: pass.lo,
                    hi: pass.hi,
                    c: pass.c,
                    total: total as u32,
                    src_base: batch.offset as u32,
                    dst_base: batch.offset as u32,
                    tw_base: offsets[&log_size] as u32,
                    transposed: u32::from(k == 0 && log_size > CACHED_FFT_LOG_SIZE),
                    n_wg_x,
                    n_wg,
                    pad: [0; 5],
                };
                items.push((b, k, params));
            }
        }
        let mergeable = |b: usize| match merge {
            Merge::Off => false,
            Merge::Small => layout.batches[b].log_size <= TILE_LOG,
            Merge::All => true,
        };

        let mut params: Vec<Params> = vec![];
        let mut merged_params: Vec<MergedParams> = vec![];
        let mut dispatches = vec![];
        let mut single = |dispatches: &mut Vec<Dispatch>, b: usize, k: usize, p: Params| {
            dispatches.push(Dispatch {
                kind: Kind { in_place: k > 0, merged: false },
                buffer: layout.batches[b].buffer,
                params_offset: (params.len() as u64 * kernels.params_stride) as u32,
                workgroups: kernels.grid(p.n_wg),
            });
            params.push(p);
        };
        if merge == Merge::Off {
            // Batch by batch, each batch's passes in order.
            for &(b, k, p) in &items {
                single(&mut dispatches, b, k, p);
            }
        } else {
            // Pass by pass (a batch's passes stay in order), buffer by buffer: the mergeable
            // batches' k-th passes of a buffer in one dispatch (several past MAX_MERGED).
            let max_passes = items.iter().map(|&(_, k, _)| k + 1).max().unwrap_or(0);
            for k in 0..max_passes {
                for buffer in 0..layout.buffer_words.len() {
                    let at = |&&(b, kk, _): &&(usize, usize, Params)| {
                        kk == k && layout.batches[b].buffer == buffer
                    };
                    let (to_merge, alone): (Vec<_>, Vec<_>) =
                        items.iter().filter(at).partition(|&&(b, _, _)| mergeable(b));
                    for &&(b, k, p) in &alone {
                        single(&mut dispatches, b, k, p);
                    }
                    for chunk in to_merge.chunks(MAX_MERGED) {
                        if let &[&(b, k, p)] = chunk {
                            single(&mut dispatches, b, k, p);
                            continue;
                        }
                        let mut m = MergedParams::zeroed();
                        let mut n_wg = 0u32;
                        for (j, &&(_, _, p)) in chunk.iter().enumerate() {
                            m.batches[j] = Params { pad: [n_wg, 0, 0, 0, 0], ..p };
                            n_wg = n_wg.checked_add(p.n_wg).unwrap();
                        }
                        let workgroups = kernels.grid(n_wg);
                        m.n_wg_x = workgroups.0;
                        m.n_wg = n_wg;
                        m.n_batches = chunk.len() as u32;
                        dispatches.push(Dispatch {
                            kind: Kind { in_place: k > 0, merged: true },
                            buffer,
                            params_offset: (merged_params.len() as u64 * kernels.merged_stride)
                                as u32,
                            workgroups,
                        });
                        merged_params.push(m);
                    }
                }
            }
        }

        /// One uniform buffer of `params`, each at a multiple of `stride` bytes.
        fn uniforms<T: Pod>(
            gpu: &GpuContext,
            params: &[T],
            stride: u64,
            label: &str,
        ) -> wgpu::Buffer {
            let stride = stride as usize;
            let mut bytes = vec![0u8; (params.len() * stride).max(stride.max(size_of::<T>()))];
            for (i, p) in params.iter().enumerate() {
                bytes[i * stride..i * stride + size_of::<T>()]
                    .copy_from_slice(bytemuck::bytes_of(p));
            }
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: &bytes,
                usage: wgpu::BufferUsages::UNIFORM,
            })
        }
        let params_buffers = [
            uniforms(gpu, &params, kernels.params_stride, "extension params"),
            uniforms(gpu, &merged_params, kernels.merged_stride, "extension merged params"),
        ];

        let entry = |binding, resource| wgpu::BindGroupEntry { binding, resource };
        let groups = Kind::ALL.map(|kind| {
            let (size, params_buffer) = if kind.merged {
                (size_of::<MergedParams>(), &params_buffers[1])
            } else {
                (size_of::<Params>(), &params_buffers[0])
            };
            let params_binding = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: params_buffer,
                offset: 0,
                size: wgpu::BufferSize::new(size as u64),
            });
            twiddles
                .iter()
                .map(|tw| {
                    (0..layout.buffer_words.len())
                        .map(|b| {
                            let mut entries = vec![];
                            if !kind.in_place {
                                entries.push(entry(0, coeffs.buffers[b].as_entire_binding()));
                            }
                            entries.push(entry(1, out.buffers[b].as_entire_binding()));
                            entries.push(entry(2, tw.as_entire_binding()));
                            entries.push(entry(3, params_binding.clone()));
                            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                                label: None,
                                layout: &kernels.pipelines[kind.index()].1,
                                entries: &entries,
                            })
                        })
                        .collect()
                })
                .collect()
        });
        Self { log_blowup, twiddles, _params: params_buffers, groups, dispatches }
    }

    pub fn n_stripes(&self) -> usize {
        1 << self.log_blowup
    }

    /// The number of dispatches of a stripe.
    pub fn n_dispatches(&self) -> usize {
        self.dispatches.len()
    }

    /// Queues the writes of the twiddles of every stripe again (`twiddles` as in
    /// [`Self::with_twiddle_words`]).
    pub fn write_twiddles(&self, gpu: &GpuContext, twiddles: &[Vec<u32>]) {
        assert_eq!(twiddles.len(), self.twiddles.len());
        for (buffer, words) in self.twiddles.iter().zip(twiddles) {
            gpu.queue.write_buffer(buffer, 0, bytemuck::cast_slice(words));
        }
    }

    /// The bytes of twiddles on the GPU, all stripes.
    pub fn twiddle_bytes(&self) -> u64 {
        self.twiddles.iter().map(|buffer| buffer.size()).sum()
    }

    /// Frees the twiddles' and the parameters' buffers now (see `StripedCommitment::destroy`).
    pub fn destroy(&self) {
        for buffer in self.twiddles.iter().chain(&self._params) {
            buffer.destroy();
        }
    }

    /// Records the evaluation of stripe `stripe` of every column into `out`, in one compute pass.
    pub fn encode(
        &self,
        kernels: &ExtensionKernels,
        encoder: &mut wgpu::CommandEncoder,
        stripe: usize,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("extension"),
            timestamp_writes: None,
        });
        self.record(kernels, &mut pass, stripe);
    }

    /// Records the evaluation of stripe `stripe` of every column into `out` into `pass` (whose
    /// pipeline and bind group it changes).
    pub fn record(
        &self,
        kernels: &ExtensionKernels,
        pass: &mut wgpu::ComputePass<'_>,
        stripe: usize,
    ) {
        let mut current = None;
        for dispatch in &self.dispatches {
            let kind = dispatch.kind.index();
            if current != Some(kind) {
                pass.set_pipeline(&kernels.pipelines[kind].0);
                current = Some(kind);
            }
            pass.set_bind_group(
                0,
                &self.groups[kind][stripe][dispatch.buffer],
                &[dispatch.params_offset],
            );
            pass.dispatch_workgroups(dispatch.workgroups.0, dispatch.workgroups.1, 1);
        }
    }

    /// Submits the evaluation of stripe `stripe` (without waiting for it).
    pub fn submit(&self, gpu: &GpuContext, kernels: &ExtensionKernels, stripe: usize) {
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        self.encode(kernels, &mut encoder, stripe);
        gpu.queue.submit([encoder.finish()]);
    }
}
