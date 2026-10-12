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
//! - [`StripeExtension`]: the dispatches for one tree, from a [`GpuColumns`] of coefficients into
//!   a [`GpuColumns`] of evaluations, with the twiddles of every stripe uploaded once.
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
                let fill = buffer_words.last().map(|&fill| fill.next_multiple_of(BATCH_ALIGN_WORDS));
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
            slots[i] = ColumnSlot { buffer: batch.buffer, offset: buffer_words[batch.buffer], log_size };
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
        for (slot, words) in self.layout.slots.iter().zip(columns) {
            assert_eq!(words.len() as u64, 1 << slot.log_size);
            gpu.queue.write_buffer(&self.buffers[slot.buffer], slot.offset * 4, bytemuck::cast_slice(words));
        }
    }

    /// Every column's words, after all submitted work.
    pub fn read(&self, gpu: &GpuContext) -> Vec<Vec<u32>> {
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        let staging = self
            .buffers
            .iter()
            .zip(&self.layout.buffer_words)
            .map(|(buffer, &words)| {
                let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("readback"),
                    size: (words * 4).max(4),
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, words * 4);
                staging
            })
            .collect::<Vec<_>>();
        gpu.queue.submit([encoder.finish()]);
        for staging in &staging {
            staging.slice(..).map_async(wgpu::MapMode::Read, |result| result.expect("map"));
        }
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        let mut columns = vec![vec![]; self.layout.slots.len()];
        for (b, staging) in staging.iter().enumerate() {
            {
                let view = staging.slice(..).get_mapped_range().expect("mapped range");
                let words: &[u32] = bytemuck::cast_slice(&view[..]);
                for (column, slot) in columns.iter_mut().zip(&self.layout.slots) {
                    if slot.buffer == b {
                        let start = slot.offset as usize;
                        *column = words[start..start + (1 << slot.log_size)].to_vec();
                    }
                }
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
                    "        {{ let p = m31_mul(x{b}, t); let a = x{a}; x{a} = m31_add(a, p); x{b} = m31_sub(a, p); }}"
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

/// The kernel source, out of place (from the coefficients) or in place.
fn shader_source(in_place: bool) -> String {
    let (bindings, load) = if in_place {
        ("@group(0) @binding(1) var<storage, read_write> dst: array<u32>;", "dst[params.dst_base + g]")
    } else {
        (
            "@group(0) @binding(0) var<storage, read> src: array<u32>;\n\
             @group(0) @binding(1) var<storage, read_write> dst: array<u32>;",
            "src[src_index(g)]",
        )
    };
    let radix = (1..=4).map(radix_function).collect::<String>();
    let body = SHADER_TEMPLATE
        .replace("/*BINDINGS*/", bindings)
        .replace("/*LOAD*/", load)
        .replace("/*RADIX*/", &radix);
    format!("{M31_WGSL}\n{body}")
}

/// The compute pipelines of the extension, for one device.
pub struct ExtensionKernels {
    out_of_place: wgpu::ComputePipeline,
    in_place: wgpu::ComputePipeline,
    out_of_place_layout: wgpu::BindGroupLayout,
    in_place_layout: wgpu::BindGroupLayout,
    /// The distance between two dispatches' parameters, in bytes.
    params_stride: u64,
    max_workgroups_per_dimension: u32,
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
        let params = wgpu::BindGroupLayoutEntry {
            binding: 3,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: wgpu::BufferSize::new(size_of::<Params>() as u64),
            },
            count: None,
        };
        let make = |label: &str, in_place: bool, entries: &[wgpu::BindGroupLayoutEntry]| {
            let layout = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries,
            });
            let pipeline_layout = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
            let module = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(shader_source(in_place).into()),
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
        let (out_of_place, out_of_place_layout) = make(
            "extension out of place",
            false,
            &[storage(0, true), storage(1, false), storage(2, true), params],
        );
        let (in_place, in_place_layout) =
            make("extension in place", true, &[storage(1, false), storage(2, true), params]);
        Self {
            out_of_place,
            in_place,
            out_of_place_layout,
            in_place_layout,
            params_stride: (size_of::<Params>() as u64)
                .next_multiple_of(u64::from(limits.min_uniform_buffer_offset_alignment)),
            max_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
        }
    }

    /// The WGSL of the kernels (out of place, in place), for inspection.
    pub fn sources() -> (String, String) {
        (shader_source(false), shader_source(true))
    }
}

struct Dispatch {
    in_place: bool,
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
    _params: wgpu::Buffer,
    /// Per stripe, per buffer of the layout.
    out_of_place_groups: Vec<Vec<wgpu::BindGroup>>,
    in_place_groups: Vec<Vec<wgpu::BindGroup>>,
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
                    contents: bytemuck::cast_slice(if words.is_empty() { &[0u32][..] } else { words }),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                })
            })
            .collect::<Vec<_>>();

        let mut params = vec![];
        let mut dispatches = vec![];
        for batch in &layout.batches {
            let log_size = batch.log_size;
            let total = batch.n_words();
            assert!(total < 1 << 32);
            for (k, pass) in passes(log_size).into_iter().enumerate() {
                let n_wg = total.div_ceil(1 << TILE_LOG) as u32;
                let n_wg_x = n_wg.min(kernels.max_workgroups_per_dimension);
                let n_wg_y = n_wg.div_ceil(n_wg_x);
                assert!(n_wg_y <= kernels.max_workgroups_per_dimension);
                dispatches.push(Dispatch {
                    in_place: k > 0,
                    buffer: batch.buffer,
                    params_offset: (params.len() as u64 * kernels.params_stride) as u32,
                    workgroups: (n_wg_x, n_wg_y),
                });
                params.push(Params {
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
                });
            }
        }
        let stride = kernels.params_stride as usize;
        let mut params_bytes = vec![0u8; (params.len() * stride).max(stride)];
        for (i, p) in params.iter().enumerate() {
            params_bytes[i * stride..i * stride + size_of::<Params>()].copy_from_slice(bytemuck::bytes_of(p));
        }
        let params_buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("extension params"),
            contents: &params_bytes,
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let params_binding = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &params_buffer,
            offset: 0,
            size: wgpu::BufferSize::new(size_of::<Params>() as u64),
        });

        let entry = |binding, resource| wgpu::BindGroupEntry { binding, resource };
        let out_of_place_groups = twiddles
            .iter()
            .map(|tw| {
                (0..layout.buffer_words.len())
                    .map(|b| {
                        gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: None,
                            layout: &kernels.out_of_place_layout,
                            entries: &[
                                entry(0, coeffs.buffers[b].as_entire_binding()),
                                entry(1, out.buffers[b].as_entire_binding()),
                                entry(2, tw.as_entire_binding()),
                                entry(3, params_binding.clone()),
                            ],
                        })
                    })
                    .collect()
            })
            .collect();
        let in_place_groups = twiddles
            .iter()
            .map(|tw| {
                (0..layout.buffer_words.len())
                    .map(|b| {
                        gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: None,
                            layout: &kernels.in_place_layout,
                            entries: &[
                                entry(1, out.buffers[b].as_entire_binding()),
                                entry(2, tw.as_entire_binding()),
                                entry(3, params_binding.clone()),
                            ],
                        })
                    })
                    .collect()
            })
            .collect();
        Self {
            log_blowup,
            twiddles,
            _params: params_buffer,
            out_of_place_groups,
            in_place_groups,
            dispatches,
        }
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

    /// Records the evaluation of stripe `stripe` of every column into `out`, in one compute pass.
    pub fn encode(&self, kernels: &ExtensionKernels, encoder: &mut wgpu::CommandEncoder, stripe: usize) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("extension"),
            timestamp_writes: None,
        });
        let mut current = None;
        for dispatch in &self.dispatches {
            if current != Some(dispatch.in_place) {
                pass.set_pipeline(if dispatch.in_place { &kernels.in_place } else { &kernels.out_of_place });
                current = Some(dispatch.in_place);
            }
            let groups = if dispatch.in_place { &self.in_place_groups } else { &self.out_of_place_groups };
            pass.set_bind_group(0, &groups[stripe][dispatch.buffer], &[dispatch.params_offset]);
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
