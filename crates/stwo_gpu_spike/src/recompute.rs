//! A2: stripe 1 of a request's columns recomputed on the GPU from their stripe 0 (see
//! [`crate::parts`]), nothing kept between requests: stripe 0 goes up through the upload ring, is
//! interpolated (`shaders/ifft.wgsl`, as `SimdBackend::interpolate`: the coefficients in stwo's
//! storage order), evaluated on stripe 1 (the commitment's extension, [`StripeExtension`]), and
//! comes back through the readback ring. Every value is canonical, and the coefficients are the
//! column's (the unique polynomial of its degree through stripe 0), so stripe 1 is the one the
//! commitment computed, word for word.
//!
//! A spike's kernels: radix 2, the low 12 layers of the inverse FFT in workgroup memory (one
//! pass), each layer above in a pass of its own, and a pass for the scaling.

use std::collections::{BTreeSet, HashMap};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::time::{Duration, Instant};

use bytemuck::{Pod, Zeroable};
use itertools::Itertools;
use stwo::core::fields::m31::BaseField;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::poly::utils::domain_line_twiddles_from_tree;
use stwo::prover::backend::Column;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::simd::column::BaseColumn;
use stwo::prover::backend::simd::fft::CACHED_FFT_LOG_SIZE;
use stwo::prover::poly::BitReversedOrder;
use stwo::prover::poly::circle::{CircleEvaluation, PolyOps, part_domain};
use wgpu::util::DeviceExt;

use crate::committer::GpuCommitter;
use crate::extension::{
    ColumnsLayout, ExtensionKernels, GpuColumns, Merge, StripeExtension, StripeTwiddles,
    max_buffer_words,
};
use crate::transfer::{Readback, ReadbackRing, Upload, UploadRing};
use crate::{GpuContext, M31_WGSL};

/// The inverse FFT's layers done in workgroup memory.
const TILE_LOG: u32 = 12;
/// The most workgroups of a row of a dispatch grid.
const MAX_WG_X: u32 = 32768;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    log_n: u32,
    k: u32,
    total: u32,
    base: u32,
    dst_base: u32,
    inv: u32,
    transposed: u32,
    n_wg_x: u32,
}

/// The inverse FFT's pipelines.
pub struct IfftKernels {
    low: wgpu::ComputePipeline,
    layer: wgpu::ComputePipeline,
    scale: wgpu::ComputePipeline,
}

impl IfftKernels {
    /// The kernels' source (M31_WGSL prepended).
    pub fn source() -> String {
        format!("{M31_WGSL}\n{}", include_str!("shaders/ifft.wgsl"))
    }

    pub fn new(gpu: &GpuContext) -> Self {
        let source = Self::source();
        Self {
            low: gpu.pipeline("ifft low", &source, "low"),
            layer: gpu.pipeline("ifft layer", &source, "layer"),
            scale: gpu.pipeline("ifft scale", &source, "scale"),
        }
    }
}

/// The twiddles of columns of `2^log_size` coefficients committed at blowup `log_blowup`: the
/// inverse line twiddles of stripe 0's domain (as `trace_part` interpolates with), and every
/// stripe's for the extension (made only if the commitments' are not at hand).
struct Tables {
    itw: wgpu::Buffer,
    extension: Option<Arc<StripeTwiddles>>,
}

impl Tables {
    fn new(gpu: &GpuContext, log_size: u32, log_blowup: u32) -> Self {
        let domain = CanonicCoset::new(log_size + log_blowup).circle_domain();
        // As trace_part: the twiddles of the first part's half coset, their inverses.
        let first_part = part_domain(domain, log_blowup, 0);
        let tree = SimdBackend::precompute_twiddles(first_part.half_coset);
        let layers = domain_line_twiddles_from_tree(first_part, &tree.itwiddles);
        assert_eq!(layers.len(), log_size as usize - 1);
        let mut words = vec![0u32; 1 << (log_size - 1)];
        for (j, layer) in layers.iter().enumerate() {
            assert_eq!(layer.len(), 1 << (log_size as usize - 2 - j));
            let start = (1 << (log_size - 1)) - (1 << (log_size as usize - 1 - j));
            for (word, &double) in words[start..start + layer.len()].iter_mut().zip(*layer) {
                assert_eq!(double & 1, 0, "stwo keeps the twiddles doubled");
                *word = double >> 1;
            }
        }
        let itw = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("inverse twiddles"),
            contents: bytemuck::cast_slice(&words),
            usage: wgpu::BufferUsages::STORAGE,
        });
        Self { itw, extension: None }
    }

    /// Every stripe's twiddles of the domain, for the extension.
    fn extension(
        &mut self,
        gpu: &GpuContext,
        log_size: u32,
        log_blowup: u32,
    ) -> Arc<StripeTwiddles> {
        self.extension
            .get_or_insert_with(|| {
                let domain = CanonicCoset::new(log_size + log_blowup).circle_domain();
                let tree = SimdBackend::precompute_twiddles(domain.half_coset);
                Arc::new(StripeTwiddles::new(gpu, &BTreeSet::from([log_size]), log_blowup, &tree))
            })
            .clone()
    }
}

/// What A2 keeps: its kernels, the twiddles of every shape it saw, and its working buffers (the
/// largest a request needed so far).
pub struct Recomputer {
    kernels: IfftKernels,
    tables: HashMap<(u32, u32), Tables>,
    work: Vec<wgpu::Buffer>,
    coeffs: Vec<wgpu::Buffer>,
}

/// `buffers` grown to hold `layout`'s: each at least its buffer's words (a smaller one replaced).
fn grown(gpu: &GpuContext, buffers: &mut Vec<wgpu::Buffer>, layout: &ColumnsLayout, label: &str) {
    for (i, &words) in layout.buffer_words.iter().enumerate() {
        let bytes = (words * 4).max(4);
        if buffers.get(i).is_some_and(|buffer| buffer.size() >= bytes) {
            continue;
        }
        let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        match buffers.get_mut(i) {
            Some(old) => std::mem::replace(old, buffer).destroy(),
            None => buffers.push(buffer),
        }
    }
}

/// Where a request's time went (wall clock): the upload ring's writes (the CPU's copies into its
/// chunks), the GPU's work (from the submission of the inverse FFT and the extension to their end,
/// the upload's copies on the GPU included), the readback (the GPU's copies and the CPU's copies
/// out), and the setup (buffers, plans, twiddles).
#[derive(Clone, Copy, Debug, Default)]
pub struct RecomputeTimes {
    pub setup: Duration,
    pub upload: Duration,
    pub compute: Duration,
    pub readback: Duration,
    pub bytes: u64,
}

impl RecomputeTimes {
    pub fn add(&mut self, other: &Self) {
        self.setup += other.setup;
        self.upload += other.upload;
        self.compute += other.compute;
        self.readback += other.readback;
        self.bytes += other.bytes;
    }
}

impl Recomputer {
    /// The GPU memory it keeps: its working buffers and its twiddles.
    pub fn bytes(&self) -> u64 {
        let buffers = self.work.iter().chain(&self.coeffs).map(wgpu::Buffer::size).sum::<u64>();
        let tables = self
            .tables
            .values()
            .map(|t| t.itw.size() + t.extension.as_ref().map_or(0, |e| e.bytes()))
            .sum::<u64>();
        buffers + tables
    }

    pub fn new(gpu: &GpuContext) -> Self {
        Self {
            kernels: IfftKernels::new(gpu),
            tables: HashMap::new(),
            work: vec![],
            coeffs: vec![],
        }
    }

    /// Stripe 1 of `columns` (each holding stripe 0, `2^log_size` values, of its domain of log size
    /// `log_size + log_blowup`), by interpolation and evaluation on the GPU, with the extension's
    /// twiddles `extension_twiddles` (the commitments', covering `log_size`) or its own; and, if
    /// `coefficients`, the coefficients too (as stwo stores them; for tests).
    #[allow(clippy::too_many_arguments)]
    pub fn stripe1(
        &mut self,
        gpu: &GpuContext,
        extension_kernels: &ExtensionKernels,
        upload: &mut UploadRing,
        readback: &mut ReadbackRing,
        columns: &[&BaseColumn],
        log_size: u32,
        log_blowup: u32,
        extension_twiddles: Option<Arc<StripeTwiddles>>,
        coefficients: bool,
    ) -> (Vec<BaseColumn>, Option<Vec<BaseColumn>>, RecomputeTimes) {
        let mut times = RecomputeTimes::default();
        let start = Instant::now();
        assert!(log_size >= 4, "a column of at least 16 values");
        for column in columns {
            assert_eq!(column.length, 1 << log_size);
        }
        let tables = self
            .tables
            .entry((log_size, log_blowup))
            .or_insert_with(|| Tables::new(gpu, log_size, log_blowup));
        let layout =
            Arc::new(ColumnsLayout::new(&vec![log_size; columns.len()], max_buffer_words(gpu)));
        grown(gpu, &mut self.work, &layout, "recompute work");
        grown(gpu, &mut self.coeffs, &layout, "recompute coefficients");
        let n_buffers = layout.buffer_words.len();
        let work = GpuColumns::with_buffers(self.work[..n_buffers].to_vec(), layout.clone());
        let coeffs = GpuColumns::with_buffers(self.coeffs[..n_buffers].to_vec(), layout.clone());
        let extension_twiddles = match extension_twiddles {
            Some(twiddles) => twiddles,
            None => tables.extension(gpu, log_size, log_blowup),
        };
        let extension = StripeExtension::with_twiddles(
            gpu,
            extension_kernels,
            &coeffs,
            &work,
            extension_twiddles,
            Merge::Off,
        );
        // The passes of every batch: (pipeline, buffer, params, workgroups).
        let inv = BaseField::from(1u32 << log_size).inverse().0;
        let transposed = u32::from(log_size > CACHED_FFT_LOG_SIZE);
        let grid = |threads: u64| {
            let workgroups = threads.div_ceil(256).max(1) as u32;
            let x = workgroups.min(MAX_WG_X);
            (x, workgroups.div_ceil(x))
        };
        let mut passes = vec![];
        for batch in &layout.batches {
            let total = batch.n_words();
            let params = |k: u32, n_wg_x: u32| Params {
                log_n: log_size,
                k,
                total: total as u32,
                base: batch.offset as u32,
                dst_base: batch.offset as u32,
                inv,
                transposed,
                n_wg_x,
            };
            let tiles = total.div_ceil(1 << TILE_LOG) * 256;
            let (x, y) = grid(tiles);
            passes.push((&self.kernels.low, batch.buffer, params(0, x), (x, y)));
            for k in TILE_LOG..log_size {
                let (x, y) = grid(total / 2);
                passes.push((&self.kernels.layer, batch.buffer, params(k, x), (x, y)));
            }
            let (x, y) = grid(total);
            passes.push((&self.kernels.scale, batch.buffer, params(0, x), (x, y)));
        }
        let uniforms = passes
            .iter()
            .map(|(_, _, params, _)| {
                gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("ifft params"),
                    contents: bytemuck::bytes_of(params),
                    usage: wgpu::BufferUsages::UNIFORM,
                })
            })
            .collect_vec();
        let groups = passes
            .iter()
            .zip(&uniforms)
            .map(|((pipeline, buffer, ..), uniform)| {
                let scale = std::ptr::eq(*pipeline, &self.kernels.scale);
                let mut entries = vec![
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: work.buffers[*buffer].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry { binding: 2, resource: uniform.as_entire_binding() },
                ];
                if scale {
                    entries.push(wgpu::BindGroupEntry {
                        binding: 3,
                        resource: coeffs.buffers[*buffer].as_entire_binding(),
                    });
                } else {
                    entries.push(wgpu::BindGroupEntry {
                        binding: 1,
                        resource: tables.itw.as_entire_binding(),
                    });
                }
                gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &pipeline.get_bind_group_layout(0),
                    entries: &entries,
                })
            })
            .collect_vec();
        times.setup = start.elapsed();

        // Stripe 0 up, into the working buffers.
        let start = Instant::now();
        let pieces = columns
            .iter()
            .zip(&layout.slots)
            .map(|(column, slot)| Upload {
                dst: &work.buffers[slot.buffer],
                dst_offset: slot.offset * 4,
                src: bytemuck::cast_slice::<BaseField, u8>(column.as_slice()),
            })
            .collect_vec();
        upload.upload(gpu, &pieces);
        times.upload = start.elapsed();
        times.bytes = columns.iter().map(|column| 4 * column.length as u64).sum();

        // The inverse FFT into the coefficients, then stripe 1 into the working buffers.
        let start = Instant::now();
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ifft"),
                timestamp_writes: None,
            });
            for ((pipeline, _, _, (x, y)), group) in passes.iter().zip(&groups) {
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, group, &[]);
                pass.dispatch_workgroups(*x, *y, 1);
            }
        }
        // The coefficients, for a test, before the extension.
        let coefficient_columns = coefficients.then(|| {
            let mut columns = (0..columns.len())
                .map(|_| unsafe { BaseColumn::uninitialized(1 << log_size) })
                .collect_vec();
            gpu.queue.submit([std::mem::replace(
                &mut encoder,
                gpu.device.create_command_encoder(&Default::default()),
            )
            .finish()]);
            let pieces = columns
                .iter_mut()
                .zip(&layout.slots)
                .map(|(column, slot)| Readback {
                    src: &coeffs.buffers[slot.buffer],
                    src_offset: slot.offset * 4,
                    dst: bytemuck::cast_slice_mut::<BaseField, u8>(column.as_mut_slice()),
                })
                .collect_vec();
            readback.read(gpu, pieces);
            columns
        });
        extension.encode(extension_kernels, &mut encoder, 1);
        gpu.queue.submit([encoder.finish()]);
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        times.compute = start.elapsed();

        // Stripe 1 back.
        let start = Instant::now();
        // Safety: every value is read back.
        let mut out = (0..columns.len())
            .map(|_| unsafe { BaseColumn::uninitialized(1 << log_size) })
            .collect_vec();
        let pieces = out
            .iter_mut()
            .zip(&layout.slots)
            .map(|(column, slot)| Readback {
                src: &work.buffers[slot.buffer],
                src_offset: slot.offset * 4,
                dst: bytemuck::cast_slice_mut::<BaseField, u8>(column.as_mut_slice()),
            })
            .collect_vec();
        readback.read(gpu, pieces);
        times.readback = start.elapsed();

        extension.destroy();
        for uniform in uniforms {
            uniform.destroy();
        }
        (out, coefficient_columns, times)
    }
}

/// [`GpuCommitter::trace_part`] in [`crate::parts::PartsMode::Recompute`]: every column of the
/// request (one component's: one size, one blowup) recomputed; `None` for all if the request is
/// not of that shape, or the GPU fails.
pub fn trace_part(
    committer: &GpuCommitter,
    part: usize,
    log_parts: u32,
    columns: &[(usize, &CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>)],
) -> Vec<Option<BaseColumn>> {
    let none = || columns.iter().map(|_| None).collect();
    let Some((_, first)) = columns.first() else { return none() };
    let length = first.values.length;
    let domain_log_size = first.domain.log_size();
    if part != 1
        || !length.is_power_of_two()
        || length.ilog2() + log_parts != domain_log_size
        || columns.iter().any(|(_, column)| {
            column.values.length != length || column.domain.log_size() != domain_log_size
        })
    {
        return none();
    }
    let log_size = length.ilog2();
    let values = columns.iter().map(|(_, column)| &column.values).collect_vec();
    // In chunks of at most `chunk_bytes()` (one column at least), so that the GPU holds two
    // chunks' worth of working buffers rather than two of the largest request.
    let per_chunk = (chunk_bytes() / (4 << log_size)).max(1) as usize;
    let mut answers = Vec::with_capacity(columns.len());
    for chunk in values.chunks(per_chunk) {
        match committer.recompute_stripe1(chunk, log_size, log_parts) {
            Some(stripes) => answers.extend(stripes.into_iter().map(Some)),
            None => answers.extend(chunk.iter().map(|_| None)),
        }
    }
    answers
}

/// The most stripe-0 bytes A2 sends up at once: `STWO_GPU_PARTS_CHUNK_MIB` (default 128 MiB,
/// WebGPU's default binding size); 0 for no limit.
fn chunk_bytes() -> u64 {
    let mib = std::env::var("STWO_GPU_PARTS_CHUNK_MIB")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(128);
    if mib == 0 { u64::MAX } else { mib << 20 }
}

/// Runs `f`, catching a panic (a GPU failure) as `None`.
pub(crate) fn guarded<T>(f: impl FnOnce() -> T) -> Option<T> {
    catch_unwind(AssertUnwindSafe(f)).ok()
}
