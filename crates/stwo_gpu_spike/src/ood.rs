//! The out-of-domain samples of a proof's columns on the GPU, from their coefficients resident
//! there (see [`crate::parts::PartsMode::Coefficients`]).
//!
//! stwo samples a column without stored coefficients at a point by barycentric sums over its
//! first stripe (`PolyOps::subdomain_eval_group`): the value at the point of the polynomial the
//! stripe interpolates, which is the polynomial whose coefficients the GPU keeps. That value is
//! `sum_k c_k prod_{j in bits(k)} m_j` over the coefficients `c_k` (in stwo's storage order), the
//! point's mappings `m_j` (in QM31) as `SimdBackend::evaluation_mappings` gives them. One pass over
//! a column folds it into 16 sums of the coefficient vectors' weighted sums (`shaders/ood.wgsl`),
//! and the value is then a sum of 16 products, done here. The arithmetic is exact in the field:
//! the values are those stwo computes, word for word.

use itertools::Itertools;
use stwo::core::circle::CirclePoint;
use stwo::core::fields::m31::{BaseField, P};
use stwo::core::fields::qm31::SecureField;
use stwo::prover::backend::simd::SimdBackend;
use wgpu::util::DeviceExt;

use crate::constraint::wgsl::QM31_WGSL;
use crate::{GpuContext, M31_WGSL};

/// The mappings of a coefficient vector's 16 coefficients.
const LOG_VECTOR: usize = 4;
/// The most vectors a chunk folds in workgroup memory.
const MAX_LO_LOG: u32 = 8;
/// The most chunks a workgroup folds.
const MAX_SPAN: usize = 16;

/// The samples' pipelines.
pub struct OodKernels {
    fold: wgpu::ComputePipeline,
    reduce: wgpu::ComputePipeline,
}

impl OodKernels {
    /// The kernels' source (`M31_WGSL` and `QM31_WGSL` prepended).
    pub fn source() -> String {
        format!("{M31_WGSL}\n{QM31_WGSL}\n{}", include_str!("shaders/ood.wgsl"))
    }

    pub fn new(gpu: &GpuContext) -> Self {
        let source = Self::source();
        Self {
            fold: gpu.pipeline("ood fold", &source, "fold"),
            reduce: gpu.pipeline("ood reduce", &source, "reduce"),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    lo_log: u32,
    span: u32,
    n_spans: u32,
    n_columns: u32,
    wg_x: u32,
    pad: [u32; 3],
}

/// The products of `factors` over the set bits of every index below `1 << factors.len()`.
fn tensor_products(factors: &[SecureField]) -> Vec<SecureField> {
    let mut products = vec![SecureField::from(1u32); 1 << factors.len()];
    for (j, &factor) in factors.iter().enumerate() {
        let half = 1 << j;
        for i in 0..half {
            products[i + half] = products[i] * factor;
        }
    }
    products
}

fn words(values: &[SecureField]) -> Vec<u32> {
    values.iter().flat_map(|v| v.to_m31_array().map(|m| m.0)).collect()
}

/// The smallest columns the GPU samples (the others are left to the CPU).
pub const MIN_LOG_SIZE: u32 = LOG_VECTOR as u32 + 1;

/// One group of columns to sample: `columns` (word offsets in `buffer`) of `2^log_size`
/// coefficients, at `point`.
pub struct OodJob<'a> {
    pub buffer: &'a wgpu::Buffer,
    pub columns: &'a [u32],
    pub log_size: u32,
    pub point: CirclePoint<SecureField>,
}

/// A job's plan: its point's mappings and weight tables.
struct Plan {
    leading: Vec<SecureField>,
    lo_log: u32,
    span: usize,
    n_spans: usize,
    lo: Vec<u32>,
    hi: Vec<u32>,
}

fn plan(job: &OodJob<'_>) -> Plan {
    assert!(job.log_size >= MIN_LOG_SIZE, "a column of more than 16 coefficients");
    let mappings = SimdBackend::evaluation_mappings(job.point, job.log_size);
    let n_vectors_log = job.log_size - LOG_VECTOR as u32;
    let lo_log = n_vectors_log.min(MAX_LO_LOG);
    let n_chunks = 1usize << (n_vectors_log - lo_log);
    let span = n_chunks.min(MAX_SPAN);
    let lo_end = LOG_VECTOR + lo_log as usize;
    Plan {
        leading: tensor_products(&mappings[..LOG_VECTOR]),
        lo_log,
        span,
        n_spans: n_chunks / span,
        lo: words(&tensor_products(&mappings[LOG_VECTOR..lo_end])),
        hi: words(&tensor_products(&mappings[lo_end..])),
    }
}

/// A word the GPU wrote as an M31 value (`P` read as zero).
fn m31(word: u32) -> BaseField {
    BaseField::from_u32_unchecked(word % P)
}

/// The values of every job's columns at its point (per job, per column): what stwo's
/// out-of-domain sampling computes for each column. All the jobs in one submission and one
/// readback.
pub fn eval_jobs(
    gpu: &GpuContext,
    kernels: &OodKernels,
    jobs: &[OodJob<'_>],
) -> Vec<Vec<SecureField>> {
    let plans = jobs.iter().map(|job| (!job.columns.is_empty()).then(|| plan(job))).collect_vec();
    let init = |label: &str, words: &[u32], usage: wgpu::BufferUsages| {
        gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(words),
            usage,
        })
    };
    let storage = wgpu::BufferUsages::STORAGE;
    // Per job, its first output word: 64 per column (4 planes of 16 sums).
    let mut out_words = 0usize;
    let layout = jobs
        .iter()
        .map(|job| {
            let at = out_words;
            out_words += job.columns.len() * 64;
            at
        })
        .collect_vec();
    if out_words == 0 {
        return jobs.iter().map(|_| vec![]).collect();
    }
    let out = gpu.storage("ood out", out_words);
    let mut keep = vec![];
    struct Dispatch {
        bind_fold: wgpu::BindGroup,
        bind_reduce: wgpu::BindGroup,
        fold: (u32, u32),
        reduce: (u32, u32),
    }
    let mut dispatches = vec![];
    for ((job, plan), &out_at) in jobs.iter().zip(&plans).zip(&layout) {
        let Some(plan) = plan else { continue };
        let n_columns = job.columns.len();
        let lo = init("ood lo", &plan.lo, storage);
        let hi = init("ood hi", &plan.hi, storage);
        let columns = init("ood columns", job.columns, storage);
        let partial = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ood partial"),
            size: (n_columns * plan.n_spans * 64 * 4) as u64,
            usage: storage,
            mapped_at_creation: false,
        });
        let total_wg = (n_columns * plan.n_spans) as u64;
        let wg_x = total_wg.min(32768) as u32;
        let wg_y = total_wg.div_ceil(wg_x as u64) as u32;
        let params = Params {
            lo_log: plan.lo_log,
            span: plan.span as u32,
            n_spans: plan.n_spans as u32,
            n_columns: n_columns as u32,
            wg_x,
            pad: [0; 3],
        };
        let uniform =
            init("ood params", bytemuck::cast_slice(&[params]), wgpu::BufferUsages::UNIFORM);
        let out_view = wgpu::BufferBinding {
            buffer: &out,
            offset: (out_at * 4) as u64,
            size: wgpu::BufferSize::new((n_columns * 64 * 4) as u64),
        };
        let bind = |pipeline: &wgpu::ComputePipeline,
                    list: Vec<(u32, wgpu::BindingResource<'_>)>| {
            let entries = list
                .into_iter()
                .map(|(binding, resource)| wgpu::BindGroupEntry { binding, resource })
                .collect_vec();
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &entries,
            })
        };
        let bind_fold = bind(
            &kernels.fold,
            vec![
                (0, job.buffer.as_entire_binding()),
                (1, columns.as_entire_binding()),
                (2, lo.as_entire_binding()),
                (3, hi.as_entire_binding()),
                (4, partial.as_entire_binding()),
                (5, uniform.as_entire_binding()),
            ],
        );
        let bind_reduce = bind(
            &kernels.reduce,
            vec![
                (4, partial.as_entire_binding()),
                (5, uniform.as_entire_binding()),
                (6, wgpu::BindingResource::Buffer(out_view)),
            ],
        );
        let reduce_wg = ((n_columns * 64) as u64).div_ceil(256);
        let reduce_x = reduce_wg.min(32768) as u32;
        dispatches.push(Dispatch {
            bind_fold,
            bind_reduce,
            fold: (wg_x, wg_y),
            reduce: (reduce_x, reduce_wg.div_ceil(reduce_x as u64) as u32),
        });
        keep.extend([lo, hi, columns, partial, uniform]);
    }
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        for dispatch in &dispatches {
            pass.set_pipeline(&kernels.fold);
            pass.set_bind_group(0, &dispatch.bind_fold, &[]);
            pass.dispatch_workgroups(dispatch.fold.0, dispatch.fold.1, 1);
            pass.set_pipeline(&kernels.reduce);
            pass.set_bind_group(0, &dispatch.bind_reduce, &[]);
            pass.dispatch_workgroups(dispatch.reduce.0, dispatch.reduce.1, 1);
        }
    }
    gpu.queue.submit([encoder.finish()]);
    let sums = gpu.read(&out, out_words);
    for buffer in keep {
        buffer.destroy();
    }
    out.destroy();

    // Every column: the point's leading mappings' products against its 16 sums.
    jobs.iter()
        .zip(&plans)
        .zip(&layout)
        .map(|((job, plan), &at)| {
            let Some(plan) = plan else { return vec![] };
            (0..job.columns.len())
                .map(|c| {
                    let base = at + c * 64;
                    plan.leading
                        .iter()
                        .enumerate()
                        .map(|(t, &product)| {
                            let sum = SecureField::from_m31(
                                m31(sums[base + t]),
                                m31(sums[base + 16 + t]),
                                m31(sums[base + 32 + t]),
                                m31(sums[base + 48 + t]),
                            );
                            product * sum
                        })
                        .sum()
                })
                .collect()
        })
        .collect()
}

/// The values at `point` of the columns of `2^log_size` coefficients at the word offsets
/// `columns` of `buffer` (stwo's storage order).
pub fn eval_at_point(
    gpu: &GpuContext,
    kernels: &OodKernels,
    buffer: &wgpu::Buffer,
    columns: &[u32],
    log_size: u32,
    point: CirclePoint<SecureField>,
) -> Vec<SecureField> {
    eval_jobs(gpu, kernels, &[OodJob { buffer, columns, log_size, point }]).remove(0)
}
