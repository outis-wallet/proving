//! The decommitment of a striped tree's columns on the GPU, from their coefficients resident
//! there (see [`crate::parts::PartsMode::Coefficients`]).
//!
//! A tree that keeps only its columns' first stripe opens the other rows its decommitment reads
//! by evaluating each column there (stwo's `decommit_from_stripes`: the coefficients interpolated
//! from the first stripe, then `PolyOps::eval_at_domain_indices`). The value of a polynomial of
//! `2^n` coefficients `c_k` (in stwo's storage order) at a point is `sum_k c_k prod_{j in bits(k)}
//! m_j`, the mappings `m_j` of the point as `SimdBackend::evaluation_mappings` gives them. The
//! points that share their mappings past the first four (a group) share the weights of the
//! coefficient vectors (16 coefficients each): one pass over a column folds it into 16 sums per
//! group (`shaders/fold.wgsl`), and each point is then a sum of 16 products, done here. The
//! arithmetic is exact in the field: the values are those stwo computes, word for word.

use itertools::Itertools;
use stwo::core::fields::m31::BaseField;
use stwo::core::poly::circle::CircleDomain;
use stwo::core::utils::bit_reverse_index;
use stwo::prover::backend::simd::SimdBackend;
use wgpu::util::DeviceExt;

use crate::{GpuContext, M31_WGSL};

/// The mappings shared within a group (those past the first four).
const LOG_GROUP: usize = 4;
/// The most vectors a chunk folds in workgroup memory.
const MAX_LO_LOG: u32 = 8;
/// The most bytes of partial sums one dispatch writes.
const MAX_PARTIAL_BYTES: u64 = 64 << 20;

/// The fold's pipelines.
pub struct FoldKernels {
    fold: wgpu::ComputePipeline,
    reduce: wgpu::ComputePipeline,
}

impl FoldKernels {
    /// The kernels' source (`M31_WGSL` prepended).
    pub fn source() -> String {
        format!("{M31_WGSL}\n{}", include_str!("shaders/fold.wgsl"))
    }

    pub fn new(gpu: &GpuContext) -> Self {
        let source = Self::source();
        Self {
            fold: gpu.pipeline("decommit fold", &source, "fold"),
            reduce: gpu.pipeline("decommit reduce", &source, "reduce"),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    lo_log: u32,
    n_chunks: u32,
    n_groups: u32,
    n_columns: u32,
    wg_x: u32,
    pad: [u32; 3],
}

/// The products of `factors` over the set bits of every index below `1 << factors.len()`.
fn tensor_products(factors: &[BaseField]) -> Vec<BaseField> {
    let mut products = vec![BaseField::from(1u32); 1 << factors.len()];
    for (j, &factor) in factors.iter().enumerate() {
        let half = 1 << j;
        for i in 0..half {
            products[i + half] = products[i] * factor;
        }
    }
    products
}

/// One group of columns to evaluate: `columns` (word offsets in `buffer`) of `2^log_size`
/// coefficients, at the points of `domain` at the bit-reversed positions `indices`.
pub struct FoldJob<'a> {
    pub buffer: &'a wgpu::Buffer,
    pub columns: &'a [u32],
    pub log_size: u32,
    pub domain: CircleDomain,
    pub indices: &'a [usize],
}

/// A job's plan: its points' mappings, groups and weight tables.
struct Plan {
    mappings: Vec<Vec<BaseField>>,
    group_of_point: Vec<usize>,
    n_groups: usize,
    lo_log: u32,
    n_chunks: usize,
    lo: Vec<u32>,
    hi: Vec<u32>,
}

fn plan(job: &FoldJob<'_>) -> Plan {
    assert!(job.log_size as usize > LOG_GROUP, "a column of more than 16 coefficients");
    let mappings = job
        .indices
        .iter()
        .map(|&index| {
            let point = job.domain.at(bit_reverse_index(index, job.domain.log_size()));
            SimdBackend::evaluation_mappings(point, job.log_size)
        })
        .collect_vec();
    // The groups of points with the same mappings past the first four.
    let mut groups: Vec<&[BaseField]> = vec![];
    let group_of_point = mappings
        .iter()
        .map(|m| {
            let shared = &m[LOG_GROUP..];
            groups.iter().position(|&g| g == shared).unwrap_or_else(|| {
                groups.push(shared);
                groups.len() - 1
            })
        })
        .collect_vec();
    let n_vectors_log = job.log_size - LOG_GROUP as u32;
    let lo_log = n_vectors_log.min(MAX_LO_LOG);
    let n_chunks = 1usize << (n_vectors_log - lo_log);
    let lo: Vec<u32> = groups
        .iter()
        .flat_map(|g| tensor_products(&g[..lo_log as usize]).into_iter().map(|w| w.0))
        .collect();
    let hi: Vec<u32> = groups
        .iter()
        .flat_map(|g| tensor_products(&g[lo_log as usize..]).into_iter().map(|w| w.0))
        .collect();
    let n_groups = groups.len();
    Plan { mappings, group_of_point, n_groups, lo_log, n_chunks, lo, hi }
}

/// The values of every job's columns at its points (per job, per column): what
/// `SimdBackend::eval_at_domain_indices` returns for each column. All the jobs in one
/// submission and one readback.
pub fn eval_jobs(gpu: &GpuContext, kernels: &FoldKernels, jobs: &[FoldJob<'_>]) -> Vec<Vec<Vec<BaseField>>> {
    let plans = jobs
        .iter()
        .map(|job| (!job.indices.is_empty() && !job.columns.is_empty()).then(|| plan(job)))
        .collect_vec();
    // Every dispatch's buffers: the tables, the columns, the partial sums, the uniform; the sums
    // of all of them in one output buffer.
    let init = |label: &str, words: &[u32], usage: wgpu::BufferUsages| {
        gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(words),
            usage,
        })
    };
    let storage = wgpu::BufferUsages::STORAGE;
    struct Dispatch {
        bind_fold: wgpu::BindGroup,
        bind_reduce: wgpu::BindGroup,
        fold: (u32, u32),
        reduce: (u32, u32),
        /// The output's words this dispatch writes from, and how many.
        out_at: usize,
    }
    let mut out_words = 0usize;
    let mut layout = vec![]; // per job: (first output word, columns per batch)
    let mut pieces = vec![]; // per batch: (job, first column, n columns, out word)
    for (j, (job, plan)) in jobs.iter().zip(&plans).enumerate() {
        let Some(plan) = plan else {
            layout.push(None);
            continue;
        };
        let per_column_partial = (plan.n_chunks * plan.n_groups * 16 * 4) as u64;
        let batch = ((MAX_PARTIAL_BYTES / per_column_partial).max(1) as usize).min(job.columns.len());
        layout.push(Some(out_words));
        for first in (0..job.columns.len()).step_by(batch) {
            let n = batch.min(job.columns.len() - first);
            pieces.push((j, first, n, out_words));
            out_words += n * plan.n_groups * 16;
        }
    }
    if out_words == 0 {
        return jobs.iter().map(|job| job.columns.iter().map(|_| vec![]).collect()).collect();
    }
    let out = gpu.storage("fold out", out_words);
    let mut keep = vec![];
    let mut dispatches = vec![];
    for &(j, first, n_columns, out_at) in &pieces {
        let job = &jobs[j];
        let plan = plans[j].as_ref().unwrap();
        let lo = init("fold lo", &plan.lo, storage);
        let hi = init("fold hi", &plan.hi, storage);
        let columns = init("fold columns", &job.columns[first..first + n_columns], storage);
        let partial = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fold partial"),
            size: (n_columns * plan.n_chunks * plan.n_groups * 16 * 4) as u64,
            usage: storage,
            mapped_at_creation: false,
        });
        let n_blocks = plan.n_groups.div_ceil(16);
        let total_wg = (n_columns * plan.n_chunks * n_blocks) as u64;
        let wg_x = total_wg.min(32768) as u32;
        let wg_y = total_wg.div_ceil(wg_x as u64) as u32;
        let params = Params {
            lo_log: plan.lo_log,
            n_chunks: plan.n_chunks as u32,
            n_groups: plan.n_groups as u32,
            n_columns: n_columns as u32,
            wg_x,
            pad: [0; 3],
        };
        let uniform =
            init("fold params", bytemuck::cast_slice(&[params]), wgpu::BufferUsages::UNIFORM);
        let out_view = wgpu::BufferBinding {
            buffer: &out,
            offset: (out_at * 4) as u64,
            size: wgpu::BufferSize::new((n_columns * plan.n_groups * 16 * 4) as u64),
        };
        let entries = |pipeline: &wgpu::ComputePipeline, list: Vec<(u32, wgpu::BindingResource<'_>)>| {
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
        let bind_fold = entries(
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
        let bind_reduce = entries(
            &kernels.reduce,
            vec![
                (4, partial.as_entire_binding()),
                (5, uniform.as_entire_binding()),
                (6, wgpu::BindingResource::Buffer(out_view)),
            ],
        );
        let reduce_wg = ((n_columns * plan.n_groups * 16) as u64).div_ceil(256);
        let reduce_x = reduce_wg.min(32768) as u32;
        dispatches.push(Dispatch {
            bind_fold,
            bind_reduce,
            fold: (wg_x, wg_y),
            reduce: (reduce_x, reduce_wg.div_ceil(reduce_x as u64) as u32),
            out_at,
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
            let _ = dispatch.out_at;
        }
    }
    gpu.queue.submit([encoder.finish()]);
    let sums = gpu.read(&out, out_words);
    for buffer in keep {
        buffer.destroy();
    }
    out.destroy();

    // Every point: its leading mappings' products against its group's sums.
    jobs.iter()
        .zip(&plans)
        .zip(&layout)
        .map(|((job, plan), at)| {
            let (Some(plan), Some(at)) = (plan, at) else {
                return job.columns.iter().map(|_| vec![]).collect();
            };
            let leading = plan.mappings.iter().map(|m| tensor_products(&m[..LOG_GROUP])).collect_vec();
            (0..job.columns.len())
                .map(|c| {
                    let base = at + c * plan.n_groups * 16;
                    plan.group_of_point
                        .iter()
                        .zip(&leading)
                        .map(|(&g, products)| {
                            let sums = &sums[base + g * 16..base + (g + 1) * 16];
                            products
                                .iter()
                                .zip(sums)
                                .map(|(&product, &sum)| product * BaseField::from_u32_unchecked(sum))
                                .sum()
                        })
                        .collect()
                })
                .collect()
        })
        .collect()
}

/// The values at the points of `domain` at the bit-reversed positions `indices` of the columns of
/// `2^log_size` coefficients at the word offsets `columns` of `buffer` (stwo's storage order): per
/// column, what `SimdBackend::eval_at_domain_indices` returns.
pub fn eval_at_domain_indices(
    gpu: &GpuContext,
    kernels: &FoldKernels,
    buffer: &wgpu::Buffer,
    columns: &[u32],
    log_size: u32,
    domain: CircleDomain,
    indices: &[usize],
) -> Vec<Vec<BaseField>> {
    eval_jobs(gpu, kernels, &[FoldJob { buffer, columns, log_size, domain, indices }]).remove(0)
}
