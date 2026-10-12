//! The FRI quotients on the GPU (stwo's `compute_fri_quotients`), from the columns' coefficients
//! resident there (see [`crate::parts::PartsMode::Coefficients`]).
//!
//! stwo, per log size `l` of the committed columns and per sample point `p` (a batch), sums the
//! numerators `c_m f_m(x) - b_m` of the batch's members over the first stripe of the domain (`2^n`
//! points, `n = l - log_blowup`), lifts these sums to the first stripe of the lifting domain
//! (`2^N` points), adds per point `-A_p y`, divides by the point's line through `x`, sums over the
//! points, and extends the result (its 4 coordinates) from that stripe to the lifting domain.
//!
//! Here the sum `sum_m c_m f_m` of a batch is formed from the members' coefficients
//! (`shaders/fri.wgsl`, `combine`, coordinate by coordinate) and evaluated once on its stripe 0
//! (the commitment's extension, [`StripeExtension`]): the polynomial of the sum is the sum of the
//! polynomials, so its values are stwo's sums. The columns not kept as coefficients are added
//! from their stripe-0 values. The constants (the samples with their random coefficients, the
//! batches and their line coefficients) are stwo's own
//! (`build_samples_with_randomness_and_periodicity`, `ColumnSampleBatch::new_vec`,
//! `quotient_constants`), computed on the CPU. The quotients are formed per row of the lifting
//! domain's stripe 0 (`quotients`), interpolated (the part extension's inverse FFT, [`IfftPlan`])
//! and evaluated on every stripe of the lifting domain (the extension). The arithmetic is exact in
//! the field and every value is canonical: the result is stwo's, word for word.

use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::Instant;

use itertools::Itertools;
use stwo::core::circle::CirclePoint;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::core::pcs::TreeVec;
use stwo::core::pcs::quotients::{
    ColumnSampleBatch, PointSample, build_samples_with_randomness_and_periodicity,
    quotient_constants,
};
use stwo::core::poly::circle::CanonicCoset;
use stwo::prover::backend::Column;
use stwo::prover::backend::simd::column::BaseColumn;
use wgpu::util::DeviceExt;

use crate::constraint::wgsl::QM31_WGSL;
use crate::extension::{
    ColumnsLayout, ExtensionKernels, GpuColumns, MIN_LOG_SIZE, Merge, StripeExtension,
    StripeTwiddles,
};
use crate::recompute::{IfftKernels, IfftPlan, Tables};
use crate::transfer::{Readback, Upload};
use crate::{GpuContext, M31_WGSL};

/// The most bytes of stripe-0 values of columns not kept as coefficients the GPU takes (past it
/// the quotients stay on the CPU).
pub const MAX_VALUE_BYTES: u64 = 256 << 20;
/// The most workgroups of a row of a dispatch grid.
const MAX_WG_X: u32 = 32768;

/// The quotients' pipelines.
pub struct FriKernels {
    combine: wgpu::ComputePipeline,
    quotients: wgpu::ComputePipeline,
}

impl FriKernels {
    /// The kernels' source (`M31_WGSL` and `QM31_WGSL` prepended).
    pub fn source() -> String {
        format!("{M31_WGSL}\n{QM31_WGSL}\n{}", include_str!("shaders/fri.wgsl"))
    }

    pub fn new(gpu: &GpuContext) -> Self {
        let source = Self::source();
        Self {
            combine: gpu.pipeline("fri combine", &source, "combine"),
            quotients: gpu.pipeline("fri quotients", &source, "quotients"),
        }
    }
}

/// The quotients' kernels and what they keep: the inverse FFT's kernels and twiddles.
pub struct FriState {
    kernels: FriKernels,
    ifft: IfftKernels,
    /// The inverse twiddles of the lifting domains' stripe 0, by (log size, blowup).
    tables: Vec<((u32, u32), Tables)>,
}

impl FriState {
    pub fn new(gpu: &GpuContext) -> Self {
        Self { kernels: FriKernels::new(gpu), ifft: IfftKernels::new(gpu), tables: vec![] }
    }

    fn itw(&mut self, gpu: &GpuContext, log_size: u32, log_blowup: u32) -> &wgpu::Buffer {
        let key = (log_size, log_blowup);
        let at = match self.tables.iter().position(|(k, _)| *k == key) {
            Some(at) => at,
            None => {
                // A few shapes at most (one per leg).
                if self.tables.len() >= 4 {
                    self.tables.remove(0);
                }
                self.tables.push((key, Tables::new(gpu, log_size, log_blowup)));
                self.tables.len() - 1
            }
        };
        &self.tables[at].1.itw
    }
}

/// Where a column's data is.
#[derive(Clone, Copy)]
pub enum Source<'a> {
    /// Its `2^n` coefficients (stwo's storage order) at word `offset` of `buffer`.
    Coefficients { buffer: &'a wgpu::Buffer, offset: u32 },
    /// Its values on the CPU, stripe 0 first (the first `2^n` words).
    Values(&'a [BaseField]),
}

/// One committed column: its data, and its domain's log size.
#[derive(Clone, Copy)]
pub struct FriColumn<'a> {
    pub source: Source<'a>,
    pub log_size: u32,
}

/// What the quotients take from their caller besides the kernels.
pub trait FriResources {
    /// A storage buffer (`STORAGE | COPY_SRC | COPY_DST`) of at least `bytes`, for the use
    /// `slot` (one per use: the same slot is not asked for twice in one call).
    fn buffer(&self, slot: usize, bytes: u64) -> wgpu::Buffer;
    /// Every stripe's twiddles at `log_blowup` covering `log_sizes` (the commitments').
    fn stripe_twiddles(
        &self,
        log_sizes: &BTreeSet<u32>,
        log_blowup: u32,
    ) -> Option<Arc<StripeTwiddles>>;
    /// Writes `pieces` (their copies submitted before any later submission).
    fn upload(&self, pieces: &[Upload<'_>]) -> Result<(), String>;
    /// Reads `pieces` back (after the work submitted so far).
    fn read(&self, pieces: Vec<Readback<'_, '_>>) -> Result<(), String>;
}

/// The buffers' uses ([`FriResources::buffer`]'s slots). The quotients' buffers reuse the
/// numerators' once these are done with: the inverse FFT's work and the stripes' evaluations the
/// combined coefficients', the quotients' coefficients the numerators' evaluations'.
pub mod slot {
    pub const COMBINED: usize = 0;
    pub const EVALUATIONS: usize = 1;
    pub const VALUES: usize = 2;
    pub const N_SLOTS: usize = 3;
}

/// The bytes of [`FriResources::buffer`]'s slots a call would take (in [`slot`]'s order), for a
/// policy check before it, or why the GPU doesn't take it.
pub fn plan_bytes(
    columns: &TreeVec<Vec<FriColumn<'_>>>,
    samples: &TreeVec<Vec<Vec<PointSample>>>,
    lifting_log_size: u32,
    log_blowup: u32,
    max_buffer_words: u64,
) -> Result<[u64; slot::N_SLOTS], String> {
    let plan = Plan::new(columns, samples, SecureField::from(1u32), lifting_log_size, log_blowup)?;
    plan.bytes(max_buffer_words)
}

/// Where the time of a call went (wall clock, seconds): the plan (stwo's constants, on the CPU),
/// the setup (buffers, the values' upload, the dispatches), the GPU (submission to the end of the
/// readback) and the readback's copies.
#[derive(Clone, Copy, Debug, Default)]
pub struct FriTimes {
    pub plan: f64,
    pub setup: f64,
    pub gpu: f64,
    pub batches: usize,
    pub value_columns: usize,
}

impl std::ops::AddAssign for FriTimes {
    fn add_assign(&mut self, other: Self) {
        self.plan += other.plan;
        self.setup += other.setup;
        self.gpu += other.gpu;
        self.batches += other.batches;
        self.value_columns += other.value_columns;
    }
}

/// A batch: one log size, one sample point.
struct Batch<'a> {
    /// Its columns' stripe-0 log size.
    log_size: u32,
    point: CirclePoint<SecureField>,
    /// Its members: their data and their `c` coefficient.
    members: Vec<(Source<'a>, SecureField)>,
    a_sum: SecureField,
    b_sum: SecureField,
}

/// stwo's constants of the quotients, by batch.
struct Plan<'a> {
    batches: Vec<Batch<'a>>,
    /// The lifting domain's stripe-0 log size.
    log_n: u32,
}

impl<'a> Plan<'a> {
    fn new(
        columns: &TreeVec<Vec<FriColumn<'a>>>,
        samples: &TreeVec<Vec<Vec<PointSample>>>,
        random_coeff: SecureField,
        lifting_log_size: u32,
        log_blowup: u32,
    ) -> Result<Self, String> {
        if columns.len() != samples.len()
            || columns.iter().zip(samples.iter()).any(|(c, s)| c.len() != s.len())
        {
            return Err("columns and samples of different shapes".into());
        }
        // As compute_fri_quotients: the samples with their random coefficients, the columns by
        // log size (stable), the batches of each log size and their line coefficients.
        let samples_with_randomness = build_samples_with_randomness_and_periodicity(
            samples,
            columns.iter().map(|tree| tree.iter().map(|column| column.log_size)).collect(),
            lifting_log_size,
            random_coeff,
        );
        let mut batches = vec![];
        let groups = columns
            .iter()
            .flatten()
            .zip(samples_with_randomness.iter().flatten())
            .sorted_by_key(|(column, _)| column.log_size)
            .group_by(|(column, _)| column.log_size);
        for (log_size, group) in &groups {
            let (group_columns, group_samples): (Vec<&FriColumn<'a>>, Vec<_>) = group.unzip();
            if log_size < log_blowup || log_size - log_blowup < MIN_LOG_SIZE {
                return Err(format!("a column of log size {log_size}"));
            }
            let sample_batches = ColumnSampleBatch::new_vec(&group_samples);
            let constants = quotient_constants(&sample_batches);
            for (batch, coeffs) in sample_batches.iter().zip(&constants.line_coeffs) {
                let members = batch
                    .cols_vals_randpows
                    .iter()
                    .zip(coeffs)
                    .map(|(numerator, &(_, _, c))| {
                        (group_columns[numerator.column_index].source, c)
                    })
                    .collect_vec();
                batches.push(Batch {
                    log_size: log_size - log_blowup,
                    point: batch.point,
                    members,
                    a_sum: coeffs.iter().map(|(a, ..)| *a).sum(),
                    b_sum: coeffs.iter().map(|(_, b, _)| *b).sum(),
                });
            }
        }
        if lifting_log_size < log_blowup + MIN_LOG_SIZE {
            return Err(format!("a lifting domain of log size {lifting_log_size}"));
        }
        let log_n = lifting_log_size - log_blowup;
        if batches.iter().any(|batch| batch.log_size > log_n) {
            return Err("a column larger than the lifting domain".into());
        }
        Ok(Self { batches, log_n })
    }

    /// The layout of the batches' numerators (4 columns each).
    fn layout(&self, max_buffer_words: u64) -> ColumnsLayout {
        let log_sizes =
            self.batches.iter().flat_map(|batch| [batch.log_size; 4]).collect::<Vec<_>>();
        ColumnsLayout::new(&log_sizes, max_buffer_words)
    }

    /// The stripe-0 values the GPU takes from the CPU: words, with each column at a multiple
    /// of 64 words.
    fn value_words(&self) -> u64 {
        self.batches
            .iter()
            .flat_map(|batch| batch.members.iter().map(move |m| (batch.log_size, m.0)))
            .filter(|(_, source)| matches!(source, Source::Values(_)))
            .map(|(log_size, _)| (1u64 << log_size).next_multiple_of(64))
            .sum()
    }

    fn bytes(&self, max_buffer_words: u64) -> Result<[u64; slot::N_SLOTS], String> {
        let layout = self.layout(max_buffer_words);
        let [words] = layout.buffer_words.as_slice() else {
            return Err("the numerators past one buffer".into());
        };
        let value_bytes = 4 * self.value_words();
        if value_bytes > MAX_VALUE_BYTES {
            return Err(format!("{} MiB of values not kept as coefficients", value_bytes >> 20));
        }
        let quotient = ColumnsLayout::new(&[self.log_n; 4], max_buffer_words);
        let [quotient_words] = quotient.buffer_words.as_slice() else {
            return Err("the quotients past one buffer".into());
        };
        let mut bytes = [0u64; slot::N_SLOTS];
        bytes[slot::COMBINED] = words.max(quotient_words) * 4;
        bytes[slot::EVALUATIONS] = words.max(quotient_words) * 4;
        bytes[slot::VALUES] = value_bytes.max(256);
        Ok(bytes)
    }
}

fn qm31_words(value: SecureField) -> [u32; 4] {
    value.to_m31_array().map(|m| m.0)
}

fn grid(threads: u64) -> (u32, u32) {
    let workgroups = threads.div_ceil(256).max(1) as u32;
    let x = workgroups.min(MAX_WG_X);
    (x, workgroups.div_ceil(x))
}

/// stwo's `compute_fri_quotients` of `columns` (by tree, with their committed log sizes) and their
/// out-of-domain `samples`: the 4 coordinate columns of the quotients' evaluation on the lifting
/// domain (`CanonicCoset::new(lifting_log_size).circle_domain()`, bit-reversed), word for word.
#[allow(clippy::too_many_arguments)]
pub fn fri_quotients(
    gpu: &GpuContext,
    state: &mut FriState,
    extension_kernels: &ExtensionKernels,
    resources: &dyn FriResources,
    columns: &TreeVec<Vec<FriColumn<'_>>>,
    samples: &TreeVec<Vec<Vec<PointSample>>>,
    random_coeff: SecureField,
    lifting_log_size: u32,
    log_blowup: u32,
) -> Result<([BaseColumn; 4], FriTimes), String> {
    let mut times = FriTimes::default();
    let started = Instant::now();
    let plan = Plan::new(columns, samples, random_coeff, lifting_log_size, log_blowup)?;
    let max_words = crate::extension::max_buffer_words(gpu);
    let bytes = plan.bytes(max_words)?;
    let log_n = plan.log_n;
    times.plan = started.elapsed().as_secs_f64();
    times.batches = plan.batches.len();
    let started = Instant::now();

    // The twiddles: the batches' log sizes and the lifting domain's stripe 0.
    let log_sizes: BTreeSet<u32> =
        plan.batches.iter().map(|batch| batch.log_size).chain([log_n]).collect();
    let twiddles = resources
        .stripe_twiddles(&log_sizes, log_blowup)
        .ok_or_else(|| "no stripe twiddles".to_string())?;

    // The numerators: their coefficients (combined), then their stripe-0 values.
    let layout = Arc::new(plan.layout(max_words));
    let combined = GpuColumns::with_buffers(
        vec![resources.buffer(slot::COMBINED, bytes[slot::COMBINED])],
        layout.clone(),
    );
    let evaluations = GpuColumns::with_buffers(
        vec![resources.buffer(slot::EVALUATIONS, bytes[slot::EVALUATIONS])],
        layout.clone(),
    );
    let values = resources.buffer(slot::VALUES, bytes[slot::VALUES]);

    // The members' tables: per (batch, source), a run of (offset, c); the values' uploads.
    struct Run<'b> {
        batch: usize,
        src: &'b wgpu::Buffer,
        first: u32,
        count: u32,
        /// Into the evaluations (values) or the combined coefficients.
        values: bool,
    }
    let mut offsets: Vec<u32> = vec![];
    let mut coeffs: Vec<[u32; 4]> = vec![];
    let mut runs: Vec<Run<'_>> = vec![];
    let mut uploads: Vec<Upload<'_>> = vec![];
    let mut value_at = 0u64;
    for (b, batch) in plan.batches.iter().enumerate() {
        let n = 1usize << batch.log_size;
        // Members by source buffer (the values' buffer last), in their order otherwise.
        type Members = Vec<(u32, [u32; 4])>;
        let mut by_source: Vec<(Option<&wgpu::Buffer>, Members)> = vec![];
        for &(source, c) in &batch.members {
            let (key, offset) = match source {
                Source::Coefficients { buffer, offset } => (Some(buffer), offset),
                Source::Values(column) => {
                    if column.len() < n {
                        return Err("a column shorter than its stripe 0".into());
                    }
                    let at = value_at;
                    value_at += (n as u64).next_multiple_of(64);
                    uploads.push(Upload {
                        dst: &values,
                        dst_offset: at * 4,
                        src: bytemuck::cast_slice::<BaseField, u8>(&column[..n]),
                    });
                    times.value_columns += 1;
                    (None, at as u32)
                }
            };
            let same = |k: &Option<&wgpu::Buffer>| *k == key;
            match by_source.iter_mut().find(|(k, _)| same(k)) {
                Some((_, list)) => list.push((offset, qm31_words(c))),
                None => by_source.push((key, vec![(offset, qm31_words(c))])),
            }
        }
        for (key, list) in by_source {
            runs.push(Run {
                batch: b,
                src: key.unwrap_or(&values),
                first: offsets.len() as u32,
                count: list.len() as u32,
                values: key.is_none(),
            });
            for (offset, c) in list {
                offsets.push(offset);
                coeffs.push(c);
            }
        }
    }
    resources.upload(&uploads)?;
    drop(uploads);
    let init = |label: &str, contents: &[u8], usage: wgpu::BufferUsages| {
        gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents,
            usage,
        })
    };
    let storage = wgpu::BufferUsages::STORAGE;
    let uniform = wgpu::BufferUsages::UNIFORM;
    let offsets_buffer = init("fri member offsets", bytemuck::cast_slice(&offsets), storage);
    let coeffs_buffer = init("fri member coefficients", bytemuck::cast_slice(&coeffs), storage);
    let mut keep = vec![];
    // Each run's dispatch: (values or coefficients, bind group, workgroups).
    let mut combine_dispatches = vec![];
    for run in &runs {
        let batch = &plan.batches[run.batch];
        let n_words = 1u32 << batch.log_size;
        let (x, y) = grid(n_words as u64);
        let outs: [u32; 4] = std::array::from_fn(|k| layout.slots[4 * run.batch + k].offset as u32);
        let params: [u32; 8] =
            [n_words, run.first, run.count, x, outs[0], outs[1], outs[2], outs[3]];
        let params = init("fri combine params", bytemuck::cast_slice(&params), uniform);
        let out = if run.values { &evaluations.buffers[0] } else { &combined.buffers[0] };
        let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &state.kernels.combine.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: run.src.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: offsets_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: coeffs_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: out.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: params.as_entire_binding() },
            ],
        });
        combine_dispatches.push((run.values, group, (x, y)));
        keep.push(params);
    }
    let extension = StripeExtension::with_twiddles(
        gpu,
        extension_kernels,
        &combined,
        &evaluations,
        twiddles.clone(),
        Merge::Off,
    );

    // The quotients on the lifting domain's stripe 0: the points, their batches, the domain.
    let quotient_layout = Arc::new(ColumnsLayout::new(&[log_n; 4], max_words));
    // The combined coefficients are done with once evaluated, the numerators' evaluations once
    // the quotients are formed.
    let work = GpuColumns::with_buffers(combined.buffers.clone(), quotient_layout.clone());
    let coefficients =
        GpuColumns::with_buffers(evaluations.buffers.clone(), quotient_layout.clone());
    let stripe = GpuColumns::with_buffers(combined.buffers.clone(), quotient_layout.clone());
    let points = plan.batches.iter().map(|batch| batch.point).unique().collect_vec();
    let mut point_words: Vec<u32> = vec![];
    let mut point_batches: Vec<[u32; 2]> = vec![];
    let mut batch_words: Vec<u32> = vec![];
    for point in &points {
        let members = plan.batches.iter().enumerate().filter(|(_, b)| b.point == *point);
        let first = batch_words.len() as u32 / 8;
        let (mut a, mut b) = (SecureField::from(0u32), SecureField::from(0u32));
        for (g, batch) in members {
            a += batch.a_sum;
            b += batch.b_sum;
            let planes: [u32; 4] = std::array::from_fn(|k| layout.slots[4 * g + k].offset as u32);
            batch_words.extend(planes);
            batch_words.extend([batch.log_size, 0, 0, 0]);
        }
        point_batches.push([first, batch_words.len() as u32 / 8 - first]);
        let cm = |v: stwo::core::fields::cm31::CM31| [v.0.0, v.1.0];
        point_words.extend(cm(point.x.0));
        point_words.extend(cm(point.y.0));
        point_words.extend(cm(point.x.1));
        point_words.extend(cm(point.y.1));
        point_words.extend(qm31_words(a));
        point_words.extend(qm31_words(b));
    }
    let lifting_domain = CanonicCoset::new(lifting_log_size).circle_domain();
    let (stripe_domain, _) = lifting_domain.split(log_blowup);
    let half = stripe_domain.half_coset;
    let mut domain_words = vec![half.initial.x.0, half.initial.y.0];
    let mut step = half.step;
    for _ in 0..log_n.saturating_sub(1) {
        domain_words.extend([step.x.0, step.y.0]);
        step = step.double();
    }
    let (x, _) = grid(1u64 << log_n);
    let q_outs: [u32; 4] = std::array::from_fn(|k| quotient_layout.slots[k].offset as u32);
    let quotient_params: [u32; 8] =
        [log_n, points.len() as u32, x, q_outs[0], q_outs[1], q_outs[2], q_outs[3], 0];
    let buffers = [
        init("fri points", bytemuck::cast_slice(&point_words), storage),
        init("fri point batches", bytemuck::cast_slice(&point_batches), storage),
        init("fri batches", bytemuck::cast_slice(&batch_words), storage),
        init("fri domain", bytemuck::cast_slice(&domain_words), storage),
        init("fri quotient params", bytemuck::cast_slice(&quotient_params), uniform),
    ];
    let quotient_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &state.kernels.quotients.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 10,
                resource: evaluations.buffers[0].as_entire_binding(),
            },
            wgpu::BindGroupEntry { binding: 11, resource: buffers[0].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 12, resource: buffers[1].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 13, resource: buffers[2].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 14, resource: buffers[3].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 15, resource: work.buffers[0].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 16, resource: buffers[4].as_entire_binding() },
        ],
    });
    // Interpolated, then evaluated on every stripe of the lifting domain.
    let itw = state.itw(gpu, log_n, log_blowup).clone();
    let ifft = IfftPlan::new(gpu, &state.ifft, &itw, &work, &coefficients, log_n);
    let final_extension = StripeExtension::with_twiddles(
        gpu,
        extension_kernels,
        &coefficients,
        &stripe,
        twiddles,
        Merge::Off,
    );

    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    encoder.clear_buffer(&combined.buffers[0], 0, Some(layout.buffer_words[0] * 4));
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&state.kernels.combine);
        for (values, group, (x, y)) in &combine_dispatches {
            if !values {
                pass.set_bind_group(0, group, &[]);
                pass.dispatch_workgroups(*x, *y, 1);
            }
        }
    }
    extension.encode(extension_kernels, &mut encoder, 0);
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&state.kernels.combine);
        for (values, group, (x, y)) in &combine_dispatches {
            if *values {
                pass.set_bind_group(0, group, &[]);
                pass.dispatch_workgroups(*x, *y, 1);
            }
        }
        pass.set_pipeline(&state.kernels.quotients);
        pass.set_bind_group(0, &quotient_group, &[]);
        let (x, y) = grid(1u64 << log_n);
        pass.dispatch_workgroups(x, y, 1);
    }
    ifft.encode(&state.ifft, &mut encoder);
    times.setup = started.elapsed().as_secs_f64();
    let started = Instant::now();

    // Stripe by stripe, evaluated and read back into place: stripe s of the lifting domain is
    // rows s 2^N .. (s + 1) 2^N (bit-reversed).
    // Safety: every value is read back.
    let mut result: [BaseColumn; 4] =
        std::array::from_fn(|_| unsafe { BaseColumn::uninitialized(1 << lifting_log_size) });
    let stripe_words = 1usize << log_n;
    let mut encoder = Some(encoder);
    for s in 0..1usize << log_blowup {
        let mut stripe_encoder = encoder
            .take()
            .unwrap_or_else(|| gpu.device.create_command_encoder(&Default::default()));
        final_extension.encode(extension_kernels, &mut stripe_encoder, s);
        gpu.queue.submit([stripe_encoder.finish()]);
        let pieces = result
            .iter_mut()
            .enumerate()
            .map(|(k, column)| Readback {
                src: &stripe.buffers[0],
                src_offset: quotient_layout.slots[k].offset * 4,
                dst: bytemuck::cast_slice_mut::<BaseField, u8>(
                    &mut column.as_mut_slice()[s * stripe_words..(s + 1) * stripe_words],
                ),
            })
            .collect_vec();
        resources.read(pieces)?;
    }
    times.gpu = started.elapsed().as_secs_f64();

    extension.destroy();
    final_extension.destroy();
    ifft.destroy();
    for buffer in keep.iter().chain(&buffers).chain([&offsets_buffer, &coeffs_buffer]) {
        buffer.destroy();
    }
    Ok((result, times))
}
