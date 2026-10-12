//! The generated constraint kernels on a device: their pipelines, the inputs' buffers, and the
//! dispatches, with timestamps.

use std::ops::Range;
use std::time::{Duration, Instant};

use stwo::core::fields::qm31::QM31;
use wgpu::util::DeviceExt;

use super::ir::{NodeId, Program, words};
use super::wgsl::{ColumnGroups, Kernel, emit, uniform_nodes};
use crate::GpuContext;

#[derive(Clone, Copy, Debug)]
pub struct KernelOptions {
    pub workgroup_size: u32,
    /// The number of kernels the constraints are split over (contiguous ranges of about equal
    /// numbers of constraints), each adding its share.
    pub split: usize,
    /// The largest column binding, in bytes (default: the device's limit).
    pub max_binding_bytes: Option<u64>,
    /// Pipelines without wgpu's runtime bounds checks (see [`GpuContext::pipeline_unchecked`]).
    pub unchecked: bool,
}

impl Default for KernelOptions {
    fn default() -> Self {
        Self { workgroup_size: 256, split: 1, max_binding_bytes: None, unchecked: false }
    }
}

pub struct ConstraintKernels {
    pub groups: ColumnGroups,
    pub kernels: Vec<(Range<usize>, Kernel)>,
    pipelines: Vec<wgpu::ComputePipeline>,
    layout: wgpu::BindGroupLayout,
    pub uniform_nodes: Vec<NodeId>,
    pub trace_log: u32,
    pub eval_log: u32,
    pub workgroup_size: u32,
    /// Generating the WGSL.
    pub emit_time: Duration,
    /// Creating the shader modules and pipelines (the driver's compilation).
    pub compile_time: Duration,
}

/// The inputs of a component's evaluation on the device.
pub struct GpuInputs {
    pub columns: Vec<wgpu::Buffer>,
    pub uniforms: wgpu::Buffer,
    pub coeffs: wgpu::Buffer,
    pub denom_inv: wgpu::Buffer,
    pub accum: wgpu::Buffer,
}

impl GpuInputs {
    /// Their size in bytes.
    pub fn bytes(&self) -> u64 {
        self.columns.iter().map(|b| b.size()).sum::<u64>()
            + self.uniforms.size()
            + self.coeffs.size()
            + self.denom_inv.size()
            + self.accum.size()
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

impl ConstraintKernels {
    pub fn new(
        gpu: &GpuContext,
        program: &Program,
        n_columns: &[usize],
        trace_log: u32,
        eval_log: u32,
        options: KernelOptions,
    ) -> Self {
        let limits = gpu.device.limits();
        let device_max =
            (limits.max_storage_buffer_binding_size as u64).min(limits.max_buffer_size);
        let max_binding = options.max_binding_bytes.unwrap_or(device_max).min(device_max);
        let column_bytes = 4u64 << eval_log;
        let max_columns = (max_binding / column_bytes).max(1) as usize;
        let groups = ColumnGroups::plan(n_columns, max_columns);
        assert!(
            groups.groups.len() + 4 <= limits.max_storage_buffers_per_shader_stage as usize,
            "{} column bindings exceed the device's storage buffers",
            groups.groups.len()
        );
        let n = program.constraints.len();
        let split = options.split.clamp(1, n.max(1));
        let ranges: Vec<Range<usize>> =
            (0..split).map(|k| k * n / split..(k + 1) * n / split).collect();
        let uniforms = uniform_nodes(program, 0..n);

        let start = Instant::now();
        let kernels: Vec<(Range<usize>, Kernel)> = ranges
            .into_iter()
            .map(|range| {
                let kernel = emit(
                    program,
                    range.clone(),
                    &groups,
                    &uniforms,
                    trace_log,
                    eval_log,
                    options.workgroup_size,
                );
                (range, kernel)
            })
            .collect();
        let emit_time = start.elapsed();

        let entries: Vec<wgpu::BindGroupLayoutEntry> = (0..4 + groups.groups.len() as u32)
            .map(|binding| storage_entry(binding, binding != 3))
            .collect();
        let layout = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("constraint eval"),
            entries: &entries,
        });
        let pipeline_layout = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("constraint eval"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let start = Instant::now();
        let pipelines = kernels
            .iter()
            .map(|(_, kernel)| {
                let descriptor = wgpu::ShaderModuleDescriptor {
                    label: Some("constraint eval"),
                    source: wgpu::ShaderSource::Wgsl(kernel.source.as_str().into()),
                };
                let module = if options.unchecked {
                    // Safety: every index is a row below N into a column of N words, a coefficient
                    // or uniform the program's own, or a part index below `denom_inv`'s length.
                    unsafe {
                        gpu.device.create_shader_module_trusted(
                            descriptor,
                            wgpu::ShaderRuntimeChecks::unchecked(),
                        )
                    }
                } else {
                    gpu.device.create_shader_module(descriptor)
                };
                gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("constraint eval"),
                    layout: Some(&pipeline_layout),
                    module: &module,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                })
            })
            .collect();
        // Pipeline creation is synchronous natively; wait for the device anyway.
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        let compile_time = start.elapsed();
        Self {
            groups,
            kernels,
            pipelines,
            layout,
            uniform_nodes: uniforms,
            trace_log,
            eval_log,
            workgroup_size: options.workgroup_size,
            emit_time,
            compile_time,
        }
    }

    /// The uniforms buffer's words, from [`Program::bind`]'s values.
    pub fn uniform_words(&self, values: &[Option<QM31>]) -> Vec<u32> {
        let mut words_out: Vec<u32> =
            self.uniform_nodes.iter().flat_map(|id| words(values[*id as usize].unwrap())).collect();
        if words_out.is_empty() {
            words_out = vec![0; 4];
        }
        words_out
    }

    /// Creates the buffers and uploads the inputs (`columns[interaction][index]`, `accum`
    /// `[coordinate][row]`), waiting until the device has them.
    pub fn upload(
        &self,
        gpu: &GpuContext,
        columns: &[Vec<&[u32]>],
        uniform_values: &[Option<QM31>],
        coeffs: &[QM31],
        denom_inv: &[u32],
        accum: &[&[u32]; 4],
    ) -> GpuInputs {
        let n = 1usize << self.eval_log;
        let create = |label: &str, words: usize| {
            gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (words.max(4) * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let column_buffers: Vec<wgpu::Buffer> = self
            .groups
            .groups
            .iter()
            .map(|group| {
                let buffer = create("constraint columns", group.len() * n);
                for (k, (i, j)) in group.iter().enumerate() {
                    let column = columns[*i as usize][*j as usize];
                    assert_eq!(column.len(), n, "a column on the evaluation domain");
                    gpu.queue.write_buffer(&buffer, (k * n * 4) as u64, bytemuck::cast_slice(column));
                }
                buffer
            })
            .collect();
        let uniforms = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("uniforms"),
            contents: bytemuck::cast_slice(&self.uniform_words(uniform_values)),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let coeff_words: Vec<u32> = coeffs.iter().flat_map(|c| words(*c)).collect();
        let coeffs = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("coeffs"),
            contents: bytemuck::cast_slice(&coeff_words),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let denom = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("denom_inv"),
            contents: bytemuck::cast_slice(denom_inv),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let accum_buffer = create("accum", 4 * n);
        for (c, coordinate) in accum.iter().enumerate() {
            gpu.queue.write_buffer(&accum_buffer, (c * n * 4) as u64, bytemuck::cast_slice(coordinate));
        }
        gpu.queue.submit([]);
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        GpuInputs { columns: column_buffers, uniforms, coeffs, denom_inv: denom, accum: accum_buffer }
    }

    /// Writes `accum` (`[coordinate][row]`) into the accumulator buffer again.
    pub fn reset_accum(&self, gpu: &GpuContext, inputs: &GpuInputs, accum: &[&[u32]; 4]) {
        let n = 1usize << self.eval_log;
        for (c, coordinate) in accum.iter().enumerate() {
            gpu.queue.write_buffer(&inputs.accum, (c * n * 4) as u64, bytemuck::cast_slice(coordinate));
        }
        gpu.queue.submit([]);
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    }

    pub fn bind_group(&self, gpu: &GpuContext, inputs: &GpuInputs) -> wgpu::BindGroup {
        let mut buffers = vec![&inputs.uniforms, &inputs.coeffs, &inputs.denom_inv, &inputs.accum];
        buffers.extend(inputs.columns.iter());
        let entries: Vec<wgpu::BindGroupEntry> = buffers
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("constraint eval"),
            layout: &self.layout,
            entries: &entries,
        })
    }

    fn workgroups(&self) -> (u32, u32) {
        let groups = (1u64 << self.eval_log).div_ceil(self.workgroup_size as u64);
        let x = groups.min(32768);
        (x as u32, groups.div_ceil(x) as u32)
    }

    /// Runs the kernels once on resident inputs and waits: the GPU time of the passes (from
    /// timestamps, if the device has them) and the wall-clock time of submit-to-idle.
    pub fn run(&self, gpu: &GpuContext, bind_group: &wgpu::BindGroup) -> (Option<Duration>, Duration) {
        let timestamps = gpu.device.features().contains(wgpu::Features::TIMESTAMP_QUERY).then(|| {
            let count = 2 * self.pipelines.len() as u32;
            let query_set = gpu.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("constraint eval timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count,
            });
            let resolve = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 8 * count as u64,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 8 * count as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            (query_set, resolve, readback, count)
        });
        let (x, y) = self.workgroups();
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        for (k, pipeline) in self.pipelines.iter().enumerate() {
            let timestamp_writes = timestamps.as_ref().map(|(query_set, ..)| {
                wgpu::ComputePassTimestampWrites {
                    query_set,
                    beginning_of_pass_write_index: Some(2 * k as u32),
                    end_of_pass_write_index: Some(2 * k as u32 + 1),
                }
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("constraint eval"),
                timestamp_writes,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(x, y, 1);
        }
        if let Some((query_set, resolve, readback, count)) = &timestamps {
            encoder.resolve_query_set(query_set, 0..*count, resolve, 0);
            encoder.copy_buffer_to_buffer(resolve, 0, readback, 0, 8 * *count as u64);
        }
        let start = Instant::now();
        gpu.queue.submit([encoder.finish()]);
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        let wall = start.elapsed();
        let gpu_time = timestamps.map(|(_, _, readback, count)| {
            let slice = readback.slice(..);
            slice.map_async(wgpu::MapMode::Read, |result| result.expect("map"));
            gpu.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
            let ticks: Vec<u64> =
                bytemuck::cast_slice(&slice.get_mapped_range().expect("mapped")[..]).to_vec();
            readback.unmap();
            let period = gpu.queue.get_timestamp_period() as f64;
            let ns: f64 = (0..count as usize / 2)
                .map(|k| ticks[2 * k + 1].wrapping_sub(ticks[2 * k]) as f64 * period)
                .sum();
            Duration::from_nanos(ns as u64)
        });
        (gpu_time, wall)
    }

    /// The accumulator buffer's words, `[coordinate][row]`.
    pub fn read_accum(&self, gpu: &GpuContext, inputs: &GpuInputs) -> [Vec<u32>; 4] {
        let n = 1usize << self.eval_log;
        let words = gpu.read(&inputs.accum, 4 * n);
        std::array::from_fn(|c| words[c * n..(c + 1) * n].to_vec())
    }
}
