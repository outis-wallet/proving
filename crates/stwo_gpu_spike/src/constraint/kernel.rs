//! The generated constraint kernels on a device: their pipelines, bind groups and dispatches,
//! and, for fixtures and tests, buffers holding uploaded inputs.

use std::ops::Range;
use std::time::{Duration, Instant};

use stwo::core::fields::qm31::QM31;
use wgpu::util::DeviceExt;

use super::ir::{NodeId, Program, words};
use super::wgsl::{ColumnTable, Kernel, META_HEADER_WORDS, emit, uniform_nodes};
use crate::GpuContext;

#[derive(Clone, Copy, Debug)]
pub struct KernelOptions {
    pub workgroup_size: u32,
    /// The number of kernels the constraints are split over (contiguous ranges of about equal
    /// numbers of constraints), each adding its share; 0: by [`Self::nodes_per_kernel`].
    pub split: usize,
    /// With `split` 0: about this many row nodes per kernel (at most `max_split` kernels).
    pub nodes_per_kernel: usize,
    pub max_split: usize,
    /// Pipelines without wgpu's runtime bounds checks (see [`GpuContext::pipeline_unchecked`]).
    pub unchecked: bool,
}

impl Default for KernelOptions {
    fn default() -> Self {
        Self {
            workgroup_size: 128,
            split: 0,
            nodes_per_kernel: 150,
            max_split: 8,
            unchecked: false,
        }
    }
}

impl KernelOptions {
    /// The constraint ranges of the kernels of `program`.
    pub fn ranges(&self, program: &Program) -> Vec<Range<usize>> {
        let n = program.constraints.len();
        let split = match self.split {
            0 => {
                let nodes = program.row_nodes(0..n).len();
                nodes.div_ceil(self.nodes_per_kernel.max(1)).clamp(1, self.max_split.max(1))
            }
            split => split,
        }
        .clamp(1, n.max(1));
        (0..split).map(|k| k * n / split..(k + 1) * n / split).collect()
    }
}

/// A program's kernels, compiled.
pub struct ConstraintKernels {
    pub table: ColumnTable,
    pub kernels: Vec<(Range<usize>, Kernel)>,
    pipelines: Vec<wgpu::ComputePipeline>,
    layout: wgpu::BindGroupLayout,
    pub uniform_nodes: Vec<NodeId>,
    pub workgroup_size: u32,
    /// Generating the WGSL.
    pub emit_time: Duration,
    /// Creating the shader modules and pipelines (the driver's compilation).
    pub compile_time: Duration,
}

/// Inputs uploaded for a kernel (fixtures, tests): one buffer per interaction bound, its columns
/// one after the other, and the small buffers.
pub struct GpuInputs {
    pub trees: Vec<wgpu::Buffer>,
    pub uniforms: wgpu::Buffer,
    pub coeffs: wgpu::Buffer,
    pub denom_inv: wgpu::Buffer,
    pub out: wgpu::Buffer,
    pub meta: wgpu::Buffer,
    pub eval_log: u32,
}

impl GpuInputs {
    /// Their size in bytes.
    pub fn bytes(&self) -> u64 {
        self.trees.iter().map(|b| b.size()).sum::<u64>()
            + self.uniforms.size()
            + self.coeffs.size()
            + self.denom_inv.size()
            + self.out.size()
            + self.meta.size()
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

/// `meta`'s words: the sizes, then the columns' word offsets in the table's order.
pub fn meta_words(trace_log: u32, eval_log: u32, offsets: &[u32]) -> Vec<u32> {
    let mut words = vec![1u32 << eval_log, trace_log, eval_log, 0];
    debug_assert_eq!(words.len(), META_HEADER_WORDS);
    words.extend_from_slice(offsets);
    words
}

impl ConstraintKernels {
    pub fn new(
        gpu: &GpuContext,
        program: &Program,
        options: KernelOptions,
        cache: Option<&wgpu::PipelineCache>,
    ) -> Self {
        let table = ColumnTable::of(program);
        let n = program.constraints.len();
        let uniforms = uniform_nodes(program, 0..n);
        let start = Instant::now();
        let kernels: Vec<(Range<usize>, Kernel)> = options
            .ranges(program)
            .into_iter()
            .map(|range| {
                let kernel = emit(program, range.clone(), &uniforms, options.workgroup_size);
                (range, kernel)
            })
            .collect();
        let emit_time = start.elapsed();

        let entries: Vec<wgpu::BindGroupLayoutEntry> = (0..5 + table.interactions.len() as u32)
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
                    // Safety: every index is a row below n into a column the table places in its
                    // tree, a coefficient or uniform of the program's own, or a part index below
                    // `denom_inv`'s length.
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
                    cache,
                })
            })
            .collect();
        let compile_time = start.elapsed();
        Self {
            table,
            kernels,
            pipelines,
            layout,
            uniform_nodes: uniforms,
            workgroup_size: options.workgroup_size,
            emit_time,
            compile_time,
        }
    }

    /// The uniforms buffer's words, from [`Program::bind`]'s values.
    pub fn uniform_words(&self, values: &[Option<QM31>]) -> Vec<u32> {
        let mut out: Vec<u32> =
            self.uniform_nodes.iter().flat_map(|id| words(values[*id as usize].unwrap())).collect();
        if out.is_empty() {
            out = vec![0; 4];
        }
        out
    }

    /// The bind group of a run: `trees` in the table's `interactions` order.
    pub fn bind_group(
        &self,
        gpu: &GpuContext,
        uniforms: &wgpu::Buffer,
        coeffs: &wgpu::Buffer,
        denom_inv: &wgpu::Buffer,
        out: wgpu::BufferBinding<'_>,
        meta: &wgpu::Buffer,
        trees: &[&wgpu::Buffer],
    ) -> wgpu::BindGroup {
        assert_eq!(trees.len(), self.table.interactions.len(), "a buffer per interaction");
        let mut entries = vec![
            wgpu::BindGroupEntry { binding: 0, resource: uniforms.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: coeffs.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: denom_inv.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Buffer(out) },
            wgpu::BindGroupEntry { binding: 4, resource: meta.as_entire_binding() },
        ];
        for (k, tree) in trees.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: 5 + k as u32,
                resource: tree.as_entire_binding(),
            });
        }
        gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("constraint eval"),
            layout: &self.layout,
            entries: &entries,
        })
    }

    fn workgroups(&self, eval_log: u32) -> (u32, u32) {
        let groups = (1u64 << eval_log).div_ceil(self.workgroup_size as u64);
        let x = groups.min(32768);
        (x as u32, groups.div_ceil(x) as u32)
    }

    /// Records the kernels' passes, with timestamps at `queries` (2 per kernel from `first`).
    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        eval_log: u32,
        queries: Option<(&wgpu::QuerySet, u32)>,
    ) {
        let (x, y) = self.workgroups(eval_log);
        for (k, pipeline) in self.pipelines.iter().enumerate() {
            let timestamp_writes = queries.map(|(query_set, first)| {
                wgpu::ComputePassTimestampWrites {
                    query_set,
                    beginning_of_pass_write_index: Some(first + 2 * k as u32),
                    end_of_pass_write_index: Some(first + 2 * k as u32 + 1),
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
    }

    /// Uploads inputs (`columns[interaction][index]` on the evaluation domain, the accumulator
    /// `[coordinate][row]` to add to), waiting until the device has them.
    #[allow(clippy::too_many_arguments)]
    pub fn upload(
        &self,
        gpu: &GpuContext,
        columns: &[Vec<&[u32]>],
        uniform_values: &[Option<QM31>],
        coeffs: &[QM31],
        denom_inv: &[u32],
        accum: &[&[u32]; 4],
        trace_log: u32,
        eval_log: u32,
    ) -> GpuInputs {
        let n = 1usize << eval_log;
        let storage = |label: &str, words: usize| {
            gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (words.max(4) * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let init = |label: &str, words: &[u32]| {
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(words),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            })
        };
        let mut offsets = vec![0u32; self.table.total()];
        let trees = self
            .table
            .interactions
            .iter()
            .map(|&i| {
                let k = self.table.n_columns[i as usize];
                let buffer = storage("constraint columns", k * n);
                for j in 0..k {
                    let column = columns[i as usize][j];
                    assert_eq!(column.len(), n, "a column on the evaluation domain");
                    gpu.queue.write_buffer(&buffer, (j * n * 4) as u64, bytemuck::cast_slice(column));
                    offsets[self.table.index(i, j as u32)] = (j * n) as u32;
                }
                buffer
            })
            .collect();
        let coeff_words: Vec<u32> = coeffs.iter().flat_map(|c| words(*c)).collect();
        let out = storage("out", 4 * n);
        for (c, coordinate) in accum.iter().enumerate() {
            gpu.queue.write_buffer(&out, (c * n * 4) as u64, bytemuck::cast_slice(coordinate));
        }
        let inputs = GpuInputs {
            trees,
            uniforms: init("uniforms", &self.uniform_words(uniform_values)),
            coeffs: init("coeffs", &coeff_words),
            denom_inv: init("denom_inv", denom_inv),
            out,
            meta: init("meta", &meta_words(trace_log, eval_log, &offsets)),
            eval_log,
        };
        gpu.queue.submit([]);
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        inputs
    }

    pub fn bind_inputs(&self, gpu: &GpuContext, inputs: &GpuInputs) -> wgpu::BindGroup {
        let trees: Vec<&wgpu::Buffer> = inputs.trees.iter().collect();
        self.bind_group(
            gpu,
            &inputs.uniforms,
            &inputs.coeffs,
            &inputs.denom_inv,
            inputs.out.as_entire_buffer_binding(),
            &inputs.meta,
            &trees,
        )
    }

    /// Runs the kernels once on uploaded inputs and waits: the GPU time of the passes (from
    /// timestamps, if the device has them) and the wall-clock time of submit-to-idle.
    pub fn run(
        &self,
        gpu: &GpuContext,
        bind_group: &wgpu::BindGroup,
        eval_log: u32,
    ) -> (Option<Duration>, Duration) {
        let count = 2 * self.pipelines.len() as u32;
        let timestamps = gpu.device.features().contains(wgpu::Features::TIMESTAMP_QUERY).then(|| {
            let query_set = gpu.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("constraint eval timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count,
            });
            let buffer = |usage| {
                gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: 8 * count as u64,
                    usage,
                    mapped_at_creation: false,
                })
            };
            let resolve =
                buffer(wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC);
            let readback = buffer(wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
            (query_set, resolve, readback)
        });
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        self.encode(&mut encoder, bind_group, eval_log, timestamps.as_ref().map(|t| (&t.0, 0)));
        if let Some((query_set, resolve, readback)) = &timestamps {
            encoder.resolve_query_set(query_set, 0..count, resolve, 0);
            encoder.copy_buffer_to_buffer(resolve, 0, readback, 0, 8 * count as u64);
        }
        let start = Instant::now();
        gpu.queue.submit([encoder.finish()]);
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        let wall = start.elapsed();
        let gpu_time = timestamps.map(|(_, _, readback)| read_timestamps(gpu, &readback, count));
        (gpu_time, wall)
    }

    /// The output buffer's words, `[coordinate][row]`.
    pub fn read_out(&self, gpu: &GpuContext, inputs: &GpuInputs) -> [Vec<u32>; 4] {
        let n = 1usize << inputs.eval_log;
        let words = gpu.read(&inputs.out, 4 * n);
        std::array::from_fn(|c| words[c * n..(c + 1) * n].to_vec())
    }

    /// Writes `accum` (`[coordinate][row]`) into the output buffer again.
    pub fn reset_out(&self, gpu: &GpuContext, inputs: &GpuInputs, accum: &[&[u32]; 4]) {
        let n = 1usize << inputs.eval_log;
        for (c, coordinate) in accum.iter().enumerate() {
            gpu.queue.write_buffer(&inputs.out, (c * n * 4) as u64, bytemuck::cast_slice(coordinate));
        }
        gpu.queue.submit([]);
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    }
}

/// The sum of the `count / 2` passes' spans in a resolved timestamp buffer (mappable).
pub fn read_timestamps(gpu: &GpuContext, readback: &wgpu::Buffer, count: u32) -> Duration {
    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |result| result.expect("map"));
    gpu.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let ticks: Vec<u64> = bytemuck::cast_slice(&slice.get_mapped_range().expect("mapped")[..]).to_vec();
    readback.unmap();
    let period = gpu.queue.get_timestamp_period() as f64;
    let ns: f64 = (0..count as usize / 2)
        .map(|k| ticks[2 * k + 1].wrapping_sub(ticks[2 * k]) as f64 * period)
        .sum();
    Duration::from_nanos(ns as u64)
}
