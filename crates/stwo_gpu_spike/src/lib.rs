//! A spike: stwo's commitment (polynomial extension and Merkle hashing) on the GPU, through wgpu
//! (Vulkan, Metal or DX12 natively, WebGPU in browsers), checked bit for bit against stwo's SIMD
//! backend. It changes nothing in stwo: it reads stwo's inputs and compares with stwo's outputs.
//!
//! The adapter comes from the environment (`wgpu::util::initialize_adapter_from_env_or_default`):
//! `WGPU_ADAPTER_NAME=llvmpipe` picks Mesa's software Vulkan (lavapipe), slow but exact, for
//! checking results on a machine without a working GPU; `WGPU_BACKEND=vulkan|metal|dx12|gl`
//! restricts the backends.
//!
//! - [`blake2s`]: Blake2s-256 in WGSL, as stwo hashes.
//! - [`extension`]: the polynomial extension (circle FFT) of a stripe of a tree's columns.
//! - [`commit`]: the striped commitment of a tree, extension and hashing stripe by stripe on the
//!   GPU, equal to `CommitmentTreeProver::new_striped`'s.
//! - [`merkle`]: the lifted Merkle tree of a commitment (leaves of a stripe's columns, its subtree,
//!   the levels above the stripes), its input and output layouts, on the GPU.
//! - [`shapes`]: the Cairo leg's trace shapes; [`reference`](mod@reference): stwo's SIMD results to
//!   compare with.
//! - `bench_merkle`, `bench_extension`, `bench_commit` (`src/bin`): the Merkle hashing, the
//!   extension, and the whole striped commitment of the Cairo base trace, GPU against SIMD.

use wgpu::util::DeviceExt;

pub mod blake2s;
pub mod commit;
pub mod committer;
pub mod extension;
pub mod merkle;
pub mod parts;
pub mod recompute;
pub mod reference;
pub mod shapes;
pub mod stream;
pub mod transfer;
pub mod warm;

/// M31 arithmetic in WGSL (`m31_add`, `m31_sub`, `m31_mul`, ...), to prepend to a kernel's source.
pub const M31_WGSL: &str = include_str!("shaders/m31.wgsl");

/// A device and its queue, with blocking helpers.
pub struct GpuContext {
    pub adapter_info: wgpu::AdapterInfo,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl GpuContext {
    /// The adapter the environment picks (see the crate docs), with its own limits.
    pub fn new() -> Self {
        pollster::block_on(Self::new_async())
    }

    pub async fn new_async() -> Self {
        Self::with_limits_async(None).await
    }

    /// The adapter the environment picks, with `limits` instead of its own (e.g.
    /// `wgpu::Limits::default()`, WebGPU's defaults, to check that kernels fit in a browser's).
    pub fn with_limits(limits: wgpu::Limits) -> Self {
        pollster::block_on(Self::with_limits_async(Some(limits)))
    }

    pub async fn with_limits_async(limits: Option<wgpu::Limits>) -> Self {
        Self::create_async(limits, wgpu::Features::empty()).await
    }

    /// The adapter the environment picks, with its own limits, and those of the `optional`
    /// features it has (e.g. `wgpu::Features::TIMESTAMP_QUERY`, for the benchmarks): see
    /// `device.features()` for the ones it got.
    pub fn with_optional_features(optional: wgpu::Features) -> Self {
        pollster::block_on(Self::create_async(None, optional))
    }

    async fn create_async(limits: Option<wgpu::Limits>, optional: wgpu::Features) -> Self {
        Self::try_create_async(limits, optional).await.unwrap_or_else(|error| panic!("{error}"))
    }

    /// [`Self::new`] that reports a missing adapter or device instead of panicking.
    pub fn try_new() -> Result<Self, String> {
        pollster::block_on(Self::try_create_async(None, wgpu::Features::empty()))
    }

    pub async fn try_create_async(
        limits: Option<wgpu::Limits>,
        optional: wgpu::Features,
    ) -> Result<Self, String> {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = wgpu::util::initialize_adapter_from_env_or_default(&instance, None)
            .await
            .map_err(|error| format!("no GPU adapter: {error}"))?;
        let adapter_info = adapter.get_info();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("stwo-gpu-spike"),
                required_features: optional & adapter.features(),
                required_limits: limits.unwrap_or_else(|| adapter.limits()),
                ..Default::default()
            })
            .await
            .map_err(|error| format!("no GPU device: {error}"))?;
        Ok(Self { adapter_info, device, queue })
    }

    /// A storage buffer holding `data`, readable back (COPY_SRC) and writable (COPY_DST).
    pub fn upload(&self, label: &str, data: &[u32]) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        })
    }

    /// An uninitialized storage buffer of `len` words.
    pub fn storage(&self, label: &str, len: usize) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (len * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// The first `len` words of `buffer`, after all submitted work.
    pub fn read(&self, buffer: &wgpu::Buffer, len: usize) -> Vec<u32> {
        let size = (len * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
        self.queue.submit([encoder.finish()]);
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |result| result.expect("map"));
        self.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        let words =
            bytemuck::cast_slice::<u8, u32>(&slice.get_mapped_range().expect("mapped range")[..])
                .to_vec();
        staging.unmap();
        words
    }

    /// A compute pipeline for `entry` in `source`, with its bind group layout inferred.
    pub fn pipeline(&self, label: &str, source: &str, entry: &str) -> wgpu::ComputePipeline {
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: None,
            module: &module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        })
    }

    /// [`Self::pipeline`] without the runtime checks wgpu adds to a shader (bounds checks on
    /// every buffer access, loop bounding), which WebGPU in browsers always keeps.
    ///
    /// # Safety
    ///
    /// The kernel must never index out of bounds nor loop forever, whatever it is dispatched on.
    pub unsafe fn pipeline_unchecked(
        &self,
        label: &str,
        source: &str,
        entry: &str,
    ) -> wgpu::ComputePipeline {
        // Safety: the caller's.
        let module = unsafe {
            self.device.create_shader_module_trusted(
                wgpu::ShaderModuleDescriptor {
                    label: Some(label),
                    source: wgpu::ShaderSource::Wgsl(source.into()),
                },
                wgpu::ShaderRuntimeChecks::unchecked(),
            )
        };
        self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: None,
            module: &module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        })
    }

    /// Runs `pipeline` once over `workgroups` with `buffers` bound in order to group 0, and
    /// waits for it.
    pub fn run(
        &self,
        pipeline: &wgpu::ComputePipeline,
        buffers: &[&wgpu::Buffer],
        workgroups: (u32, u32, u32),
    ) {
        let layout = pipeline.get_bind_group_layout(0);
        let entries = buffers
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect::<Vec<_>>();
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &entries,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
        }
        self.queue.submit([encoder.finish()]);
        self.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    }
}

impl Default for GpuContext {
    fn default() -> Self {
        Self::new()
    }
}
