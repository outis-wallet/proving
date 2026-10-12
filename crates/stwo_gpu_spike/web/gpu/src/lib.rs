//! The GPU worker's module: the bridge's ops, run with wgpu (WebGPU backend) on the prover's
//! data, which it reads and writes through JS views of the prover's shared memory.
//!
//! The JS worker (../gpu.worker.mjs) owns the job table's protocol (waiting on the doorbell,
//! slot states, notifications, failure) and calls [`Gpu::execute`] for each job, with the
//! memory's current `SharedArrayBuffer`. This module only runs ops:
//!
//! - inputs: `GPUQueue.writeBuffer` straight from a `Uint8Array` view of the shared memory
//!   (WebGPU takes `AllowSharedBufferSource`; Chrome accepts it), so no copy into this module;
//! - outputs: the mapped range's `Uint8Array` (`BufferView::as_uint8array`) `.set()` into a view
//!   of the shared memory: the one copy any readback needs.
//!
//! The kernels' host code is shared with native through `stwo_gpu_spike` (the `GpuContext`, the
//! WGSL): only these I/O edges differ (see [`upload`] and [`read_into`]).

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use js_sys::{Float64Array, Object, Promise, Reflect, SharedArrayBuffer, Uint8Array};
use stwo_gpu_bridge as bridge;
use stwo_gpu_spike::{GpuContext, M31_WGSL};
use wasm_bindgen::prelude::*;

#[wasm_bindgen(inline_js = "export function perf_now() { return performance.now(); }")]
extern "C" {
    fn perf_now() -> f64;
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console, js_name = error)]
    fn console_error(message: &str);
}

/// Set once the device is lost: every job fails from then on.
static LOST: AtomicBool = AtomicBool::new(false);

#[wasm_bindgen(start)]
fn start() {
    std::panic::set_hook(Box::new(|info| console_error(&format!("gpu module panic: {info}"))));
}

/// The bridge's layout, for the worker's JS (one source of truth: the bridge crate).
#[wasm_bindgen]
pub fn layout() -> Object {
    let object = Object::new();
    let set = |key: &str, value: f64| {
        Reflect::set(&object, &key.into(), &value.into()).unwrap();
    };
    set("MAGIC", bridge::MAGIC as f64);
    set("VERSION", bridge::VERSION as f64);
    set("NSLOTS", bridge::NSLOTS as f64);
    set("SLOT_WORDS", bridge::SLOT_WORDS as f64);
    set("HEADER_WORDS", bridge::HEADER_WORDS as f64);
    set("REGION_WORDS", bridge::REGION_WORDS as f64);
    set("H_MAGIC", bridge::H_MAGIC as f64);
    set("H_VERSION", bridge::H_VERSION as f64);
    set("H_NSLOTS", bridge::H_NSLOTS as f64);
    set("H_SLOT_WORDS", bridge::H_SLOT_WORDS as f64);
    set("H_DOORBELL", bridge::H_DOORBELL as f64);
    set("H_SERVER", bridge::H_SERVER as f64);
    set("H_COMPLETED", bridge::H_COMPLETED as f64);
    set("SERVER_NONE", bridge::SERVER_NONE as f64);
    set("SERVER_READY", bridge::SERVER_READY as f64);
    set("SERVER_LOST", bridge::SERVER_LOST as f64);
    set("S_STATE", bridge::S_STATE as f64);
    set("S_OP", bridge::S_OP as f64);
    set("S_NARGS", bridge::S_NARGS as f64);
    set("S_ERROR_LEN", bridge::S_ERROR_LEN as f64);
    set("S_ARGS", bridge::S_ARGS as f64);
    set("MAX_ARGS", bridge::MAX_ARGS as f64);
    set("S_TIMES", bridge::S_TIMES as f64);
    set("S_ERROR", bridge::S_ERROR as f64);
    set("ERROR_BYTES", bridge::ERROR_BYTES as f64);
    set("T_SUBMIT", bridge::T_SUBMIT as f64);
    set("T_WAKE", bridge::T_WAKE as f64);
    set("T_DONE", bridge::T_DONE as f64);
    set("T_UPLOAD", bridge::T_UPLOAD as f64);
    set("FREE", bridge::FREE as f64);
    set("CLAIMED", bridge::CLAIMED as f64);
    set("SUBMITTED", bridge::SUBMITTED as f64);
    set("RUNNING", bridge::RUNNING as f64);
    set("DONE", bridge::DONE as f64);
    set("ERROR", bridge::ERROR as f64);
    object
}

const M31_MUL_KERNEL: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read> b: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;

@compute @workgroup_size(256)
fn m31_mul_kernel(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&out)) { return; }
    out[i] = m31_mul(a[i], b[i]);
}
"#;

const WORKGROUP: u64 = 256;

/// GPU buffers kept between jobs (one per role), grown as needed.
#[derive(Default)]
struct Pool {
    a: Option<wgpu::Buffer>,
    b: Option<wgpu::Buffer>,
    out: Option<wgpu::Buffer>,
    staging: Option<wgpu::Buffer>,
}

fn pooled(
    device: &wgpu::Device,
    slot: &mut Option<wgpu::Buffer>,
    label: &str,
    size: u64,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    let size = size.max(4).next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT);
    if let Some(buffer) = slot.as_ref()
        && buffer.size() >= size
    {
        return buffer.clone();
    }
    if let Some(old) = slot.take() {
        old.destroy();
    }
    let buffer = device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage, mapped_at_creation: false });
    *slot = Some(buffer.clone());
    buffer
}

const STORAGE: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE
    .union(wgpu::BufferUsages::COPY_SRC)
    .union(wgpu::BufferUsages::COPY_DST);
const STAGING: wgpu::BufferUsages = wgpu::BufferUsages::MAP_READ.union(wgpu::BufferUsages::COPY_DST);

struct Inner {
    gpu: GpuContext,
    m31_mul: wgpu::ComputePipeline,
    pool: RefCell<Pool>,
    /// Elements per dispatch: what one binding and one dimension of workgroups allow.
    chunk_words: u64,
    /// The largest single `writeBuffer` (bounds the browser's staging copies).
    upload_chunk: u64,
}

type JobResult = Result<[f64; 3], String>;

/// The GPU worker's device, its pipelines and buffers.
#[wasm_bindgen]
pub struct Gpu {
    inner: Rc<Inner>,
}

#[wasm_bindgen]
impl Gpu {
    /// The adapter's device, with the adapter's own limits (a 512 MiB binding needs more than
    /// WebGPU's defaults: 128 MiB per binding, 256 MiB per buffer).
    pub async fn create() -> Result<Gpu, JsValue> {
        let gpu = context().await.map_err(|e| JsValue::from_str(&e))?;
        gpu.device.on_uncaptured_error(Arc::new(|error: wgpu::Error| {
            console_error(&format!("uncaptured WebGPU error: {error}"));
        }));
        gpu.device.set_device_lost_callback(|reason, message| {
            LOST.store(true, Ordering::SeqCst);
            console_error(&format!("WebGPU device lost ({reason:?}): {message}"));
        });
        let source = format!("{M31_WGSL}\n{M31_MUL_KERNEL}");
        let m31_mul = gpu.pipeline("m31_mul", &source, "m31_mul_kernel");
        let limits = gpu.device.limits();
        // Bindings start at multiples of minStorageBufferOffsetAlignment (256 bytes = 64 words).
        let align_words = (limits.min_storage_buffer_offset_alignment as u64 / 4).max(1);
        let by_binding = limits.max_storage_buffer_binding_size as u64 / 4;
        let by_dispatch = limits.max_compute_workgroups_per_dimension as u64 * WORKGROUP;
        let chunk_words = by_binding.min(by_dispatch) / align_words * align_words;
        Ok(Gpu {
            inner: Rc::new(Inner {
                gpu,
                m31_mul,
                pool: RefCell::default(),
                chunk_words,
                upload_chunk: 64 << 20,
            }),
        })
    }

    /// The adapter and the limits that matter here, as JSON.
    pub fn info(&self) -> String {
        let info = &self.inner.gpu.adapter_info;
        let limits = self.inner.gpu.device.limits();
        format!(
            "{{\"name\":{:?},\"vendor\":{},\"device\":{},\"backend\":{:?},\"driver\":{:?},\"max_buffer_size\":{},\"max_storage_buffer_binding_size\":{},\"max_compute_workgroups_per_dimension\":{},\"max_compute_invocations_per_workgroup\":{},\"chunk_words\":{}}}",
            info.name,
            info.vendor,
            info.device,
            format!("{:?}", info.backend),
            info.driver,
            limits.max_buffer_size,
            limits.max_storage_buffer_binding_size,
            limits.max_compute_workgroups_per_dimension,
            limits.max_compute_invocations_per_workgroup,
            self.inner.chunk_words,
        )
    }

    /// Whether the device was lost.
    pub fn lost(&self) -> bool {
        LOST.load(Ordering::SeqCst)
    }

    /// Runs bridge op `op` with `args` on `memory` (the prover memory's current buffer). Resolves
    /// to the phases' durations in ms, `[upload, compute, readback]`, or rejects with a message.
    pub fn execute(&self, op: u32, args: Vec<u32>, memory: SharedArrayBuffer) -> Promise {
        let inner = self.inner.clone();
        wasm_bindgen_futures::future_to_promise(async move {
            match inner.execute(op, &args, &memory).await {
                Ok(times) => Ok(Float64Array::from(&times[..]).into()),
                Err(message) => Err(JsValue::from_str(&message)),
            }
        })
    }

    /// Frees the pooled buffers (between runs: they can be large).
    pub fn release(&self) {
        let mut pool = self.inner.pool.borrow_mut();
        for buffer in [pool.a.take(), pool.b.take(), pool.out.take(), pool.staging.take()].into_iter().flatten() {
            buffer.destroy();
        }
    }
}

/// `GpuContext::new_async`, with failures returned rather than panicking (a panic in a future
/// would leave its promise pending): the browser's WebGPU adapter, with its own limits.
async fn context() -> Result<GpuContext, String> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::BROWSER_WEBGPU,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .map_err(|e| format!("no WebGPU adapter: {e}"))?;
    let adapter_info = adapter.get_info();
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("stwo-gpu-web"),
            required_limits: adapter.limits(),
            ..Default::default()
        })
        .await
        .map_err(|e| format!("no WebGPU device: {e}"))?;
    Ok(GpuContext { adapter_info, device, queue })
}

/// A view of `len` bytes at `ptr` in `memory`, checked against its current length.
fn view(memory: &SharedArrayBuffer, ptr: u32, len: u32) -> Result<Uint8Array, String> {
    let size = memory.byte_length() as u64;
    if ptr as u64 + len as u64 > size {
        return Err(format!("range {ptr}+{len} outside the memory ({size} bytes)"));
    }
    if ptr % 4 != 0 || len % 4 != 0 {
        return Err(format!("range {ptr}+{len} not 4-byte aligned"));
    }
    Ok(Uint8Array::new_with_byte_offset_and_length(memory, ptr, len))
}

/// Resolves when the queue has finished all work submitted so far.
async fn queue_done(queue: &wgpu::Queue) {
    let (sender, receiver) = futures_channel::oneshot::channel();
    queue.on_submitted_work_done(move || {
        let _ = sender.send(());
    });
    let _ = receiver.await;
}

/// `writeBuffer` of `source` (a view of the shared memory) into `buffer` at 0, in chunks.
/// This is the web side of `GpuContext::upload`: the data goes from the prover's memory to the
/// browser's staging without passing through this module.
fn upload(inner: &Inner, buffer: &wgpu::Buffer, source: &Uint8Array) -> Result<(), String> {
    let queue = inner.gpu.queue.as_webgpu().ok_or("not on the WebGPU backend")?;
    let target = buffer.as_webgpu().ok_or("not on the WebGPU backend")?;
    let len = source.length() as u64;
    let mut offset = 0;
    while offset < len {
        let end = (offset + inner.upload_chunk).min(len);
        let part = source.subarray(offset as u32, end as u32);
        queue
            .write_buffer_with_f64_and_u8_array(target, offset as f64, &part)
            .map_err(|e| format!("writeBuffer: {e:?}"))?;
        offset = end;
    }
    Ok(())
}

/// Copies the first `dest.length()` bytes of `buffer` into `dest` (a view of the shared memory),
/// through the pooled staging buffer. The web side of `GpuContext::read`.
async fn read_into(inner: &Inner, buffer: &wgpu::Buffer, dest: &Uint8Array, already_copied: bool) -> Result<(), String> {
    let len = dest.length() as u64;
    if len == 0 {
        return Ok(());
    }
    let staging = if already_copied {
        inner.pool.borrow().staging.clone().ok_or("no staging buffer")?
    } else {
        let staging = pooled(&inner.gpu.device, &mut inner.pool.borrow_mut().staging, "staging", len, STAGING);
        let mut encoder = inner.gpu.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, len);
        inner.gpu.queue.submit([encoder.finish()]);
        staging
    };
    let slice = staging.slice(..len);
    let (sender, receiver) = futures_channel::oneshot::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    receiver.await.map_err(|_| "map callback dropped")?.map_err(|e| format!("mapAsync: {e}"))?;
    {
        let mapped = slice.get_mapped_range().map_err(|e| format!("getMappedRange: {e:?}"))?;
        dest.set(mapped.as_uint8array(), 0);
    }
    staging.unmap();
    Ok(())
}

impl Inner {
    async fn execute(&self, op: u32, args: &[u32], memory: &SharedArrayBuffer) -> JobResult {
        if LOST.load(Ordering::SeqCst) {
            return Err("device lost".into());
        }
        // The two probes run without error scopes: popping one is a round trip to the GPU
        // process (~0.1 ms on Chrome), which would hide what they measure.
        match op {
            bridge::op::PING => return Ok([0.0; 3]),
            bridge::op::SUBMIT => {
                let start = perf_now();
                self.gpu.queue.submit([]);
                queue_done(&self.gpu.queue).await;
                return Ok([0.0, perf_now() - start, 0.0]);
            }
            _ => {}
        }
        // A validation or out-of-memory error means the outputs are garbage: the job fails, and
        // the prover falls back to its CPU path.
        let device = &self.gpu.device;
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let oom = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let result = match op {
            bridge::op::UPLOAD => self.upload_op(args, memory).await,
            bridge::op::READBACK => self.readback_op(args, memory).await,
            bridge::op::M31_MUL => self.m31_mul(args, memory).await,
            _ => Err(format!("unknown op {op}")),
        };
        // Pops run in reverse order of the pushes.
        let oom = oom.pop().await;
        let validation = scope.pop().await;
        if let Some(error) = oom.or(validation) {
            return Err(format!("WebGPU: {error}"));
        }
        if LOST.load(Ordering::SeqCst) {
            return Err("device lost".into());
        }
        result
    }

    async fn upload_op(&self, args: &[u32], memory: &SharedArrayBuffer) -> JobResult {
        let [src, len] = args else { return Err("upload takes [src, len]".into()) };
        let source = view(memory, *src, *len)?;
        let buffer = pooled(&self.gpu.device, &mut self.pool.borrow_mut().a, "a", *len as u64, STORAGE);
        let start = perf_now();
        upload(self, &buffer, &source)?;
        queue_done(&self.gpu.queue).await;
        Ok([perf_now() - start, 0.0, 0.0])
    }

    async fn readback_op(&self, args: &[u32], memory: &SharedArrayBuffer) -> JobResult {
        let [dst, len] = args else { return Err("readback takes [dst, len]".into()) };
        let dest = view(memory, *dst, *len)?;
        let buffer = pooled(&self.gpu.device, &mut self.pool.borrow_mut().out, "out", *len as u64, STORAGE);
        let start = perf_now();
        read_into(self, &buffer, &dest, false).await?;
        Ok([0.0, 0.0, perf_now() - start])
    }

    /// Records `out = a * b` over `n` words in chunks: each dispatch binds sub-ranges of at most
    /// `chunk_words` (maxStorageBufferBindingSize, and maxComputeWorkgroupsPerDimension x 256).
    fn encode_m31_mul(&self, encoder: &mut wgpu::CommandEncoder, a: &wgpu::Buffer, b: &wgpu::Buffer, out: &wgpu::Buffer, n: u64) {
        let layout = self.m31_mul.get_bind_group_layout(0);
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.m31_mul);
        let mut start = 0;
        while start < n {
            let len = self.chunk_words.min(n - start);
            fn binding(buffer: &wgpu::Buffer, binding: u32, start: u64, len: u64) -> wgpu::BindGroupEntry<'_> {
                wgpu::BindGroupEntry {
                    binding,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer,
                        offset: start * 4,
                        size: wgpu::BufferSize::new(len * 4),
                    }),
                }
            }
            let group = self.gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &[binding(a, 0, start, len), binding(b, 1, start, len), binding(out, 2, start, len)],
            });
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(len.div_ceil(WORKGROUP) as u32, 1, 1);
            start += len;
        }
    }

    async fn m31_mul(&self, args: &[u32], memory: &SharedArrayBuffer) -> JobResult {
        let [a_ptr, b_ptr, out_ptr, n, rest @ ..] = args else {
            return Err("m31_mul takes [a, b, out, n, flags?]".into());
        };
        let pipelined = rest.first().is_some_and(|flags| flags & 1 != 0);
        let n = *n as u64;
        let bytes = (n * 4) as u32;
        let (a_view, b_view, out_view) = (view(memory, *a_ptr, bytes)?, view(memory, *b_ptr, bytes)?, view(memory, *out_ptr, bytes)?);
        let device = &self.gpu.device;
        let (a, b, out) = {
            let mut pool = self.pool.borrow_mut();
            (
                pooled(device, &mut pool.a, "a", n * 4, STORAGE),
                pooled(device, &mut pool.b, "b", n * 4, STORAGE),
                pooled(device, &mut pool.out, "out", n * 4, STORAGE),
            )
        };
        if n == 0 {
            return Ok([0.0; 3]);
        }
        if pipelined {
            // One submit (uploads, dispatches, copy to staging), one map: what a kernel would do.
            let start = perf_now();
            upload(self, &a, &a_view)?;
            upload(self, &b, &b_view)?;
            let staging = pooled(device, &mut self.pool.borrow_mut().staging, "staging", n * 4, STAGING);
            let mut encoder = device.create_command_encoder(&Default::default());
            self.encode_m31_mul(&mut encoder, &a, &b, &out, n);
            encoder.copy_buffer_to_buffer(&out, 0, &staging, 0, n * 4);
            self.gpu.queue.submit([encoder.finish()]);
            read_into(self, &out, &out_view, true).await?;
            return Ok([0.0, perf_now() - start, 0.0]);
        }
        let t0 = perf_now();
        upload(self, &a, &a_view)?;
        upload(self, &b, &b_view)?;
        queue_done(&self.gpu.queue).await;
        let t1 = perf_now();
        let mut encoder = device.create_command_encoder(&Default::default());
        self.encode_m31_mul(&mut encoder, &a, &b, &out, n);
        self.gpu.queue.submit([encoder.finish()]);
        queue_done(&self.gpu.queue).await;
        let t2 = perf_now();
        read_into(self, &out, &out_view, false).await?;
        let t3 = perf_now();
        Ok([t1 - t0, t2 - t1, t3 - t2])
    }
}
