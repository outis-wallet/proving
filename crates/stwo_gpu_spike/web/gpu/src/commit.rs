//! The bridge's `COMMIT_STRIPED` op: stwo's `CommitmentTreeProver::new_striped` on the GPU, for a
//! prover thread blocked on the bridge, with its inputs and outputs in the prover's memory.
//!
//! The commitment itself is the native one, [`StripedCommitment`] (the same layout, plans and
//! kernels, recorded by its own `encode`, one submission). Only its I/O edges are the web's, as in
//! the rest of this module:
//!
//! - the coefficients go from the prover's memory to the coefficients' buffers through a ring of
//!   mappable buffers kept for the page's life ([`UploadMode::Mapped`]): each `upload_chunk` of
//!   them is copied into a mapped one and copied on the GPU, where native code writes them into
//!   one staging buffer (`writeBuffer`s straight from the prover's memory measured 2-4x slower
//!   in Chrome, see [`upload_coefficients`]);
//! - the waits are awaits (`onSubmittedWorkDone`, `mapAsync`), not `device.poll(wait)`, which
//!   returns at once on the web;
//! - the outputs come back through a ring of small mappable buffers, straight into the Vecs the
//!   prover allocated (stwo's own columns and layers), where native code reads them back through
//!   its own ring, waiting with `device.poll`;
//! - every buffer of the commitment is destroyed at the end: on the web a buffer otherwise lives
//!   until its JS object is garbage collected.
//!
//! The twiddles come from the prover's twiddle tree (the tail of it the largest domain reads, see
//! `domain_line_twiddles_from_tree`), or, configured so, are computed here once and kept for the
//! page's life.
//!
//! Everything a mismatched descriptor could make the commitment assert is checked first: a panic
//! here is a trap that ends the GPU worker (every later job fails), where an error only sends
//! this commitment to the CPU.

use std::cell::{Cell, RefCell};
use std::collections::VecDeque;

use js_sys::{SharedArrayBuffer, Uint8Array, Uint32Array};
use wasm_bindgen::JsCast;
use stwo::core::poly::circle::CanonicCoset;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::poly::circle::PolyOps;
use stwo::prover::poly::twiddles::TwiddleTree;
use stwo_gpu_bridge::commit as d;
use stwo_gpu_spike::commit::{
    CommitKernels, CommitOptions, StripedCommitment, kept_whole, takes_striped_path,
};
use stwo_gpu_spike::extension::{MAX_LOG_SIZE, MIN_LOG_SIZE};
use stwo_gpu_spike::merkle::{TreeLayout, stwo_min_stored_level};

use crate::{Inner, JobResult, STAGING, perf_now, queue_done, view};

/// Where the twiddles come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TwiddleSource {
    /// The prover's twiddle tree, read from its memory for each commitment.
    Prover,
    /// Computed here (stwo's `precompute_twiddles`) and kept for the page's life.
    Computed,
}

/// How the coefficients go to the GPU (see [`upload_coefficients`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UploadMode {
    /// `writeBuffer` straight from the prover's memory, a call per column (or chunk of one).
    Write,
    /// `writeBuffer`, the small columns gathered into `upload_chunk`-byte writes first.
    Gather,
    /// Through a ring of mappable buffers kept for the page's life: the columns are copied into
    /// a mapped one, which is then copied into the coefficients' buffer on the GPU.
    Mapped,
}

/// The ring [`UploadMode::Mapped`] goes through: buffers of `upload_chunk` bytes, each with the
/// `mapAsync` that makes it writable again once the GPU has copied it.
struct UploadRing {
    chunk: u64,
    slots: Vec<(wgpu::Buffer, Option<wasm_bindgen_futures::JsFuture>)>,
}

/// The rings' default depths: the chunks in flight between this worker and the GPU.
const UPLOAD_RING: usize = 4;
const READBACK_RING: usize = 4;

/// The commitment's kernels on the worker's device, and what it keeps between commitments.
pub struct Committer {
    kernels: CommitKernels,
    pub twiddles: Cell<TwiddleSource>,
    /// How the coefficients are uploaded (see [`upload_coefficients`]).
    pub upload: Cell<UploadMode>,
    upload_ring: RefCell<Option<UploadRing>>,
    /// The time spent waiting for a ring's buffer, in the current phase (ms).
    waits: Cell<f64>,
    /// The rings' depths.
    pub upload_ring_depth: Cell<usize>,
    pub readback_ring_depth: Cell<usize>,
    /// The readback ring's buffers (`readback_chunk` bytes each), kept for the page's life.
    readback_ring: RefCell<Vec<wgpu::Buffer>>,
    /// Commitments whose coefficients and stripe evaluations would take more device memory than
    /// this (MiB; 0: no limit) are refused, for the CPU (e.g. on phones, where WebGPU's buffers
    /// count against the tab's memory).
    pub max_gpu_mib: Cell<u32>,
    /// [`UploadMode::Gather`]'s array, `upload_chunk` bytes (a JS array, not this module's
    /// memory).
    gather_array: RefCell<Option<Uint8Array>>,
    /// [`TwiddleSource::Computed`]'s tree, with the log size of the largest domain it covers.
    computed: RefCell<Option<(u32, TwiddleTree<SimdBackend>)>>,
}

impl Committer {
    pub fn new(inner_gpu: &stwo_gpu_spike::GpuContext) -> Self {
        Self {
            kernels: CommitKernels::new(inner_gpu),
            twiddles: Cell::new(TwiddleSource::Prover),
            upload: Cell::new(UploadMode::Mapped),
            upload_ring: RefCell::new(None),
            waits: Cell::new(0.0),
            upload_ring_depth: Cell::new(UPLOAD_RING),
            readback_ring_depth: Cell::new(READBACK_RING),
            readback_ring: RefCell::new(vec![]),
            max_gpu_mib: Cell::new(0),
            gather_array: RefCell::new(None),
            computed: RefCell::new(None),
        }
    }

    /// The gathering array, of `bytes` bytes.
    fn gather_buffer(&self, bytes: u64) -> Uint8Array {
        let mut array = self.gather_array.borrow_mut();
        if array.as_ref().is_none_or(|array| array.length() as u64 != bytes) {
            *array = Some(Uint8Array::new_with_length(bytes as u32));
        }
        array.clone().unwrap()
    }
}

/// A column of the descriptor.
struct Column {
    coeffs: u32,
    log_size: u32,
    out: u32,
    out_words: u32,
}

/// The descriptor, checked against the memory's size; the layers as stwo orders them (root
/// first).
struct Descriptor {
    log_blowup: u32,
    lifting_log_size: u32,
    twiddles: (u32, u32),
    columns: Vec<Column>,
    layers: Vec<(u32, u32)>,
}

fn word(words: &[u32], index: usize) -> Result<u32, String> {
    words.get(index).copied().ok_or_else(|| "descriptor too short".to_string())
}

/// Checks that `[ptr, ptr + bytes)` is inside the memory and 4-byte aligned.
fn check_range(memory_bytes: u64, ptr: u32, bytes: u64, what: &str) -> Result<(), String> {
    if ptr % 4 != 0 || ptr as u64 + bytes > memory_bytes {
        return Err(format!("{what}: range {ptr}+{bytes} outside the memory or unaligned"));
    }
    Ok(())
}

impl Descriptor {
    fn read(memory: &SharedArrayBuffer, ptr: u32, bytes: u32) -> Result<Self, String> {
        let memory_bytes = memory.byte_length() as u64;
        check_range(memory_bytes, ptr, bytes as u64, "descriptor")?;
        let words = Uint32Array::new_with_byte_offset_and_length(memory, ptr, bytes / 4).to_vec();
        if word(&words, d::D_VERSION)? != d::DESCRIPTOR_VERSION {
            return Err(format!("descriptor version {} != {}", words[0], d::DESCRIPTOR_VERSION));
        }
        let n_columns = word(&words, d::D_N_COLUMNS)? as usize;
        let log_blowup = word(&words, d::D_LOG_BLOWUP)?;
        let lifting_log_size = word(&words, d::D_LIFTING_LOG_SIZE)?;
        if n_columns == 0 || log_blowup > 8 || lifting_log_size > 31 {
            return Err(format!(
                "descriptor: {n_columns} columns, log blowup {log_blowup}, lifting log size \
                 {lifting_log_size}"
            ));
        }
        if words.len() < d::descriptor_words(n_columns, lifting_log_size as usize) {
            return Err("descriptor too short".into());
        }
        let twiddles = (word(&words, d::D_TWIDDLES_PTR)?, word(&words, d::D_TWIDDLES_LEN)?);
        if twiddles.0 != 0 {
            check_range(memory_bytes, twiddles.0, twiddles.1 as u64 * 4, "twiddles")?;
        }
        let columns = (0..n_columns)
            .map(|i| {
                let at = d::HEADER_WORDS + i * d::COLUMN_WORDS;
                let column = Column {
                    coeffs: words[at + d::C_COEFFS],
                    log_size: words[at + d::C_LOG_SIZE],
                    out: words[at + d::C_OUT],
                    out_words: words[at + d::C_OUT_WORDS],
                };
                if !(MIN_LOG_SIZE..=MAX_LOG_SIZE).contains(&column.log_size) {
                    return Err(format!("column {i}: log size {}", column.log_size));
                }
                check_range(memory_bytes, column.coeffs, 4 << column.log_size, "coefficients")?;
                check_range(memory_bytes, column.out, column.out_words as u64 * 4, "evaluations")?;
                Ok(column)
            })
            .collect::<Result<Vec<_>, String>>()?;
        let layers_at = d::HEADER_WORDS + n_columns * d::COLUMN_WORDS;
        let layers = (0..=lifting_log_size as usize)
            .map(|j| {
                let at = layers_at + j * d::LAYER_WORDS;
                let (ptr, hashes) = (words[at + d::L_PTR], words[at + d::L_HASHES]);
                check_range(memory_bytes, ptr, hashes as u64 * 32, "layer")?;
                Ok((ptr, hashes))
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Self { log_blowup, lifting_log_size, twiddles, columns, layers })
    }

    fn log_sizes(&self) -> Vec<u32> {
        self.columns.iter().map(|column| column.log_size).collect()
    }

    /// Checks the outputs' sizes against `new_striped`'s (and the commitment's) and what the
    /// device can bind, before anything that would assert on them.
    fn check_shapes(&self, gpu: &stwo_gpu_spike::GpuContext) -> Result<TreeLayout, String> {
        let log_sizes = self.log_sizes();
        if !takes_striped_path(&log_sizes, self.log_blowup, self.lifting_log_size) {
            return Err("new_striped wouldn't stripe these columns".into());
        }
        for (i, column) in self.columns.iter().enumerate() {
            let words = if kept_whole(column.log_size, self.log_blowup) {
                1u64 << (column.log_size + self.log_blowup)
            } else {
                1u64 << column.log_size
            };
            if column.out_words as u64 != words {
                return Err(format!("column {i}: {} words of output, not {words}", column.out_words));
            }
        }
        let stripe_height = self.lifting_log_size - self.log_blowup;
        let mut sorted = log_sizes.clone();
        sorted.sort();
        let tree = TreeLayout::new(
            stripe_height,
            self.log_blowup,
            stwo_min_stored_level(&sorted, stripe_height),
        );
        for (j, &(_, hashes)) in self.layers.iter().enumerate() {
            let level = self.lifting_log_size - j as u32;
            let expected = tree.level_offset(level).map_or(0, |_| tree.level_len(level));
            if hashes as usize != expected {
                return Err(format!("layer {j}: {hashes} hashes, not {expected}"));
            }
        }
        // The planners assert that the tree and a column fit in a binding.
        let limits = gpu.device.limits();
        let max_binding = limits.max_storage_buffer_binding_size.min(limits.max_buffer_size);
        let largest_column = 4u64 << log_sizes.iter().max().unwrap();
        if tree.total_words * 4 > max_binding || largest_column > max_binding {
            return Err(format!("the tree or a column exceeds a binding ({max_binding} bytes)"));
        }
        Ok(tree)
    }
}

/// Destroys the commitment's buffers however the job ends.
struct Destroy<'a>(&'a StripedCommitment);

impl Drop for Destroy<'_> {
    fn drop(&mut self) {
        self.0.destroy();
    }
}

/// The prover's twiddles that `domain_line_twiddles_from_tree` reads for a domain of
/// `2^max_domain_log` points (the tail of its tree), copied out of its memory.
fn prover_twiddles(
    memory: &SharedArrayBuffer,
    (ptr, len): (u32, u32),
    max_domain_log: u32,
) -> Result<TwiddleTree<SimdBackend>, String> {
    let need = 1u32 << (max_domain_log - 1);
    if len < need {
        return Err(format!("the prover's twiddles ({len} words) cover less than {need}"));
    }
    let tail = Uint32Array::new_with_byte_offset_and_length(memory, ptr + 4 * (len - need), need);
    Ok(TwiddleTree {
        root_coset: CanonicCoset::new(max_domain_log).circle_domain().half_coset,
        twiddles: tail.to_vec(),
        itwiddles: vec![],
    })
}

/// Runs the op: `args = [descriptor, descriptor bytes]`.
pub async fn commit_striped(inner: &Inner, args: &[u32], memory: &SharedArrayBuffer) -> JobResult {
    let started = perf_now();
    let [desc_ptr, desc_bytes] = args else {
        return Err("commit_striped takes [descriptor, bytes]".into());
    };
    let desc = Descriptor::read(memory, *desc_ptr, *desc_bytes)?;
    let gpu = &inner.gpu;
    let committer = &inner.committer;
    let tree_layout = desc.check_shapes(gpu)?;
    let log_sizes = desc.log_sizes();
    // The coefficients and a stripe's evaluations (the same size), the tree's stored levels.
    let estimate = 2 * log_sizes.iter().map(|&log_size| 4u64 << log_size).sum::<u64>()
        + tree_layout.total_words * 4;
    let limit = committer.max_gpu_mib.get() as u64;
    if limit > 0 && estimate > limit << 20 {
        return Err(format!("{} MiB of GPU memory, over the limit ({limit} MiB)", estimate >> 20));
    }
    let mut stats = [0f64; d::NSTATS];

    // The twiddles, and the commitment: its buffers, its plans, and the stripes' twiddles, laid
    // out from the tree and uploaded.
    let t0 = perf_now();
    let max_domain_log = log_sizes.iter().max().unwrap() + desc.log_blowup;
    let from_prover = committer.twiddles.get() == TwiddleSource::Prover && desc.twiddles.0 != 0;
    let prover_tree = if from_prover {
        Some(prover_twiddles(memory, desc.twiddles, max_domain_log)?)
    } else {
        let covers = committer
            .computed
            .borrow()
            .as_ref()
            .is_some_and(|(log_size, _)| *log_size >= max_domain_log);
        if !covers {
            let coset = CanonicCoset::new(max_domain_log).circle_domain().half_coset;
            *committer.computed.borrow_mut() =
                Some((max_domain_log, SimdBackend::precompute_twiddles(coset)));
        }
        None
    };
    stats[d::ST_TWIDDLES_FROM_PROVER] = from_prover as u32 as f64;
    let t1 = perf_now();
    let options = CommitOptions::default();
    let commitment = {
        let computed = committer.computed.borrow();
        let tree = match &prover_tree {
            Some(tree) => tree,
            None => &computed.as_ref().unwrap().1,
        };
        StripedCommitment::with_options(
            gpu,
            &committer.kernels,
            &log_sizes,
            desc.log_blowup,
            desc.lifting_log_size,
            tree,
            options,
        )
    };
    drop(prover_tree);
    let destroy = Destroy(&commitment);
    let t2 = perf_now();
    // The stripes' twiddle words are laid out inside `with_options`, so the setup includes them;
    // the twiddles' stat is getting the tree.
    stats[d::ST_TWIDDLES] = t1 - t0;
    stats[d::ST_SETUP] = t2 - t1;
    stats[d::ST_GPU_BYTES] = commitment.gpu_bytes().device() as f64;

    // The coefficients, from the prover's memory.
    committer.waits.set(0.0);
    let uploaded = upload_coefficients(inner, &desc, &commitment, memory).await?;
    stats[d::ST_UPLOAD_WAIT] = committer.waits.replace(0.0);
    let t3 = perf_now();
    stats[d::ST_UPLOAD] = t3 - t2;
    stats[d::ST_UPLOAD_BYTES] = uploaded as f64;

    // Every stripe's extension and hashing, then the levels above: one submission.
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    commitment.encode(&committer.kernels, &mut encoder, None);
    gpu.queue.submit([encoder.finish()]);
    queue_done(&gpu.queue).await;
    let t4 = perf_now();
    stats[d::ST_COMPUTE] = t4 - t3;

    // The outputs, straight into the prover's columns and layers.
    let segments = output_segments(&desc, &commitment);
    let read = ring_readback(inner, &source_buffers(&commitment), &segments, memory).await?;
    stats[d::ST_READBACK_WAIT] = committer.waits.replace(0.0);
    let t5 = perf_now();
    stats[d::ST_READBACK] = t5 - t4;
    stats[d::ST_READBACK_BYTES] = read as f64;

    drop(destroy);
    let t6 = perf_now();
    stats[d::ST_DESTROY] = t6 - t5;
    stats[d::ST_TOTAL] = t6 - started;
    write_stats(memory, *desc_ptr, &stats);
    Ok([t3 - t2, t4 - t3, t5 - t4])
}

/// Writes the columns' coefficients from the prover's memory into the commitment's coefficient
/// buffers. Returns the bytes written.
///
/// In Chrome 154 on an RTX 3080, for a proof's 2.1 GiB of coefficients (the Cairo base trace
/// alone: 2608 columns, 200 KiB on average), until the GPU had them: `writeBuffer`s
/// ([`UploadMode::Write`], a call per column) and the same gathered into 8 MiB writes
/// ([`UploadMode::Gather`]) ran at 0.7 to 1.5 GiB/s, whatever the write and window sizes (1 to
/// 32 MiB); the ring of mapped buffers ([`UploadMode::Mapped`], the default) at 2.2 to 2.9 GiB/s.
/// Chrome sends a write under 4 MiB through its command stream, and makes a shared-memory region
/// for each larger one; the ring's buffers keep theirs. [`UploadMode::Write`] and
/// [`UploadMode::Gather`] wait for the queue after each `upload_window` bytes.
async fn upload_coefficients(
    inner: &Inner,
    desc: &Descriptor,
    commitment: &StripedCommitment,
    memory: &SharedArrayBuffer,
) -> Result<u64, String> {
    // Every column, by its place in the coefficients' buffers.
    let layout = commitment.layout();
    let mut segments = desc
        .columns
        .iter()
        .zip(&layout.slots)
        .map(|(column, slot)| Segment {
            buffer: slot.buffer,
            src: slot.offset * 4,
            len: 4u64 << column.log_size,
            dst: column.coeffs,
        })
        .collect::<Vec<_>>();
    segments.sort_by_key(|segment| (segment.buffer, segment.src));
    let uploaded = segments.iter().map(|segment| segment.len).sum();
    match inner.committer.upload.get() {
        UploadMode::Mapped => upload_mapped(inner, commitment, &segments, memory).await?,
        mode => upload_written(inner, commitment, &segments, memory, mode == UploadMode::Gather).await?,
    }
    Ok(uploaded)
}

/// [`UploadMode::Mapped`]: the segments packed into chunks of the ring's size (a chunk is a run of
/// one coefficient buffer, gaps included), each copied into a mapped ring buffer, unmapped and
/// copied on the GPU; the buffer's next `mapAsync` resolves once that copy is done.
async fn upload_mapped(
    inner: &Inner,
    commitment: &StripedCommitment,
    segments: &[Segment],
    memory: &SharedArrayBuffer,
) -> Result<(), String> {
    let gpu = &inner.gpu;
    let chunk_bytes = inner.io.get().upload_chunk;
    let mut ring = inner.committer.upload_ring.borrow_mut();
    let depth = inner.committer.upload_ring_depth.get().max(1);
    if ring.as_ref().is_none_or(|ring| ring.chunk != chunk_bytes || ring.slots.len() != depth) {
        let slots = (0..depth)
            .map(|_| {
                let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("upload ring"),
                    size: chunk_bytes,
                    usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });
                let mapped = map_write(&buffer)?;
                Ok((buffer, Some(mapped)))
            })
            .collect::<Result<Vec<_>, String>>()?;
        if let Some(old) = ring.replace(UploadRing { chunk: chunk_bytes, slots }) {
            for (buffer, _) in old.slots {
                buffer.destroy();
            }
        }
    }
    let ring = ring.as_mut().unwrap();
    for (k, chunk) in chunks(segments, chunk_bytes).iter().enumerate() {
        let (staging, pending) = &mut ring.slots[k % depth];
        if let Some(mapped) = pending.take() {
            let waited = perf_now();
            mapped.await.map_err(|e| format!("mapAsync(WRITE): {e:?}"))?;
            inner.committer.waits.set(inner.committer.waits.get() + perf_now() - waited);
        }
        let raw = staging.as_webgpu().ok_or("not on the WebGPU backend")?;
        let range = raw
            .get_mapped_range_with_f64_and_f64(0.0, chunk.len as f64)
            .map_err(|e| format!("getMappedRange: {e:?}"))?;
        let target = Uint8Array::new(&range);
        for &(at, len, src) in &chunk.pieces {
            target.set(&Uint8Array::new_with_byte_offset_and_length(memory, src, len as u32), at as u32);
        }
        raw.unmap();
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(
            staging,
            0,
            &commitment.coefficient_buffers()[chunk.buffer],
            chunk.src,
            chunk.len,
        );
        gpu.queue.submit([encoder.finish()]);
        *pending = Some(map_write(staging)?);
    }
    Ok(())
}

/// `mapAsync(WRITE)` of the whole buffer, raw (wgpu's own map state is never used for the ring).
fn map_write(buffer: &wgpu::Buffer) -> Result<wasm_bindgen_futures::JsFuture, String> {
    const MAP_WRITE: u32 = 2;
    let raw = buffer.as_webgpu().ok_or("not on the WebGPU backend")?;
    Ok(wasm_bindgen_futures::JsFuture::from(raw.map_async(MAP_WRITE).unchecked_into::<js_sys::Promise>()))
}

/// [`UploadMode::Write`] and [`UploadMode::Gather`]: `writeBuffer`s, waiting for the queue after
/// each `upload_window` bytes. With `gather`, the columns that sit side by side in a coefficient
/// buffer are copied into one `upload_chunk`-byte array first and written together; a column of
/// half a chunk or more is written straight from the prover's memory.
async fn upload_written(
    inner: &Inner,
    commitment: &StripedCommitment,
    segments: &[Segment],
    memory: &SharedArrayBuffer,
    gather: bool,
) -> Result<(), String> {
    let io = inner.io.get();
    let gpu = &inner.gpu;
    let queue = gpu.queue.as_webgpu().ok_or("not on the WebGPU backend")?;
    let staging = if gather { Some(inner.committer.gather_buffer(io.upload_chunk)) } else { None };
    let mut unwaited = 0u64;
    // A run of columns gathered into `staging`: its buffer, its offset there, its bytes.
    let mut run: Option<(usize, u64, u64)> = None;
    // Writes `source` at byte `at` of coefficient buffer `buffer`.
    let write = |buffer: usize, at: u64, source: &Uint8Array| -> Result<(), String> {
        let target = commitment.coefficient_buffers()[buffer]
            .as_webgpu()
            .ok_or("not on the WebGPU backend")?;
        queue
            .write_buffer_with_f64_and_u8_array(target, at as f64, source)
            .map_err(|e| format!("writeBuffer: {e:?}"))
    };
    for segment in segments {
        let (bytes, at) = (segment.len, segment.src);
        let source = view(memory, segment.dst, bytes as u32)?;
        match &staging {
            Some(staging) if bytes * 2 <= io.upload_chunk => {
                // Flush the run unless this column continues it and fits.
                if let Some((buffer, start, len)) = run
                    && (buffer != segment.buffer || start + len != at || len + bytes > io.upload_chunk)
                {
                    write(buffer, start, &staging.subarray(0, len as u32))?;
                    unwaited += len;
                    run = None;
                }
                let (_, _, len) = run.get_or_insert((segment.buffer, at, 0));
                staging.set(&source, *len as u32);
                *len += bytes;
            }
            _ => {
                let mut offset = 0u64;
                while offset < bytes {
                    let end = (offset + io.upload_chunk).min(bytes);
                    write(segment.buffer, at + offset, &source.subarray(offset as u32, end as u32))?;
                    unwaited += end - offset;
                    offset = end;
                }
            }
        }
        if io.upload_window > 0 && unwaited >= io.upload_window {
            queue_done(&gpu.queue).await;
            unwaited = 0;
        }
    }
    if let (Some((buffer, start, len)), Some(staging)) = (run, &staging) {
        write(buffer, start, &staging.subarray(0, len as u32))?;
    }
    Ok(())
}

/// The stats into the descriptor, each f64 as two words.
fn write_stats(memory: &SharedArrayBuffer, desc_ptr: u32, stats: &[f64; d::NSTATS]) {
    let words = Uint32Array::new_with_byte_offset_and_length(
        memory,
        desc_ptr + 4 * d::D_STATS as u32,
        2 * d::NSTATS as u32,
    );
    for (k, value) in stats.iter().enumerate() {
        let bits = value.to_bits();
        words.set_index(2 * k as u32, bits as u32);
        words.set_index(2 * k as u32 + 1, (bits >> 32) as u32);
    }
}

/// A run of bytes to read back: from `buffer` (an index into the commitment's buffers, see
/// [`output_segments`]) at `src`, to the prover's memory at `dst`.
#[derive(Clone, Copy, Debug)]
struct Segment {
    buffer: usize,
    src: u64,
    len: u64,
    dst: u32,
}

/// The buffers the outputs come from: the evaluation buffers, then the whole columns', then the
/// tree's.
fn source_buffers(commitment: &StripedCommitment) -> Vec<&wgpu::Buffer> {
    commitment
        .evaluation_buffers()
        .iter()
        .chain(commitment.whole_buffer())
        .chain([commitment.tree_buffer()])
        .collect()
}

/// Every output's bytes, by source buffer and offset: each column's stripe 0 (or every stripe of
/// a column kept whole, stripe 0 first, as `new_striped` concatenates them), each stored layer.
fn output_segments(desc: &Descriptor, commitment: &StripedCommitment) -> Vec<Segment> {
    let n_out = commitment.evaluation_buffers().len();
    let whole = commitment.whole_buffer().map(|_| n_out);
    let tree = n_out + whole.is_some() as usize;
    let n_stripes = 1u64 << desc.log_blowup;
    let mut segments = vec![];
    for (i, (column, slot)) in desc.columns.iter().zip(&commitment.layout().slots).enumerate() {
        let len = 4u64 << column.log_size;
        match commitment.whole_slot(i) {
            None => segments.push(Segment {
                buffer: slot.buffer,
                src: slot.offset * 4,
                len,
                dst: column.out,
            }),
            Some(offset) => {
                for stripe in 0..n_stripes {
                    segments.push(Segment {
                        buffer: whole.unwrap(),
                        src: (stripe * commitment.whole_stride() + offset) * 4,
                        len,
                        dst: column.out + (stripe * len) as u32,
                    });
                }
            }
        }
    }
    let layout = commitment.tree_layout();
    for (j, &(ptr, hashes)) in desc.layers.iter().enumerate() {
        let level = desc.lifting_log_size - j as u32;
        if let (Some(offset), true) = (layout.level_offset(level), hashes > 0) {
            segments.push(Segment { buffer: tree, src: offset * 4, len: hashes as u64 * 32, dst: ptr });
        }
    }
    segments.sort_by_key(|segment| (segment.buffer, segment.src));
    segments
}

/// A staging chunk: `len` bytes of `buffer` from `src`, and where its pieces go (offset in the
/// chunk, bytes, destination).
struct Chunk {
    buffer: usize,
    src: u64,
    len: u64,
    pieces: Vec<(u64, u64, u32)>,
}

/// The segments packed into chunks of at most `chunk` bytes of one buffer each (the gaps between
/// segments within a chunk are copied too).
fn chunks(segments: &[Segment], chunk: u64) -> Vec<Chunk> {
    let mut chunks: Vec<Chunk> = vec![];
    for segment in segments {
        let (mut src, mut dst, mut left) = (segment.src, segment.dst, segment.len);
        while left > 0 {
            let fits = chunks.last().is_some_and(|last| {
                last.buffer == segment.buffer && src >= last.src && src < last.src + chunk
            });
            if !fits {
                chunks.push(Chunk { buffer: segment.buffer, src, len: 0, pieces: vec![] });
            }
            let last = chunks.last_mut().unwrap();
            let take = left.min(last.src + chunk - src);
            last.pieces.push((src - last.src, take, dst));
            last.len = last.len.max(src + take - last.src);
            src += take;
            dst += take as u32;
            left -= take;
        }
    }
    chunks
}

/// A chunk in flight: its staging buffer, its map's result, the chunk.
type InFlight<'a> = (
    wgpu::Buffer,
    futures_channel::oneshot::Receiver<Result<(), wgpu::BufferAsyncError>>,
    &'a Chunk,
);

/// Copies the mapped chunk's pieces into the prover's memory, and unmaps it.
async fn finish(
    inner: &Inner,
    memory: &SharedArrayBuffer,
    (staging, mapped, chunk): InFlight<'_>,
) -> Result<(), String> {
    let waited = perf_now();
    mapped.await.map_err(|_| "map callback dropped")?.map_err(|e| format!("mapAsync: {e}"))?;
    inner.committer.waits.set(inner.committer.waits.get() + perf_now() - waited);
    {
        let range = staging.slice(..chunk.len).get_mapped_range().map_err(|e| format!("getMappedRange: {e:?}"))?;
        let source = range.as_uint8array();
        for &(at, len, dst) in &chunk.pieces {
            Uint8Array::new_with_byte_offset_and_length(memory, dst, len as u32)
                .set(&source.subarray(at as u32, (at + len) as u32), 0);
        }
    }
    staging.unmap();
    Ok(())
}

/// Reads the segments of `sources` back through a ring of mappable buffers of `readback_chunk`
/// bytes, kept for the page's life (chunk k + 1 is copied on the GPU and mapped while chunk k is
/// copied out). Returns the bytes copied out.
async fn ring_readback(
    inner: &Inner,
    sources: &[&wgpu::Buffer],
    segments: &[Segment],
    memory: &SharedArrayBuffer,
) -> Result<u64, String> {
    let io = inner.io.get();
    let chunk_bytes = if io.readback_chunk == 0 { 64 << 20 } else { io.readback_chunk };
    let chunks = chunks(segments, chunk_bytes);
    let depth = inner.committer.readback_ring_depth.get().max(1);
    let ring = {
        let mut ring = inner.committer.readback_ring.borrow_mut();
        if ring.len() != depth || ring.iter().any(|buffer| buffer.size() != chunk_bytes) {
            for buffer in ring.drain(..) {
                buffer.destroy();
            }
            ring.extend((0..depth).map(|_| {
                inner.gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("readback ring"),
                    size: chunk_bytes,
                    usage: STAGING,
                    mapped_at_creation: false,
                })
            }));
        }
        ring.clone()
    };
    let mut in_flight: VecDeque<InFlight> = VecDeque::new();
    let mut bytes = 0;
    for (k, chunk) in chunks.iter().enumerate() {
        if in_flight.len() == depth {
            finish(inner, memory, in_flight.pop_front().unwrap()).await?;
        }
        let staging = ring[k % depth].clone();
        let mut encoder = inner.gpu.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(sources[chunk.buffer], chunk.src, &staging, 0, chunk.len);
        inner.gpu.queue.submit([encoder.finish()]);
        let (sender, receiver) = futures_channel::oneshot::channel();
        staging.slice(..chunk.len).map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        in_flight.push_back((staging, receiver, chunk));
        bytes += chunk.pieces.iter().map(|piece| piece.1).sum::<u64>();
    }
    while let Some(next) = in_flight.pop_front() {
        finish(inner, memory, next).await?;
    }
    Ok(bytes)
}
