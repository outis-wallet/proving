//! The bridge's `COMMIT_STRIPED` op: stwo's `CommitmentTreeProver::new_striped` on the GPU, for a
//! prover thread blocked on the bridge, with its inputs and outputs in the prover's memory.
//!
//! The commitment is the native committer's ([`stwo_gpu_spike::committer`]): the same plans and
//! kernels, on device buffers and twiddles kept from one commitment to the next, and streamed
//! ([`StreamedCommitment`]): the columns go to the GPU in groups, the coefficients of a group
//! going up while the GPU works on the one before, and each group's stripe 0 coming back while it
//! works on later ones. Only the I/O edges are the web's:
//!
//! - the waits are awaits (`mapAsync`, `onSubmittedWorkDone`), where native code waits with
//!   `device.poll`, which returns at once on the web; the work is in the same order, and the
//!   queue's order is what makes it correct (a group's readback is submitted before the next
//!   group's work, which overwrites it);
//! - the coefficients go up through a ring of mappable buffers kept for the page's life
//!   ([`UploadRing`]): the worker copies them from the prover's memory into a mapped chunk, which
//!   the GPU then copies into place (or, [`UploadMode::Write`], `writeBuffer`s straight from the
//!   prover's memory);
//! - the outputs come back through a ring of mappable buffers ([`ReadbackRing`]), each chunk copied
//!   out straight into the Vecs the prover allocated (stwo's own columns and layers);
//! - buffers are destroyed when they are replaced, and the commitment's plans when it is done: on
//!   the web a buffer otherwise lives until its JS object is garbage collected.
//!
//! The worker is one thread: the copies into and out of the rings are its own work, while the GPU
//! (and the browser's GPU process) does the rest. A tree whose columns don't fit a group goes
//! stripe by stripe ([`StripedCommitment`]), its upload, compute and readback one after the
//! other.
//!
//! The twiddles come from the prover's twiddle tree (or, configured so, are computed here), once:
//! every stripe's twiddles of every log size up to the largest committed, per blowup, are kept
//! ([`TwiddleCache`]), so a proof's commitments after the first of a blowup, and every commitment
//! of later proofs, find them ready.
//!
//! Everything a mismatched descriptor could make the commitment assert is checked first: a panic
//! here is a trap that ends the GPU worker (every later job fails), where an error only sends
//! this commitment to the CPU. After an error, everything kept is dropped (a ring's chunk may be
//! left mapped): the next commitment starts afresh.

use std::cell::{Cell, RefCell};
use std::collections::{BTreeSet, VecDeque};
use std::sync::Arc;

use js_sys::{ArrayBuffer, SharedArrayBuffer, Uint8Array, Uint32Array};
use stwo::core::poly::circle::CanonicCoset;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::poly::circle::PolyOps;
use stwo::prover::poly::twiddles::TwiddleTree;
use stwo_gpu_bridge::commit as d;
use stwo_gpu_spike::GpuContext;
use stwo_gpu_spike::commit::{
    CommitBuffers, CommitKernels, CommitOptions, StripedCommitment, TwiddleCache, kept_whole,
    takes_striped_path,
};
use stwo_gpu_spike::extension::{MAX_LOG_SIZE, MIN_LOG_SIZE, StripeTwiddles};
use stwo_gpu_spike::merkle::{TreeLayout, stwo_min_stored_level};
use stwo_gpu_spike::stream::{Place, StreamedCommitment, group_words};
use stwo_gpu_spike::transfer::{copies, pack};
use wasm_bindgen::JsCast;

use crate::{Inner, JobResult, perf_now, queue_done};

/// Where the twiddles come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TwiddleSource {
    /// The prover's twiddle tree, read from its memory when none kept covers a commitment.
    Prover,
    /// Computed here (stwo's `precompute_twiddles`).
    Computed,
}

/// How the coefficients go to the GPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UploadMode {
    /// Through the [`UploadRing`]: copied into a mapped chunk, which the GPU copies into place.
    Mapped,
    /// `writeBuffer` straight from the prover's memory, a call per column (or `chunk` of one).
    Write,
}

/// The commitment's options (see `Gpu::configure_commit`).
#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub twiddles: TwiddleSource,
    pub upload: UploadMode,
    /// The rings' chunk (bytes): also a group's coefficients, when streamed.
    pub chunk: u64,
    pub upload_depth: usize,
    pub readback_depth: usize,
    /// Commitments that would take more device memory than this (MiB; 0: no limit) are refused,
    /// for the CPU (e.g. on phones, where WebGPU's buffers count against the tab's memory).
    pub max_gpu_mib: u32,
    /// Streamed in groups when the columns fit (false: stripe by stripe, always).
    pub streamed: bool,
    /// Keep the buffers, twiddles and plans from one commitment to the next (false: every
    /// commitment makes its own and frees them after).
    pub keep: bool,
    /// Wait for each phase to end before the next (a diagnostic: each phase's time alone).
    pub serial: bool,
    /// The most commitments' plans kept, by shape.
    pub max_plans: usize,
    /// Time the GPU's own work with timestamp queries, if the device has them (a diagnostic).
    pub timestamps: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            twiddles: TwiddleSource::Prover,
            upload: UploadMode::Mapped,
            chunk: 32 << 20,
            upload_depth: 4,
            readback_depth: 4,
            max_gpu_mib: 0,
            streamed: true,
            keep: true,
            serial: false,
            max_plans: 16,
            timestamps: false,
        }
    }
}

/// The commitment's kernels on the worker's device, and what it keeps between commitments.
pub struct Committer {
    kernels: CommitKernels,
    pub options: Cell<Options>,
    /// What commitments reuse: `None` until the first, after an error, or after a change of
    /// options that makes it stale.
    kept: RefCell<Option<Kept>>,
    /// Set when the options changed what is kept: dropped at the next commitment.
    stale: Cell<bool>,
    /// [`TwiddleSource::Computed`]'s tree, with the log size of the largest domain it covers.
    computed: RefCell<Option<(u32, TwiddleTree<SimdBackend>)>>,
}

/// What a [`Committer`] keeps from one commitment to the next.
struct Kept {
    buffers: CommitBuffers,
    twiddles: TwiddleCache,
    upload: UploadRing,
    readback: ReadbackRing,
    plans: Plans,
}

impl Kept {
    fn new(gpu: &GpuContext, options: &Options) -> Result<Self, String> {
        Ok(Self {
            buffers: CommitBuffers::new(),
            twiddles: TwiddleCache::new(),
            upload: UploadRing::new(gpu, options.chunk, options.upload_depth)?,
            readback: ReadbackRing::new(gpu, options.chunk, options.readback_depth),
            plans: Plans::default(),
        })
    }

    /// Frees everything now.
    fn destroy(mut self) {
        self.plans.clear();
        self.buffers.destroy();
        self.twiddles.destroy();
        self.upload.destroy();
        self.readback.destroy();
    }

    fn held_bytes(&self) -> u64 {
        self.buffers.bytes() + self.twiddles.bytes() + self.upload.bytes() + self.readback.bytes()
    }
}

/// A commitment's plans, on the kept buffers and twiddles.
enum Planned {
    Streamed(StreamedCommitment),
    Striped(StripedCommitment),
}

impl Planned {
    fn destroy_plans(&self) {
        match self {
            Planned::Streamed(commitment) => commitment.destroy_plans(),
            Planned::Striped(commitment) => commitment.destroy_plans(),
        }
    }

    fn gpu_bytes(&self) -> u64 {
        match self {
            Planned::Streamed(commitment) => commitment.gpu_bytes().device(),
            Planned::Striped(commitment) => commitment.gpu_bytes().device(),
        }
    }
}

/// A tree's shape: what its plans depend on, with the buffers and twiddles they are bound to.
#[derive(Clone, PartialEq, Eq)]
struct Shape {
    log_sizes: Vec<u32>,
    log_blowup: u32,
    lifting_log_size: u32,
    /// The group's words when streamed, 0 stripe by stripe.
    group_words: u64,
}

/// Commitments' plans kept, by shape, the most recently used last: valid while the buffers they
/// are bound to are (none replaced since: [`CommitBuffers::generation`]) and their twiddles are
/// still kept ([`TwiddleCache::holds`]).
#[derive(Default)]
struct Plans {
    entries: Vec<(Shape, u64, Arc<StripeTwiddles>, Planned)>,
}

impl Plans {
    /// The plans of `shape` on `twiddles`, if kept and still valid; the invalid ones are freed.
    fn take(
        &mut self,
        shape: &Shape,
        generation: u64,
        cache: &TwiddleCache,
        twiddles: &Arc<StripeTwiddles>,
    ) -> Option<Planned> {
        let mut found = None;
        for (s, g, t, planned) in std::mem::take(&mut self.entries) {
            if g != generation || !cache.holds(&t) {
                discard(planned, t);
            } else if found.is_none() && s == *shape && Arc::ptr_eq(&t, twiddles) {
                found = Some(planned);
            } else {
                self.entries.push((s, g, t, planned));
            }
        }
        found
    }

    fn put(
        &mut self,
        shape: Shape,
        generation: u64,
        twiddles: Arc<StripeTwiddles>,
        planned: Planned,
        max: usize,
    ) {
        self.entries.push((shape, generation, twiddles, planned));
        while self.entries.len() > max {
            let (_, _, twiddles, planned) = self.entries.remove(0);
            discard(planned, twiddles);
        }
    }

    fn clear(&mut self) {
        for (_, _, twiddles, planned) in std::mem::take(&mut self.entries) {
            discard(planned, twiddles);
        }
    }
}

/// Frees a commitment's plans, and its twiddles if nothing else holds them any more (a table the
/// [`TwiddleCache`] replaced while these plans held it).
fn discard(planned: Planned, twiddles: Arc<StripeTwiddles>) {
    planned.destroy_plans();
    drop(planned);
    if let Ok(table) = Arc::try_unwrap(twiddles) {
        table.destroy();
    }
}

impl Committer {
    pub fn new(gpu: &GpuContext) -> Self {
        Self {
            kernels: CommitKernels::new(gpu),
            options: Cell::new(Options::default()),
            kept: RefCell::new(None),
            stale: Cell::new(false),
            computed: RefCell::new(None),
        }
    }

    /// Sets the options. A change of the rings' sizes, or of `keep`, drops what is kept (at the
    /// next commitment, so never under a running one).
    pub fn configure(&self, options: Options) {
        let old = self.options.replace(options);
        if (old.chunk, old.upload_depth, old.readback_depth, old.keep)
            != (options.chunk, options.upload_depth, options.readback_depth, options.keep)
        {
            self.stale.set(true);
        }
    }

    /// Frees what is kept between commitments (between proofs: the next commitment makes it
    /// again). Not while a commitment runs.
    pub fn release(&self) {
        if let Ok(mut kept) = self.kept.try_borrow_mut()
            && let Some(kept) = kept.take()
        {
            kept.destroy();
        }
    }

    /// The options, as JSON (for `Gpu::info`).
    pub fn options_json(&self) -> String {
        let o = self.options.get();
        format!(
            "{{\"twiddles\":\"{:?}\",\"upload\":\"{:?}\",\"chunk\":{},\"upload_ring\":{},\"\
             readback_ring\":{},\"max_gpu_mib\":{},\"streamed\":{},\"keep\":{},\"serial\":{},\"\
             max_plans\":{},\"timestamps\":{}}}",
            o.twiddles,
            o.upload,
            o.chunk,
            o.upload_depth,
            o.readback_depth,
            o.max_gpu_mib,
            o.streamed,
            o.keep,
            o.serial,
            o.max_plans,
            o.timestamps
        )
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
    if !ptr.is_multiple_of(4) || ptr as u64 + bytes > memory_bytes {
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
    fn check_shapes(&self, gpu: &GpuContext) -> Result<TreeLayout, String> {
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
                return Err(format!(
                    "column {i}: {} words of output, not {words}",
                    column.out_words
                ));
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

    /// The bytes of column `i`'s coefficients (and of its stripe 0).
    fn column_bytes(&self, i: usize) -> u64 {
        4 << self.columns[i].log_size
    }

    /// Where the tree's stored levels come back from (`tree`, laid out as `layout`) and go.
    fn layer_reads<'a>(&self, tree: &'a wgpu::Buffer, layout: &TreeLayout) -> Vec<Read<'a>> {
        let mut reads = vec![];
        for (j, &(ptr, hashes)) in self.layers.iter().enumerate() {
            let level = self.lifting_log_size - j as u32;
            if let (Some(offset), true) = (layout.level_offset(level), hashes > 0) {
                reads.push(Read {
                    src: tree,
                    src_offset: offset * 4,
                    len: hashes as u64 * 32,
                    dst: ptr,
                });
            }
        }
        reads
    }
}

/// The prover's twiddle tree (`ptr`, `len` words), copied out of its memory: stwo's
/// `precompute_twiddles` of the half coset of the canonic coset of `2 * len` points, the root
/// whose tail `domain_line_twiddles_from_tree` reads for each smaller domain.
fn prover_twiddles(
    memory: &SharedArrayBuffer,
    (ptr, len): (u32, u32),
) -> Result<TwiddleTree<SimdBackend>, String> {
    Ok(TwiddleTree {
        root_coset: prover_root(len)?,
        twiddles: Uint32Array::new_with_byte_offset_and_length(memory, ptr, len).to_vec(),
        itwiddles: vec![],
    })
}

/// The root coset of the prover's twiddle tree of `len` words.
fn prover_root(len: u32) -> Result<stwo::core::circle::Coset, String> {
    if !len.is_power_of_two() || len < 2 {
        return Err(format!("the prover's twiddles: {len} words, not a power of two"));
    }
    Ok(CanonicCoset::new(len.ilog2() + 1).circle_domain().half_coset)
}

/// Runs the op: `args = [descriptor, descriptor bytes]`.
pub async fn commit_striped(inner: &Inner, args: &[u32], memory: &SharedArrayBuffer) -> JobResult {
    let started = perf_now();
    let [desc_ptr, desc_bytes] = args else {
        return Err("commit_striped takes [descriptor, bytes]".into());
    };
    let desc = Descriptor::read(memory, *desc_ptr, *desc_bytes)?;
    let committer = &inner.committer;
    let options = committer.options.get();
    let tree_layout = desc.check_shapes(&inner.gpu)?;
    let mut stats = Stats::default();
    stats.set(d::ST_SERIAL, options.serial as u32 as f64);

    // Taken for the job (jobs run one at a time), put back after it.
    let mut kept = committer.kept.take();
    if committer.stale.replace(false)
        && let Some(old) = kept.take()
    {
        old.destroy();
    }
    let mut kept = match kept {
        Some(kept) => kept,
        None => Kept::new(&inner.gpu, &options)?,
    };
    let result = commit(inner, &desc, &tree_layout, memory, &mut kept, &options, &mut stats).await;
    match &result {
        Ok(()) | Err(Failure::Refused(_)) if options.keep => {
            stats.set(d::ST_HELD_BYTES, kept.held_bytes() as f64);
            *committer.kept.borrow_mut() = Some(kept);
        }
        // After a failure a ring's chunk may be left mapped, a map pending: start afresh.
        _ => kept.destroy(),
    }
    result.map_err(|(Failure::Refused(message) | Failure::Failed(message))| message)?;
    stats.set(d::ST_TOTAL, perf_now() - started);
    stats.write(memory, *desc_ptr);
    Ok([stats.get(d::ST_UPLOAD), stats.get(d::ST_COMPUTE), stats.get(d::ST_READBACK)])
}

/// Why a commitment went to the CPU: refused before any GPU work (what is kept stays), or failed
/// on the GPU (what is kept is dropped).
enum Failure {
    Refused(String),
    Failed(String),
}

/// The stats of [`d`], as the job collects them.
struct Stats([f64; d::NSTATS]);

impl Default for Stats {
    fn default() -> Self {
        Self([0.0; d::NSTATS])
    }
}

impl Stats {
    fn get(&self, k: usize) -> f64 {
        self.0[k]
    }

    fn set(&mut self, k: usize, value: f64) {
        self.0[k] = value;
    }

    fn add(&mut self, k: usize, value: f64) {
        self.0[k] += value;
    }

    /// Into the descriptor, each f64 as two words.
    fn write(&self, memory: &SharedArrayBuffer, desc_ptr: u32) {
        let words = Uint32Array::new_with_byte_offset_and_length(
            memory,
            desc_ptr + 4 * d::D_STATS as u32,
            2 * d::NSTATS as u32,
        );
        for (k, value) in self.0.iter().enumerate() {
            let bits = value.to_bits();
            words.set_index(2 * k as u32, bits as u32);
            words.set_index(2 * k as u32 + 1, (bits >> 32) as u32);
        }
    }
}

/// The commitment on `kept`: its twiddles, its plans (kept ones if any), the commitment itself,
/// streamed or stripe by stripe.
async fn commit(
    inner: &Inner,
    desc: &Descriptor,
    tree_layout: &TreeLayout,
    memory: &SharedArrayBuffer,
    kept: &mut Kept,
    options: &Options,
    stats: &mut Stats,
) -> Result<(), Failure> {
    let gpu = &inner.gpu;
    let committer = &inner.committer;
    let log_sizes = desc.log_sizes();

    // The stripes' twiddles: kept ones, or computed from the prover's tree (or one of ours).
    let t0 = perf_now();
    let distinct: BTreeSet<u32> = log_sizes.iter().copied().collect();
    let max_domain_log = log_sizes.iter().max().unwrap() + desc.log_blowup;
    let mut computed = false;
    let twiddles = if options.twiddles == TwiddleSource::Prover && desc.twiddles.0 != 0 {
        let root = prover_root(desc.twiddles.1).map_err(Failure::Refused)?;
        if root.log_size() + 1 < max_domain_log {
            return Err(Failure::Refused(format!(
                "the prover's twiddles cover less than 2^{max_domain_log} points"
            )));
        }
        kept.twiddles
            .try_get_with(gpu, &distinct, desc.log_blowup, root, || {
                computed = true;
                stats.set(d::ST_TWIDDLES_FROM_PROVER, 1.0);
                prover_twiddles(memory, desc.twiddles)
            })
            .map_err(Failure::Refused)?
    } else {
        let mut ours = committer.computed.borrow_mut();
        if ours.as_ref().is_none_or(|(log_size, _)| *log_size < max_domain_log) {
            let coset = CanonicCoset::new(max_domain_log).circle_domain().half_coset;
            *ours = Some((max_domain_log, SimdBackend::precompute_twiddles(coset)));
        }
        let tree = &ours.as_ref().unwrap().1;
        kept.twiddles
            .try_get_with(gpu, &distinct, desc.log_blowup, tree.root_coset, || {
                computed = true;
                Ok::<_, String>(TwiddleTree {
                    root_coset: tree.root_coset,
                    twiddles: tree.twiddles.clone(),
                    itwiddles: vec![],
                })
            })
            .map_err(Failure::Refused)?
    };
    stats.set(d::ST_TWIDDLES_COMPUTED, computed as u32 as f64);
    let t1 = perf_now();
    stats.set(d::ST_TWIDDLES, t1 - t0);

    // The plans: kept ones of this shape, or new ones (on the kept buffers, grown if needed).
    let group_words = if options.streamed {
        group_words(
            gpu,
            &log_sizes,
            desc.log_blowup,
            desc.lifting_log_size,
            options.chunk,
            options.chunk,
        )
    } else {
        None
    };
    let shape = Shape {
        log_sizes: log_sizes.clone(),
        log_blowup: desc.log_blowup,
        lifting_log_size: desc.lifting_log_size,
        group_words: group_words.unwrap_or(0),
    };
    let planned =
        match kept.plans.take(&shape, kept.buffers.generation(), &kept.twiddles, &twiddles) {
            Some(planned) => {
                stats.set(d::ST_PLANS_KEPT, 1.0);
                planned
            }
            None => match group_words {
                Some(group_words) => Planned::Streamed(StreamedCommitment::with_buffers(
                    gpu,
                    &committer.kernels,
                    &mut kept.buffers,
                    twiddles.clone(),
                    &log_sizes,
                    desc.log_blowup,
                    desc.lifting_log_size,
                    group_words,
                )),
                None => Planned::Striped(StripedCommitment::with_buffers(
                    gpu,
                    &committer.kernels,
                    &mut kept.buffers,
                    twiddles.clone(),
                    &log_sizes,
                    desc.log_blowup,
                    desc.lifting_log_size,
                    CommitOptions::default(),
                )),
            },
        };
    let t2 = perf_now();
    stats.set(d::ST_SETUP, t2 - t1);
    stats.set(d::ST_GPU_BYTES, planned.gpu_bytes() as f64);
    let limit = options.max_gpu_mib as u64;
    if limit > 0 && planned.gpu_bytes() + twiddles.bytes() > limit << 20 {
        planned.destroy_plans();
        // The buffers may have grown for it: free them too.
        kept.plans.clear();
        kept.buffers.destroy();
        return Err(Failure::Refused(format!(
            "{} MiB of GPU memory, over the limit ({limit} MiB)",
            (planned.gpu_bytes() + twiddles.bytes()) >> 20
        )));
    }

    let timestamps = if options.timestamps { Timestamps::new(gpu, 4096) } else { None };
    let mut io = Io { inner, memory, desc, kept, options, stats, timestamps };
    let mut result = match &planned {
        Planned::Streamed(commitment) => io.streamed(commitment).await,
        Planned::Striped(commitment) => io.striped(commitment, tree_layout).await,
    };
    let Io { kept, stats, timestamps, .. } = io;
    stats.set(d::ST_COMMIT, perf_now() - t2);
    if let (Ok(()), Some(timestamps)) = (&result, timestamps) {
        match timestamps.read(gpu).await {
            Ok(gpu_times) => {
                for (k, ms) in [
                    d::ST_GPU_UPLOAD,
                    d::ST_GPU_COMPUTE,
                    d::ST_GPU_READBACK,
                    d::ST_GPU_BUSY,
                    d::ST_GPU_SPAN,
                ]
                .into_iter()
                .zip(gpu_times)
                {
                    stats.set(k, ms);
                }
            }
            Err(error) => result = Err(error),
        }
    }
    let t3 = perf_now();
    match (&result, options.keep) {
        (Ok(()), true) => {
            let generation = kept.buffers.generation();
            kept.plans.put(shape, generation, twiddles, planned, options.max_plans);
        }
        _ => planned.destroy_plans(),
    }
    stats.set(d::ST_DESTROY, perf_now() - t3);
    result.map_err(Failure::Failed)
}

/// Bytes of the prover's memory to write at `dst_offset` of `dst`.
struct Write<'a> {
    dst: &'a wgpu::Buffer,
    dst_offset: u64,
    src: u32,
    len: u64,
}

/// Bytes at `src_offset` of `src` to read into the prover's memory.
struct Read<'a> {
    src: &'a wgpu::Buffer,
    src_offset: u64,
    len: u64,
    dst: u32,
}

/// A commitment's transfers and submissions.
struct Io<'a> {
    inner: &'a Inner,
    memory: &'a SharedArrayBuffer,
    desc: &'a Descriptor,
    kept: &'a mut Kept,
    options: &'a Options,
    stats: &'a mut Stats,
    timestamps: Option<Timestamps>,
}

impl Io<'_> {
    /// The coefficients of the columns at `places`.
    fn writes<'b>(&self, places: Vec<Place<'b>>) -> Vec<Write<'b>> {
        places
            .into_iter()
            .map(|(i, dst, dst_offset)| Write {
                dst,
                dst_offset,
                src: self.desc.columns[i].coeffs,
                len: self.desc.column_bytes(i),
            })
            .collect()
    }

    /// Stripe 0 of the columns at `places`.
    fn reads<'b>(&self, places: Vec<Place<'b>>) -> Vec<Read<'b>> {
        places
            .into_iter()
            .map(|(i, src, src_offset)| Read {
                src,
                src_offset,
                len: self.desc.column_bytes(i),
                dst: self.desc.columns[i].out,
            })
            .collect()
    }

    async fn upload(&mut self, writes: &[Write<'_>]) -> Result<(), String> {
        let start = perf_now();
        let (copy, wait) = match self.options.upload {
            UploadMode::Mapped => {
                self.kept
                    .upload
                    .upload(&self.inner.gpu, self.memory, writes, self.timestamps.as_mut())
                    .await?
            }
            UploadMode::Write => {
                (write_buffers(&self.inner.gpu, self.memory, writes, self.options.chunk)?, 0.0)
            }
        };
        self.stats.add(d::ST_UPLOAD_BYTES, writes.iter().map(|w| w.len).sum::<u64>() as f64);
        self.stats.add(d::ST_UPLOAD_WAIT, wait);
        if self.options.serial {
            queue_done(&self.inner.gpu.queue).await;
            self.stats.add(d::ST_UPLOAD, perf_now() - start);
        } else {
            self.stats.add(d::ST_UPLOAD, copy);
        }
        Ok(())
    }

    /// Submits what `record` records.
    async fn submit(&mut self, record: impl FnOnce(&mut wgpu::CommandEncoder)) {
        let start = perf_now();
        let mut encoder = self.inner.gpu.device.create_command_encoder(&Default::default());
        let span = self.timestamps.as_mut().and_then(|t| t.begin(&mut encoder, Phase::Compute));
        record(&mut encoder);
        if let (Some(t), Some(span)) = (self.timestamps.as_mut(), span) {
            t.end(&mut encoder, span);
        }
        self.inner.gpu.queue.submit([encoder.finish()]);
        if self.options.serial {
            queue_done(&self.inner.gpu.queue).await;
        }
        self.stats.add(d::ST_COMPUTE, perf_now() - start);
    }

    /// Starts reading `reads` back (after the work submitted so far).
    async fn read(&mut self, reads: &[Read<'_>]) -> Result<(), String> {
        let start = perf_now();
        let (copy, wait) = self
            .kept
            .readback
            .start(&self.inner.gpu, self.memory, reads, self.timestamps.as_mut())
            .await?;
        self.stats.add(d::ST_READBACK_BYTES, reads.iter().map(|r| r.len).sum::<u64>() as f64);
        if self.options.serial {
            // The readback alone: its copies and its waits.
            let (_, drained) = self.kept.readback.drain(self.memory).await?;
            self.stats.add(d::ST_READBACK_WAIT, wait + drained);
            self.stats.add(d::ST_READBACK, perf_now() - start);
        } else {
            self.add_readback(copy, wait);
        }
        Ok(())
    }

    async fn drain(&mut self) -> Result<(), String> {
        let (copy, wait) = self.kept.readback.drain(self.memory).await?;
        self.add_readback(copy, wait);
        Ok(())
    }

    fn add_readback(&mut self, copy: f64, wait: f64) {
        self.stats.add(d::ST_READBACK, copy);
        self.stats.add(d::ST_READBACK_WAIT, wait);
    }

    /// Streamed: group by group (each group's upload, its work, its stripe 0's readback), then
    /// the columns of the largest size stripe by stripe (stripe 0's readback after its work),
    /// then the levels, the columns kept whole and the layers. As `StreamedCommitment::commit`
    /// does natively, in the same order.
    async fn streamed(&mut self, commitment: &StreamedCommitment) -> Result<(), String> {
        let kernels = &self.inner.committer.kernels;
        self.stats.set(d::ST_STREAMED, 1.0);
        self.stats.set(d::ST_GROUPS, commitment.n_groups() as f64);
        for group in 0..commitment.n_groups() {
            self.upload(&self.writes(commitment.group_uploads(group))).await?;
            self.submit(|encoder| commitment.encode_group(kernels, encoder, group)).await;
            // Submitted before the next group's work, which overwrites them.
            self.read(&self.reads(commitment.group_reads(group))).await?;
        }
        let top = commitment.top_uploads();
        if !top.is_empty() {
            self.upload(&self.writes(top)).await?;
            for stripe in 0..commitment.n_stripes() {
                self.submit(|encoder| commitment.encode_top(kernels, encoder, stripe)).await;
                if stripe == 0 {
                    // Submitted before stripe 1's work, which overwrites them.
                    self.read(&self.reads(commitment.top_reads())).await?;
                }
            }
        }
        let tail = perf_now();
        self.submit(|encoder| commitment.encode_levels(encoder)).await;
        let mut finals = vec![];
        if let Some(whole) = commitment.whole_buffer() {
            for (i, stripe, src_offset) in commitment.whole_reads() {
                let len = self.desc.column_bytes(i);
                let dst = self.desc.columns[i].out + (stripe as u64 * len) as u32;
                finals.push(Read { src: whole, src_offset, len, dst });
            }
        }
        finals.extend(self.desc.layer_reads(commitment.tree_buffer(), commitment.tree_layout()));
        self.read(&finals).await?;
        self.drain().await?;
        self.stats.set(d::ST_TAIL, perf_now() - tail);
        Ok(())
    }

    /// Stripe by stripe: the upload, the work (one submission), the readback.
    async fn striped(
        &mut self,
        commitment: &StripedCommitment,
        tree_layout: &TreeLayout,
    ) -> Result<(), String> {
        let kernels = &self.inner.committer.kernels;
        let layout = commitment.layout();
        let mut writes = vec![];
        for batch in &layout.batches {
            for (k, &i) in batch.columns.iter().enumerate() {
                let dst_offset = (batch.offset + ((k as u64) << batch.log_size)) * 4;
                writes.push((i, &commitment.coefficient_buffers()[batch.buffer], dst_offset));
            }
        }
        self.upload(&self.writes(writes)).await?;
        let tail = perf_now();
        self.submit(|encoder| commitment.encode(kernels, encoder, None)).await;
        let mut reads = vec![];
        let n_stripes = commitment.n_stripes() as u64;
        for (i, slot) in layout.slots.iter().enumerate() {
            let len = self.desc.column_bytes(i);
            match commitment.whole_slot(i) {
                None => reads.push(Read {
                    src: &commitment.evaluation_buffers()[slot.buffer],
                    src_offset: slot.offset * 4,
                    len,
                    dst: self.desc.columns[i].out,
                }),
                Some(offset) => {
                    for stripe in 0..n_stripes {
                        reads.push(Read {
                            src: commitment.whole_buffer().unwrap(),
                            src_offset: (stripe * commitment.whole_stride() + offset) * 4,
                            len,
                            dst: self.desc.columns[i].out + (stripe * len) as u32,
                        });
                    }
                }
            }
        }
        reads.extend(self.desc.layer_reads(commitment.tree_buffer(), tree_layout));
        self.read(&reads).await?;
        self.drain().await?;
        self.stats.set(d::ST_TAIL, perf_now() - tail);
        Ok(())
    }
}

/// [`UploadMode::Write`]: a `writeBuffer` straight from the prover's memory per write (or chunk
/// of one). Returns the time spent.
fn write_buffers(
    gpu: &GpuContext,
    memory: &SharedArrayBuffer,
    writes: &[Write<'_>],
    chunk: u64,
) -> Result<f64, String> {
    let start = perf_now();
    let queue = gpu.queue.as_webgpu().ok_or("not on the WebGPU backend")?;
    for write in writes {
        let target = write.dst.as_webgpu().ok_or("not on the WebGPU backend")?;
        let mut done = 0;
        while done < write.len {
            let len = chunk.min(write.len - done);
            let source = Uint8Array::new_with_byte_offset_and_length(
                memory,
                write.src + done as u32,
                len as u32,
            );
            queue
                .write_buffer_with_f64_and_u8_array(
                    target,
                    (write.dst_offset + done) as f64,
                    &source,
                )
                .map_err(|e| format!("writeBuffer: {e:?}"))?;
            done += len;
        }
    }
    Ok(perf_now() - start)
}

/// The distinct buffers of `buffers`, and each one's index among them.
fn buffer_indices<'a>(
    buffers: impl Iterator<Item = &'a wgpu::Buffer>,
) -> (Vec<&'a wgpu::Buffer>, Vec<usize>) {
    let mut distinct: Vec<&wgpu::Buffer> = vec![];
    let indices = buffers
        .map(|buffer| match distinct.iter().position(|&known| known == buffer) {
            Some(i) => i,
            None => {
                distinct.push(buffer);
                distinct.len() - 1
            }
        })
        .collect();
    (distinct, indices)
}

const MAP_READ: u32 = 1;
const MAP_WRITE: u32 = 2;

/// `mapAsync(mode, 0, bytes)` on the raw buffer (wgpu's own map state is never used for the
/// rings).
fn map(
    buffer: &wgpu::Buffer,
    mode: u32,
    bytes: u64,
) -> Result<wasm_bindgen_futures::JsFuture, String> {
    let raw = buffer.as_webgpu().ok_or("not on the WebGPU backend")?;
    let promise = raw.map_async_with_f64_and_f64(mode, 0.0, bytes as f64);
    Ok(wasm_bindgen_futures::JsFuture::from(promise.unchecked_into::<js_sys::Promise>()))
}

/// The mapped range `[0, bytes)` of the raw buffer.
fn mapped_range(buffer: &wgpu::Buffer, bytes: u64) -> Result<ArrayBuffer, String> {
    let raw = buffer.as_webgpu().ok_or("not on the WebGPU backend")?;
    raw.get_mapped_range_with_f64_and_f64(0.0, bytes as f64)
        .map_err(|e| format!("getMappedRange: {e:?}"))
}

fn unmap(buffer: &wgpu::Buffer) {
    if let Some(raw) = buffer.as_webgpu() {
        raw.unmap();
    }
}

/// Mappable chunks (`MAP_WRITE`) kept for the page's life, the coefficients go to the GPU
/// through: each one filled from the prover's memory by the worker, then copied into place by
/// the GPU; its next `mapAsync(WRITE)` resolves once that copy is done.
struct UploadRing {
    chunk: u64,
    slots: Vec<(wgpu::Buffer, Option<wasm_bindgen_futures::JsFuture>)>,
    next: usize,
}

impl UploadRing {
    fn new(gpu: &GpuContext, chunk: u64, depth: usize) -> Result<Self, String> {
        let slots = (0..depth.max(1))
            .map(|_| {
                let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("upload ring"),
                    size: chunk,
                    usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });
                let mapped = map(&buffer, MAP_WRITE, chunk)?;
                Ok((buffer, Some(mapped)))
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Self { chunk, slots, next: 0 })
    }

    fn bytes(&self) -> u64 {
        self.chunk * self.slots.len() as u64
    }

    fn destroy(&self) {
        for (buffer, _) in &self.slots {
            buffer.destroy();
        }
    }

    /// Writes every piece into its buffer, chunk by chunk; each chunk's copy is submitted before
    /// this returns. Returns (the copies' time, the waits' time) in ms.
    async fn upload(
        &mut self,
        gpu: &GpuContext,
        memory: &SharedArrayBuffer,
        writes: &[Write<'_>],
        mut timestamps: Option<&mut Timestamps>,
    ) -> Result<(f64, f64), String> {
        let (buffers, indices) = buffer_indices(writes.iter().map(|write| write.dst));
        let places = writes
            .iter()
            .zip(&indices)
            .map(|(write, &b)| (b, write.dst_offset))
            .collect::<Vec<_>>();
        let lens = writes.iter().map(|write| write.len as usize).collect::<Vec<_>>();
        let (mut copy, mut wait) = (0.0, 0.0);
        for runs in pack(&places, &lens, self.chunk) {
            let bytes = runs.iter().map(|run| run.len).sum::<u64>();
            let depth = self.slots.len();
            let (staging, pending) = &mut self.slots[self.next];
            self.next = (self.next + 1) % depth;
            if let Some(mapped) = pending.take() {
                let start = perf_now();
                mapped.await.map_err(|e| format!("mapAsync(WRITE): {e:?}"))?;
                wait += perf_now() - start;
            }
            let start = perf_now();
            let range = mapped_range(staging, bytes)?;
            let target = Uint8Array::new_with_byte_offset_and_length(&range, 0, bytes as u32);
            for run in &runs {
                let src = writes[run.piece].src + run.piece_offset as u32;
                target.set(
                    &Uint8Array::new_with_byte_offset_and_length(memory, src, run.len as u32),
                    run.chunk_offset as u32,
                );
            }
            unmap(staging);
            copy += perf_now() - start;
            let mut encoder = gpu.device.create_command_encoder(&Default::default());
            let span = timestamps.as_mut().and_then(|t| t.begin(&mut encoder, Phase::Upload));
            for (chunk_offset, buffer, offset, len) in copies(&runs) {
                encoder.copy_buffer_to_buffer(staging, chunk_offset, buffers[buffer], offset, len);
            }
            if let (Some(t), Some(span)) = (timestamps.as_mut(), span) {
                t.end(&mut encoder, span);
            }
            gpu.queue.submit([encoder.finish()]);
            *pending = Some(map(staging, MAP_WRITE, self.chunk)?);
        }
        Ok((copy, wait))
    }
}

/// A chunk being read back: its slot, its map, and where its bytes go (chunk offset, bytes,
/// destination).
struct InFlight {
    slot: usize,
    mapped: wasm_bindgen_futures::JsFuture,
    bytes: u64,
    pieces: Vec<(u64, u64, u32)>,
}

/// Mappable chunks (`MAP_READ`) kept for the page's life, the outputs come back through: each
/// chunk's copy is its own submission, after the work submitted before it; the worker copies a
/// chunk out (straight into the prover's memory) while the GPU works on later ones.
struct ReadbackRing {
    chunk: u64,
    slots: Vec<wgpu::Buffer>,
    free: Vec<usize>,
    in_flight: VecDeque<InFlight>,
}

impl ReadbackRing {
    fn new(gpu: &GpuContext, chunk: u64, depth: usize) -> Self {
        let slots = (0..depth.max(1))
            .map(|_| {
                gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("readback ring"),
                    size: chunk,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect::<Vec<_>>();
        let free = (0..slots.len()).rev().collect();
        Self { chunk, slots, free, in_flight: VecDeque::new() }
    }

    fn bytes(&self) -> u64 {
        self.chunk * self.slots.len() as u64
    }

    fn destroy(&self) {
        for buffer in &self.slots {
            buffer.destroy();
        }
    }

    /// Starts the readback of `reads`, chunk by chunk, each in a submission of its own after the
    /// work submitted before; when no chunk is free, the oldest is copied out first. Returns
    /// (the copies' time, the waits' time) in ms.
    async fn start(
        &mut self,
        gpu: &GpuContext,
        memory: &SharedArrayBuffer,
        reads: &[Read<'_>],
        mut timestamps: Option<&mut Timestamps>,
    ) -> Result<(f64, f64), String> {
        let (buffers, indices) = buffer_indices(reads.iter().map(|read| read.src));
        let places =
            reads.iter().zip(&indices).map(|(read, &b)| (b, read.src_offset)).collect::<Vec<_>>();
        let lens = reads.iter().map(|read| read.len as usize).collect::<Vec<_>>();
        let (mut copy, mut wait) = (0.0, 0.0);
        for runs in pack(&places, &lens, self.chunk) {
            if self.free.is_empty() {
                let (c, w) = self.finish(memory).await?;
                copy += c;
                wait += w;
            }
            let slot = self.free.pop().unwrap();
            let bytes = runs.iter().map(|run| run.len).sum::<u64>();
            let mut encoder = gpu.device.create_command_encoder(&Default::default());
            let span = timestamps.as_mut().and_then(|t| t.begin(&mut encoder, Phase::Readback));
            for (chunk_offset, buffer, offset, len) in copies(&runs) {
                encoder.copy_buffer_to_buffer(
                    buffers[buffer],
                    offset,
                    &self.slots[slot],
                    chunk_offset,
                    len,
                );
            }
            if let (Some(t), Some(span)) = (timestamps.as_mut(), span) {
                t.end(&mut encoder, span);
            }
            gpu.queue.submit([encoder.finish()]);
            let mapped = map(&self.slots[slot], MAP_READ, bytes)?;
            let pieces = runs
                .iter()
                .map(|run| {
                    (run.chunk_offset, run.len, reads[run.piece].dst + run.piece_offset as u32)
                })
                .collect();
            self.in_flight.push_back(InFlight { slot, mapped, bytes, pieces });
        }
        Ok((copy, wait))
    }

    /// Waits for the oldest chunk and copies it out. Returns (copy, wait) in ms.
    async fn finish(&mut self, memory: &SharedArrayBuffer) -> Result<(f64, f64), String> {
        let InFlight { slot, mapped, bytes, pieces } =
            self.in_flight.pop_front().ok_or("no chunk being read back")?;
        let start = perf_now();
        mapped.await.map_err(|e| format!("mapAsync(READ): {e:?}"))?;
        let wait = perf_now() - start;
        let start = perf_now();
        let buffer = &self.slots[slot];
        let range = mapped_range(buffer, bytes)?;
        let source = Uint8Array::new_with_byte_offset_and_length(&range, 0, bytes as u32);
        for (at, len, dst) in pieces {
            Uint8Array::new_with_byte_offset_and_length(memory, dst, len as u32)
                .set(&source.subarray(at as u32, (at + len) as u32), 0);
        }
        unmap(buffer);
        self.free.push(slot);
        Ok((perf_now() - start, wait))
    }

    /// Copies every chunk still being read back out. Returns (copy, wait) in ms.
    async fn drain(&mut self, memory: &SharedArrayBuffer) -> Result<(f64, f64), String> {
        let (mut copy, mut wait) = (0.0, 0.0);
        while !self.in_flight.is_empty() {
            let (c, w) = self.finish(memory).await?;
            copy += c;
            wait += w;
        }
        Ok((copy, wait))
    }
}

/// What a span of the GPU's time was.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Upload,
    Compute,
    Readback,
}

/// The GPU's own time (a diagnostic, `Options::timestamps`): a timestamp before and after each
/// submission's work, written by empty compute passes, resolved and read back at the end.
struct Timestamps {
    set: wgpu::QuerySet,
    capacity: u32,
    /// Each span's phase and its first query (the second is the next).
    spans: Vec<(Phase, u32)>,
}

impl Timestamps {
    /// `None` if the device has no timestamp queries.
    fn new(gpu: &GpuContext, capacity: u32) -> Option<Self> {
        if !gpu.device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return None;
        }
        let set = gpu.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("commit timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: capacity,
        });
        Some(Self { set, capacity, spans: vec![] })
    }

    fn write(&self, encoder: &mut wgpu::CommandEncoder, index: u32) {
        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set: &self.set,
                beginning_of_pass_write_index: Some(index),
                end_of_pass_write_index: None,
            }),
        });
    }

    /// Writes a span's first timestamp (`None` once the set is full).
    fn begin(&mut self, encoder: &mut wgpu::CommandEncoder, phase: Phase) -> Option<u32> {
        let index = 2 * self.spans.len() as u32;
        if index + 2 > self.capacity {
            return None;
        }
        self.write(encoder, index);
        self.spans.push((phase, index));
        Some(index)
    }

    fn end(&mut self, encoder: &mut wgpu::CommandEncoder, index: u32) {
        self.write(encoder, index + 1);
    }

    /// The GPU's time in ms: (upload, compute, readback, any of them, first to last).
    async fn read(self, gpu: &GpuContext) -> Result<[f64; 5], String> {
        let n = 2 * self.spans.len() as u64;
        if n == 0 {
            return Ok([0.0; 5]);
        }
        let resolve = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("timestamps resolve"),
            size: n * 8,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("timestamps readback"),
            size: n * 8,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        encoder.resolve_query_set(&self.set, 0..n as u32, &resolve, 0);
        encoder.copy_buffer_to_buffer(&resolve, 0, &staging, 0, n * 8);
        gpu.queue.submit([encoder.finish()]);
        let result = async {
            map(&staging, MAP_READ, n * 8)?.await.map_err(|e| format!("mapAsync(READ): {e:?}"))?;
            let range = mapped_range(&staging, n * 8)?;
            let words = Uint32Array::new(&range).to_vec();
            unmap(&staging);
            Ok::<_, String>(words)
        }
        .await;
        for buffer in [&resolve, &staging] {
            buffer.destroy();
        }
        self.set.destroy();
        let words = result?;
        let ns = |i: usize| (words[2 * i] as u64 | (words[2 * i + 1] as u64) << 32) as f64;
        let period = gpu.queue.get_timestamp_period() as f64 / 1e6;
        let mut times = [0.0; 5];
        let mut spans = self
            .spans
            .iter()
            .map(|&(phase, i)| (phase, ns(i as usize) * period, ns(i as usize + 1) * period))
            .collect::<Vec<_>>();
        for &(phase, a, b) in &spans {
            times[phase as usize] += (b - a).max(0.0);
        }
        spans.sort_by(|x, y| x.1.total_cmp(&y.1));
        let (mut busy, mut end) = (0.0, f64::MIN);
        for &(_, a, b) in &spans {
            if b > end {
                busy += b - a.max(end);
                end = b;
            }
        }
        times[3] = busy;
        times[4] = end - spans[0].1;
        Ok(times)
    }
}
