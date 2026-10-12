//! The composition's constraint evaluation on the GPU, as constraint-framework's constraint
//! evaluation hook: each component of a proof is evaluated by its generated kernels on its
//! columns resident on the GPU (the commitments keep their stripes 0 and 1, see
//! [`crate::parts::PartsMode::Resident`]), and its share added into the accumulator; a component
//! the GPU can't take declines, and the CPU evaluates it as without a hook.
//!
//! A component's kernels are found by a [`KernelKey`]: its type, and a fingerprint of its
//! constraints (constraint-framework's `ConstraintsFingerprint`, cheap: it doesn't trace them),
//! so that instances of one type with different constraints get different kernels. A key seen
//! for the first time is traced and lowered once (`ExprEvaluator`, [`lower`]), written to the
//! kernel library on disk, and compiled in the background; until its pipelines are ready the
//! component declines: a proof never waits for a compilation. At its creation the composer
//! compiles the library's programs in the background, with a pipeline cache persisted next to
//! them where the backend has one (Vulkan): a later process's kernels are ready within
//! milliseconds of its start.
//!
//! Shadow mode evaluates on the GPU, declines, and compares the CPU's result with the GPU's, word
//! for word, for debugging.
//!
//! What a component's evaluation adds is the same, word for word, as the SIMD evaluation's: the
//! arithmetic is exact in the field and every value canonical (see the crate's
//! `tests/constraint.rs` and the proofs' identity checks).

use std::collections::HashMap;
use std::fs;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use itertools::Itertools;
use rayon::prelude::*;
use stwo::core::fields::m31::{M31, P};
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::secure_column::SecureColumnByCoords;
use stwo_constraint_framework::constraint_hook::{ConstraintEvalJob, ConstraintEvalOutcome};

use super::ir::{Program, lower, param_values_of, words};
use super::kernel::{ConstraintKernels, KernelOptions, meta_words, read_timestamps};
use crate::GpuContext;
use crate::committer::GpuCommitter;
use crate::parts::Placement;

/// What identifies a component's kernels.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct KernelKey {
    /// The component's `FrameworkEval` type.
    pub component: String,
    /// The constraints' fingerprint.
    pub fingerprint: [u32; 4],
    pub n_constraints: usize,
    pub n_columns: [usize; 4],
}

impl KernelKey {
    fn line(&self) -> String {
        let f = self.fingerprint;
        let c = self.n_columns;
        format!(
            "key {} {} {} {} {} {} {} {} {} {}",
            self.component, f[0], f[1], f[2], f[3], self.n_constraints, c[0], c[1], c[2], c[3]
        )
    }

    fn parse(line: &str) -> Option<Self> {
        let w: Vec<&str> = line.split_whitespace().collect();
        if w.len() != 11 || w[0] != "key" {
            return None;
        }
        let n = |i: usize| w[i].parse::<u64>().ok();
        Some(Self {
            component: w[1].to_owned(),
            fingerprint: [n(2)? as u32, n(3)? as u32, n(4)? as u32, n(5)? as u32],
            n_constraints: n(6)? as usize,
            n_columns: [n(7)? as usize, n(8)? as usize, n(9)? as usize, n(10)? as usize],
        })
    }

    /// The library file's name.
    fn file_name(&self) -> String {
        let module = short_name(&self.component);
        let f = self.fingerprint;
        format!("{module}-{:08x}{:08x}{:08x}{:08x}.prog", f[0], f[1], f[2], f[3])
    }
}

/// Where a component's kernels are.
enum KernelState {
    Compiling,
    Ready(Arc<ReadyKernel>),
    Failed(String),
}

struct ReadyKernel {
    program: Program,
    kernels: ConstraintKernels,
}

#[derive(Clone, Debug)]
pub struct ComposerConfig {
    /// The kernel library (programs) and the pipeline cache: `None`, nothing on disk.
    pub library: Option<PathBuf>,
    /// Background compilation threads.
    pub compile_threads: usize,
    pub options: KernelOptions,
    /// Evaluate on the GPU, decline, and compare with the CPU's result.
    pub shadow: bool,
    /// Only the components whose type contains one of these (all if empty).
    pub only: Vec<String>,
}

impl Default for ComposerConfig {
    fn default() -> Self {
        Self {
            library: None,
            compile_threads: 2,
            options: KernelOptions::default(),
            shadow: false,
            only: vec![],
        }
    }
}

/// Counts and times of the composer's work since they were last taken.
#[derive(Clone, Debug, Default)]
pub struct ComposerStats {
    /// Components evaluated on the GPU, and their rows.
    pub evaluated: u64,
    pub rows: u64,
    /// Components declined, by reason.
    pub declined: HashMap<&'static str, u64>,
    /// Programs traced and lowered (first sight of a key), and the time.
    pub traced: u64,
    pub trace_time: Duration,
    /// Kernels compiled, failed, and the compile time (background threads).
    pub compiled: u64,
    pub compile_failed: u64,
    pub compile_time: Duration,
    /// Time in the hook for the components evaluated: in all, the fingerprints, the GPU's passes
    /// (timestamps), waiting for the GPU (submit to done), the readback, the CPU's add.
    pub hook_time: Duration,
    pub fingerprint_time: Duration,
    pub lookup_time: Duration,
    pub setup_time: Duration,
    pub timestamps_time: Duration,
    pub gpu_time: Duration,
    pub wait_time: Duration,
    pub readback_time: Duration,
    pub add_time: Duration,
    /// Components whose columns were assembled (not all resident with both stripes), the bytes
    /// uploaded for them, and the times the commitment buffers were released to make room.
    pub assembled: u64,
    pub uploaded_bytes: u64,
    pub releases: u64,
    /// Shadow mode: components compared, and words that differ (canonically, and as stored).
    pub shadow_compared: u64,
    pub shadow_different: u64,
    pub shadow_raw_different: u64,
    /// Per component short name: (count, hook seconds).
    pub per_component: HashMap<String, (u64, f64)>,
}

/// Where a component's columns of one interaction are read from.
enum Source {
    /// All in `buffer`, both stripes, at these byte offsets.
    Resident { buffer: wgpu::Buffer, byte_offsets: Vec<u64> },
    /// Assembled into a scratch buffer: the `k` columns from the `first` of the component's.
    Assembled { first: usize, k: usize },
}

/// Buffers reused by the runs.
#[derive(Default)]
struct RunBuffers {
    out: Option<wgpu::Buffer>,
    host: Vec<u32>,
    /// uniforms, coeffs, denom_inv, info: rewritten for each component.
    small: Vec<Option<wgpu::Buffer>>,
    /// The timestamp queries (with `OUTIS_GPU_COMPOSITION_TIMESTAMPS=1`), for up to the count.
    timestamps: Option<(wgpu::QuerySet, wgpu::Buffer, wgpu::Buffer, u32)>,
}

impl RunBuffers {
    /// Small buffer `k`, holding at least `bytes` bytes, with `words` written at its start.
    fn small(&mut self, gpu: &GpuContext, k: usize, words: &[u32]) -> wgpu::Buffer {
        if self.small.len() <= k {
            self.small.resize(k + 1, None);
        }
        let bytes = (words.len().max(4) * 4) as u64;
        if self.small[k].as_ref().is_none_or(|b| b.size() < bytes) {
            self.small[k] = Some(gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("constraint eval small"),
                size: bytes.next_power_of_two().max(64 << 10),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        let buffer = self.small[k].clone().unwrap();
        gpu.queue.write_buffer(&buffer, 0, bytemuck::cast_slice(words));
        buffer
    }

    /// The timestamp queries, for at least `count`.
    fn timestamps(
        &mut self,
        gpu: &GpuContext,
        count: u32,
    ) -> &(wgpu::QuerySet, wgpu::Buffer, wgpu::Buffer, u32) {
        if self.timestamps.as_ref().is_none_or(|t| t.3 < count) {
            let count = count.next_power_of_two().max(16);
            let query_set = gpu.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: None,
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
            self.timestamps = Some((
                query_set,
                buffer(wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC),
                buffer(wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST),
                count,
            ));
        }
        self.timestamps.as_ref().unwrap()
    }
}

struct CompileJob {
    key: KernelKey,
    program: Program,
}

/// See the module docs.
pub struct GpuComposer {
    committer: Arc<GpuCommitter>,
    config: ComposerConfig,
    registry: Arc<Mutex<HashMap<KernelKey, KernelState>>>,
    queue: Mutex<Option<Sender<CompileJob>>>,
    run: Mutex<RunBuffers>,
    stats: Arc<Mutex<ComposerStats>>,
    shadow: AtomicBool,
    /// Timestamps around the kernels' passes (`OUTIS_GPU_COMPOSITION_TIMESTAMPS=1`), for timing.
    timestamps: bool,
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn panic_message(panic: &Box<dyn std::any::Any + Send>) -> String {
    match (panic.downcast_ref::<&str>(), panic.downcast_ref::<String>()) {
        (Some(s), _) => s.to_string(),
        (_, Some(s)) => s.clone(),
        _ => "panic".to_string(),
    }
}

/// The default library: `$OUTIS_GPU_KERNEL_DIR`, else `$XDG_CACHE_HOME/outis-gpu-kernels`, else
/// `~/.cache/outis-gpu-kernels`.
pub fn default_library() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("OUTIS_GPU_KERNEL_DIR") {
        return Some(dir.into());
    }
    let cache = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| Path::new(&home).join(".cache")))?;
    Some(cache.join("outis-gpu-kernels"))
}

/// Writes `bytes` to `path` through a temporary file and a rename.
fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = path.with_extension(format!("tmp{}", std::process::id()));
    fs::write(&tmp, bytes)?;
    fs::rename(tmp, path)
}

fn short_name(component: &str) -> String {
    super::capture::short_name(component)
}

impl GpuComposer {
    /// A composer on `committer`'s device; starts compiling the library's programs in the
    /// background.
    pub fn new(committer: Arc<GpuCommitter>, config: ComposerConfig) -> Arc<Self> {
        let registry = Arc::new(Mutex::new(HashMap::new()));
        let stats = Arc::new(Mutex::new(ComposerStats::default()));
        let (sender, receiver) = channel::<CompileJob>();
        let receiver = Arc::new(Mutex::new(receiver));
        if let Some(library) = &config.library {
            let _ = fs::create_dir_all(library);
        }
        let cache = Self::pipeline_cache(committer.gpu(), config.library.as_deref());
        for _ in 0..config.compile_threads.max(1) {
            let receiver = receiver.clone();
            let registry = registry.clone();
            let stats = stats.clone();
            let committer = committer.clone();
            let cache = cache.clone();
            let options = config.options;
            std::thread::Builder::new()
                .name("constraint-compile".into())
                .spawn(move || {
                    compile_loop(&committer, &receiver, &registry, &stats, cache, options)
                })
                .expect("compile thread");
        }
        let shadow = AtomicBool::new(config.shadow);
        let timestamps = std::env::var("OUTIS_GPU_COMPOSITION_TIMESTAMPS").is_ok_and(|v| v == "1");
        let composer = Arc::new(Self {
            committer,
            shadow,
            timestamps,
            config,
            registry,
            queue: Mutex::new(Some(sender)),
            run: Mutex::new(RunBuffers::default()),
            stats,
        });
        composer.load_library();
        composer
    }

    /// The pipeline cache of the library's directory, if the device has the feature: with the
    /// data of an earlier process, if any (`fallback`: invalid data gives an empty cache).
    fn pipeline_cache(
        gpu: &GpuContext,
        library: Option<&Path>,
    ) -> Option<(wgpu::PipelineCache, PathBuf)> {
        let library = library?;
        if !gpu.device.features().contains(wgpu::Features::PIPELINE_CACHE) {
            return None;
        }
        let name = wgpu::util::pipeline_cache_key(&gpu.adapter_info)?;
        let path = library.join(name);
        let data = fs::read(&path).ok();
        // Safety: the data is what `PipelineCache::get_data` returned for this adapter (the key
        // names the adapter and driver), or nothing; `fallback` covers corrupt data.
        let cache = unsafe {
            gpu.device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
                label: Some("constraint kernels"),
                data: data.as_deref(),
                fallback: true,
            })
        };
        Some((cache, path))
    }

    /// Queues the library's programs for compilation.
    fn load_library(&self) {
        let Some(library) = &self.config.library else { return };
        let Ok(entries) = fs::read_dir(library) else { return };
        let mut files: Vec<PathBuf> = entries
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| path.extension().is_some_and(|e| e == "prog"))
            .collect();
        files.sort();
        for path in files {
            let Ok(text) = fs::read_to_string(&path) else { continue };
            let Some((first, rest)) = text.split_once('\n') else { continue };
            let (Some(key), Ok(program)) = (KernelKey::parse(first), Program::from_text(rest))
            else {
                continue;
            };
            if program.constraints.len() != key.n_constraints {
                continue;
            }
            self.queue_compile(key, program);
        }
    }

    fn queue_compile(&self, key: KernelKey, program: Program) {
        let mut registry = lock(&self.registry);
        if registry.contains_key(&key) {
            return;
        }
        registry.insert(key.clone(), KernelState::Compiling);
        if let Some(queue) = lock(&self.queue).as_ref() {
            let _ = queue.send(CompileJob { key, program });
        }
    }

    /// Sets shadow mode (see the module docs).
    pub fn set_shadow(&self, shadow: bool) {
        self.shadow.store(shadow, Ordering::Relaxed);
    }

    /// Stops the compilation threads once their queue is done.
    pub fn shutdown(&self) {
        lock(&self.queue).take();
    }

    /// The counts since the last call.
    pub fn take_stats(&self) -> ComposerStats {
        std::mem::take(&mut *lock(&self.stats))
    }

    /// The number of kernels ready, compiling and failed.
    pub fn kernel_counts(&self) -> (usize, usize, usize) {
        let registry = lock(&self.registry);
        let ready = registry.values().filter(|s| matches!(s, KernelState::Ready(_))).count();
        let compiling = registry.values().filter(|s| matches!(s, KernelState::Compiling)).count();
        (ready, compiling, registry.len() - ready - compiling)
    }

    /// The kernels that failed, and why.
    pub fn failures(&self) -> Vec<(String, String)> {
        lock(&self.registry)
            .iter()
            .filter_map(|(key, state)| match state {
                KernelState::Failed(why) => Some((short_name(&key.component), why.clone())),
                _ => None,
            })
            .collect()
    }

    /// Waits until no kernel is compiling (or `timeout`): true if none is.
    pub fn wait_compiled(&self, timeout: Duration) -> bool {
        let start = Instant::now();
        loop {
            if self.kernel_counts().1 == 0 {
                return true;
            }
            if start.elapsed() > timeout {
                return false;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    fn decline(&self, reason: &'static str) -> ConstraintEvalOutcome {
        *lock(&self.stats).declined.entry(reason).or_default() += 1;
        ConstraintEvalOutcome::Declined(None)
    }

    /// The constraint evaluation hook: see the module docs.
    pub fn evaluate(
        &self,
        job: &ConstraintEvalJob<'_>,
        accum: &mut SecureColumnByCoords<SimdBackend>,
    ) -> ConstraintEvalOutcome {
        let start = Instant::now();
        if !self.config.only.is_empty()
            && !self.config.only.iter().any(|o| job.component.contains(o.as_str()))
        {
            return self.decline("not selected");
        }
        if job.eval_log_size != job.trace_log_size + 1 {
            return self.decline("evaluation domain");
        }
        let fingerprint = job.fingerprint();
        let fingerprint_time = start.elapsed();
        if fingerprint.n_constraints != job.n_constraints {
            return self.decline("constraint count");
        }
        let n_columns = fingerprint.n_columns;
        let key = KernelKey {
            component: job.component.to_owned(),
            fingerprint: words(fingerprint.value),
            n_constraints: job.n_constraints,
            n_columns: [n_columns[0], n_columns[1], n_columns[2], n_columns[3]],
        };
        let state = lock(&self.registry).get(&key).map(|state| match state {
            KernelState::Ready(kernel) => Ok(kernel.clone()),
            KernelState::Compiling => Err("compiling"),
            KernelState::Failed(_) => Err("kernel failed"),
        });
        let kernel = match state {
            Some(Ok(kernel)) => kernel,
            Some(Err(reason)) => return self.decline(reason),
            None => {
                self.trace(job, key);
                return self.decline("first sight");
            }
        };

        // The columns, resident.
        let table = &kernel.kernels.table;
        let mut columns = vec![];
        for &i in &table.interactions {
            let Some(polys) = job.polys.get(i as usize) else {
                return self.decline("columns");
            };
            if polys.len() < table.n_columns[i as usize] {
                return self.decline("columns");
            }
            for poly in &polys[..table.n_columns[i as usize]] {
                columns.push((&poly.evals.values, poly.evals.domain.log_size()));
            }
        }
        let Some(domain_log_size) = columns.first().map(|c| c.1) else {
            return self.decline("columns");
        };
        let log_blowup = domain_log_size - job.trace_log_size;
        let lookup_start = Instant::now();
        let placements = self.committer.placements(&columns, log_blowup);
        let lookup_time = lookup_start.elapsed();
        // Per interaction: its columns' buffer, if they all hold both stripes in one; otherwise
        // they are assembled into a scratch buffer (stripe 0 from the host, stripe 1 or both from
        // the GPU, the whole evaluation from the host for a column the tree keeps whole).
        let column_bytes = 8u64 << job.trace_log_size;
        let mut sources: Vec<Source> = vec![];
        let mut scratch_bytes = 0u64;
        let mut at = 0;
        for &i in &table.interactions {
            let k = table.n_columns[i as usize];
            let mut whole = None;
            let mut offsets = vec![];
            for (j, placement) in placements[at..at + k].iter().enumerate() {
                match placement {
                    Some(Placement::Both { buffer, byte_offset })
                        if j == 0 || whole.as_ref() == Some(buffer) =>
                    {
                        whole = Some(buffer.clone());
                        offsets.push(*byte_offset);
                    }
                    _ => {
                        whole = None;
                        break;
                    }
                }
            }
            match whole {
                Some(buffer) if offsets.len() == k => {
                    sources.push(Source::Resident { buffer, byte_offsets: offsets })
                }
                _ => {
                    for (j, placement) in placements[at..at + k].iter().enumerate() {
                        let held = columns[at + j].0.length as u64 * 4;
                        if placement.is_none() && held < column_bytes {
                            return self.decline("not resident");
                        }
                    }
                    sources.push(Source::Assembled { first: at, k });
                    scratch_bytes += k as u64 * column_bytes;
                }
            }
            at += k;
        }
        if scratch_bytes > 0 {
            // Within the video memory policy; the commitment buffers are idle until the next
            // commitment (the composition's): released if that makes room.
            if !self.committer.vram_allows(scratch_bytes) {
                self.committer.release_buffers();
                lock(&self.stats).releases += 1;
                if !self.committer.vram_allows(scratch_bytes) {
                    return self.decline("video memory");
                }
            }
        }

        let params =
            param_values_of(&fingerprint.relation_values, job.claimed_sum, job.trace_log_size);
        let Ok(uniform_values) = kernel.program.bind(&params) else {
            return self.decline("parameters");
        };
        let coeffs: Vec<u32> = job.random_coeff_powers.iter().flat_map(|c| words(*c)).collect();
        let denom_inv: Vec<u32> = job.denom_inv.iter().map(|d| d.0).collect();
        let eval_log = job.eval_log_size;
        let n = 1usize << eval_log;

        let mut run = lock(&self.run);
        let gpu = self.committer.gpu();
        let setup_start = Instant::now();
        let result = catch_unwind(AssertUnwindSafe(|| {
            let out_bytes = 16u64 << eval_log;
            if run.out.as_ref().is_none_or(|out| out.size() < out_bytes) {
                run.out = Some(gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("constraint eval out"),
                    size: out_bytes,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
            }
            let out = run.out.clone().unwrap();
            let uniforms = run.small(gpu, 0, &kernel.kernels.uniform_words(&uniform_values));
            let coeffs = run.small(gpu, 1, &coeffs);
            let denom = run.small(gpu, 2, &denom_inv);
            // The columns' buffers and word offsets, assembling those not resident whole.
            let mut encoder = gpu.device.create_command_encoder(&Default::default());
            let mut trees: Vec<wgpu::Buffer> = vec![];
            let mut offsets: Vec<u32> = vec![];
            let mut uploaded = 0u64;
            for source in &sources {
                match source {
                    Source::Resident { buffer, byte_offsets } => {
                        trees.push(buffer.clone());
                        for offset in byte_offsets {
                            offsets.push(u32::try_from(offset / 4).expect("an offset in words"));
                        }
                    }
                    Source::Assembled { first, k } => {
                        let scratch = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                            label: Some("constraint eval assembled columns"),
                            size: *k as u64 * column_bytes,
                            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                            mapped_at_creation: false,
                        });
                        let half = column_bytes / 2;
                        for j in 0..*k {
                            let dst = j as u64 * column_bytes;
                            let host = columns[first + j].0.as_slice();
                            match &placements[first + j] {
                                Some(Placement::Both { buffer, byte_offset }) => {
                                    encoder.copy_buffer_to_buffer(
                                        buffer,
                                        *byte_offset,
                                        &scratch,
                                        dst,
                                        column_bytes,
                                    );
                                }
                                Some(Placement::Stripe1 { buffer, byte_offset }) => {
                                    let stripe0 = &host[..(half / 4) as usize];
                                    gpu.queue.write_buffer(
                                        &scratch,
                                        dst,
                                        bytemuck::cast_slice(stripe0),
                                    );
                                    uploaded += half;
                                    encoder.copy_buffer_to_buffer(
                                        buffer,
                                        *byte_offset,
                                        &scratch,
                                        dst + half,
                                        half,
                                    );
                                }
                                None => {
                                    // A column the tree keeps whole: stripes 0 and 1 from the host.
                                    let both = &host[..(column_bytes / 4) as usize];
                                    gpu.queue.write_buffer(
                                        &scratch,
                                        dst,
                                        bytemuck::cast_slice(both),
                                    );
                                    uploaded += column_bytes;
                                }
                            }
                            offsets.push((j as u64 * column_bytes / 4) as u32);
                        }
                        trees.push(scratch);
                    }
                }
            }
            let info = run.small(gpu, 3, &meta_words(job.trace_log_size, eval_log, &offsets));
            let tree_refs: Vec<&wgpu::Buffer> = trees.iter().collect();
            let bind_group = kernel.kernels.bind_group(
                gpu,
                &uniforms,
                &coeffs,
                &denom,
                wgpu::BufferBinding {
                    buffer: &out,
                    offset: 0,
                    size: wgpu::BufferSize::new(out_bytes),
                },
                &info,
                &tree_refs,
            );
            let count = 2 * kernel.kernels.kernels.len() as u32;
            let timestamps = (self.timestamps
                && gpu.device.features().contains(wgpu::Features::TIMESTAMP_QUERY))
            .then(|| run.timestamps(gpu, count).clone());
            encoder.clear_buffer(&out, 0, Some(out_bytes));
            kernel.kernels.encode(
                &mut encoder,
                &bind_group,
                eval_log,
                timestamps.as_ref().map(|t| (&t.0, 0)),
            );
            if let Some((query_set, resolve, readback, _)) = &timestamps {
                encoder.resolve_query_set(query_set, 0..count, resolve, 0);
                encoder.copy_buffer_to_buffer(resolve, 0, readback, 0, 8 * count as u64);
            }
            let setup = setup_start.elapsed();
            let submitted = Instant::now();
            gpu.queue.submit([encoder.finish()]);
            gpu.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
            let wait = submitted.elapsed();
            let timestamps_start = Instant::now();
            let gpu_time = timestamps
                .map(|(_, _, readback, _)| read_timestamps(gpu, &readback, count))
                .unwrap_or_default();
            let timestamps_time = timestamps_start.elapsed();
            // The component's share, read back.
            let read_start = Instant::now();
            if run.host.len() < 4 * n {
                run.host.resize(4 * n, 0);
            }
            let host = &mut run.host[..4 * n];
            self.committer.read_back(&out, bytemuck::cast_slice_mut(host))?;
            Ok::<_, String>((
                gpu_time,
                wait,
                read_start.elapsed(),
                setup,
                timestamps_time,
                uploaded,
            ))
        }));
        let (gpu_time, wait, readback, setup, timestamps_time, uploaded) = match result {
            Ok(Ok(times)) => times,
            Ok(Err(_)) | Err(_) => {
                drop(run);
                return self.decline("gpu failed");
            }
        };
        let share = &run.host[..4 * n];

        let add_start = Instant::now();
        let shadow = self.shadow.load(Ordering::Relaxed);
        let outcome = if shadow {
            let before: [Vec<u32>; 4] = std::array::from_fn(|c| {
                bytemuck::cast_slice::<M31, u32>(accum.columns[c].as_slice()).to_vec()
            });
            let share = share.to_vec();
            let stats = self.stats.clone();
            let name = short_name(job.component);
            ConstraintEvalOutcome::Declined(Some(Box::new(move |after| {
                let (mut different, mut raw_different) = (0u64, 0u64);
                for c in 0..4 {
                    let after = bytemuck::cast_slice::<M31, u32>(after.columns[c].as_slice());
                    for (r, word) in after.iter().enumerate().take(n) {
                        let expected = (M31(before[c][r] % P) + M31(share[c * n + r])).0;
                        different += (word % P != expected) as u64;
                        raw_different += (*word != expected) as u64;
                    }
                }
                if different > 0 || raw_different > 0 {
                    eprintln!(
                        "constraint shadow: {name}: {different} words differ ({raw_different} as \
                         stored)"
                    );
                }
                let mut stats = lock(&stats);
                stats.shadow_compared += 1;
                stats.shadow_different += different;
                stats.shadow_raw_different += raw_different;
            })))
        } else {
            // accum += share, coordinate by coordinate, in parallel.
            for c in 0..4 {
                let column = accum.columns[c].as_mut_slice();
                column[..n]
                    .par_chunks_mut(1 << 14)
                    .zip(share[c * n..(c + 1) * n].par_chunks(1 << 14))
                    .for_each(|(dst, src)| {
                        for (d, s) in dst.iter_mut().zip(src) {
                            *d += M31(*s);
                        }
                    });
            }
            ConstraintEvalOutcome::Evaluated
        };
        let add_time = add_start.elapsed();
        drop(run);
        let mut stats = lock(&self.stats);
        if !shadow {
            stats.evaluated += 1;
            stats.rows += n as u64;
        }
        let hook_time = start.elapsed();
        stats.hook_time += hook_time;
        stats.fingerprint_time += fingerprint_time;
        stats.lookup_time += lookup_time;
        if scratch_bytes > 0 {
            stats.assembled += 1;
            stats.uploaded_bytes += uploaded;
        }
        stats.setup_time += setup;
        stats.timestamps_time += timestamps_time;
        stats.gpu_time += gpu_time;
        stats.wait_time += wait;
        stats.readback_time += readback;
        stats.add_time += add_time;
        let entry = stats.per_component.entry(short_name(job.component)).or_default();
        entry.0 += 1;
        entry.1 += hook_time.as_secs_f64();
        outcome
    }

    /// Traces and lowers a component seen for the first time, writes its program to the library
    /// and queues its compilation.
    fn trace(&self, job: &ConstraintEvalJob<'_>, key: KernelKey) {
        let start = Instant::now();
        let traced = catch_unwind(AssertUnwindSafe(|| lower(&job.constraints())));
        {
            let mut stats = lock(&self.stats);
            stats.traced += 1;
            stats.trace_time += start.elapsed();
        }
        let program = match traced {
            Ok(program) if program.constraints.len() == job.n_constraints => program,
            Ok(_) => {
                lock(&self.registry).insert(key, KernelState::Failed("constraint count".into()));
                return;
            }
            Err(panic) => {
                lock(&self.registry).insert(key, KernelState::Failed(panic_message(&panic)));
                return;
            }
        };
        if let Some(library) = &self.config.library {
            let text = format!("{}\n{}", key.line(), program.to_text());
            let _ = write_atomic(&library.join(key.file_name()), text.as_bytes());
        }
        self.queue_compile(key, program);
    }
}

fn compile_loop(
    committer: &GpuCommitter,
    receiver: &Mutex<Receiver<CompileJob>>,
    registry: &Mutex<HashMap<KernelKey, KernelState>>,
    stats: &Mutex<ComposerStats>,
    cache: Option<(wgpu::PipelineCache, PathBuf)>,
    options: KernelOptions,
) {
    loop {
        let job = { lock(receiver).recv() };
        let Ok(CompileJob { key, program }) = job else { return };
        let start = Instant::now();
        let gpu = committer.gpu();
        let compiled = catch_unwind(AssertUnwindSafe(|| {
            ConstraintKernels::new(gpu, &program, options, cache.as_ref().map(|c| &c.0))
        }));
        let elapsed = start.elapsed();
        let state = match compiled {
            Ok(kernels) => {
                let mut stats = lock(stats);
                stats.compiled += 1;
                stats.compile_time += elapsed;
                KernelState::Ready(Arc::new(ReadyKernel { program, kernels }))
            }
            Err(panic) => {
                lock(stats).compile_failed += 1;
                let message = panic_message(&panic);
                eprintln!("constraint kernel {}: {message}", key.component);
                KernelState::Failed(message)
            }
        };
        lock(registry).insert(key, state);
        // Persist the pipeline cache when the queue is done (another thread may still add).
        let idle = !lock(registry).values().any(|s| matches!(s, KernelState::Compiling));
        if let (true, Some((cache, path))) = (idle, &cache)
            && let Some(data) = cache.get_data()
        {
            let _ = write_atomic(path, &data);
        }
    }
}

/// The components' kernels in a summary line, and the times.
pub fn describe(stats: &ComposerStats) -> String {
    let declined = stats.declined.iter().sorted().map(|(r, n)| format!("{r} {n}")).join(", ");
    format!(
        "constraint eval on the GPU: {} components ({} rows), declined [{declined}]; traced {} \
         ({:.3} s), compiled {} ({:.2} s, {} failed); hook {:.3} s: fingerprints {:.3}, lookups \
         {:.3}, setup {:.3}, GPU {:.3} (wait {:.3}, timestamps {:.3}), readback {:.3}, add {:.3}; \
         assembled {} ({} MiB uploaded, {} releases); shadow {} compared, {} words differ ({} as \
         stored)",
        stats.evaluated,
        stats.rows,
        stats.traced,
        stats.trace_time.as_secs_f64(),
        stats.compiled,
        stats.compile_time.as_secs_f64(),
        stats.compile_failed,
        stats.hook_time.as_secs_f64(),
        stats.fingerprint_time.as_secs_f64(),
        stats.lookup_time.as_secs_f64(),
        stats.setup_time.as_secs_f64(),
        stats.gpu_time.as_secs_f64(),
        stats.wait_time.as_secs_f64(),
        stats.timestamps_time.as_secs_f64(),
        stats.readback_time.as_secs_f64(),
        stats.add_time.as_secs_f64(),
        stats.assembled,
        stats.uploaded_bytes >> 20,
        stats.releases,
        stats.shadow_compared,
        stats.shadow_different,
        stats.shadow_raw_different,
    )
}
