//! The GPU striped commitment as a prover uses it: one long-lived [`GpuCommitter`] per process
//! (the device and the compiled kernels, and whatever else can be reused between commitments),
//! whose [`GpuCommitter::commit_striped`] either returns exactly what
//! `CommitmentTreeProver::new_striped` returns for the same arguments, or gives the polynomials
//! back with the reason, for the caller to commit them on the CPU.
//!
//! This is the API the prover's hook calls; what is behind it can change without the hook
//! changing. Behind it now: the device buffers, kept from one commitment to the next and grown
//! when a tree needs more ([`CommitBuffers`]); every blowup's twiddles, computed once
//! ([`TwiddleCache`]); and two staging rings, through which the coefficients go up and the
//! result comes back, by all the CPU's threads ([`crate::transfer`]). Only the plans (uniforms
//! and bind groups) are built per commitment. Natively, as an option (off by default:
//! `STWO_GPU_KEEP_WARM_MS`, [`GpuCommitter::set_keep_warm`]), a thread keeps the GPU's clocks up
//! for a few seconds after each commitment, as the next one of a proof usually follows within
//! them, and after the committer's creation, for the first ([`crate::warm`]).

use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use itertools::Itertools;
use stwo::core::ColumnVec;
use stwo::core::circle::Coset;
use stwo::core::fields::m31::BaseField;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sM31MerkleChannel;
use stwo::prover::CommitmentTreeProver;
use stwo::prover::backend::Column;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::simd::column::BaseColumn;
use stwo::prover::poly::BitReversedOrder;
use stwo::prover::poly::circle::{CircleCoefficients, CircleEvaluation};
use stwo::prover::poly::twiddles::TwiddleTree;

use crate::GpuContext;
use crate::commit::{
    CommitBuffers, CommitKernels, CommitOptions, StripedCommitment, TwiddleCache,
    takes_striped_path,
};
use crate::memory::VramPolicy;
use crate::parts::{Found, PartStats, PartStore, PartsMode};
use crate::recompute::Recomputer;
use crate::stream::{DeviceStripes, KeepStripe1, StreamedCommitment, StripeContent, group_words};
use crate::transfer::{Readback, ReadbackRing, RingTimes, Upload, UploadRing};
use crate::warm::{KeepWarm, window_from_env};

/// The tree `new_striped` returns.
pub type StripedTree = CommitmentTreeProver<SimdBackend, Blake2sM31MerkleChannel>;

/// Why a commitment wasn't made on the GPU: the caller commits on the CPU instead.
#[derive(Debug)]
pub enum Declined {
    /// `new_striped` wouldn't take its striped path for these arguments.
    NotStriped,
    /// The GPU failed (a validation error, a lost device, ...).
    Gpu(String),
}

/// What [`GpuCommitter::commit_striped`] returns when it declines: the polynomials, untouched.
pub type DeclinedCommit = (ColumnVec<CircleCoefficients<SimdBackend>>, Declined);

/// A device and its compiled kernels, for a process's commitments, one at a time.
pub struct GpuCommitter {
    /// `None` if its thread could not start (the commitments go on without it).
    warm: Option<KeepWarm>,
    gpu: GpuContext,
    kernels: CommitKernels,
    /// What commitments reuse; one commitment at a time. `None` after a failure (rebuilt by the
    /// next commitment: a failure may leave a ring's chunk mapped, or a buffer lost).
    state: Mutex<Option<Reused>>,
    /// Where the last commitment's time went.
    times: Mutex<Option<CommitTimes>>,
    /// Stripe 1 of the columns committed, for constraint evaluation (see [`crate::parts`]).
    /// Locked after `state` when both are.
    parts: Mutex<PartStore>,
    parts_mode: Mutex<PartsMode>,
    /// A2's kernels and twiddles, made at its first request (locked after `state` and `parts`).
    recomputer: Mutex<Option<Recomputer>>,
    /// The root coset of the twiddle tree of the last commitment at each blowup, whose stripe
    /// twiddles the commitments' cache keeps (A2 reuses them).
    root_cosets: Mutex<HashMap<u32, Coset>>,
    /// The bound on the video memory kept for constraint evaluation.
    vram_policy: Mutex<VramPolicy>,
    /// Whether the decommitments read the coefficients kept (see [`Self::decommit_values`]):
    /// then they stay until their tree's decommitment, the composition's tree's too.
    decommit: std::sync::atomic::AtomicBool,
    /// Device buffers lent to constraint evaluation between commitments ([`Self::transient`]):
    /// grow-only per slot, released at every commitment's start and on [`Self::clear_parts`].
    transient: Mutex<Vec<Option<wgpu::Buffer>>>,
    /// The decommitment's kernels, made at its first use.
    fold: Mutex<Option<crate::decommit::FoldKernels>>,
    decommit_stats: Mutex<DecommitStats>,
    /// The out-of-domain samples' kernels, made at their first use.
    ood: Mutex<Option<crate::ood::OodKernels>>,
    /// The FRI quotients' kernels and tables, made at their first use, and their counts.
    fri: Mutex<Option<crate::fri::FriState>>,
    fri_stats: Mutex<FriStats>,
    ood_stats: Mutex<OodStats>,
    /// EXPERIMENT (lean): commit with nothing held between commitments and the coefficients
    /// moved to the GPU, not copied (see [`Self::commit_lean`]). `STWO_GPU_LEAN=1`, or
    /// [`Self::set_lean`].
    lean: std::sync::atomic::AtomicBool,
    /// EXPERIMENT: the streamed path's buffers, twiddles and rings dropped after every
    /// commitment (`STWO_GPU_RELEASE=1`).
    release_after: bool,
    /// The lean commitments so far (for `STWO_GPU_LEAN_FAULT`).
    lean_calls: std::sync::atomic::AtomicUsize,
}

/// The out-of-domain samples served on the GPU (see [`GpuCommitter::ood_values`]), since the
/// stats were last taken.
#[derive(Clone, Debug, Default)]
pub struct OodStats {
    /// The samplings asked for, their groups and columns.
    pub calls: u64,
    pub groups: u64,
    pub columns: u64,
    /// The columns sampled on the GPU, those left to the CPU (not kept as coefficients, too
    /// small, or kept whole), and those the GPU failed.
    pub served: u64,
    pub not_resident: u64,
    pub failed: u64,
    /// The time in all (wall clock), and the GPU evaluations' phases.
    pub seconds: f64,
    pub times: crate::EvalTimes,
}

/// The decommitments served on the GPU (see [`GpuCommitter::decommit_values`]), since the stats
/// were last taken.
#[derive(Clone, Debug, Default)]
pub struct DecommitStats {
    /// Trees asked for, served, and why not.
    pub trees: u64,
    pub served: u64,
    pub not_resident: u64,
    pub failed: u64,
    /// The columns and positions evaluated, and the time in all (wall clock).
    pub columns: u64,
    pub positions: u64,
    pub seconds: f64,
    /// The GPU evaluations' phases.
    pub times: crate::EvalTimes,
}

/// The FRI quotients served on the GPU (see [`GpuCommitter::fri_quotients`]), since the stats
/// were last taken.
#[derive(Clone, Debug, Default)]
pub struct FriStats {
    /// The quotients asked for, served, and why not: the GPU doesn't take their shape (or too
    /// many values not kept as coefficients), the video memory policy, the GPU failed.
    pub calls: u64,
    pub served: u64,
    pub declined: u64,
    pub refused_vram: u64,
    pub failed: u64,
    /// The last reason one was declined or failed.
    pub reason: Option<String>,
    /// The columns of the served ones, those taken from their values.
    pub columns: u64,
    pub value_columns: u64,
    /// The time in all (wall clock), and the phases'.
    pub seconds: f64,
    pub times: crate::fri::FriTimes,
}

/// The transient slots of the FRI quotients' buffers (past the composer's).
const FRI_TRANSIENT_BASE: usize = 16;

/// [`crate::fri::FriResources`] from the committer: its transient buffers, its twiddles, its
/// rings.
struct CommitterFri<'a>(&'a GpuCommitter);

impl crate::fri::FriResources for CommitterFri<'_> {
    fn buffer(&self, slot: usize, bytes: u64) -> wgpu::Buffer {
        self.0.transient(FRI_TRANSIENT_BASE + slot, bytes)
    }

    fn stripe_twiddles(
        &self,
        log_sizes: &std::collections::BTreeSet<u32>,
        log_blowup: u32,
    ) -> Option<std::sync::Arc<crate::extension::StripeTwiddles>> {
        self.0.stripe_twiddles_covering(log_sizes, log_blowup)
    }

    fn upload(&self, pieces: &[Upload<'_>]) -> Result<(), String> {
        self.0.upload(pieces).map(|_| ())
    }

    fn read(&self, pieces: Vec<Readback<'_, '_>>) -> Result<(), String> {
        self.0.read_back_pieces(pieces).map(|_| ())
    }
}

/// A transient buffer's size for `bytes`: rounded up to 16 MiB (so that it serves the next
/// requests of about that size), never past a storage binding.
fn transient_size(bytes: u64, gpu: &GpuContext) -> u64 {
    let limits = gpu.device.limits();
    let max = limits.max_storage_buffer_binding_size.min(limits.max_buffer_size) / 256 * 256;
    bytes.next_multiple_of(16 << 20).min(max).max(bytes).max(256)
}

/// Kept free besides a tree's resident columns when checking the policy: the composer's buffers
/// (its output of the largest evaluation domain, 64 MiB at 2^22 rows).
const RESIDENT_MARGIN: u64 = 128 << 20;

/// Where a commitment's time went (wall clock).
#[derive(Clone, Copy, Debug, Default)]
pub struct CommitTimes {
    /// Whether the tree was streamed in groups ([`StreamedCommitment`]), or committed stripe by
    /// stripe ([`StripedCommitment`]: its largest column exceeds a group).
    pub streamed: bool,
    /// The plans (uniforms, bind groups), the buffers grown if the tree is the largest yet, the
    /// twiddles computed if new.
    pub setup: Duration,
    /// The upload, the compute and the readback: overlapping when streamed (the coefficients
    /// going up while the GPU works on earlier groups, stripe 0's evaluations copied out while it
    /// works on later ones); stripe by stripe, the upload, then the compute (while the CPU
    /// allocates the result), then the readback.
    pub commit: Duration,
    /// Of `commit`: the upload ring's (the CPU's writes into its chunks, and its waits for a free
    /// one).
    pub upload: RingTimes,
    /// Of `commit`: the readback ring's (its waits for the GPU's copies, and the CPU's copies out
    /// into stwo's columns and layers).
    pub readback: RingTimes,
    /// Of `commit`, streamed: the groups (their uploads and submissions, the copies out of the
    /// chunks read back meanwhile), the columns of the largest size (their upload, every stripe's
    /// submission, stripe 0's readback started), the copies out of the chunks still being read
    /// back (while the GPU hashes the levels), the readback of the columns kept whole and the
    /// layers.
    pub phases: [Duration; 4],
    /// The tree's assembly.
    pub assembly: Duration,
    /// The drop of the polynomials (kept until then, to give them back on a failure): natively,
    /// only handing them to a thread that frees them (about 40 ms for the Cairo base trace's
    /// 516 MiB, off the prover's way); in wasm, freeing them.
    pub drop_polynomials: Duration,
    pub total: Duration,
}

/// What a [`GpuCommitter`] keeps from one commitment to the next.
struct Reused {
    buffers: CommitBuffers,
    twiddles: TwiddleCache,
    upload: UploadRing,
    readback: ReadbackRing,
}

impl Reused {
    fn new(gpu: &GpuContext) -> Self {
        Self {
            buffers: CommitBuffers::new(),
            twiddles: TwiddleCache::new(),
            upload: UploadRing::with_defaults(gpu),
            readback: ReadbackRing::with_defaults(gpu),
        }
    }
}

/// The memory a [`GpuCommitter`] holds between commitments.
#[derive(Clone, Copy, Debug, Default)]
pub struct HeldBytes {
    /// The device buffers of the largest tree so far (coefficients, evaluations, Merkle states,
    /// layers, whole columns).
    pub buffers: u64,
    /// The twiddles of the blowups committed with, on the device.
    pub twiddles: u64,
    /// The upload ring's chunks (host-visible: the BAR, or the host's memory).
    pub upload: u64,
    /// The readback ring's chunks (the host's memory).
    pub readback: u64,
}

impl GpuCommitter {
    /// The committer on the adapter the environment picks (see the crate docs), or why there is
    /// none.
    pub fn new() -> Result<Self, String> {
        // With the optional features constraint evaluation uses if the adapter has them: a
        // pipeline cache (see `constraint::composer`) and timestamps for its timings.
        let gpu = pollster::block_on(GpuContext::try_create_async(
            None,
            wgpu::Features::PIPELINE_CACHE
                | wgpu::Features::TIMESTAMP_QUERY
                | wgpu::Features::MAPPABLE_PRIMARY_BUFFERS,
        ))?;
        let (kernels, reused) =
            catch_unwind(AssertUnwindSafe(|| (CommitKernels::new(&gpu), Reused::new(&gpu))))
                .map_err(|panic| format!("kernels: {}", panic_message(&panic)))?;
        let warm = catch_unwind(AssertUnwindSafe(|| KeepWarm::new(&gpu, window_from_env())))
            .ok()
            .and_then(Result::ok);
        // A committer is made for proving: the first commitment usually follows within the
        // window (a prover's process makes its committer, then proves). Nothing if keep-warm is
        // off (the default: no window).
        if let Some(warm) = &warm {
            warm.warm_up();
        }
        Ok(Self {
            warm,
            gpu,
            kernels,
            state: Mutex::new((!lean_from_env()).then_some(reused)),
            times: Mutex::new(None),
            parts: Mutex::new(PartStore::new()),
            parts_mode: Mutex::new(PartsMode::from_env()),
            recomputer: Mutex::new(None),
            root_cosets: Mutex::new(HashMap::new()),
            vram_policy: Mutex::new(VramPolicy::from_env()),
            decommit: std::sync::atomic::AtomicBool::new(false),
            fold: Mutex::new(None),
            transient: Mutex::new(vec![]),
            decommit_stats: Mutex::new(DecommitStats::default()),
            ood: Mutex::new(None),
            fri: Mutex::new(None),
            fri_stats: Mutex::new(FriStats::default()),
            ood_stats: Mutex::new(OodStats::default()),
            lean: std::sync::atomic::AtomicBool::new(lean_from_env()),
            release_after: std::env::var("STWO_GPU_RELEASE").is_ok_and(|v| v == "1"),
            lean_calls: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    /// How long the GPU's clocks are kept up after each commitment (see [`crate::warm`]): `None`
    /// (never) unless `STWO_GPU_KEEP_WARM_MS` (in ms) or [`Self::set_keep_warm`] set a window.
    /// Natively only; in wasm the window is kept but nothing is dispatched.
    pub fn keep_warm(&self) -> Option<Duration> {
        self.warm.as_ref().and_then(KeepWarm::window)
    }

    /// Sets how long the GPU's clocks are kept up after each commitment (`None`: never; it stops
    /// at once): e.g. [`crate::warm::DEFAULT_WINDOW`] to turn keep-warm on.
    pub fn set_keep_warm(&self, window: Option<Duration>) {
        if let Some(warm) = &self.warm {
            warm.set_window(window);
        }
    }

    /// Keeps the GPU's clocks up from now, for the window (see [`Self::set_keep_warm`]): for a
    /// caller that knows a commitment is coming (e.g. at the start of a proof, whose first
    /// commitment would otherwise start on an idle GPU).
    pub fn warm_up(&self) {
        if let Some(warm) = &self.warm {
            warm.warm_up();
        }
    }

    /// Stops keeping the GPU warm, for good, and waits for the keep-warm thread to end: call it
    /// before the process exits (a submission in flight while the driver is torn down crashes
    /// it). Commitments still work after it, without keep-warm.
    pub fn shutdown(&self) {
        if let Some(warm) = &self.warm {
            warm.stop();
        }
    }

    /// The keep-warm dispatches so far.
    pub fn keep_warm_dispatches(&self) -> u64 {
        self.warm.as_ref().map_or(0, KeepWarm::dispatches)
    }

    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.gpu.adapter_info
    }

    /// `CommitmentTreeProver::new_striped(polynomials, log_blowup_factor, twiddles,
    /// lifting_log_size, pool)` on the GPU: the same tree (every layer, every column's kept
    /// evaluations), or the polynomials back with the reason.
    pub fn commit_striped(
        &self,
        polynomials: ColumnVec<CircleCoefficients<SimdBackend>>,
        log_blowup_factor: u32,
        twiddles: &TwiddleTree<SimdBackend>,
        lifting_log_size: u32,
    ) -> Result<StripedTree, DeclinedCommit> {
        let log_sizes = polynomials.iter().map(|poly| poly.log_size()).collect_vec();
        if !takes_striped_path(&log_sizes, log_blowup_factor, lifting_log_size) {
            return Err((polynomials, Declined::NotStriped));
        }
        if self.lean()
            && *self.parts_mode.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
                == PartsMode::Off
            && !self.decommit_enabled()
        {
            if lean_moves() {
                return self.commit_lean(polynomials, log_blowup_factor, twiddles, lifting_log_size);
            }
            return self.commit_lean_in_place(
                polynomials,
                log_blowup_factor,
                twiddles,
                lifting_log_size,
            );
        }
        let mut state = self.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut parts = self.parts.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let parts_mode_now =
            *self.parts_mode.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        self.release_transient();
        let after_lookup = parts.commitment_starts(
            parts_mode_now == PartsMode::Coefficients && self.decommit_enabled(),
        );
        let parts_mode = *self.parts_mode.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(warm) = &self.warm {
            warm.begin();
        }
        let start = Instant::now();
        let mut times = CommitTimes::default();
        // The polynomials are kept until the tree is read back, so that a failure can still
        // give them back.
        let tree = catch_unwind(AssertUnwindSafe(|| {
            let reused = state.get_or_insert_with(|| Reused::new(&self.gpu));
            reused.upload.take_times();
            reused.readback.take_times();
            let distinct = log_sizes.iter().copied().collect();
            self.root_cosets
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .insert(log_blowup_factor, twiddles.root_coset);
            let twiddles = reused.twiddles.get(&self.gpu, &distinct, log_blowup_factor, twiddles);
            let group_words = group_words(
                &self.gpu,
                &log_sizes,
                log_blowup_factor,
                lifting_log_size,
                reused.upload.chunk_bytes(),
                reused.readback.chunk_bytes(),
            );
            let tree = match group_words {
                Some(group_words) => {
                    times.streamed = true;
                    let commitment = StreamedCommitment::with_buffers(
                        &self.gpu,
                        &self.kernels,
                        &mut reused.buffers,
                        twiddles,
                        &log_sizes,
                        log_blowup_factor,
                        lifting_log_size,
                        group_words,
                    );
                    // Stripe 1 kept for constraint evaluation, as the mode says.
                    let device_stripes = self.stripes_to_keep(
                        &mut parts,
                        parts_mode,
                        after_lookup,
                        &log_sizes,
                        log_blowup_factor,
                    );
                    let keep = match (parts_mode, &device_stripes) {
                        (PartsMode::Host, _) => KeepStripe1::Host,
                        (_, Some(stripes)) => KeepStripe1::Device(stripes),
                        _ => KeepStripe1::No,
                    };
                    times.setup = start.elapsed();
                    let (output, phases, host_stripes) = commitment.commit_keeping(
                        &self.gpu,
                        &self.kernels,
                        &mut reused.upload,
                        &mut reused.readback,
                        &polynomials,
                        keep,
                    );
                    times.commit = start.elapsed() - times.setup;
                    times.phases = phases;
                    let tree = commitment.tree_prover(output);
                    let stripe0 =
                        tree.polynomials.iter().map(|poly| &poly.evals.values).collect_vec();
                    if let Some(stripes) = host_stripes {
                        parts.keep_host(&stripe0, stripes, &log_sizes, log_blowup_factor);
                    }
                    if let Some(stripes) = device_stripes {
                        parts.keep_device(&stripe0, stripes, &log_sizes, log_blowup_factor);
                    }
                    tree
                }
                None => {
                    let commitment = StripedCommitment::with_buffers(
                        &self.gpu,
                        &self.kernels,
                        &mut reused.buffers,
                        twiddles,
                        &log_sizes,
                        log_blowup_factor,
                        lifting_log_size,
                        CommitOptions::default(),
                    );
                    times.setup = start.elapsed();
                    commitment.upload_through(&self.gpu, &mut reused.upload, &polynomials);
                    let output = commitment.run_and_read_through(
                        &self.gpu,
                        &self.kernels,
                        &mut reused.readback,
                    );
                    times.commit = start.elapsed() - times.setup;
                    commitment.tree_prover(output)
                }
            };
            times.upload = reused.upload.take_times();
            times.readback = reused.readback.take_times();
            times.assembly = start.elapsed() - times.setup - times.commit;
            tree
        }));
        if let Some(warm) = &self.warm {
            warm.end();
        }
        match tree {
            Ok(tree) => {
                let before = start.elapsed();
                free(polynomials);
                times.drop_polynomials = start.elapsed() - before;
                times.total = start.elapsed();
                *self.times.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(times);
                if self.release_after {
                    *state = None;
                }
                Ok(tree)
            }
            Err(panic) => {
                // Whatever the failure left behind goes; the next commitment starts afresh.
                *state = None;
                parts.clear();
                Err((polynomials, Declined::Gpu(panic_message(&panic))))
            }
        }
    }

    /// Whether commitments are lean (see [`Self::commit_lean`]).
    pub fn lean(&self) -> bool {
        self.lean.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Turns lean commitments on or off for the commitments from now on.
    pub fn set_lean(&self, lean: bool) {
        self.lean.store(lean, std::sync::atomic::Ordering::Relaxed);
    }

    /// EXPERIMENT (lean): the commitment for a unified-memory device, within the memory
    /// `new_striped` itself uses. Stripe by stripe ([`StripedCommitment`]: one stripe's
    /// evaluations and Merkle states, not every stripe's), on buffers of its own, freed when it
    /// returns: nothing is held between commitments. The coefficients are moved to the GPU, not
    /// copied: the host's go once uploaded (as `new_striped`'s go after stripe 0), and the GPU's
    /// once the work is done; stripe 0 is then read back buffer by buffer, each buffer freed once
    /// read. Only plain commitments (no parts kept for constraint evaluation, no decommitment from
    /// coefficients). A failure once the host's coefficients are gone cannot give them back: it
    /// panics.
    fn commit_lean(
        &self,
        polynomials: ColumnVec<CircleCoefficients<SimdBackend>>,
        log_blowup: u32,
        twiddles: &TwiddleTree<SimdBackend>,
        lifting_log_size: u32,
    ) -> Result<StripedTree, DeclinedCommit> {
        let log_sizes = polynomials.iter().map(|poly| poly.log_size()).collect_vec();
        let mut state = self.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        *state = None;
        self.release_transient();
        if let Some(warm) = &self.warm {
            warm.begin();
        }
        let start = Instant::now();
        let verbose = std::env::var("STWO_GPU_LEAN_VERBOSE").is_ok_and(|v| v == "1");
        if verbose {
            eprintln!("lean enter: footprint {} MiB", footprint_mib());
        }
        let mut times = CommitTimes::default();
        let chunk = lean_env_mib("STWO_GPU_LEAN_CHUNK_MIB", 8) << 20;
        // 0: the device's largest binding (the default layout): the largest size's columns then
        // go in one dispatch, the leaves' last absorption, and the Merkle states stay at the
        // sizes below it. Split, they carry states at the stripe's full height.
        let buffer_mib = lean_env_mib("STWO_GPU_LEAN_BUFFER_MIB", 0);
        let buffer_words = (buffer_mib > 0).then_some((buffer_mib << 20) / 4);
        let prepared = catch_unwind(AssertUnwindSafe(|| {
            self.root_cosets
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .insert(log_blowup, twiddles.root_coset);
            let distinct = log_sizes.iter().copied().collect();
            let stripe_twiddles = std::sync::Arc::new(crate::extension::StripeTwiddles::new(
                &self.gpu, &distinct, log_blowup, twiddles,
            ));
            let mut buffers = CommitBuffers::new();
            let commitment = StripedCommitment::with_buffers(
                &self.gpu,
                &self.kernels,
                &mut buffers,
                stripe_twiddles,
                &log_sizes,
                log_blowup,
                lifting_log_size,
                CommitOptions { max_buffer_words: buffer_words, ..Default::default() },
            );
            let mut upload = UploadRing::new(&self.gpu, chunk, 2);
            commitment.upload_through(&self.gpu, &mut upload, &polynomials);
            if std::env::var("STWO_GPU_LEAN_VERBOSE").is_ok_and(|v| v == "1") {
                eprintln!("lean commitment of {} columns up to 2^{}: {:?}", log_sizes.len(), log_sizes.iter().max().unwrap_or(&0), commitment.gpu_bytes());
            }
            commitment
        }));
        let commitment = match prepared {
            Ok(commitment) => commitment,
            Err(panic) => {
                if let Some(warm) = &self.warm {
                    warm.end();
                }
                return Err((polynomials, Declined::Gpu(panic_message(&panic))));
            }
        };
        times.setup = start.elapsed();
        if verbose {
            eprintln!("lean uploaded: footprint {} MiB at {:.3} s", footprint_mib(), start.elapsed().as_secs_f64());
        }
        // The coefficients are on the GPU (in the ring's chunks or copied on): the host's go,
        // here (not on a thread), so that their pages are gone before the GPU's work touches the
        // evaluations' and the states'.
        drop(polynomials);
        if verbose {
            eprintln!("lean host coefficients dropped: footprint {} MiB at {:.3} s", footprint_mib(), start.elapsed().as_secs_f64());
        }
        let tree = catch_unwind(AssertUnwindSafe(|| {
            commitment.submit(&self.gpu, &self.kernels);
            self.gpu.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
            times.commit = start.elapsed() - times.setup;
            if verbose {
                eprintln!("lean computed: footprint {} MiB at {:.3} s", footprint_mib(), start.elapsed().as_secs_f64());
            }
            commitment.release_inputs();
            let _ = self.gpu.device.poll(wgpu::PollType::wait_indefinitely());
            if verbose {
                eprintln!("lean inputs released: footprint {} MiB at {:.3} s", footprint_mib(), start.elapsed().as_secs_f64());
            }
            let mut readback = ReadbackRing::new(&self.gpu, chunk, 2);
            let output = commitment.read_through_releasing(&self.gpu, &mut readback);
            if verbose {
                eprintln!("lean read back: footprint {} MiB at {:.3} s", footprint_mib(), start.elapsed().as_secs_f64());
            }
            drop(readback);
            commitment.tree_prover(output)
        }));
        commitment.destroy();
        // wgpu frees destroyed buffers (and the staging of queued writes) at its next
        // maintenance: now, not at the next commitment.
        let _ = self.gpu.device.poll(wgpu::PollType::wait_indefinitely());
        // EXPERIMENT: Metal frees a released buffer's memory a while after (not at the release):
        // STWO_GPU_LEAN_SETTLE_MS waits for it, to see what prompt frees would give.
        if let Some(ms) = std::env::var("STWO_GPU_LEAN_SETTLE_MS").ok().and_then(|v| v.parse().ok()) {
            std::thread::sleep(Duration::from_millis(ms));
        }
        if verbose {
            eprintln!("lean destroyed: footprint {} MiB at {:.3} s", footprint_mib(), start.elapsed().as_secs_f64());
        }
        if let Some(warm) = &self.warm {
            warm.end();
        }
        match tree {
            Ok(tree) => {
                times.assembly = start.elapsed() - times.setup - times.commit;
                times.total = start.elapsed();
                *self.times.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(times);
                Ok(tree)
            }
            Err(panic) => panic!(
                "lean GPU commitment failed after its coefficients left the host: {}",
                panic_message(&panic)
            ),
        }
    }

    /// EXPERIMENT (lean, zero-copy): [`Self::commit_lean`] with the host's coefficients read in
    /// place and kept until the commitment is done, so that any failure gives them back to
    /// `new_striped`, as every commitment does. The GPU reads the columns that start on a page
    /// straight from the host's memory (Metal: [`wrap_host`]); the others go up, packed, into one
    /// buffer. Stripe by stripe ([`StripedCommitment::with_sources`]); then the levels and the
    /// columns kept whole come back, the evaluations' buffers are mapped (on a device with
    /// `MAPPABLE_PRIMARY_BUFFERS`), and only then is stripe 0 copied into the coefficients' own
    /// columns, which become the tree's ([`StripedCommitment::read_into_coefficients`]).
    fn commit_lean_in_place(
        &self,
        polynomials: ColumnVec<CircleCoefficients<SimdBackend>>,
        log_blowup: u32,
        twiddles: &TwiddleTree<SimdBackend>,
        lifting_log_size: u32,
    ) -> Result<StripedTree, DeclinedCommit> {
        let log_sizes = polynomials.iter().map(|poly| poly.log_size()).collect_vec();
        let mut state = self.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        *state = None;
        self.release_transient();
        if let Some(warm) = &self.warm {
            warm.begin();
        }
        let start = Instant::now();
        let verbose = std::env::var("STWO_GPU_LEAN_VERBOSE").is_ok_and(|v| v == "1");
        let fault = std::env::var("STWO_GPU_LEAN_FAULT").ok();
        let mut times = CommitTimes::default();
        let chunk = lean_env_mib("STWO_GPU_LEAN_CHUNK_MIB", 8) << 20;
        let mappable = self.gpu.device.features().contains(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS);
        let index = self.lean_calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let faulty = |at: &str| {
            fault.as_deref().is_some_and(|f| f == format!("{at}:{index}") || f == format!("{at}:all"))
        };
        let worked = catch_unwind(AssertUnwindSafe(|| {
            self.root_cosets
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .insert(log_blowup, twiddles.root_coset);
            let distinct = log_sizes.iter().copied().collect();
            let stripe_twiddles = std::sync::Arc::new(crate::extension::StripeTwiddles::new(
                &self.gpu, &distinct, log_blowup, twiddles,
            ));
            // The sources: the host's columns in place where they can be, else packed (in the
            // tree's order: by size, then input order, so that a batch's run is one dispatch).
            let mut sources: Vec<Option<crate::extension::ColumnSource>> = polynomials
                .iter()
                .map(|poly| {
                    let words = bytemuck::cast_slice::<BaseField, u32>(poly.coeffs.as_slice());
                    wrap_host(&self.gpu, words)
                        .map(|buffer| crate::extension::ColumnSource { buffer, word_offset: 0 })
                })
                .collect();
            let packed_columns = (0..log_sizes.len())
                .filter(|&i| sources[i].is_none())
                .sorted_by_key(|&i| log_sizes[i])
                .collect_vec();
            let packed_words: u64 = packed_columns.iter().map(|&i| 1u64 << log_sizes[i]).sum();
            let packed = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("packed coefficients"),
                size: (packed_words * 4).max(4),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut offset = 0;
            let mut pieces = vec![];
            for &i in &packed_columns {
                sources[i] =
                    Some(crate::extension::ColumnSource { buffer: packed.clone(), word_offset: offset });
                pieces.push(Upload {
                    dst: &packed,
                    dst_offset: offset * 4,
                    src: bytemuck::cast_slice::<BaseField, u8>(polynomials[i].coeffs.as_slice()),
                });
                offset += 1 << log_sizes[i];
            }
            if !pieces.is_empty() {
                UploadRing::new(&self.gpu, chunk, 2).upload(&self.gpu, &pieces);
            }
            let mut buffers = CommitBuffers::new();
            let commitment = StripedCommitment::with_sources(
                &self.gpu,
                &self.kernels,
                &mut buffers,
                stripe_twiddles,
                &log_sizes,
                log_blowup,
                lifting_log_size,
                CommitOptions::default(),
                sources.into_iter().map(Option::unwrap).collect(),
                mappable,
            );
            if verbose {
                eprintln!(
                    "lean in place {} columns up to 2^{}: {} packed ({} MiB), {:?}; footprint {} MiB",
                    log_sizes.len(),
                    log_sizes.iter().max().unwrap_or(&0),
                    packed_columns.len(),
                    (packed_words * 4) >> 20,
                    commitment.gpu_bytes(),
                    footprint_mib()
                );
            }
            commitment.submit(&self.gpu, &self.kernels);
            self.gpu.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
            assert!(!faulty("compute"), "STWO_GPU_LEAN_FAULT: a fault after the compute");
            // The GPU is done with the host's columns: their wrappers go before anything writes.
            commitment.release_inputs();
            packed.destroy();
            let _ = self.gpu.device.poll(wgpu::PollType::wait_indefinitely());
            if verbose {
                eprintln!("lean in place computed, inputs released: footprint {} MiB", footprint_mib());
            }
            commitment
        }));
        times.commit = start.elapsed();
        let done = |commitment: Option<&StripedCommitment>| {
            if let Some(commitment) = commitment {
                commitment.destroy();
            }
            let _ = self.gpu.device.poll(wgpu::PollType::wait_indefinitely());
            if let Some(warm) = &self.warm {
                warm.end();
            }
        };
        let commitment = match worked {
            Ok(commitment) => commitment,
            Err(panic) => {
                done(None);
                return Err((polynomials, Declined::Gpu(panic_message(&panic))));
            }
        };
        let mut readback = ReadbackRing::new(&self.gpu, chunk, 2);
        let output = if mappable {
            if faulty("readback") {
                // A failure before the commit point: the polynomials come back untouched.
                Err((polynomials, "STWO_GPU_LEAN_FAULT: a fault in the readback".to_string()))
            } else {
                commitment.read_into_coefficients(&self.gpu, &mut readback, polynomials, faulty("map"))
            }
        } else {
            // Without mappable buffers, stripe 0 into new columns; the coefficients go after.
            match catch_unwind(AssertUnwindSafe(|| commitment.read_through_releasing(&self.gpu, &mut readback))) {
                Ok(output) => {
                    drop(polynomials);
                    Ok(output)
                }
                Err(panic) => Err((polynomials, panic_message(&panic))),
            }
        };
        drop(readback);
        let tree = output.map(|output| commitment.tree_prover(output));
        done(Some(&commitment));
        if verbose {
            eprintln!("lean in place done: footprint {} MiB at {:.3} s", footprint_mib(), start.elapsed().as_secs_f64());
        }
        match tree {
            Ok(tree) => {
                times.total = start.elapsed();
                *self.times.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(times);
                Ok(tree)
            }
            Err((polynomials, message)) => Err((polynomials, Declined::Gpu(message))),
        }
    }

    /// What to keep of a tree being committed for constraint evaluation, as `mode` says, within
    /// the video memory policy: A3 (`Device`) its stripe 1; `Resident` both its stripes if they
    /// fit, else its stripe 1 only (the composer uploads stripe 0 for its components), else
    /// nothing; nothing of the commitment that follows constraint evaluation's lookups (the
    /// composition's, which it never reads).
    fn stripes_to_keep(
        &self,
        parts: &mut PartStore,
        mode: PartsMode,
        after_lookup: bool,
        log_sizes: &[u32],
        log_blowup: u32,
    ) -> Option<DeviceStripes> {
        // The ways to try, in order.
        let ways: &[StripeContent] = match mode {
            PartsMode::Device => &[StripeContent::Stripe1],
            PartsMode::Resident if !after_lookup => {
                &[StripeContent::Stripes01, StripeContent::Stripe1]
            }
            // Every tree's, the composition's too if the decommitments read them.
            PartsMode::Coefficients if !after_lookup || self.decommit_enabled() => {
                &[StripeContent::Coefficients]
            }
            _ => return None,
        };
        let limits = self.gpu.device.limits();
        let memory = crate::memory::DeviceMemory::query(&self.gpu);
        let reserved = crate::memory::process_reserved(&self.gpu);
        for &content in ways {
            // Storage buffers the kernels bind (a tree in one, if it fits); stripe 1 only: copy
            // sources.
            let max_bytes = match content {
                StripeContent::Stripe1 => limits.max_buffer_size.min(1 << 30),
                _ => limits.max_buffer_size.min(limits.max_storage_buffer_binding_size) / 256 * 256,
            };
            let bytes = DeviceStripes::planned_bytes(log_sizes, log_blowup, max_bytes, content);
            // The process's share of the device, the driver's budget, the cap: the GPU may drive
            // a display.
            let allowed = self.vram_policy().allows(
                bytes + RESIDENT_MARGIN,
                parts.held_bytes(),
                reserved,
                memory.as_ref(),
            );
            parts.offered(bytes, content, allowed.is_ok(), reserved.unwrap_or(0));
            match allowed {
                Ok(()) => {
                    return Some(DeviceStripes::new(
                        &self.gpu, log_sizes, log_blowup, max_bytes, content,
                    ));
                }
                Err(reason) => parts.refused(reason),
            }
        }
        None
    }

    /// Where stripe 1 of the columns committed from now on comes from, for constraint evaluation
    /// (see [`crate::parts`]); `STWO_GPU_PARTS` sets it at the committer's creation.
    pub fn set_parts_mode(&self, mode: PartsMode) {
        *self.parts_mode.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = mode;
    }

    pub fn parts_mode(&self) -> PartsMode {
        *self.parts_mode.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// The device.
    pub fn gpu(&self) -> &GpuContext {
        &self.gpu
    }

    /// Sets the video memory policy for what the commitments keep for constraint evaluation
    /// (see [`crate::memory::VramPolicy`]); by default [`VramPolicy::from_env`].
    pub fn set_vram_policy(&self, policy: VramPolicy) {
        *self.vram_policy.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = policy;
    }

    pub fn vram_policy(&self) -> VramPolicy {
        *self.vram_policy.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Sets the policy's explicit cap on what the commitments keep (`None`: none).
    pub fn set_parts_budget(&self, bytes: Option<u64>) {
        let mut policy = self.vram_policy.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        policy.cap = bytes;
    }

    /// The GPU memory the commitments keep for constraint evaluation now.
    pub fn parts_held_bytes(&self) -> u64 {
        self.parts.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).held_bytes()
    }

    /// The device's memory and this process's share now: `(memory, reserved, the process's
    /// bound)`, for reports.
    pub fn vram_report(&self) -> (Option<crate::memory::DeviceMemory>, Option<u64>, u64) {
        let memory = crate::memory::DeviceMemory::query(&self.gpu);
        let bound = self.vram_policy().process_bound(memory.as_ref());
        (memory, crate::memory::process_reserved(&self.gpu), bound)
    }

    /// The columns `columns` (each the values a tree holds and its committed domain's log size)
    /// resident on the GPU with their stripes 0 and 1 (in [`PartsMode::Resident`]), in order;
    /// `None` if one is not.
    pub fn resident_columns(
        &self,
        columns: &[(&BaseColumn, u32)],
        log_blowup: u32,
    ) -> Option<Vec<crate::parts::ResidentColumn>> {
        let mut parts = self.parts.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        columns
            .iter()
            .map(|(values, domain_log_size)| parts.resident(values, *domain_log_size, log_blowup))
            .collect()
    }

    /// Where the columns `columns` (each the values a tree holds and its committed domain's log
    /// size) are on the GPU, in order (see [`crate::parts::PartStore::placement`]).
    pub fn placements(
        &self,
        columns: &[(&BaseColumn, u32)],
        log_blowup: u32,
    ) -> Vec<Option<crate::parts::Placement>> {
        let mut parts = self.parts.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        columns
            .iter()
            .map(|(values, domain_log_size)| parts.placement(values, *domain_log_size, log_blowup))
            .collect()
    }

    /// Whether `bytes` more video memory may be taken now (the policy, with what is kept and
    /// the process's reserved memory).
    pub fn vram_allows(&self, bytes: u64) -> bool {
        let held = self.parts_held_bytes();
        self.vram_policy()
            .allows(
                bytes,
                held,
                crate::memory::process_reserved(&self.gpu),
                crate::memory::DeviceMemory::query(&self.gpu).as_ref(),
            )
            .is_ok()
    }

    /// Writes `pieces` into their buffers through the committer's upload ring (its fixed chunks:
    /// no staging memory beyond them); the copies are submitted when this returns, before any
    /// later submission. `Err` if the GPU failed (the ring is rebuilt by the next user).
    pub fn upload(&self, pieces: &[Upload<'_>]) -> Result<RingTimes, String> {
        let mut state = self.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let written = catch_unwind(AssertUnwindSafe(|| {
            let reused = state.get_or_insert_with(|| Reused::new(&self.gpu));
            reused.upload.take_times();
            reused.upload.upload(&self.gpu, pieces);
            reused.upload.take_times()
        }));
        written.map_err(|panic| {
            *state = None;
            panic_message(&panic)
        })
    }

    /// Reads `src` (from byte 0) back into `dst` through the committer's readback ring; `Err`
    /// if the GPU failed (the ring is rebuilt by the next user).
    pub fn read_back(&self, src: &wgpu::Buffer, dst: &mut [u8]) -> Result<RingTimes, String> {
        let mut state = self.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let read = catch_unwind(AssertUnwindSafe(|| {
            let reused = state.get_or_insert_with(|| Reused::new(&self.gpu));
            reused.readback.take_times();
            reused.readback.read(&self.gpu, vec![Readback { src, src_offset: 0, dst }]);
            reused.readback.take_times()
        }));
        read.map_err(|panic| {
            *state = None;
            panic_message(&panic)
        })
    }

    /// Drops whatever the commitments kept for constraint evaluation (e.g. at a proof's start, or
    /// when a commitment is made elsewhere).
    pub fn clear_parts(&self) {
        self.release_transient();
        self.parts.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).clear();
    }

    /// The counts of the part extension's lookups since the last call.
    pub fn take_part_stats(&self) -> PartStats {
        self.parts.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).take_stats()
    }

    /// Stripe 1 of `columns` (`(tree, committed column)`, each holding its stripe 0 of the
    /// committed domain split into `2^log_parts` parts), as constraint-framework's
    /// `TracePartHook` asks for it: each column's evaluations on part `part`, or `None` (the CPU
    /// evaluates it). Only part 1 is kept.
    pub fn trace_part(
        &self,
        part: usize,
        log_parts: u32,
        columns: &[(usize, &CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>)],
    ) -> Vec<Option<BaseColumn>> {
        let mode = self.parts_mode();
        let none = || columns.iter().map(|_| None).collect();
        if mode == PartsMode::Off || part != 1 {
            return none();
        }
        if mode == PartsMode::Recompute {
            return crate::recompute::trace_part(self, part, log_parts, columns);
        }
        let start = Instant::now();
        // The readback ring, for stripes kept on the GPU (`state` before `parts`).
        let mut state = self.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut parts = self.parts.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        parts.request(columns.len());
        let mut answers: Vec<Option<BaseColumn>> = Vec::with_capacity(columns.len());
        let mut device_reads = vec![];
        for (k, (tree, column)) in columns.iter().enumerate() {
            let found = parts.find(*tree, &column.values, column.domain.log_size(), log_parts);
            answers.push(match found {
                Some(Found::Host(values)) => Some(values),
                Some(Found::Device { set, buffer, offset }) => {
                    device_reads.push((k, set, buffer, offset));
                    // Safety: read back below, every word.
                    Some(unsafe { BaseColumn::uninitialized(column.values.length) })
                }
                None => None,
            });
        }
        let mut transfer = Duration::ZERO;
        if !device_reads.is_empty() {
            let transfer_start = Instant::now();
            let read = catch_unwind(AssertUnwindSafe(|| {
                let reused = state.get_or_insert_with(|| Reused::new(&self.gpu));
                let mut destinations: Vec<Option<&mut [u8]>> = answers
                    .iter_mut()
                    .map(|answer| {
                        answer.as_mut().map(|values| {
                            bytemuck::cast_slice_mut::<BaseField, u8>(values.as_mut_slice())
                        })
                    })
                    .collect();
                let pieces = device_reads
                    .iter()
                    .map(|&(k, set, buffer, offset)| Readback {
                        src: parts.device_buffer(set, buffer),
                        src_offset: offset,
                        dst: destinations[k].take().expect("a column read back"),
                    })
                    .collect_vec();
                reused.readback.take_times();
                reused.readback.read(&self.gpu, pieces);
                reused.readback.take_times()
            }));
            transfer = transfer_start.elapsed();
            if let Ok(ring) = &read {
                parts.add_readback_times(ring.wait, ring.copy);
            }
            if read.is_err() {
                // The columns read back are dropped: the CPU evaluates them.
                *state = None;
                for &(k, ..) in &device_reads {
                    answers[k] = None;
                }
            }
        }
        parts.add_lookup_time(start.elapsed(), transfer);
        answers
    }

    /// Stripe 1 of `values` (each the stripe 0, of `2^log_size` values, of a column committed at
    /// blowup `log_blowup`), recomputed on the GPU (A2, see [`crate::recompute`]); `None` if the
    /// GPU failed.
    pub fn recompute_stripe1(
        &self,
        values: &[&BaseColumn],
        log_size: u32,
        log_blowup: u32,
    ) -> Option<Vec<BaseColumn>> {
        let start = Instant::now();
        let mut state = self.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut parts = self.parts.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut recomputer =
            self.recomputer.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        parts.request(values.len());
        let root_coset = self
            .root_cosets
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&log_blowup)
            .copied();
        let result = crate::recompute::guarded(|| {
            let reused = state.get_or_insert_with(|| Reused::new(&self.gpu));
            let recomputer = recomputer.get_or_insert_with(|| Recomputer::new(&self.gpu));
            // The commitments' stripe twiddles of this blowup, if they cover the size.
            let cached = root_coset.and_then(|root_coset| {
                let log_sizes = [log_size].into_iter().collect();
                reused
                    .twiddles
                    .try_get_with(&self.gpu, &log_sizes, log_blowup, root_coset, || Err(()))
                    .ok()
            });
            recomputer.stripe1(
                &self.gpu,
                &self.kernels.extension,
                &mut reused.upload,
                &mut reused.readback,
                values,
                log_size,
                log_blowup,
                cached,
                false,
            )
        });
        let Some((stripes, _, times)) = result else {
            *state = None;
            *recomputer = None;
            parts.add_lookup_time(start.elapsed(), Duration::ZERO);
            return None;
        };
        let kept = recomputer.as_ref().map_or(0, Recomputer::bytes);
        parts.recomputed(values.len(), times.bytes, &times, kept);
        parts.add_lookup_time(start.elapsed(), times.upload + times.readback);
        Some(stripes)
    }

    /// [`Self::recompute_stripe1`], with the coefficients interpolated on the GPU too (for tests),
    /// and the times.
    pub fn recompute_stripe1_with_coefficients(
        &self,
        values: &[&BaseColumn],
        log_size: u32,
        log_blowup: u32,
    ) -> (Vec<BaseColumn>, Vec<BaseColumn>, crate::recompute::RecomputeTimes) {
        let mut state = self.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut recomputer =
            self.recomputer.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let reused = state.get_or_insert_with(|| Reused::new(&self.gpu));
        let recomputer = recomputer.get_or_insert_with(|| Recomputer::new(&self.gpu));
        let (stripes, coefficients, times) = recomputer.stripe1(
            &self.gpu,
            &self.kernels.extension,
            &mut reused.upload,
            &mut reused.readback,
            values,
            log_size,
            log_blowup,
            None,
            true,
        );
        (stripes, coefficients.unwrap(), times)
    }

    /// Where the last commitment made on the GPU spent its time.
    pub fn last_times(&self) -> Option<CommitTimes> {
        *self.times.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// The memory held between commitments (see [`HeldBytes`]).
    pub fn held_bytes(&self) -> HeldBytes {
        let state = self.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        state.as_ref().map_or_else(HeldBytes::default, |reused| HeldBytes {
            buffers: reused.buffers.bytes(),
            twiddles: reused.twiddles.bytes(),
            upload: reused.upload.bytes(),
            readback: reused.readback.bytes(),
        })
    }

    /// Frees the commitment buffers held between commitments (the next commitment allocates
    /// them again), keeping the twiddles and the staging rings: for constraint evaluation, which
    /// needs the room while the next commitment is still to come.
    pub fn release_commit_buffers(&self) {
        let mut state = self.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(reused) = state.as_mut() {
            reused.buffers = CommitBuffers::new();
        }
    }

    /// Sets whether the decommitments read the coefficients kept (see
    /// [`Self::decommit_values`]).
    pub fn set_decommit(&self, enabled: bool) {
        self.decommit.store(enabled, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn decommit_enabled(&self) -> bool {
        self.decommit.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// The decommitments' counts since the last call.
    pub fn take_decommit_stats(&self) -> DecommitStats {
        std::mem::take(&mut *self.decommit_stats.lock().unwrap_or_else(|p| p.into_inner()))
    }

    /// The FRI quotients' counts since the last call.
    pub fn take_fri_stats(&self) -> FriStats {
        std::mem::take(&mut *self.fri_stats.lock().unwrap_or_else(|p| p.into_inner()))
    }

    /// stwo's FRI quotients of `request` (its `FriQuotientsHook`), computed on the GPU from the
    /// coefficients kept ([`PartsMode::Coefficients`]) and the values of the columns not kept as
    /// coefficients; `None` (the CPU computes them) if the GPU doesn't take their shape, the
    /// video memory policy refuses its buffers, or the GPU fails.
    pub fn fri_quotients(
        &self,
        request: &stwo::prover::FriQuotientsRequest<'_, SimdBackend>,
    ) -> Option<stwo::prover::poly::circle::SecureEvaluation<SimdBackend, BitReversedOrder>> {
        use crate::fri::{FriColumn, Source};
        use crate::parts::Placement;
        let start = Instant::now();
        // Every column's coefficients, if kept.
        let placements: Vec<Vec<Option<(wgpu::Buffer, u32)>>> = {
            let mut parts = self.parts.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            request
                .columns
                .iter()
                .map(|tree| {
                    tree.iter()
                        .map(|column| {
                            let log_size = column.values.len().ilog2();
                            let domain_log_size = column.domain.log_size();
                            if log_size >= domain_log_size {
                                return None;
                            }
                            match parts.placement(
                                &column.values,
                                domain_log_size,
                                domain_log_size - log_size,
                            ) {
                                Some(Placement::Coefficients { buffer, byte_offset }) => {
                                    Some((buffer, (byte_offset / 4) as u32))
                                }
                                _ => None,
                            }
                        })
                        .collect()
                })
                .collect()
        };
        let columns: stwo::core::pcs::TreeVec<Vec<FriColumn<'_>>> = stwo::core::pcs::TreeVec::new(
            request
                .columns
                .iter()
                .zip(&placements)
                .map(|(tree, placements)| {
                    tree.iter()
                        .zip(placements)
                        .map(|(column, placement)| FriColumn {
                            source: match placement {
                                Some((buffer, offset)) => {
                                    Source::Coefficients { buffer, offset: *offset }
                                }
                                None => Source::Values(column.values.as_slice()),
                            },
                            log_size: column.domain.log_size(),
                        })
                        .collect()
                })
                .collect(),
        );
        let n_columns: u64 = columns.iter().map(|tree| tree.len() as u64).sum();
        self.fri_stats.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).calls += 1;
        let record = |f: &dyn Fn(&mut FriStats)| {
            let mut stats = self.fri_stats.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            f(&mut stats);
            stats.seconds += start.elapsed().as_secs_f64();
        };
        // The buffers it would take, within the policy.
        let bytes = match crate::fri::plan_bytes(
            &columns,
            request.samples,
            request.lifting_log_size,
            request.log_blowup_factor,
            crate::extension::max_buffer_words(&self.gpu),
        ) {
            Ok(bytes) => bytes,
            Err(reason) => {
                record(&|stats| {
                    stats.declined += 1;
                    stats.reason = Some(reason.clone());
                });
                return None;
            }
        };
        let requests = bytes
            .iter()
            .enumerate()
            .map(|(slot, &bytes)| (FRI_TRANSIENT_BASE + slot, bytes))
            .collect_vec();
        let growth = self.transient_growth(&requests);
        if growth > 0 && !self.vram_allows(growth) {
            record(&|stats| {
                stats.refused_vram += 1;
                stats.reason = Some(format!("video memory: {} MiB more", growth >> 20));
            });
            return None;
        }
        for &(slot, bytes) in &requests {
            self.transient(slot, bytes);
        }
        // The buffers go when the quotients are done (the decommitment and the next leg need
        // the room).
        struct Release<'a>(&'a GpuCommitter);
        impl Drop for Release<'_> {
            fn drop(&mut self) {
                self.0.release_transient_slots(
                    FRI_TRANSIENT_BASE..FRI_TRANSIENT_BASE + crate::fri::slot::N_SLOTS,
                );
            }
        }
        let _release = Release(self);
        let computed = catch_unwind(AssertUnwindSafe(|| {
            let mut state = self.fri.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            let state = state.get_or_insert_with(|| crate::fri::FriState::new(&self.gpu));
            crate::fri::fri_quotients(
                &self.gpu,
                state,
                self.extension_kernels(),
                &CommitterFri(self),
                &columns,
                request.samples,
                request.random_coeff,
                request.lifting_log_size,
                request.log_blowup_factor,
            )
        }));
        match computed {
            Ok(Ok((values, times))) => {
                record(&|stats| {
                    stats.served += 1;
                    stats.columns += n_columns;
                    stats.value_columns += times.value_columns as u64;
                    stats.times += times;
                });
                let domain = stwo::core::poly::circle::CanonicCoset::new(request.lifting_log_size)
                    .circle_domain();
                Some(stwo::prover::poly::circle::SecureEvaluation::new(
                    domain,
                    stwo::prover::secure_column::SecureColumnByCoords { columns: values },
                ))
            }
            Ok(Err(reason)) => {
                record(&|stats| {
                    stats.declined += 1;
                    stats.reason = Some(reason.clone());
                });
                None
            }
            Err(panic) => {
                let reason = panic_message(&panic);
                eprintln!("FRI quotients on the GPU failed: {reason}");
                record(&|stats| {
                    stats.failed += 1;
                    stats.reason = Some(reason.clone());
                });
                None
            }
        }
    }

    /// [`Self::read_back`] of several pieces.
    pub fn read_back_pieces(&self, pieces: Vec<Readback<'_, '_>>) -> Result<RingTimes, String> {
        let mut state = self.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let read = catch_unwind(AssertUnwindSafe(|| {
            let reused = state.get_or_insert_with(|| Reused::new(&self.gpu));
            reused.readback.take_times();
            reused.readback.read(&self.gpu, pieces);
            reused.readback.take_times()
        }));
        read.map_err(|panic| {
            *state = None;
            panic_message(&panic)
        })
    }

    /// [`Self::stripe_twiddles`] covering every log size of `log_sizes`.
    pub fn stripe_twiddles_covering(
        &self,
        log_sizes: &std::collections::BTreeSet<u32>,
        log_blowup: u32,
    ) -> Option<std::sync::Arc<crate::extension::StripeTwiddles>> {
        let root_coset = *self
            .root_cosets
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&log_blowup)?;
        let mut state = self.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let reused = state.as_mut()?;
        reused.twiddles.try_get_with(&self.gpu, log_sizes, log_blowup, root_coset, || Err(())).ok()
    }

    /// The out-of-domain samples' counts since the last call.
    pub fn take_ood_stats(&self) -> OodStats {
        std::mem::take(&mut *self.ood_stats.lock().unwrap_or_else(|p| p.into_inner()))
    }

    /// The values of the columns of every group of a proof's out-of-domain samples at the group's
    /// point (stwo's `OodSampleHook`), evaluated on the GPU from the coefficients kept
    /// ([`PartsMode::Coefficients`]): per group and column, `None` for the columns left to the
    /// CPU (not kept as coefficients, too small, or kept whole), and for all if the GPU fails.
    /// The coefficients stay (the decommitment reads them after).
    pub fn ood_values(
        &self,
        groups: &[stwo::prover::OodSampleGroup<'_, SimdBackend>],
    ) -> Vec<Vec<Option<stwo::core::fields::qm31::SecureField>>> {
        use stwo::core::circle::CirclePoint;
        use stwo::core::fields::qm31::SecureField;

        use crate::parts::Placement;
        let start = Instant::now();
        let mut answers: Vec<Vec<Option<SecureField>>> =
            groups.iter().map(|group| vec![None; group.columns.len()]).collect();
        // The columns to sample, by buffer, size and point: their group, index and word offset.
        type Key = (wgpu::Buffer, u32, CirclePoint<SecureField>);
        type Member = (usize, usize, u32);
        let mut jobs: Vec<(Key, Vec<Member>)> = vec![];
        let mut not_resident = 0;
        {
            let mut parts = self.parts.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            for (g, group) in groups.iter().enumerate() {
                for (i, column) in group.columns.iter().enumerate() {
                    let log_size = column.values.len().ilog2();
                    let domain_log_size = column.domain.log_size();
                    let placement = (log_size >= crate::ood::MIN_LOG_SIZE
                        && log_size < domain_log_size)
                        .then(|| {
                            parts.placement(
                                &column.values,
                                domain_log_size,
                                domain_log_size - log_size,
                            )
                        })
                        .flatten();
                    let Some(Placement::Coefficients { buffer, byte_offset }) = placement else {
                        not_resident += 1;
                        continue;
                    };
                    let key = (buffer, log_size, group.point);
                    let member = (g, i, (byte_offset / 4) as u32);
                    match jobs.iter_mut().find(|(k, _)| *k == key) {
                        Some((_, members)) => members.push(member),
                        None => jobs.push((key, vec![member])),
                    }
                }
            }
        }
        let n_asked: usize = jobs.iter().map(|(_, members)| members.len()).sum();
        let result = (!jobs.is_empty())
            .then(|| {
                catch_unwind(AssertUnwindSafe(|| {
                    let mut kernels =
                        self.ood.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                    let kernels =
                        kernels.get_or_insert_with(|| crate::ood::OodKernels::new(&self.gpu));
                    let offsets = jobs
                        .iter()
                        .map(|(_, members)| {
                            members.iter().map(|&(.., offset)| offset).collect_vec()
                        })
                        .collect_vec();
                    let ood_jobs = jobs
                        .iter()
                        .zip(&offsets)
                        .map(|(((buffer, log_size, point), _), offsets)| crate::ood::OodJob {
                            buffer,
                            columns: offsets,
                            log_size: *log_size,
                            point: *point,
                        })
                        .collect_vec();
                    let (values, times) =
                        crate::ood::eval_jobs_timed(&self.gpu, kernels, &ood_jobs);
                    self.ood_stats.lock().unwrap_or_else(|p| p.into_inner()).times += times;
                    values
                }))
                .ok()
            })
            .flatten();
        let served = result.is_some();
        if let Some(values) = result {
            for ((_, members), values) in jobs.iter().zip(values) {
                for (&(g, i, _), value) in members.iter().zip(values) {
                    answers[g][i] = Some(value);
                }
            }
        }
        let mut stats = self.ood_stats.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        stats.calls += 1;
        stats.groups += groups.len() as u64;
        stats.columns += groups.iter().map(|group| group.columns.len() as u64).sum::<u64>();
        stats.not_resident += not_resident;
        match served {
            true => stats.served += n_asked as u64,
            false => stats.failed += n_asked as u64,
        }
        stats.seconds += start.elapsed().as_secs_f64();
        answers
    }

    /// The values a striped tree's decommitment reads at the positions `wanted[c]` of each column
    /// `columns[c]` doesn't hold (stwo's `StripedDecommitHook`), evaluated on the GPU from the
    /// coefficients kept ([`PartsMode::Coefficients`]); `None` (the CPU evaluates them) if a column
    /// that has some isn't kept, or the GPU fails. The tree's coefficients are dropped either way:
    /// a tree is decommitted once.
    pub fn decommit_values(
        &self,
        columns: &[&CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>],
        wanted: &[Vec<usize>],
    ) -> Option<Vec<Vec<BaseField>>> {
        use crate::parts::Placement;
        let start = Instant::now();
        // The columns to evaluate, by buffer, size, domain and positions.
        type Job = (wgpu::Buffer, u32, u32, Vec<usize>);
        let mut jobs: Vec<(Job, Vec<(usize, u32)>)> = vec![];
        let mut resident = true;
        {
            let mut parts = self.parts.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            for (c, (column, wanted)) in columns.iter().zip(wanted).enumerate() {
                if wanted.is_empty() {
                    continue;
                }
                let log_size = column.values.len().ilog2();
                let domain_log_size = column.domain.log_size();
                match parts.placement(&column.values, domain_log_size, domain_log_size - log_size) {
                    Some(Placement::Coefficients { buffer, byte_offset }) => {
                        let key = (buffer, log_size, domain_log_size, wanted.clone());
                        let offset = (byte_offset / 4) as u32;
                        match jobs.iter_mut().find(|(k, _)| *k == key) {
                            Some((_, members)) => members.push((c, offset)),
                            None => jobs.push((key, vec![(c, offset)])),
                        }
                    }
                    _ => {
                        resident = false;
                        break;
                    }
                }
            }
        }
        let n_positions: u64 = jobs.iter().map(|((.., w), m)| (w.len() * m.len()) as u64).sum();
        let result = resident
            .then(|| {
                catch_unwind(AssertUnwindSafe(|| {
                    let mut fold =
                        self.fold.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                    let kernels =
                        fold.get_or_insert_with(|| crate::decommit::FoldKernels::new(&self.gpu));
                    let mut values: Vec<Vec<BaseField>> = columns.iter().map(|_| vec![]).collect();
                    let offsets = jobs
                        .iter()
                        .map(|(_, members)| members.iter().map(|&(_, offset)| offset).collect_vec())
                        .collect_vec();
                    let fold_jobs = jobs
                        .iter()
                        .zip(&offsets)
                        .map(|((key, members), offsets)| {
                            let (buffer, log_size, domain_log_size, positions) = key;
                            let domain = columns[members[0].0].domain;
                            assert_eq!(domain.log_size(), *domain_log_size);
                            crate::decommit::FoldJob {
                                buffer,
                                columns: offsets,
                                log_size: *log_size,
                                domain,
                                indices: positions,
                            }
                        })
                        .collect_vec();
                    let (evaluated, times) =
                        crate::decommit::eval_jobs_timed(&self.gpu, kernels, &fold_jobs);
                    self.decommit_stats.lock().unwrap_or_else(|p| p.into_inner()).times += times;
                    for ((_, members), job_values) in jobs.iter().zip(evaluated) {
                        for (&(c, _), column_values) in members.iter().zip(job_values) {
                            values[c] = column_values;
                        }
                    }
                    values
                }))
                .ok()
            })
            .flatten();
        // The tree's coefficients are no longer needed (after the GPU's work on them).
        drop(jobs);
        let stripe0 = columns.iter().map(|column| &column.values).collect_vec();
        self.parts.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).decommitted(&stripe0);
        let mut stats = self.decommit_stats.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        stats.trees += 1;
        match (&result, resident) {
            (Some(_), _) => {
                stats.served += 1;
                stats.columns += wanted.iter().filter(|w| !w.is_empty()).count() as u64;
                stats.positions += n_positions;
            }
            (None, false) => stats.not_resident += 1,
            (None, true) => stats.failed += 1,
        }
        stats.seconds += start.elapsed().as_secs_f64();
        result
    }

    /// The bytes [`Self::transient`] would add to give these slots these sizes (growths only).
    pub fn transient_growth(&self, requests: &[(usize, u64)]) -> u64 {
        let transient = self.transient.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        requests
            .iter()
            .map(|&(slot, bytes)| {
                let held =
                    transient.get(slot).and_then(Option::as_ref).map_or(0, wgpu::Buffer::size);
                transient_size(bytes, &self.gpu)
                    .saturating_sub(held)
                    .min(transient_size(bytes, &self.gpu))
            })
            .sum()
    }

    /// A storage buffer of at least `bytes` (rounded up to 16 MiB) in slot `slot`, kept
    /// for the next calls until the next commitment starts: constraint evaluation's scratch, which
    /// then neither allocates nor clears (wgpu zero-fills a buffer's first use) per component.
    pub fn transient(&self, slot: usize, bytes: u64) -> wgpu::Buffer {
        let mut transient = self.transient.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if transient.len() <= slot {
            transient.resize(slot + 1, None);
        }
        if transient[slot].as_ref().is_none_or(|buffer| buffer.size() < bytes) {
            // Dropped, not destroyed: work recorded on it may still be to run (wgpu frees it
            // when that is done).
            transient[slot] = None;
            transient[slot] = Some(self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("transient"),
                size: transient_size(bytes, &self.gpu),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        transient[slot].clone().unwrap()
    }

    /// The transient buffers' bytes now.
    pub fn transient_bytes(&self) -> u64 {
        let transient = self.transient.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        transient.iter().flatten().map(wgpu::Buffer::size).sum()
    }

    /// Releases the transient buffers.
    pub fn release_transient(&self) {
        let mut transient = self.transient.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        for buffer in transient.drain(..).flatten() {
            buffer.destroy();
        }
    }

    /// Frees the transient buffers of slots `slots` (after the work recorded on them: dropped,
    /// wgpu frees them when it is done).
    pub fn release_transient_slots(&self, slots: std::ops::Range<usize>) {
        let mut transient = self.transient.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        for slot in slots {
            if let Some(buffer) = transient.get_mut(slot) {
                *buffer = None;
            }
        }
    }

    /// The extension's kernels (the commitment's).
    pub fn extension_kernels(&self) -> &crate::extension::ExtensionKernels {
        &self.kernels.extension
    }

    /// Every stripe's twiddles of columns of `2^log_size` coefficients committed at blowup
    /// `log_blowup`, as the commitments evaluated them (the last commitment's twiddle tree at that
    /// blowup); `None` if no commitment kept them.
    pub fn stripe_twiddles(
        &self,
        log_size: u32,
        log_blowup: u32,
    ) -> Option<std::sync::Arc<crate::extension::StripeTwiddles>> {
        let root_coset = *self
            .root_cosets
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&log_blowup)?;
        let mut state = self.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let reused = state.as_mut()?;
        let log_sizes = [log_size].into_iter().collect();
        reused.twiddles.try_get_with(&self.gpu, &log_sizes, log_blowup, root_coset, || Err(())).ok()
    }

    /// Frees the device buffers and the twiddles held between commitments (the next commitment
    /// allocates them again), keeping the staging rings: for a process that is done proving for
    /// a while, on a GPU it shares.
    pub fn release_buffers(&self) {
        let mut state = self.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(reused) = state.as_mut() {
            reused.buffers = CommitBuffers::new();
            reused.twiddles = TwiddleCache::new();
        }
    }
}

/// Frees `polynomials`: natively, on a thread of its own, as freeing hundreds of MiB takes tens of
/// milliseconds (unmapping their pages) that the prover need not wait for; in wasm, here.
fn free(polynomials: ColumnVec<CircleCoefficients<SimdBackend>>) {
    #[cfg(not(target_family = "wasm"))]
    {
        let spawned = std::thread::Builder::new()
            .name("free polynomials".to_string())
            .spawn(move || drop(polynomials));
        // Without a thread, `spawn` drops the closure, and so the polynomials, here.
        drop(spawned);
    }
    #[cfg(target_family = "wasm")]
    drop(polynomials);
}

/// EXPERIMENT: the process's physical footprint in MiB (Apple: `proc_pid_rusage`), for the lean
/// commitment's verbose trace; 0 elsewhere.
fn footprint_mib() -> u64 {
    #[cfg(target_vendor = "apple")]
    {
        unsafe extern "C" {
            fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *mut u8) -> i32;
            fn getpid() -> i32;
        }
        let mut buffer = [0u8; 512];
        // Safety: RUSAGE_INFO_V4 (4) fills fewer than 512 bytes.
        if unsafe { proc_pid_rusage(getpid(), 4, buffer.as_mut_ptr()) } == 0 {
            return u64::from_le_bytes(buffer[72..80].try_into().unwrap()) >> 20;
        }
    }
    0
}

/// `STWO_GPU_LEAN=1` (zero-copy, [`GpuCommitter::commit_lean_in_place`]) or `move`
/// ([`GpuCommitter::commit_lean`]): lean commitments.
fn lean_from_env() -> bool {
    std::env::var("STWO_GPU_LEAN").is_ok_and(|v| v == "1" || v == "move")
}

/// Whether lean commitments move the coefficients (`STWO_GPU_LEAN=move`) instead of reading the
/// host's in place.
fn lean_moves() -> bool {
    std::env::var("STWO_GPU_LEAN").is_ok_and(|v| v == "move")
}

/// The host's memory of `words` as a read-only storage buffer, without a copy: Metal's
/// `newBufferWithBytesNoCopy` over the pages that hold it; `None` if `words` doesn't start on a
/// page (the large blocks of the prover's allocator do), or elsewhere than on Metal. The buffer
/// must be destroyed, and the GPU done with it, before `words` is written or freed.
#[cfg(target_vendor = "apple")]
fn wrap_host(gpu: &GpuContext, words: &[u32]) -> Option<wgpu::Buffer> {
    use objc2_metal::{MTLDevice, MTLResourceOptions};
    use wgpu::hal::api::Metal;
    unsafe extern "C" {
        fn getpagesize() -> i32;
    }
    // Safety: libc's.
    let page = unsafe { getpagesize() } as usize;
    let address = words.as_ptr() as usize;
    let bytes = words.len() * 4;
    if page == 0 || address % page != 0 || bytes < page {
        return None;
    }
    // The pages that hold `words`: mapped, as the last one holds its end.
    let len = bytes.next_multiple_of(page);
    // Safety: the device's own handle, used for a buffer only.
    let hal_device = unsafe { gpu.device.as_hal::<Metal>() }?;
    // Safety: `len` bytes from a page boundary, all mapped (above); no deallocator: the memory
    // stays the caller's.
    let raw = unsafe {
        hal_device.raw_device().newBufferWithBytesNoCopy_length_options_deallocator(
            std::ptr::NonNull::new(address as *mut std::ffi::c_void)?,
            len,
            MTLResourceOptions::StorageModeShared,
            None,
        )
    }?;
    drop(hal_device);
    // Safety: a buffer of this device, of `len` bytes, bound read-only as storage only.
    let hal_buffer = unsafe { wgpu::hal::metal::Device::buffer_from_raw(raw, len as u64) };
    Some(unsafe {
        gpu.device.create_buffer_from_hal::<Metal>(
            hal_buffer,
            &wgpu::BufferDescriptor {
                label: Some("host coefficients"),
                size: len as u64,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            },
        )
    })
}

#[cfg(not(target_vendor = "apple"))]
fn wrap_host(_gpu: &GpuContext, _words: &[u32]) -> Option<wgpu::Buffer> {
    None
}

/// A MiB count from the environment, else `default`.
fn lean_env_mib(name: &str, default: u64) -> u64 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn panic_message(panic: &Box<dyn std::any::Any + Send>) -> String {
    panic
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| panic.downcast_ref::<&str>().map(|message| message.to_string()))
        .unwrap_or_else(|| "a panic".to_string())
}
