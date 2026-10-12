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
//! and bind groups) are built per commitment. Natively, a thread keeps the GPU's clocks up for a
//! few seconds after each commitment, as the next one of a proof usually follows within them,
//! and after the committer's creation, for the first ([`crate::warm`]:
//! [`GpuCommitter::set_keep_warm`]).

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use itertools::Itertools;
use stwo::core::ColumnVec;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sM31MerkleChannel;
use stwo::prover::CommitmentTreeProver;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::poly::circle::CircleCoefficients;
use stwo::prover::poly::twiddles::TwiddleTree;

use crate::GpuContext;
use crate::commit::{
    CommitBuffers, CommitKernels, CommitOptions, StripedCommitment, TwiddleCache,
    takes_striped_path,
};
use crate::stream::{StreamedCommitment, group_words};
use crate::transfer::{ReadbackRing, RingTimes, UploadRing};
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
}

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
        let gpu = GpuContext::try_new()?;
        let (kernels, reused) =
            catch_unwind(AssertUnwindSafe(|| (CommitKernels::new(&gpu), Reused::new(&gpu))))
                .map_err(|panic| format!("kernels: {}", panic_message(&panic)))?;
        let warm = catch_unwind(AssertUnwindSafe(|| KeepWarm::new(&gpu, window_from_env())))
            .ok()
            .and_then(Result::ok);
        // A committer is made for proving: the first commitment usually follows within the
        // window (a prover's process makes its committer, then proves).
        if let Some(warm) = &warm {
            warm.warm_up();
        }
        Ok(Self { warm, gpu, kernels, state: Mutex::new(Some(reused)), times: Mutex::new(None) })
    }

    /// How long the GPU's clocks are kept up after each commitment (see [`crate::warm`]):
    /// `STWO_GPU_KEEP_WARM_MS` (0: never), or [`crate::warm::DEFAULT_WINDOW`]. Natively only;
    /// `None` in wasm too if set so.
    pub fn keep_warm(&self) -> Option<Duration> {
        self.warm.as_ref().and_then(KeepWarm::window)
    }

    /// Sets how long the GPU's clocks are kept up after each commitment (`None`: never; it stops
    /// at once).
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
        let mut state = self.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
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
                    times.setup = start.elapsed();
                    let (output, phases) = commitment.commit_timed(
                        &self.gpu,
                        &self.kernels,
                        &mut reused.upload,
                        &mut reused.readback,
                        &polynomials,
                    );
                    times.commit = start.elapsed() - times.setup;
                    times.phases = phases;
                    commitment.tree_prover(output)
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
                Ok(tree)
            }
            Err(panic) => {
                // Whatever the failure left behind goes; the next commitment starts afresh.
                *state = None;
                Err((polynomials, Declined::Gpu(panic_message(&panic))))
            }
        }
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

fn panic_message(panic: &Box<dyn std::any::Any + Send>) -> String {
    panic
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| panic.downcast_ref::<&str>().map(|message| message.to_string()))
        .unwrap_or_else(|| "a panic".to_string())
}
