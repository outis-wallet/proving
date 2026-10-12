//! A synchronous bridge from the prover's threads to a GPU worker, through the prover's shared
//! wasm memory.
//!
//! The prover proves inside one long synchronous wasm call, on a worker and rayon's Web Workers,
//! while WebGPU is asynchronous (its promises resolve on a JS event loop). So a dedicated GPU
//! worker owns the GPU device and serves a job table that lives in the prover's memory:
//!
//! - a prover thread claims a free slot, writes the job (an op and up to [`MAX_ARGS`] words,
//!   typically pointers and lengths into the same memory), marks it `SUBMITTED`, rings the
//!   doorbell (`memory.atomic.notify`) and blocks on its slot (`memory.atomic.wait32`);
//! - the GPU worker, parked in `Atomics.waitAsync` on the doorbell, takes the slot, reads its
//!   inputs straight from the shared memory, runs the kernels, writes the outputs straight back,
//!   marks the slot `DONE` (or `ERROR`, with a message) and `Atomics.notify`s it.
//!
//! This crate is the protocol (the layout below, shared by both sides; the GPU worker's JS reads
//! it from the GPU module's `layout()`) and the client. It has no dependencies, so the prover's
//! module (wasm-bindgen 0.2.126) and the GPU module (wgpu 30, wasm-bindgen 0.2.128) can both
//! build it. All words are little-endian u32; times are f64 in two consecutive words.
#![cfg_attr(all(target_arch = "wasm32", target_feature = "atomics"), feature(stdarch_wasm_atomic_wait))]

/// "GPUB".
pub const MAGIC: u32 = 0x4255_5047;
/// Bumped on any change to the layout or the ops.
pub const VERSION: u32 = 2;

/// Concurrent jobs (one per waiting thread; more than rayon's threads on any device).
pub const NSLOTS: usize = 64;
/// Words per slot (256 bytes).
pub const SLOT_WORDS: usize = 64;
/// Words before the first slot.
pub const HEADER_WORDS: usize = 16;
/// The whole region, in words.
pub const REGION_WORDS: usize = HEADER_WORDS + NSLOTS * SLOT_WORDS;

// Header words.
pub const H_MAGIC: usize = 0;
pub const H_VERSION: usize = 1;
pub const H_NSLOTS: usize = 2;
pub const H_SLOT_WORDS: usize = 3;
/// Incremented by each submission; the GPU worker waits on it.
pub const H_DOORBELL: usize = 4;
/// [`SERVER_NONE`], [`SERVER_READY`] or [`SERVER_LOST`], set by the GPU worker.
pub const H_SERVER: usize = 5;
/// Jobs the GPU worker has finished (for diagnostics).
pub const H_COMPLETED: usize = 6;

pub const SERVER_NONE: u32 = 0;
pub const SERVER_READY: u32 = 1;
/// The device was lost or the GPU worker failed: every job fails from now on.
pub const SERVER_LOST: u32 = 2;

// Slot words, relative to the slot.
pub const S_STATE: usize = 0;
pub const S_OP: usize = 1;
pub const S_NARGS: usize = 2;
/// Bytes of the UTF-8 error message at [`S_ERROR`].
pub const S_ERROR_LEN: usize = 3;
pub const S_ARGS: usize = 4;
pub const MAX_ARGS: usize = 16;
/// [`NTIMES`] f64 (word index; 8-byte aligned since slots are).
pub const S_TIMES: usize = 20;
pub const NTIMES: usize = 6;
pub const S_ERROR: usize = 40;
pub const ERROR_BYTES: usize = (SLOT_WORDS - S_ERROR) * 4;

// Times in a slot, in ms. The absolute ones are `performance.timeOrigin + performance.now()`,
// comparable across workers.
/// Absolute: the client submitted.
pub const T_SUBMIT: usize = 0;
/// Absolute: the GPU worker took the job.
pub const T_WAKE: usize = 1;
/// Absolute: the GPU worker finished the job (just before notifying).
pub const T_DONE: usize = 2;
/// Durations of the job's phases, as the op reports them (0 when it has no such phase).
pub const T_UPLOAD: usize = 3;
pub const T_COMPUTE: usize = 4;
pub const T_READBACK: usize = 5;

// Slot states.
pub const FREE: u32 = 0;
/// A client is writing the job.
pub const CLAIMED: u32 = 1;
pub const SUBMITTED: u32 = 2;
pub const RUNNING: u32 = 3;
pub const DONE: u32 = 4;
pub const ERROR: u32 = 5;

/// The ops. Pointers and lengths are byte offsets and byte counts in the prover's memory.
pub mod op {
    /// No GPU work: the bridge's own round trip.
    pub const PING: u32 = 0;
    /// An empty submit, then `onSubmittedWorkDone`: the GPU queue's round trip.
    pub const SUBMIT: u32 = 1;
    /// `[src, len]`: `writeBuffer` from the memory into a GPU buffer, then `onSubmittedWorkDone`.
    pub const UPLOAD: u32 = 2;
    /// `[dst, len]`: a GPU buffer to a mappable one, `mapAsync`, and the copy into the memory.
    pub const READBACK: u32 = 3;
    /// `[a, b, out, n, flags]`: `out[i] = a[i] * b[i]` in M31, `n` words. With `flags & 1`, the
    /// phases are pipelined (one submit, no waits between them) and only the total is timed.
    pub const M31_MUL: u32 = 16;
    /// `[descriptor, descriptor bytes]`: stwo's `CommitmentTreeProver::new_striped` (SimdBackend,
    /// Blake2sM31MerkleChannel) for the polynomials and outputs the descriptor lists, laid out as
    /// [`commit`](super::commit) says. The GPU worker fills every output and the stats.
    pub const COMMIT_STRIPED: u32 = 32;
    /// Panics in the GPU module: exercises the failure path (the bridge is lost after it).
    pub const TEST_TRAP: u32 = 255;
}

/// The descriptor of [`op::COMMIT_STRIPED`]: u32 words in the prover's memory (4-byte aligned),
/// a header, then a record per column (polynomial order), then a record per layer of the tree in
/// stwo's order (the root first, `lifting_log_size + 1` layers). The prover allocates every
/// output (stwo's own columns and layers, so it keeps them as they are) and the GPU worker fills
/// them; it checks every length against its own layout first, and fails the job on any
/// difference (the prover then commits on the CPU).
pub mod commit {
    /// The descriptor's own version ([`D_VERSION`]). 2: the stats of a commitment whose
    /// transfers overlap its compute (the streamed one).
    pub const DESCRIPTOR_VERSION: u32 = 2;

    // Header words.
    pub const D_VERSION: usize = 0;
    pub const D_N_COLUMNS: usize = 1;
    pub const D_LOG_BLOWUP: usize = 2;
    pub const D_LIFTING_LOG_SIZE: usize = 3;
    /// The prover's twiddle tree's `twiddles` (u32 words, stwo's doubled twiddles), or 0: the GPU
    /// worker reads it when it has none of its own for this tree yet, or computes the twiddles
    /// itself.
    pub const D_TWIDDLES_PTR: usize = 4;
    /// Its length in words.
    pub const D_TWIDDLES_LEN: usize = 5;
    /// Reserved (0).
    pub const D_FLAGS: usize = 6;
    /// [`NSTATS`] f64, each as two words (low, high), written by the GPU worker.
    pub const D_STATS: usize = 8;
    pub const NSTATS: usize = 28;
    pub const HEADER_WORDS: usize = D_STATS + 2 * NSTATS;

    /// Per column: its coefficients (pointer; `2^log_size` words), its log size, and where its
    /// kept evaluations go (pointer, words): stripe 0's `2^log_size` words, or, for a column kept
    /// whole (`log_size + log_blowup < 14`), its whole evaluation, stripe 0 first.
    pub const COLUMN_WORDS: usize = 4;
    pub const C_COEFFS: usize = 0;
    pub const C_LOG_SIZE: usize = 1;
    pub const C_OUT: usize = 2;
    pub const C_OUT_WORDS: usize = 3;

    /// Per layer, the root first: where its hashes go (pointer; 32 bytes each) and how many there
    /// are, 0 for a layer stwo leaves empty.
    pub const LAYER_WORDS: usize = 2;
    pub const L_PTR: usize = 0;
    pub const L_HASHES: usize = 1;

    /// The descriptor's words for `n_columns` columns and a tree of height `lifting_log_size`.
    pub const fn descriptor_words(n_columns: usize, lifting_log_size: usize) -> usize {
        HEADER_WORDS + n_columns * COLUMN_WORDS + (lifting_log_size + 1) * LAYER_WORDS
    }

    // The stats, in ms unless named otherwise. The upload, the compute and the readback overlap
    // (the coefficients of a group go up while the GPU works on the one before, its stripe 0
    // comes back while the GPU works on later ones), so their times are each side's own work, and
    // `ST_COMMIT` is the whole of it.
    /// Getting the stripes' twiddles: kept ones, or (a miss) read the prover's tree (or compute
    /// one) and compute every stripe's.
    pub const ST_TWIDDLES: usize = 0;
    /// The commitment's plans (uniforms, bind groups), and its buffers if they had to grow.
    pub const ST_SETUP: usize = 1;
    /// The worker's own upload work: copying the coefficients into the ring's mapped chunks (or
    /// its `writeBuffer` calls). In the serial mode ([`ST_SERIAL`]), the upload until the GPU has
    /// it.
    pub const ST_UPLOAD: usize = 2;
    /// Recording and submitting the GPU's work. In the serial mode, the GPU's compute alone.
    pub const ST_COMPUTE: usize = 3;
    /// The worker's copies out of the readback ring's chunks into the prover's memory. In the
    /// serial mode, the readback with its waits.
    pub const ST_READBACK: usize = 4;
    /// Freeing the commitment's plans (its buffers are kept).
    pub const ST_DESTROY: usize = 5;
    pub const ST_UPLOAD_BYTES: usize = 6;
    pub const ST_READBACK_BYTES: usize = 7;
    /// The device buffers the commitment ran on (bytes; kept buffers count whole).
    pub const ST_GPU_BYTES: usize = 8;
    /// The whole job, as the GPU worker timed it.
    pub const ST_TOTAL: usize = 9;
    /// 1 if the twiddles were read from the prover's tree for this commitment.
    pub const ST_TWIDDLES_FROM_PROVER: usize = 10;
    /// Waiting for an upload chunk to be writable again (the GPU still copying it).
    pub const ST_UPLOAD_WAIT: usize = 11;
    /// Waiting for a readback chunk to be mapped (the GPU still computing, or copying it).
    pub const ST_READBACK_WAIT: usize = 12;
    /// From the first upload to the last copy out: the upload, compute and readback, overlapped.
    pub const ST_COMMIT: usize = 13;
    /// 1 if streamed in groups of columns, 0 if stripe by stripe.
    pub const ST_STREAMED: usize = 14;
    /// The groups (streamed).
    pub const ST_GROUPS: usize = 15;
    /// 1 if the stripes' twiddles were computed for this commitment (none kept covered it).
    pub const ST_TWIDDLES_COMPUTED: usize = 16;
    /// What the worker keeps between commitments (bytes): the buffers, the twiddles, the rings.
    pub const ST_HELD_BYTES: usize = 17;
    /// 1 in the serial (diagnostic) mode: each phase waited for, for its time alone.
    pub const ST_SERIAL: usize = 18;
    /// From the submission of the tree's levels to the end of the readback.
    pub const ST_TAIL: usize = 19;
    /// 1 if the commitment's plans were kept from an earlier commitment of the same shape.
    pub const ST_PLANS_KEPT: usize = 20;
    /// The GPU's own time (timestamp queries; 0 without them, see `timestamps` in the GPU
    /// worker's options): its copies of the upload chunks into place, its compute (the
    /// submissions of the commitment's work), its copies into the readback chunks.
    pub const ST_GPU_UPLOAD: usize = 21;
    pub const ST_GPU_COMPUTE: usize = 22;
    pub const ST_GPU_READBACK: usize = 23;
    /// The time the GPU was busy with any of them, and from the first to the last.
    pub const ST_GPU_BUSY: usize = 24;
    pub const ST_GPU_SPAN: usize = 25;
    /// Keep-warm when the commitment began (the GPU worker's `keep_warm_ms` option): 0 off, 1
    /// warming, 2 on but its window had ended (the commitment started on an idle GPU).
    pub const ST_WARM: usize = 26;
    /// Keep-warm's dispatches since the previous commitment ended (or the proof attached).
    pub const ST_WARM_DISPATCHES: usize = 27;
}

/// Why a call failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BridgeError {
    /// No GPU worker is attached (or it never got a device): use the CPU.
    Unavailable,
    /// The GPU worker failed or lost its device, before or during the job.
    Lost,
    /// More than [`MAX_ARGS`] arguments.
    TooManyArgs,
    /// The job failed, with the GPU worker's message.
    Failed(String),
    /// The job didn't finish within the caller's deadline ([`call_with_deadline`]). The bridge
    /// is marked lost, so later calls fail at once.
    TimedOut,
}

impl BridgeError {
    /// Whether the GPU worker may still be running the job: after [`BridgeError::Lost`] or
    /// [`BridgeError::TimedOut`] it may yet write the job's outputs, so the caller must keep
    /// them allocated (leak them) rather than free memory the worker could write into.
    pub fn job_may_be_running(&self) -> bool {
        matches!(self, BridgeError::Lost | BridgeError::TimedOut)
    }
}

impl core::fmt::Display for BridgeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            BridgeError::Unavailable => write!(f, "no GPU worker attached"),
            BridgeError::Lost => write!(f, "the GPU worker failed or lost its device"),
            BridgeError::TooManyArgs => write!(f, "more than {MAX_ARGS} arguments"),
            BridgeError::Failed(message) => write!(f, "GPU job failed: {message}"),
            BridgeError::TimedOut => write!(f, "the GPU job didn't finish in time"),
        }
    }
}

impl std::error::Error for BridgeError {}

/// A finished job's times (see the `T_*` constants), and when the client saw it finish.
#[derive(Debug, Clone, Copy, Default)]
pub struct Completion {
    pub times: [f64; NTIMES],
    /// Absolute: the client woke with the result.
    pub returned: f64,
}

impl Completion {
    /// The whole round trip, as the client sees it.
    pub fn total_ms(&self) -> f64 {
        self.returned - self.times[T_SUBMIT]
    }
    /// From the doorbell to the GPU worker taking the job (includes waiting behind other jobs).
    pub fn wake_ms(&self) -> f64 {
        self.times[T_WAKE] - self.times[T_SUBMIT]
    }
    /// The GPU worker's time on the job.
    pub fn serve_ms(&self) -> f64 {
        self.times[T_DONE] - self.times[T_WAKE]
    }
    /// From the GPU worker's notify to the client running again.
    pub fn return_ms(&self) -> f64 {
        self.returned - self.times[T_DONE]
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
pub use client::*;

#[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
mod client {
    use super::*;
    use core::arch::wasm32::{memory_atomic_notify, memory_atomic_wait32};
    use core::sync::atomic::{AtomicU32, Ordering::SeqCst};

    #[repr(C, align(64))]
    struct Region([AtomicU32; REGION_WORDS]);

    static REGION: Region = Region([const { AtomicU32::new(0) }; REGION_WORDS]);
    /// Where the next claim starts looking, so concurrent clients spread over the slots.
    static NEXT: AtomicU32 = AtomicU32::new(0);

    fn word(index: usize) -> &'static AtomicU32 {
        &REGION.0[index]
    }

    fn slot(slot: usize, index: usize) -> &'static AtomicU32 {
        word(HEADER_WORDS + slot * SLOT_WORDS + index)
    }

    fn wait(cell: &AtomicU32, value: u32, timeout_ms: i64) {
        // SAFETY: the pointer is to a live, aligned static; wait only reads it. A worker may
        // wait (the main thread may not, but the prover never runs there).
        unsafe { memory_atomic_wait32(cell.as_ptr() as *mut i32, value as i32, timeout_ms * 1_000_000) };
    }

    fn notify(cell: &AtomicU32) {
        // SAFETY: as in `wait`.
        unsafe { memory_atomic_notify(cell.as_ptr() as *mut i32, u32::MAX) };
    }

    /// The region's address in the memory, with its header written: what the GPU worker needs.
    pub fn region_address() -> u32 {
        word(H_MAGIC).store(MAGIC, SeqCst);
        word(H_VERSION).store(VERSION, SeqCst);
        word(H_NSLOTS).store(NSLOTS as u32, SeqCst);
        word(H_SLOT_WORDS).store(SLOT_WORDS as u32, SeqCst);
        REGION.0.as_ptr() as u32
    }

    /// Whether a GPU worker is attached and serving.
    pub fn ready() -> bool {
        word(H_MAGIC).load(SeqCst) == MAGIC && word(H_SERVER).load(SeqCst) == SERVER_READY
    }

    /// Blocks until a GPU worker is attached, for at most `timeout_ms`.
    pub fn wait_ready(timeout_ms: i64) -> bool {
        let step = 10;
        let mut waited = 0;
        loop {
            let server = word(H_SERVER).load(SeqCst);
            if server == SERVER_READY {
                return true;
            }
            if server == SERVER_LOST || waited >= timeout_ms {
                return false;
            }
            wait(word(H_SERVER), server, step);
            waited += step;
        }
    }

    fn claim() -> usize {
        loop {
            let start = NEXT.fetch_add(1, SeqCst) as usize;
            for k in 0..NSLOTS {
                let index = (start + k) % NSLOTS;
                if slot(index, S_STATE).compare_exchange(FREE, CLAIMED, SeqCst, SeqCst).is_ok() {
                    return index;
                }
            }
            // All taken (more waiting threads than slots): try again shortly.
            wait(slot(start % NSLOTS, S_STATE), slot(start % NSLOTS, S_STATE).load(SeqCst), 1);
        }
    }

    fn read_f64(index: usize, word_index: usize) -> f64 {
        let lo = slot(index, word_index).load(SeqCst) as u64;
        let hi = slot(index, word_index + 1).load(SeqCst) as u64;
        f64::from_bits(lo | (hi << 32))
    }

    fn write_f64(index: usize, word_index: usize, value: f64) {
        let bits = value.to_bits();
        slot(index, word_index).store(bits as u32, SeqCst);
        slot(index, word_index + 1).store((bits >> 32) as u32, SeqCst);
    }

    /// Runs `op` with `args` on the GPU worker and blocks until it is done. `now` is the
    /// absolute clock (`performance.timeOrigin + performance.now()`), for the times.
    ///
    /// Memory the args point to must stay put and untouched until this returns: the GPU worker
    /// reads inputs and writes outputs there while the caller is blocked. After an error for
    /// which [`BridgeError::job_may_be_running`], the outputs must stay allocated for good.
    pub fn call(op: u32, args: &[u32], now: fn() -> f64) -> Result<Completion, BridgeError> {
        call_with_deadline(op, args, now, f64::INFINITY)
    }

    /// [`call`], giving up after `deadline_ms` (by `now`) with [`BridgeError::TimedOut`]: a GPU
    /// worker the page ended can't mark the bridge lost, and nothing else would wake the caller.
    /// A timeout marks the bridge lost, so every later call fails at once; the slot stays
    /// claimed, as the worker may still finish the job.
    pub fn call_with_deadline(
        op: u32,
        args: &[u32],
        now: fn() -> f64,
        deadline_ms: f64,
    ) -> Result<Completion, BridgeError> {
        if args.len() > MAX_ARGS {
            return Err(BridgeError::TooManyArgs);
        }
        match word(H_SERVER).load(SeqCst) {
            SERVER_READY => {}
            SERVER_LOST => return Err(BridgeError::Lost),
            _ => return Err(BridgeError::Unavailable),
        }
        let index = claim();
        slot(index, S_OP).store(op, SeqCst);
        slot(index, S_NARGS).store(args.len() as u32, SeqCst);
        for (k, &arg) in args.iter().enumerate() {
            slot(index, S_ARGS + k).store(arg, SeqCst);
        }
        for t in 0..NTIMES {
            write_f64(index, S_TIMES + 2 * t, 0.0);
        }
        slot(index, S_ERROR_LEN).store(0, SeqCst);
        let submitted = now();
        write_f64(index, S_TIMES + 2 * T_SUBMIT, submitted);
        slot(index, S_STATE).store(SUBMITTED, SeqCst);
        word(H_DOORBELL).fetch_add(1, SeqCst);
        notify(word(H_DOORBELL));

        // Wait in steps, so a GPU worker that dies (and can no longer notify) can't hang us.
        let state = loop {
            let state = slot(index, S_STATE).load(SeqCst);
            if state == DONE || state == ERROR {
                break state;
            }
            if word(H_SERVER).load(SeqCst) == SERVER_LOST {
                // The GPU worker is gone; the slot stays claimed (its memory may still be read).
                return Err(BridgeError::Lost);
            }
            if now() - submitted > deadline_ms {
                // Every later call fails at once; a live worker sees it and stops serving.
                let _ = word(H_SERVER).compare_exchange(SERVER_READY, SERVER_LOST, SeqCst, SeqCst);
                notify(word(H_SERVER));
                notify(word(H_DOORBELL));
                return Err(BridgeError::TimedOut);
            }
            wait(slot(index, S_STATE), state, 100);
        };
        // An ERROR the worker set while marking itself lost (markLost fails the jobs it was
        // running) doesn't mean the job stopped writing: the caller must treat it as lost.
        if state == ERROR && word(H_SERVER).load(SeqCst) == SERVER_LOST {
            return Err(BridgeError::Lost);
        }
        let returned = now();
        let mut completion = Completion { times: [0.0; NTIMES], returned };
        for t in 0..NTIMES {
            completion.times[t] = read_f64(index, S_TIMES + 2 * t);
        }
        let result = if state == DONE {
            Ok(completion)
        } else {
            let len = (slot(index, S_ERROR_LEN).load(SeqCst) as usize).min(ERROR_BYTES);
            let bytes: Vec<u8> = (0..len)
                .map(|i| (slot(index, S_ERROR + i / 4).load(SeqCst) >> (8 * (i % 4))) as u8)
                .collect();
            Err(BridgeError::Failed(String::from_utf8_lossy(&bytes).into_owned()))
        };
        slot(index, S_STATE).store(FREE, SeqCst);
        notify(slot(index, S_STATE));
        result
    }
}
