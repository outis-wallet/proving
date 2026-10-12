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
pub const VERSION: u32 = 1;

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
    // Reserved: 32.. for the polynomial extension (FFT), 48.. for the Merkle hashing.
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
}

impl core::fmt::Display for BridgeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            BridgeError::Unavailable => write!(f, "no GPU worker attached"),
            BridgeError::Lost => write!(f, "the GPU worker failed or lost its device"),
            BridgeError::TooManyArgs => write!(f, "more than {MAX_ARGS} arguments"),
            BridgeError::Failed(message) => write!(f, "GPU job failed: {message}"),
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
    /// reads inputs and writes outputs there while the caller is blocked.
    pub fn call(op: u32, args: &[u32], now: fn() -> f64) -> Result<Completion, BridgeError> {
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
        write_f64(index, S_TIMES + 2 * T_SUBMIT, now());
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
            wait(slot(index, S_STATE), state, 100);
        };
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
