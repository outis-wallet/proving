//! Keeping the GPU's clocks up between a proof's commitments, as an option (off unless the page
//! sets `keep_warm_ms`): the native committer's keep-warm (`stwo_gpu_spike::warm`), from the GPU
//! worker's event loop.
//!
//! A discrete GPU lowers its clocks within a second or two without work, and the prover's CPU
//! phases between its commitments are that long: each commitment then starts on a slow GPU and
//! spends most of its time computing at the clocks the GPU climbs back to (in Chrome on the RTX
//! 3080, the Cairo base trace's compute took 270-314 ms of GPU time, against 69 ms at full
//! clocks). The driver measures use as the time some kernel runs, so a dispatch kept in flight
//! keeps the clocks up: the native kernel ([`SpinKernel`], one workgroup of 64 threads in a loop),
//! submitted again a millisecond after the previous one is done (`onSubmittedWorkDone`, then a
//! timer).
//!
//! Its loop is longer than native's: Chrome's compiler (Tint) makes the same kernel run about 20
//! times faster on the RTX 3080 (0.16 ms at full clocks for native's 2^16 steps, against about
//! 3 ms through wgpu's), and a submission's round trip in Chrome (`onSubmittedWorkDone`, the timer)
//! takes about 4 ms. With 2^16 steps the GPU stayed in P8 (busy 4 % of the time); with 2^20 (2.2 ms
//! at full clocks, 4.7 ms a cycle) it stayed in P0 at 1920 MHz, about 100 W (the spin probe, 8 s).
//! A commitment waits for the dispatch in flight at most: 2.2 ms.
//!
//! It warms for the window from when a proof attaches (and when the page says a proof starts),
//! and from the end of each commitment; it pauses during a commitment, and stops when the proof
//! detaches, the window ends, the option is switched off, or the device is lost. The GPU is the
//! page's too (and its power the laptop's), hence an option, off by default.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use stwo_gpu_spike::warm::SpinKernel;
use wasm_bindgen::prelude::*;

use crate::{Inner, LOST, perf_now, queue_done};

#[wasm_bindgen(
    inline_js = "export function sleep_ms(ms) { return new Promise((r) => setTimeout(r, ms)); }"
)]
extern "C" {
    fn sleep_ms(ms: f64) -> js_sys::Promise;
}

/// The pause between a dispatch's completion and the next one's submission, in ms.
const GAP_MS: f64 = 1.0;
/// The steps of a dispatch's loop: about 2.2 ms in Chrome on the RTX 3080 at full clocks (see the
/// module docs).
pub const ITERATIONS: u32 = 1 << 20;

/// The state of keep-warm when a commitment began (`commit::ST_WARM`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WarmState {
    /// The option is off.
    Off = 0,
    /// Warming: the window from the last commitment (or the attach) had not ended.
    Warming = 1,
    /// On, but the window had ended (or no proof was attached): the commitment started cold.
    Expired = 2,
}

/// Keep-warm's state on the GPU worker's thread.
pub struct Warm {
    kernel: SpinKernel,
    /// The window in ms; 0: off.
    window_ms: Cell<f64>,
    /// Warm until then (`perf_now` ms); 0: not warming.
    until: Cell<f64>,
    /// A commitment is running.
    busy: Cell<bool>,
    /// A proof is attached.
    attached: Cell<bool>,
    /// The loop is running (its dispatch in flight, or its pause).
    running: Cell<bool>,
    /// Dispatches so far, and since the last commitment ended (or the attach).
    dispatches: Cell<u64>,
    since: Cell<u64>,
    /// When the last commitment began: the state then, and the dispatches before it.
    last: RefCell<(WarmState, u64)>,
}

impl Warm {
    pub fn new(gpu: &stwo_gpu_spike::GpuContext) -> Self {
        Self {
            kernel: SpinKernel::new(gpu, ITERATIONS),
            window_ms: Cell::new(0.0),
            until: Cell::new(0.0),
            busy: Cell::new(false),
            attached: Cell::new(false),
            running: Cell::new(false),
            dispatches: Cell::new(0),
            since: Cell::new(0),
            last: RefCell::new((WarmState::Off, 0)),
        }
    }

    /// The window in ms (0: off).
    pub fn window_ms(&self) -> f64 {
        self.window_ms.get()
    }

    /// The dispatches so far.
    pub fn dispatches(&self) -> u64 {
        self.dispatches.get()
    }

    /// The state when the last commitment began, and the dispatches between it and the one
    /// before (or the attach).
    pub fn last(&self) -> (WarmState, u64) {
        *self.last.borrow()
    }

    /// Sets the window (ms; 0: off). Off stops the warming now (after the dispatch in flight, if
    /// any); on doesn't start it (an attach or a commitment's end does).
    pub fn set_window(&self, window_ms: f64) {
        let window_ms = if window_ms.is_finite() && window_ms > 0.0 { window_ms } else { 0.0 };
        self.window_ms.set(window_ms);
        if window_ms == 0.0 {
            self.until.set(0.0);
        }
    }
}

/// Warm for the window from now (if on, attached, and no commitment is running).
fn warm_from_now(inner: &Rc<Inner>) {
    let warm = &inner.warm;
    if warm.window_ms.get() > 0.0 && warm.attached.get() && !warm.busy.get() {
        warm.until.set(perf_now() + warm.window_ms.get());
        if !warm.running.replace(true) {
            wasm_bindgen_futures::spawn_local(run(inner.clone()));
        }
    }
}

/// A proof attached: warm for the window from now.
pub fn attach(inner: &Rc<Inner>) {
    inner.warm.attached.set(true);
    inner.warm.since.set(0);
    warm_from_now(inner);
}

/// The page says a proof starts (its first commitment is coming): warm for the window from now.
pub fn proof_starts(inner: &Rc<Inner>) {
    warm_from_now(inner);
}

/// The proof detached: stop now (after the dispatch in flight, if any).
pub fn detach(inner: &Rc<Inner>) {
    inner.warm.attached.set(false);
    inner.warm.until.set(0.0);
}

/// A commitment begins: no dispatch until it ends (the one in flight, if any, runs first).
pub fn begin(inner: &Rc<Inner>) {
    let warm = &inner.warm;
    let state = if warm.window_ms.get() == 0.0 {
        WarmState::Off
    } else if warm.running.get() && warm.until.get() > perf_now() {
        WarmState::Warming
    } else {
        WarmState::Expired
    };
    *warm.last.borrow_mut() = (state, warm.since.replace(0));
    warm.busy.set(true);
}

/// A commitment ended (whatever its outcome): warm for the window from now.
pub fn end(inner: &Rc<Inner>) {
    inner.warm.busy.set(false);
    warm_from_now(inner);
}

/// The loop: a dispatch, waited for, a pause, while warming.
async fn run(inner: Rc<Inner>) {
    let warm = &inner.warm;
    loop {
        if LOST.load(std::sync::atomic::Ordering::SeqCst)
            || warm.busy.get()
            || warm.until.get() <= perf_now()
        {
            warm.running.set(false);
            return;
        }
        warm.dispatches.set(warm.dispatches.get() + 1);
        warm.since.set(warm.since.get() + 1);
        let mut encoder = inner.gpu.device.create_command_encoder(&Default::default());
        warm.kernel.record(&mut encoder);
        inner.gpu.queue.submit([encoder.finish()]);
        // One dispatch in flight at most.
        queue_done(&inner.gpu.queue).await;
        let _ = wasm_bindgen_futures::JsFuture::from(sleep_ms(GAP_MS)).await;
    }
}
