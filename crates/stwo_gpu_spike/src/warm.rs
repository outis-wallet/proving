//! Keeping the GPU's clocks up between a prover's commitments: an option, off by default
//! (`STWO_GPU_KEEP_WARM_MS`, or `GpuCommitter::set_keep_warm`).
//!
//! A discrete NVIDIA GPU lowers its clocks within a second or two of low use (on the RTX 3080:
//! P0 at 1920 MHz and 9501 MHz memory, down through P3, 240-855 MHz and 5001 MHz memory, to P8,
//! 210 and 405 MHz), and takes a few hundred milliseconds of load to climb back. A prover's
//! commitments come 0.4-1.5 s of CPU work apart, so each would start on a slow GPU: the Cairo
//! base trace's streamed commitment takes 124 ms on a GPU at full clocks, 240-330 ms after 2-3 s
//! without work (`bench_idle`), and a proof's six trees 391 ms in a row against 504-521 ms with
//! `new_striped` run between them (`bench_proof_commits`).
//!
//! The driver's measure of use is the time some kernel runs, not how much of the GPU it uses. So
//! [`KeepWarm`] keeps one tiny dispatch in flight (one workgroup of 64 threads in a loop, about
//! 3 ms at full clocks, a millisecond apart: three quarters of the time) for a window after each
//! commitment: the clocks stay up (P0, 1920 MHz, 9501 MHz memory) and the Cairo base trace's
//! commitment takes 124 ms after 3 s of idle (`bench_idle`; a dispatch every 5 ms: 130 ms, the
//! clocks at 1605-1680 MHz; every 20 ms: 190-210 ms, the memory down to 5001 MHz). The cost is
//! the power of a GPU at full clocks, mostly idle: about 104 W on the RTX 3080 against 2.5 W at
//! P8 (`nvidia-smi`), for the window after each commitment. It pauses during a commitment. When
//! on, the committer also warms for a window from its creation (a prover's process makes its
//! committer, then proves: in a proof of the Outis prover, the first commitment started 0.7 s
//! after it).
//!
//! Off unless asked for: it trades the GPU's (and on a laptop, the battery's) power for the
//! commitments' time, and the GPU may be shared (with the display, other work), which the
//! committer can't see. `STWO_GPU_KEEP_WARM_MS=N` turns it on with a window of N ms (0: off), and
//! `GpuCommitter::set_keep_warm` sets it at run time; [`DEFAULT_WINDOW`] is the window to ask for
//! when in doubt.
//!
//! Natively only: a thread with the device's handles. In wasm it does nothing; a browser's GPU
//! worker (`web/gpu`) runs the same kernel ([`SpinKernel`]) from its own event loop, as an option.

use std::time::Duration;

use crate::GpuContext;

/// The loop of a dispatch: about 3 ms on the RTX 3080 at full clocks (`bench_idle`), 10-25 ms at
/// its lowest. (The same kernel compiled by a browser runs much faster: see `web/gpu`'s
/// `warm.rs`, which loops longer.)
pub const SPIN_ITERATIONS: u32 = 1 << 16;

/// The keep-warm kernel: one workgroup of 64 threads, each in a loop of `ITERATIONS` steps of a
/// linear congruential generator, its result stored (so that the loop isn't optimized away).
pub const SPIN_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> sink: array<u32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) i: u32) {
    var x = i;
    for (var k = 0u; k < ITERATIONSu; k++) {
        x = x * 1664525u + 1013904223u;
    }
    sink[i] = x;
}
"#;

/// The keep-warm kernel on a device: its pipeline, its sink buffer and their bind group.
pub struct SpinKernel {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    sink: wgpu::Buffer,
}

impl SpinKernel {
    /// [`SPIN_WGSL`] of `iterations` ([`SPIN_ITERATIONS`] natively), compiled (checked, as every
    /// kernel here).
    pub fn new(gpu: &GpuContext, iterations: u32) -> Self {
        let source = SPIN_WGSL.replace("ITERATIONS", &iterations.to_string());
        let pipeline = gpu.pipeline("keep warm", &source, "main");
        let sink = gpu.storage("keep warm", 64);
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("keep warm"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: sink.as_entire_binding() }],
        });
        Self { pipeline, bind_group, sink }
    }

    /// Records one dispatch (one workgroup) into `encoder`.
    pub fn record(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("keep warm"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    /// Frees its buffer now (a browser would otherwise hold it until its JS object is collected).
    pub fn destroy(&self) {
        self.sink.destroy();
    }

    /// The buffer each dispatch writes: thread `i`'s [`spin_reference`] (for a test).
    pub fn sink(&self) -> &wgpu::Buffer {
        &self.sink
    }
}

/// What thread `i` of the kernel stores after `iterations` steps.
pub fn spin_reference(i: u32, iterations: u32) -> u32 {
    (0..iterations).fold(i, |x, _| x.wrapping_mul(1664525).wrapping_add(1013904223))
}

/// A window to keep the GPU warm after each commitment, for a caller turning keep-warm on
/// without one of its own: a prover's commitments come 0.4-1.5 s of CPU work apart natively (more
/// in a browser), so 3 s covers the gaps of a proof.
pub const DEFAULT_WINDOW: Duration = Duration::from_secs(3);

/// The window the environment asks for (`STWO_GPU_KEEP_WARM_MS`, in ms); `None` (off) when it is
/// unset, 0 or not a number.
pub fn window_from_env() -> Option<Duration> {
    match std::env::var("STWO_GPU_KEEP_WARM_MS").ok().and_then(|ms| ms.trim().parse::<u64>().ok()) {
        Some(ms) if ms > 0 => Some(Duration::from_millis(ms)),
        _ => None,
    }
}

#[cfg(not(target_family = "wasm"))]
pub use native::KeepWarm;
#[cfg(target_family = "wasm")]
pub use wasm::KeepWarm;

#[cfg(not(target_family = "wasm"))]
mod native {
    use std::sync::{Arc, Condvar, Mutex, MutexGuard};
    use std::thread::JoinHandle;
    use std::time::{Duration, Instant};

    use super::SpinKernel;
    use crate::GpuContext;

    /// The pause between two dispatches.
    const GAP: Duration = Duration::from_millis(1);

    struct State {
        /// How long to keep warm after a commitment; `None`: never.
        window: Option<Duration>,
        /// Keep warm until then (if not busy).
        until: Option<Instant>,
        /// A commitment is running.
        busy: bool,
        stop: bool,
        dispatches: u64,
    }

    struct Shared {
        state: Mutex<State>,
        wake: Condvar,
    }

    impl Shared {
        fn lock(&self) -> MutexGuard<'_, State> {
            self.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
        }
    }

    /// A thread keeping the GPU's clocks up for a window after each commitment (see the module
    /// docs).
    pub struct KeepWarm {
        shared: Arc<Shared>,
        thread: Mutex<Option<JoinHandle<()>>>,
    }

    impl KeepWarm {
        /// Its kernel compiled, its thread started (idle until the first commitment ends).
        pub fn new(gpu: &GpuContext, window: Option<Duration>) -> Result<Self, String> {
            let kernel = SpinKernel::new(gpu, super::SPIN_ITERATIONS);
            let shared = Arc::new(Shared {
                state: Mutex::new(State {
                    window,
                    until: None,
                    busy: false,
                    stop: false,
                    dispatches: 0,
                }),
                wake: Condvar::new(),
            });
            let (device, queue) = (gpu.device.clone(), gpu.queue.clone());
            let thread_shared = shared.clone();
            let thread = std::thread::Builder::new()
                .name("gpu keep warm".to_string())
                .spawn(move || {
                    let shared = thread_shared;
                    loop {
                        {
                            let mut state = shared.lock();
                            loop {
                                if state.stop {
                                    return;
                                }
                                match state.until {
                                    Some(until) if !state.busy && Instant::now() < until => break,
                                    Some(_) if !state.busy => state.until = None,
                                    _ => {}
                                }
                                state = shared
                                    .wake
                                    .wait(state)
                                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                            }
                            state.dispatches += 1;
                        }
                        let mut encoder =
                            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("keep warm"),
                            });
                        kernel.record(&mut encoder);
                        // One dispatch in flight at most, waited for (a second at most).
                        let index = queue.submit([encoder.finish()]);
                        let wait = wgpu::PollType::Wait {
                            submission_index: Some(index),
                            timeout: Some(Duration::from_secs(1)),
                        };
                        if device.poll(wait).is_err() {
                            // A lost device: the commitments will fail and say so.
                            return;
                        }
                        std::thread::sleep(GAP);
                    }
                })
                .map_err(|error| format!("keep-warm thread: {error}"))?;
            Ok(Self { shared, thread: Mutex::new(Some(thread)) })
        }

        /// How long it keeps the GPU warm after a commitment (`None`: never).
        pub fn window(&self) -> Option<Duration> {
            self.shared.lock().window
        }

        pub fn set_window(&self, window: Option<Duration>) {
            let mut state = self.shared.lock();
            state.window = window;
            if window.is_none() {
                state.until = None;
            }
            self.shared.wake.notify_all();
        }

        /// A commitment starts: no more dispatches until it ends (one may still run, 3 ms at
        /// most at full clocks).
        pub fn begin(&self) {
            self.shared.lock().busy = true;
        }

        /// A commitment ended: warm for the window from now.
        pub fn end(&self) {
            let mut state = self.shared.lock();
            state.busy = false;
            state.until = state.window.map(|window| Instant::now() + window);
            self.shared.wake.notify_all();
        }

        /// Warm for the window from now, unless a commitment is running (which warms after it):
        /// for a caller that knows a commitment is coming.
        pub fn warm_up(&self) {
            let mut state = self.shared.lock();
            if !state.busy {
                state.until = state.window.map(|window| Instant::now() + window);
                self.shared.wake.notify_all();
            }
        }

        /// The dispatches so far.
        pub fn dispatches(&self) -> u64 {
            self.shared.lock().dispatches
        }
    }

    impl KeepWarm {
        /// Ends the thread, after its dispatch in flight if any, and waits for it: no more
        /// keep-warm work on the device. For a process about to exit, whose driver must not be torn
        /// down under a submission (that crashed os_native_probe at exit).
        pub fn stop(&self) {
            self.shared.lock().stop = true;
            self.shared.wake.notify_all();
            let thread = self.thread.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).take();
            if let Some(thread) = thread {
                let _ = thread.join();
            }
        }
    }

    impl Drop for KeepWarm {
        fn drop(&mut self) {
            self.stop();
        }
    }
}

#[cfg(target_family = "wasm")]
mod wasm {
    use std::sync::Mutex;
    use std::time::Duration;

    use crate::GpuContext;

    /// In wasm, nothing: it only keeps its window (a browser's GPU worker keeps warm from its
    /// event loop, see `web/gpu`).
    pub struct KeepWarm {
        window: Mutex<Option<Duration>>,
    }

    impl KeepWarm {
        pub fn new(_gpu: &GpuContext, window: Option<Duration>) -> Result<Self, String> {
            Ok(Self { window: Mutex::new(window) })
        }

        pub fn stop(&self) {}

        pub fn window(&self) -> Option<Duration> {
            *self.window.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
        }

        pub fn set_window(&self, window: Option<Duration>) {
            *self.window.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = window;
        }

        pub fn begin(&self) {}

        pub fn end(&self) {}

        pub fn warm_up(&self) {}

        pub fn dispatches(&self) -> u64 {
            0
        }
    }
}
