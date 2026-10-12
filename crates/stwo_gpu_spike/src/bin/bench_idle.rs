//! How a streamed commitment's time depends on how long the GPU was idle before it, and whether
//! light work submitted meanwhile keeps it fast.
//!
//! An idle NVIDIA GPU drops to its lowest performance state (on the RTX 3080: P8, 210 MHz, the
//! PCIe link at gen 1) and takes a while to climb back under load: a prover's commitments, a
//! second or more of CPU work apart, each start on an idle GPU. This measures the Cairo base
//! trace's commitment (end to end, as the committer's) after each of `--idles` milliseconds of
//! sleep, with no work meanwhile ("off"), then with a thread submitting a small compute dispatch
//! `P` ms after the previous one completed during the sleep (`--warm P[:I],..`, each dispatch
//! `--warm-words` words, `I` multiply-adds each (default 64), waited for: one in flight at most).
//! The thread pauses during the commitments.
//!
//! ```text
//! bench_idle [--scale N] [--idles 0,100,..] [--reps R] [--warm P1,P2,..] [--warm-words W]
//!     [--smi]
//! ```
//!
//! `--smi` prints `nvidia-smi`'s performance state, clocks and PCIe link before each commitment
//! (its query takes some time of its own, taken from the sleep).
//!
//! Each mode's keep-warm is this bench's own thread, set explicitly by `--warm` (`off` included):
//! the committer's (`warm.rs`, off by default) isn't involved.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use itertools::Itertools;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use stwo::core::fields::m31::{BaseField, P};
use stwo::core::poly::circle::CanonicCoset;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::simd::column::BaseColumn;
use stwo::prover::poly::circle::{CircleCoefficients, PolyOps};
use stwo_gpu_spike::GpuContext;
use stwo_gpu_spike::commit::{CommitBuffers, CommitKernels, TwiddleCache};
use stwo_gpu_spike::shapes::{base_trace_log_sizes, cairo_pay_components};
use stwo_gpu_spike::stream::{StreamedCommitment, group_words};
use stwo_gpu_spike::transfer::{ReadbackRing, UploadRing};

const WARM_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> data: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= arrayLength(&data)) {
        return;
    }
    var x = data[id.x];
    for (var i = 0u; i < ITERSu; i++) {
        x = x * 1664525u + 1013904223u;
    }
    data[id.x] = x;
}
"#;

struct Args {
    scale: u32,
    idles: Vec<u64>,
    reps: usize,
    /// `(period ms, loop iterations)`.
    warms: Vec<Option<(u64, u32)>>,
    warm_words: u32,
    smi: bool,
}

fn parse_args() -> Args {
    let mut args = Args {
        scale: 0,
        idles: vec![0, 50, 100, 200, 400, 1000, 2000],
        reps: 2,
        warms: vec![None, Some((20, 64)), Some((100, 64))],
        warm_words: 1 << 20,
        smi: false,
    };
    let list = |s: String| s.split(',').map(|v| v.trim().to_string()).collect_vec();
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut value = |name: &str| it.next().unwrap_or_else(|| panic!("{name} needs a value"));
        match arg.as_str() {
            "--scale" => args.scale = value("--scale").parse().expect("--scale N"),
            "--idles" => {
                args.idles =
                    list(value("--idles")).iter().map(|v| v.parse().expect("--idles")).collect()
            }
            "--reps" => args.reps = value("--reps").parse().expect("--reps R"),
            "--warm" => {
                args.warms = list(value("--warm"))
                    .iter()
                    .map(|v| {
                        (v != "off").then(|| {
                            let (p, i) = v.split_once(':').unwrap_or((v, "64"));
                            (p.parse().expect("--warm P[:I],.. or off"), i.parse().expect("I"))
                        })
                    })
                    .collect()
            }
            "--warm-words" => {
                args.warm_words = value("--warm-words").parse().expect("--warm-words W")
            }
            "--smi" => args.smi = true,
            other => panic!("unknown argument {other}"),
        }
    }
    args
}

/// `nvidia-smi`'s performance state, SM and memory clocks, PCIe link generation.
fn smi() -> String {
    std::process::Command::new("timeout")
        .args([
            "5",
            "nvidia-smi",
            "--query-gpu=pstate,clocks.sm,clocks.mem,pcie.link.gen.current,power.draw,utilization.\
             gpu",
            "--format=csv,noheader",
        ])
        .output()
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
        .unwrap_or_else(|error| format!("nvidia-smi: {error}"))
}

/// A thread submitting a small dispatch every `period` while not paused.
struct Warmer {
    stop: Arc<AtomicBool>,
    paused: Arc<AtomicBool>,
    dispatches: Arc<AtomicU64>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Warmer {
    fn new(gpu: &GpuContext, period: Duration, iterations: u32, words: u32) -> Self {
        let device = gpu.device.clone();
        let queue = gpu.queue.clone();
        let source = WARM_WGSL.replace("ITERS", &iterations.to_string());
        let pipeline = gpu.pipeline("warm", &source, "main");
        let buffer = gpu.storage("warm", words as usize);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("warm"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: buffer.as_entire_binding() }],
        });
        // One dispatch, timed (the GPU at full speed: right after a commitment).
        let start = Instant::now();
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(words.div_ceil(256), 1, 1);
        }
        queue.submit([encoder.finish()]);
        device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        println!(
            "  (a warm dispatch, submitted and waited for: {:.2} ms)",
            start.elapsed().as_secs_f64() * 1e3
        );
        let stop = Arc::new(AtomicBool::new(false));
        let paused = Arc::new(AtomicBool::new(true));
        let dispatches = Arc::new(AtomicU64::new(0));
        let (s, p, d) = (stop.clone(), paused.clone(), dispatches.clone());
        let thread = std::thread::spawn(move || {
            let _buffer = buffer;
            while !s.load(Ordering::Relaxed) {
                if !p.load(Ordering::Relaxed) {
                    let mut encoder = device.create_command_encoder(&Default::default());
                    {
                        let mut pass = encoder.begin_compute_pass(&Default::default());
                        pass.set_pipeline(&pipeline);
                        pass.set_bind_group(0, &bind_group, &[]);
                        pass.dispatch_workgroups(words.div_ceil(256), 1, 1);
                    }
                    // One dispatch in flight at most: waited for before the next.
                    let index = queue.submit([encoder.finish()]);
                    device
                        .poll(wgpu::PollType::Wait { submission_index: Some(index), timeout: None })
                        .expect("poll");
                    d.fetch_add(1, Ordering::Relaxed);
                }
                if p.load(Ordering::Relaxed) || !period.is_zero() {
                    std::thread::sleep(period.max(Duration::from_millis(1)));
                }
            }
        });
        Self { stop, paused, dispatches, thread: Some(thread) }
    }
}

impl Drop for Warmer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            thread.join().unwrap();
        }
    }
}

fn main() {
    let args = parse_args();
    let log_sizes = base_trace_log_sizes(&cairo_pay_components(), args.scale, 4);
    let log_blowup = 3;
    let lifting_log_size = log_sizes.iter().max().unwrap() + log_blowup;
    let polys: Vec<CircleCoefficients<SimdBackend>> = log_sizes
        .par_iter()
        .enumerate()
        .map(|(i, &log_size)| {
            let mut rng = SmallRng::seed_from_u64(i as u64);
            CircleCoefficients::new(
                (0..1 << log_size)
                    .map(|_| BaseField::from_u32_unchecked(rng.random_range(0..P)))
                    .collect::<BaseColumn>(),
            )
        })
        .collect();
    let twiddles = SimdBackend::precompute_twiddles(
        CanonicCoset::new(lifting_log_size).circle_domain().half_coset,
    );
    let bytes: u64 = log_sizes.iter().map(|&l| 4u64 << l).sum();

    let gpu = GpuContext::new();
    println!(
        "GPU: {} ({:?}); Cairo base trace / 2^{}: {} MiB",
        gpu.adapter_info.name,
        gpu.adapter_info.backend,
        args.scale,
        bytes >> 20
    );
    let kernels = CommitKernels::new(&gpu);
    let mut buffers = CommitBuffers::new();
    let mut twiddle_cache = TwiddleCache::new();
    let mut upload = UploadRing::with_defaults(&gpu);
    let mut readback = ReadbackRing::with_defaults(&gpu);
    let distinct = log_sizes.iter().copied().collect();
    let table = twiddle_cache.get(&gpu, &distinct, log_blowup, &twiddles);
    let words = group_words(
        &gpu,
        &log_sizes,
        log_blowup,
        lifting_log_size,
        upload.chunk_bytes(),
        readback.chunk_bytes(),
    )
    .expect("streamed");
    let commitment = StreamedCommitment::with_buffers(
        &gpu,
        &kernels,
        &mut buffers,
        table,
        &log_sizes,
        log_blowup,
        lifting_log_size,
        words,
    );
    let output = Mutex::new(None);
    let mut commit = || {
        drop(output.lock().unwrap().take());
        upload.take_times();
        readback.take_times();
        let start = Instant::now();
        let (out, phases) =
            commitment.commit_timed(&gpu, &kernels, &mut upload, &mut readback, &polys);
        let elapsed = start.elapsed();
        *output.lock().unwrap() = Some(out);
        (elapsed, upload.take_times(), readback.take_times(), phases)
    };
    for _ in 0..3 {
        commit();
    }
    let f = |d: Duration| d.as_secs_f64() * 1e3;
    for &warm in &args.warms {
        let warmer =
            warm.map(|(p, i)| Warmer::new(&gpu, Duration::from_millis(p), i, args.warm_words));
        println!(
            "-- keep-warm: {}",
            match warm {
                None => "off".to_string(),
                Some((p, i)) => {
                    let words = args.warm_words;
                    format!("a dispatch of {words} words, {i} iterations, every {p} ms")
                }
            }
        );
        let mut summary = vec![];
        for &idle in &args.idles {
            let mut times = vec![];
            for _ in 0..args.reps {
                // Back to full speed first, then idle.
                commit();
                if let Some(w) = &warmer {
                    w.paused.store(false, Ordering::Relaxed);
                }
                let state = if args.smi && idle >= 200 {
                    std::thread::sleep(Duration::from_millis(idle - 100));
                    smi()
                } else {
                    std::thread::sleep(Duration::from_millis(idle));
                    String::new()
                };
                if let Some(w) = &warmer {
                    w.paused.store(true, Ordering::Relaxed);
                }
                let (t, up, rb, phases) = commit();
                println!(
                    "  idle {idle:5} ms: {:7.1} ms | up cpy/wait {:6.1}/{:6.1} | rb wait/cpy \
                     {:6.1}/{:6.1} | phases {:6.1}/{:5.1}/{:5.1}/{:5.1} {}",
                    f(t),
                    f(up.copy),
                    f(up.wait),
                    f(rb.wait),
                    f(rb.copy),
                    f(phases[0]),
                    f(phases[1]),
                    f(phases[2]),
                    f(phases[3]),
                    if state.is_empty() { String::new() } else { format!("| before: {state}") },
                );
                times.push(t);
            }
            times.sort();
            summary.push((idle, times[times.len() / 2]));
        }
        if let Some(w) = &warmer {
            println!("  dispatches: {}", w.dispatches.load(Ordering::Relaxed));
        }
        println!(
            "  median per idle: {}",
            summary.iter().map(|(idle, t)| format!("{idle} ms: {:.1}", f(*t))).join(", ")
        );
    }
}
