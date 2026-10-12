//! The bridge's demo, on the prover's side: a threaded module (built like the Outis prover's)
//! whose rayon threads call [`gpu_m31_mul`], a synchronous function served by the GPU worker,
//! and check its results against stwo's M31 multiply on the CPU.
//!
//! The page's worker calls the `demo_*` exports, each one long synchronous call like the
//! prover's `session.prove()`; inside, the GPU calls are made from rayon's threads (Web Workers),
//! and each `demo_*` returns its report as JSON.
use rayon::prelude::*;
use stwo::core::fields::m31::{M31, P};
use stwo_gpu_bridge::{self as bridge, BridgeError, Completion, op};
use wasm_bindgen::prelude::*;

pub use wasm_bindgen_rayon::init_thread_pool;

#[wasm_bindgen(inline_js = "export function abs_now() { return performance.timeOrigin + performance.now(); }")]
extern "C" {
    /// Absolute ms, comparable across workers (each worker's `performance.now()` has its own
    /// origin). `std::time::Instant` panics on this target.
    fn abs_now() -> f64;
}

fn now() -> f64 {
    abs_now()
}

/// The bridge region's address in this module's memory, its header written: the GPU worker
/// attaches to (memory, address).
#[wasm_bindgen]
pub fn bridge_region() -> u32 {
    bridge::region_address()
}

/// Whether a GPU worker is serving the bridge.
#[wasm_bindgen]
pub fn gpu_ready() -> bool {
    bridge::ready()
}

/// The module's memory, in MiB.
#[wasm_bindgen]
pub fn memory_mib() -> u32 {
    (core::arch::wasm32::memory_size(0) * 65536 / (1 << 20)) as u32
}

/// `a[i] * b[i]` in M31 on the GPU, synchronously: callable from any prover thread (not the
/// page's main thread, which may not block). `Err` when there's no GPU worker or the job failed:
/// the caller then uses the CPU.
pub fn gpu_m31_mul(a: &[u32], b: &[u32]) -> Result<Vec<u32>, BridgeError> {
    gpu_m31_mul_timed(a, b, false).map(|(out, _)| out)
}

fn gpu_m31_mul_timed(a: &[u32], b: &[u32], pipelined: bool) -> Result<(Vec<u32>, Completion), BridgeError> {
    assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut out: Vec<u32> = Vec::with_capacity(n);
    let args = [a.as_ptr() as u32, b.as_ptr() as u32, out.as_mut_ptr() as u32, n as u32, pipelined as u32];
    let completion = bridge::call(op::M31_MUL, &args, now)?;
    // SAFETY: the job is done: the GPU worker wrote all n words of `out` (and `call`'s SeqCst
    // load of the slot's state orders its writes before ours).
    unsafe { out.set_len(n) };
    Ok((out, completion))
}

fn cpu_mul(a: u32, b: u32) -> u32 {
    (M31::from_u32_unchecked(a) * M31::from_u32_unchecked(b)).0
}

/// Mismatches between `out` and stwo's product, in parallel.
fn mismatches(a: &[u32], b: &[u32], out: &[u32]) -> usize {
    out.par_iter()
        .zip(a.par_iter().zip(b.par_iter()))
        .filter(|(o, (x, y))| **o != cpu_mul(**x, **y))
        .count()
}

const EDGES: [u32; 9] = [0, 1, 2, P - 1, P - 2, 1 << 30, (1 << 30) - 1, 1 << 16, (1 << 16) - 1];

fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// `n` canonical M31 values from `seed`, the edge-case pairs first.
fn random_m31(n: usize, seed: u64, first: bool) -> Vec<u32> {
    (0..n)
        .into_par_iter()
        .map(|i| {
            if i < EDGES.len() * EDGES.len() {
                let (row, col) = (i / EDGES.len(), i % EDGES.len());
                return EDGES[if first { row } else { col }];
            }
            (splitmix(seed ^ (i as u64).wrapping_mul(0x2545_f491_4f6c_dd1d)) % P as u64) as u32
        })
        .collect()
}

/// Runs `f` on one of rayon's threads (a Web Worker), as prover code would, and the index of
/// that thread.
fn on_rayon<R: Send>(f: impl FnOnce() -> R + Send) -> (R, Option<usize>) {
    rayon::scope(|_| (f(), rayon::current_thread_index()))
}

/// Order statistics of `values` as JSON.
fn stats(values: &mut [f64]) -> String {
    if values.is_empty() {
        return "null".into();
    }
    values.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let at = |q: f64| values[((values.len() - 1) as f64 * q).round() as usize];
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    format!(
        "{{\"n\":{},\"min\":{:.4},\"p10\":{:.4},\"median\":{:.4},\"p90\":{:.4},\"max\":{:.4},\"mean\":{:.4}}}",
        values.len(),
        values[0],
        at(0.1),
        at(0.5),
        at(0.9),
        values[values.len() - 1],
        mean
    )
}

/// A set of completions: the whole round trip and its parts, in ms.
fn breakdown(completions: &[Completion]) -> String {
    let pick = |f: &dyn Fn(&Completion) -> f64| stats(&mut completions.iter().map(f).collect::<Vec<_>>());
    format!(
        "{{\"total_ms\":{},\"wake_ms\":{},\"serve_ms\":{},\"return_ms\":{}}}",
        pick(&|c| c.total_ms()),
        pick(&|c| c.wake_ms()),
        pick(&|c| c.serve_ms()),
        pick(&|c| c.return_ms())
    )
}

fn json_index(index: Option<usize>) -> String {
    index.map_or("null".into(), |i| i.to_string())
}

fn error_json(error: &BridgeError) -> String {
    format!("{{\"error\":{:?}}}", error.to_string())
}

/// Correctness: the edge cases, then many jobs of varied sizes (not multiples of the workgroup)
/// issued concurrently from rayon's threads, each checked against stwo's M31.
#[wasm_bindgen]
pub fn demo_correctness(jobs: u32) -> String {
    let started = now();
    let sizes: Vec<usize> = (0..jobs as u64).map(|j| 81 + (splitmix(j) % 300_000) as usize).collect();
    let results: Vec<Result<(usize, usize, Option<usize>), BridgeError>> = sizes
        .par_iter()
        .enumerate()
        .map(|(j, &n)| {
            let (a, b) = (random_m31(n, 2 * j as u64 + 1, true), random_m31(n, 2 * j as u64 + 2, false));
            let out = gpu_m31_mul(&a, &b)?;
            Ok((n, mismatches(&a, &b, &out), rayon::current_thread_index()))
        })
        .collect();
    let mut threads: Vec<usize> = results.iter().filter_map(|r| r.as_ref().ok().and_then(|r| r.2)).collect();
    threads.sort();
    threads.dedup();
    let errors: Vec<String> = results.iter().filter_map(|r| r.as_ref().err().map(|e| format!("{:?}", e.to_string()))).collect();
    let ok: Vec<&(usize, usize, Option<usize>)> = results.iter().filter_map(|r| r.as_ref().ok()).collect();
    format!(
        "{{\"phase\":\"correctness\",\"jobs\":{},\"ok_jobs\":{},\"elements\":{},\"mismatches\":{},\"rayon_threads_used\":{},\"errors\":[{}],\"wall_ms\":{:.1}}}",
        jobs,
        ok.len(),
        ok.iter().map(|r| r.0).sum::<usize>(),
        ok.iter().map(|r| r.1).sum::<usize>(),
        threads.len(),
        errors.join(","),
        now() - started
    )
}

/// Failures come back as errors (and the bridge keeps serving): a range outside the memory, an
/// unknown op, then a good job.
#[wasm_bindgen]
pub fn demo_errors() -> String {
    let (results, _) = on_rayon(|| {
        let outside = bridge::call(op::M31_MUL, &[u32::MAX - 4096, 0, 0, 1024], now);
        let unknown = bridge::call(999, &[], now);
        let (a, b) = (random_m31(4096, 21, true), random_m31(4096, 22, false));
        let after = gpu_m31_mul(&a, &b).map(|out| mismatches(&a, &b, &out));
        (outside, unknown, after)
    });
    let show = |r: &Result<Completion, BridgeError>| match r {
        Ok(_) => "\"ok (unexpected)\"".to_string(),
        Err(e) => format!("{:?}", e.to_string()),
    };
    format!(
        "{{\"phase\":\"errors\",\"outside_memory\":{},\"unknown_op\":{},\"good_job_after\":{}}}",
        show(&results.0),
        show(&results.1),
        match &results.2 {
            Ok(bad) => format!("{{\"mismatches\":{bad}}}"),
            Err(e) => error_json(e),
        }
    )
}

/// The failure path: a job that traps the GPU module (the bridge is lost: run it last), then a
/// good job, which must fail too rather than hang.
#[wasm_bindgen]
pub fn demo_trap() -> String {
    let (results, _) = on_rayon(|| {
        let trap = bridge::call(op::TEST_TRAP, &[], now);
        let (a, b) = (random_m31(4096, 23, true), random_m31(4096, 24, false));
        let after = gpu_m31_mul(&a, &b).map(|out| mismatches(&a, &b, &out));
        (trap, after, bridge::ready())
    });
    format!(
        "{{\"phase\":\"trap\",\"trap\":{},\"job_after\":{},\"ready_after\":{}}}",
        match &results.0 {
            Ok(_) => "\"ok (unexpected)\"".to_string(),
            Err(e) => format!("{:?}", e.to_string()),
        },
        match &results.1 {
            Ok(bad) => format!("{{\"ok_unexpected_mismatches\":{bad}}}"),
            Err(e) => error_json(e),
        },
        results.2
    )
}

/// Latency: `iters` sequential jobs of each kind from one rayon thread, then pings from all
/// threads at once.
#[wasm_bindgen]
pub fn demo_latency(iters: u32) -> String {
    let iters = iters as usize;
    let ((ping, submit, small, small_pipelined, small_bad), thread) = on_rayon(|| {
        let run = |op: u32| -> Result<Vec<Completion>, BridgeError> {
            (0..iters).map(|_| bridge::call(op, &[], now)).collect()
        };
        // Warm up (pipelines, pooled buffers).
        let (a, b) = (random_m31(1024, 7, true), random_m31(1024, 8, false));
        let _ = run(op::PING);
        let _ = gpu_m31_mul_timed(&a, &b, false);
        let _ = gpu_m31_mul_timed(&a, &b, true);
        let ping = run(op::PING);
        let submit = run(op::SUBMIT);
        let mut bad = 0;
        let mut mul = |pipelined: bool| -> Result<Vec<Completion>, BridgeError> {
            (0..iters)
                .map(|_| {
                    let (out, c) = gpu_m31_mul_timed(&a, &b, pipelined)?;
                    bad += mismatches(&a, &b, &out);
                    Ok(c)
                })
                .collect()
        };
        let small = mul(false);
        let small_pipelined = mul(true);
        (ping, submit, small, small_pipelined, bad)
    });
    let show = |r: &Result<Vec<Completion>, BridgeError>| match r {
        Ok(completions) => breakdown(completions),
        Err(e) => error_json(e),
    };
    // Concurrent pings: every thread at once, a job table with many slots in use.
    let threads = rayon::current_num_threads();
    let started = now();
    let concurrent: Result<Vec<Completion>, BridgeError> =
        (0..threads * iters).into_par_iter().map(|_| bridge::call(op::PING, &[], now)).collect();
    let wall = now() - started;
    format!(
        "{{\"phase\":\"latency\",\"iters\":{},\"rayon_thread\":{},\"ping\":{},\"empty_submit\":{},\"m31_mul_1024\":{},\"m31_mul_1024_pipelined\":{},\"m31_mul_1024_mismatches\":{},\"concurrent_ping\":{{\"threads\":{},\"jobs\":{},\"wall_ms\":{:.2},\"jobs_per_s\":{:.0},\"per_job\":{}}}}}",
        iters,
        json_index(thread),
        show(&ping),
        show(&submit),
        show(&small),
        show(&small_pipelined),
        small_bad,
        threads,
        threads * iters,
        wall,
        (threads * iters) as f64 / wall * 1000.0,
        show(&concurrent)
    )
}

fn mib_per_s(bytes: usize, ms: f64) -> f64 {
    bytes as f64 / (1 << 20) as f64 / (ms / 1000.0)
}

/// Throughput at `mib` MiB per buffer: `reps` times the M31 product with its phases timed
/// (upload of a and b, compute, readback of out), the pipelined product, an upload alone and a
/// readback alone; the outputs checked against stwo's.
#[wasm_bindgen]
pub fn demo_throughput(mib: u32, reps: u32) -> String {
    let n = (mib as usize) << 18;
    let bytes = n * 4;
    let t = now();
    let (a, b) = (random_m31(n, 11, true), random_m31(n, 12, false));
    let fill_ms = now() - t;
    let (result, thread) = on_rayon(|| -> Result<String, BridgeError> {
        let mut phased = Vec::new();
        let mut pipelined = Vec::new();
        let mut uploads = Vec::new();
        let mut readbacks = Vec::new();
        let mut bad = 0;
        let mut check_ms = 0.0;
        for _ in 0..reps {
            let (out, c) = gpu_m31_mul_timed(&a, &b, false)?;
            let t = now();
            bad += mismatches(&a, &b, &out);
            check_ms = now() - t;
            drop(out);
            phased.push(c);
            let (out, c) = gpu_m31_mul_timed(&a, &b, true)?;
            bad += mismatches(&a, &b, &out);
            pipelined.push(c);
            uploads.push(bridge::call(op::UPLOAD, &[a.as_ptr() as u32, bytes as u32], now)?);
            let mut dest = out;
            readbacks.push(bridge::call(op::READBACK, &[dest.as_mut_ptr() as u32, bytes as u32], now)?);
        }
        let best = |cs: &[Completion], f: &dyn Fn(&Completion) -> f64| cs.iter().map(f).fold(f64::INFINITY, f64::min);
        let up = |c: &Completion| c.times[bridge::T_UPLOAD];
        let compute = |c: &Completion| c.times[bridge::T_COMPUTE];
        let rb = |c: &Completion| c.times[bridge::T_READBACK];
        let total = |c: &Completion| c.total_ms();
        let list = |cs: &[Completion], f: &dyn Fn(&Completion) -> f64| {
            cs.iter().map(|c| format!("{:.1}", f(c))).collect::<Vec<_>>().join(",")
        };
        Ok(format!(
            "\"mismatches\":{},\"cpu_check_ms\":{:.1},\
             \"m31_mul\":{{\"total_ms\":[{}],\"upload_ms\":[{}],\"compute_ms\":[{}],\"readback_ms\":[{}],\"wake_ms\":[{}],\"return_ms\":[{}],\
             \"best_upload_mib_s\":{:.0},\"best_readback_mib_s\":{:.0},\"best_end_to_end_mib_s\":{:.0}}},\
             \"m31_mul_pipelined\":{{\"total_ms\":[{}],\"best_end_to_end_mib_s\":{:.0}}},\
             \"upload_only\":{{\"ms\":[{}],\"best_mib_s\":{:.0}}},\
             \"readback_only\":{{\"ms\":[{}],\"best_mib_s\":{:.0}}}",
            bad,
            check_ms,
            list(&phased, &total),
            list(&phased, &up),
            list(&phased, &compute),
            list(&phased, &rb),
            list(&phased, &|c| c.wake_ms()),
            list(&phased, &|c| c.return_ms()),
            mib_per_s(2 * bytes, best(&phased, &up)),
            mib_per_s(bytes, best(&phased, &rb)),
            mib_per_s(3 * bytes, best(&phased, &total)),
            list(&pipelined, &total),
            mib_per_s(3 * bytes, best(&pipelined, &total)),
            list(&uploads, &up),
            mib_per_s(bytes, best(&uploads, &up)),
            list(&readbacks, &rb),
            mib_per_s(bytes, best(&readbacks, &rb)),
        ))
    });
    let body = result.unwrap_or_else(|e| format!("\"error\":{:?}", e.to_string()));
    format!(
        "{{\"phase\":\"throughput\",\"mib_per_buffer\":{},\"words\":{},\"reps\":{},\"rayon_thread\":{},\"fill_ms\":{:.1},\"wasm_memory_mib\":{},{}}}",
        mib,
        n,
        reps,
        json_index(thread),
        fill_ms,
        memory_mib(),
        body
    )
}
