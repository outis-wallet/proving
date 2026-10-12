//! The commitments of a proof, in a row, on one `GpuCommitter`, against stwo's
//! `CommitmentTreeProver::new_striped` on the CPU: per commitment and in total, end to end (the
//! committer's whole `commit_striped`: plans, upload, compute, readback into stwo's columns, the
//! polynomials' drop), the roots checked equal (and, with `--check-all`, every layer and every
//! column's kept evaluations). The GPU's split (`CommitTimes`): setup (plans; buffers and
//! twiddles when new), commit (upload, compute, readback; `s`: streamed in groups), the tree's
//! assembly, the polynomials' drop (handed to a thread).
//!
//! ```text
//! bench_proof_commits [--proofs N] [--threads T] [--check-all] [--no-cpu] [--no-gpu]
//!     [--idle-ms MS] [--keep-warm-ms MS] [--rss-breakdown]
//! ```
//!
//! The split of the GPU's `commit` (`CommitTimes`): the upload ring's CPU copies into its chunks
//! and waits for a free one, the readback ring's waits for the GPU and CPU copies out; streamed,
//! the phases (groups, the largest size's columns, the copies out of the last chunks, the
//! readback of the columns kept whole and the layers). Memory: the process's resident set
//! (`VmRSS`) after each proof, and per commitment the peak resident set above the one before it
//! (`VmHWM`, reset through `/proc/self/clear_refs`), the input polynomials (cloned before) not
//! counted. `--idle-ms MS` sleeps before each GPU commitment (the GPU idle, as while the prover
//! computes on the CPU), without the CPU's work in between that `--no-cpu` leaves out.
//! `--keep-warm-ms MS` sets the committer's keep-warm window (`GpuCommitter::set_keep_warm`, 0:
//! off; default: the committer's, `STWO_GPU_KEEP_WARM_MS` or 3 s). `--rss-breakdown` first
//! measures what each part of a committer adds to the resident set (the device, the kernels, the
//! rings, a mapped chunk written), before the committer.
//!
//! A proof's trees, as the prover commits them (random coefficients):
//! - the Cairo leg, blowup 3, lifting log size 23: the base trace and the interaction trace of
//!   `data/cairo_trace_shapes_pay.txt` (2608 and 1616 columns, 516 and 491 MiB), and a
//!   composition-like tree (8 columns of 2^20: the composition polynomial's 2 parts of 4
//!   coordinates; assumed, not captured);
//! - the circuit leg, blowup 2, lifting log size 23 (its shapes are not in `data/`): a spread of
//!   sizes like its trace's (the one `tests/commit.rs` checks, 151 columns up to 2^21, 318 MiB), an
//!   interaction-like tree (48 columns up to 2^21) and a composition-like one (8 columns of 2^21).
//!
//! The preprocessed trees are left out (their shapes are not captured either).
//!
//! `--proofs N` (default 3) proves N transactions in a row on the same committer: the odd ones
//! with other shapes (every third Cairo component one size smaller or larger, the circuit's
//! counts changed), as the next transaction's trees would be. The first proof is cold (the
//! buffers grow to the largest tree, every blowup's twiddles are computed); the others reuse
//! them. Both sides run in a rayon pool of `--threads` threads (default 18, the i9-10980XE's
//! cores), as the prover's hook would; the polynomials are cloned outside the timed region.

use std::time::{Duration, Instant};

use itertools::Itertools;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use stwo::core::fields::m31::{BaseField, P};
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sM31MerkleChannel;
use stwo::prover::CommitmentTreeProver;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::simd::column::BaseColumn;
use stwo::prover::mempool::BaseColumnPool;
use stwo::prover::poly::circle::{CircleCoefficients, PolyOps};
use stwo::prover::poly::twiddles::TwiddleTree;
use stwo_gpu_spike::committer::GpuCommitter;
use stwo_gpu_spike::shapes::{
    CAIRO_LIFTING_LOG_SIZE, CAIRO_LOG_BLOWUP, CIRCUIT_LIFTING_LOG_SIZE, CIRCUIT_LOG_BLOWUP,
    Component, base_trace_log_sizes, cairo_pay_components, interaction_trace_log_sizes,
};

type TreeProver = CommitmentTreeProver<SimdBackend, Blake2sM31MerkleChannel>;

struct Args {
    proofs: usize,
    threads: usize,
    check_all: bool,
    cpu: bool,
    gpu: bool,
    idle: Duration,
    keep_warm: Option<Option<Duration>>,
    rss_breakdown: bool,
}

fn parse_args() -> Args {
    let mut args = Args {
        proofs: 3,
        threads: 18,
        check_all: false,
        cpu: true,
        gpu: true,
        idle: Duration::ZERO,
        keep_warm: None,
        rss_breakdown: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut value = |name: &str| it.next().unwrap_or_else(|| panic!("{name} needs a value"));
        match arg.as_str() {
            "--proofs" => args.proofs = value("--proofs").parse().expect("--proofs N"),
            "--threads" => args.threads = value("--threads").parse().expect("--threads T"),
            "--check-all" => args.check_all = true,
            "--no-cpu" => args.cpu = false,
            "--no-gpu" => args.gpu = false,
            "--rss-breakdown" => args.rss_breakdown = true,
            "--keep-warm-ms" => {
                let ms: u64 = value("--keep-warm-ms").parse().expect("--keep-warm-ms MS");
                args.keep_warm = Some((ms > 0).then(|| Duration::from_millis(ms)));
            }
            "--idle-ms" => {
                args.idle = Duration::from_millis(value("--idle-ms").parse().expect("--idle-ms MS"))
            }
            "-h" | "--help" => {
                println!(
                    "bench_proof_commits [--proofs N] [--threads T] [--check-all] [--no-cpu] \
                     [--no-gpu] [--idle-ms MS] [--keep-warm-ms MS] [--rss-breakdown]"
                );
                std::process::exit(0);
            }
            other => panic!("unknown argument {other}"),
        }
    }
    args
}

/// A tree of a proof: its name, its columns' log sizes (polynomial order), its blowup and
/// lifting log size.
struct Tree {
    name: String,
    log_sizes: Vec<u32>,
    log_blowup: u32,
    lifting_log_size: u32,
}

/// Columns of `(log_size, count)` runs, interleaved as components' columns are.
fn spread(runs: &[(u32, usize)], seed: u64) -> Vec<u32> {
    let mut log_sizes = runs.iter().flat_map(|&(l, n)| std::iter::repeat_n(l, n)).collect_vec();
    let mut rng = SmallRng::seed_from_u64(seed);
    for i in (1..log_sizes.len()).rev() {
        log_sizes.swap(i, rng.random_range(0..=i));
    }
    log_sizes
}

/// The trees of transaction `tx`'s proof (see the module docs).
fn proof_trees(tx: usize) -> Vec<Tree> {
    let mut components = cairo_pay_components();
    if tx % 2 == 1 {
        // Another transaction: every third component one size smaller or larger.
        for (i, c) in components.iter_mut().enumerate().filter(|(i, _)| i % 3 == 0) {
            *c = Component {
                log_rows: if i % 2 == 0 && c.log_rows < 20 {
                    c.log_rows + 1
                } else {
                    (c.log_rows - 1).max(4)
                },
                ..c.clone()
            };
        }
    }
    let cairo = |name: &str, log_sizes: Vec<u32>| Tree {
        name: name.to_string(),
        log_sizes,
        log_blowup: CAIRO_LOG_BLOWUP,
        lifting_log_size: CAIRO_LIFTING_LOG_SIZE,
    };
    let circuit = |name: &str, log_sizes: Vec<u32>| Tree {
        name: name.to_string(),
        log_sizes,
        log_blowup: CIRCUIT_LOG_BLOWUP,
        lifting_log_size: CIRCUIT_LIFTING_LOG_SIZE,
    };
    let more = |n: usize| if tx % 2 == 1 { n + n / 3 } else { n };
    vec![
        cairo("cairo base", base_trace_log_sizes(&components, 0, 4)),
        cairo("cairo interaction", interaction_trace_log_sizes(&components, 0, 4)),
        cairo("cairo composition-like", vec![20; 8]),
        circuit(
            "circuit base-like",
            spread(
                &[
                    (21, 21),
                    (20, more(9)),
                    (19, 35),
                    (18, 3),
                    (17, more(50)),
                    (15, 7),
                    (12, more(19)),
                    (8, 5),
                    (5, 2),
                ],
                200 + tx as u64,
            ),
        ),
        circuit(
            "circuit interaction-like",
            spread(&[(21, 8), (20, more(4)), (19, 12), (17, more(16)), (12, 8)], 300 + tx as u64),
        ),
        circuit("circuit composition-like", vec![21; 8]),
    ]
}

fn random_polys(log_sizes: &[u32], seed: u64) -> Vec<CircleCoefficients<SimdBackend>> {
    log_sizes
        .par_iter()
        .enumerate()
        .map(|(i, &log_size)| {
            let mut rng = SmallRng::seed_from_u64(seed * 1_000_003 + i as u64);
            CircleCoefficients::new(
                (0..1 << log_size)
                    .map(|_| BaseField::from_u32_unchecked(rng.random_range(0..P)))
                    .collect::<BaseColumn>(),
            )
        })
        .collect()
}

/// The process's resident set and its peak (`VmRSS`, `VmHWM`), in bytes.
fn rss() -> (u64, u64) {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let field = |name: &str| {
        status
            .lines()
            .find_map(|line| line.strip_prefix(name))
            .and_then(|rest| rest.trim().trim_end_matches("kB").trim().parse::<u64>().ok())
            .map_or(0, |kb| kb << 10)
    };
    (field("VmRSS:"), field("VmHWM:"))
}

/// Resets the peak resident set (`VmHWM`) to the current one.
fn reset_peak() {
    let _ = std::fs::write("/proc/self/clear_refs", "5");
}

/// The peak resident set above the current one while `f` runs.
fn peak_above<T>(f: impl FnOnce() -> T) -> (T, u64) {
    reset_peak();
    let (before, _) = rss();
    let result = f();
    let (_, peak) = rss();
    (result, peak.saturating_sub(before))
}

fn ms(d: Duration) -> String {
    format!("{:8.1}", d.as_secs_f64() * 1e3)
}

fn mib(bytes: u64) -> String {
    format!("{:.1} MiB", bytes as f64 / (1 << 20) as f64)
}

fn assert_trees_eq(gpu: &TreeProver, cpu: &TreeProver, what: &str) {
    assert_eq!(gpu.commitment.layers, cpu.commitment.layers, "{what}: layers differ");
    assert_eq!(gpu.polynomials.len(), cpu.polynomials.len(), "{what}: number of columns");
    for (i, (g, c)) in gpu.polynomials.iter().zip(&cpu.polynomials).enumerate() {
        assert_eq!(g.evals.domain, c.evals.domain, "{what}: column {i} domain");
        assert!(
            g.evals.values.as_slice() == c.evals.values.as_slice(),
            "{what}: column {i}: kept evaluations differ"
        );
        assert!(g.coeffs.is_none() && c.coeffs.is_none(), "{what}: column {i}: coefficients");
    }
}

fn main() {
    let args = parse_args();
    let pool = rayon::ThreadPoolBuilder::new().num_threads(args.threads).build().unwrap();
    if args.gpu && args.rss_breakdown {
        rss_breakdown();
    }
    let (rss_before, _) = rss();
    let start = Instant::now();
    let committer = args.gpu.then(|| GpuCommitter::new().expect("a GPU adapter"));
    if let Some(committer) = &committer {
        if let Some(window) = args.keep_warm {
            committer.set_keep_warm(window);
        }
        let info = committer.adapter_info();
        println!(
            "GPU: {} ({:?}, driver {} {}); committer ready in {} ms, resident set +{}; keep-warm \
             {:?}; {} threads",
            info.name,
            info.backend,
            info.driver,
            info.driver_info,
            ms(start.elapsed()).trim(),
            mib(rss().0.saturating_sub(rss_before)),
            committer.keep_warm(),
            args.threads
        );
    }
    let twiddles = |lifting_log_size: u32| -> TwiddleTree<SimdBackend> {
        SimdBackend::precompute_twiddles(
            CanonicCoset::new(lifting_log_size).circle_domain().half_coset,
        )
    };
    let cairo_twiddles = twiddles(CAIRO_LIFTING_LOG_SIZE);
    let circuit_twiddles = twiddles(CIRCUIT_LIFTING_LOG_SIZE);
    // The transactions' trees and polynomials, made once.
    let transactions = (0..args.proofs.min(2))
        .map(|tx| {
            proof_trees(tx)
                .into_iter()
                .enumerate()
                .map(|(k, tree)| {
                    let polys = random_polys(&tree.log_sizes, (tx * 100 + k) as u64);
                    (tree, polys)
                })
                .collect_vec()
        })
        .collect_vec();

    let mut totals = vec![];
    for proof in 0..args.proofs {
        let tx = proof % 2;
        println!(
            "== proof {proof} (transaction {}){}",
            if tx == 0 { "A" } else { "B" },
            if proof == 0 { ", cold: buffers grow, twiddles computed" } else { "" }
        );
        println!(
            "  {:<25} {:>5} {:>10}  {:>8} {:>8}  {:>8} | {:>6} {:>8} {:>6} | {:>13} {:>13} | \
             {:>23} | {:>7} {:>7} | root",
            "tree",
            "cols",
            "coeffs",
            "CPU ms",
            "GPU ms",
            "speedup",
            "setup",
            "commit",
            "assem",
            "up cpy/wait",
            "rb wait/cpy",
            "groups/top/drain/tail",
            "CPU pk",
            "GPU pk",
        );
        let (mut cpu_total, mut gpu_total) = (Duration::ZERO, Duration::ZERO);
        for (tree, polys) in &transactions[tx] {
            let tw = if tree.log_blowup == CAIRO_LOG_BLOWUP {
                &cairo_twiddles
            } else {
                &circuit_twiddles
            };
            let bytes: u64 = tree.log_sizes.iter().map(|&l| 4u64 << l).sum();

            let mut cpu_peak = 0;
            let cpu_tree = args.cpu.then(|| {
                let input = polys.clone();
                let (result, peak) = peak_above(|| {
                    pool.install(|| {
                        let memory = BaseColumnPool::<SimdBackend>::new();
                        let start = Instant::now();
                        let tree = TreeProver::new_striped(
                            input,
                            tree.log_blowup,
                            tw,
                            tree.lifting_log_size,
                            &memory,
                        );
                        (start.elapsed(), tree)
                    })
                });
                cpu_peak = peak;
                result
            });

            let mut gpu_peak = 0;
            let gpu_tree = committer.as_ref().map(|committer| {
                let input = polys.clone();
                std::thread::sleep(args.idle);
                let (result, peak) = peak_above(|| {
                    pool.install(|| {
                        let start = Instant::now();
                        let result = committer.commit_striped(
                            input,
                            tree.log_blowup,
                            tw,
                            tree.lifting_log_size,
                        );
                        let elapsed = start.elapsed();
                        (
                            elapsed,
                            result.unwrap_or_else(|(_, declined)| panic!("declined: {declined:?}")),
                        )
                    })
                });
                gpu_peak = peak;
                (result.0, result.1, committer.last_times().unwrap())
            });

            let root = gpu_tree
                .as_ref()
                .map(|(_, tree, _)| tree.commitment.root())
                .or_else(|| cpu_tree.as_ref().map(|(_, tree)| tree.commitment.root()))
                .unwrap();
            let (cpu_ms, speedup) = match (&cpu_tree, &gpu_tree) {
                (Some((cpu_time, cpu)), Some((gpu_time, gpu_tree, _))) => {
                    assert_eq!(cpu.commitment.root(), root, "{}: roots differ", tree.name);
                    if args.check_all {
                        assert_trees_eq(gpu_tree, cpu, &tree.name);
                    }
                    (
                        ms(*cpu_time),
                        format!("{:7.1}x", cpu_time.as_secs_f64() / gpu_time.as_secs_f64()),
                    )
                }
                (Some((cpu_time, _)), None) => (ms(*cpu_time), "       -".to_string()),
                _ => ("       -".to_string(), "       -".to_string()),
            };
            if let Some((cpu_time, _)) = &cpu_tree {
                cpu_total += *cpu_time;
            }
            let f = |d: Duration| d.as_secs_f64() * 1e3;
            let (gpu_ms, split) = match &gpu_tree {
                Some((gpu_time, _, times)) => {
                    gpu_total += *gpu_time;
                    let split = format!(
                        "{:>6.1} {:>7.1}{} {:>6.1} | {:>6.1}/{:<6.1} {:>6.1}/{:<6.1} | \
                         {:>5.1}/{:>5.1}/{:>5.1}/{:>5.1}",
                        f(times.setup),
                        f(times.commit),
                        if times.streamed { "s" } else { " " },
                        f(times.assembly),
                        f(times.upload.copy),
                        f(times.upload.wait),
                        f(times.readback.wait),
                        f(times.readback.copy),
                        f(times.phases[0]),
                        f(times.phases[1]),
                        f(times.phases[2]),
                        f(times.phases[3]),
                    );
                    (ms(*gpu_time), split)
                }
                None => (
                    "       -".to_string(),
                    format!("{:>23} | {:>13} {:>13} | {:>23}", "", "", "", ""),
                ),
            };
            println!(
                "  {:<25} {:>5} {:>10}  {} {}  {} | {} | {:>7} {:>7} | {}{}",
                tree.name,
                tree.log_sizes.len(),
                mib(bytes),
                cpu_ms,
                gpu_ms,
                speedup,
                split,
                if args.cpu { mib_short(cpu_peak) } else { "-".to_string() },
                if args.gpu { mib_short(gpu_peak) } else { "-".to_string() },
                &root.to_string()[..16],
                match (&cpu_tree, &gpu_tree, args.check_all) {
                    (Some(_), Some(_), true) => " equal (all)",
                    (Some(_), Some(_), false) => " equal",
                    _ => "",
                },
            );
            drop(gpu_tree);
            drop(cpu_tree);
        }
        println!(
            "  {:<25} {:>5} {:>10}  {} {}  {}",
            "total",
            "",
            "",
            if args.cpu { ms(cpu_total) } else { "       -".to_string() },
            if args.gpu { ms(gpu_total) } else { "       -".to_string() },
            if args.cpu && args.gpu {
                format!("{:7.1}x", cpu_total.as_secs_f64() / gpu_total.as_secs_f64())
            } else {
                "       -".to_string()
            },
        );
        totals.push((cpu_total, gpu_total, rss().0));
    }
    if let Some(committer) = &committer {
        let held = committer.held_bytes();
        println!(
            "held between commitments: device buffers {}, twiddles {} (device), upload ring {} \
             (host-visible), readback ring {} (host)",
            mib(held.buffers),
            mib(held.twiddles),
            mib(held.upload),
            mib(held.readback)
        );
        println!("keep-warm dispatches: {}", committer.keep_warm_dispatches());
    }
    println!(
        "summary, per proof (CPU new_striped total, GPU end-to-end total, the process's resident \
         set after it):"
    );
    for (proof, (cpu, gpu, rss)) in totals.iter().enumerate() {
        println!(
            "  proof {proof}: CPU {} ms, GPU {} ms, resident {}",
            ms(*cpu).trim(),
            ms(*gpu).trim(),
            mib(*rss)
        );
    }
    let (rss, peak) = rss();
    println!("resident set at the end {}, peak {}", mib(rss), mib(peak));
}

/// What each part of a committer adds to the resident set, printed.
fn rss_breakdown() {
    use stwo_gpu_spike::GpuContext;
    use stwo_gpu_spike::commit::CommitKernels;
    use stwo_gpu_spike::transfer::{ReadbackRing, Upload, UploadRing};
    let mut last = rss().0;
    let mut step = |what: &str| {
        let now = rss().0;
        println!("  resident set: {what}: +{} (now {})", mib(now.saturating_sub(last)), mib(now));
        last = now;
    };
    println!("resident set, a committer's parts:");
    step("before");
    let gpu = GpuContext::new();
    step("the device (driver, instance, adapter, device)");
    let kernels = CommitKernels::new(&gpu);
    step("the kernels compiled");
    let mut upload = UploadRing::with_defaults(&gpu);
    step("the upload ring (MAP_WRITE chunks, mapped)");
    let readback = ReadbackRing::with_defaults(&gpu);
    step("the readback ring (MAP_READ chunks)");
    let target = gpu.storage("target", (upload.bytes() / 4) as usize);
    let data = vec![7u8; upload.bytes() as usize];
    step("(the upload's source, a host buffer of the upload ring's size, written)");
    upload.upload(&gpu, &[Upload { dst: &target, dst_offset: 0, src: &data }]);
    gpu.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    step("the upload ring's chunks written");
    // Written (not left to calloc's untouched pages), so that the next step is the ring's alone.
    let mut back = vec![1u8; upload.bytes() as usize];
    step("(the readback's destination, a host buffer of that size, written)");
    let mut readback = readback;
    readback.read(
        &gpu,
        vec![stwo_gpu_spike::transfer::Readback { src: &target, src_offset: 0, dst: &mut back }],
    );
    step("the readback ring's chunks mapped and read");
    drop((data, back, target, upload, readback, kernels, gpu));
    step("all dropped");
}

/// Bytes in MiB, short.
fn mib_short(bytes: u64) -> String {
    format!("{:.0}M", bytes as f64 / (1 << 20) as f64)
}
