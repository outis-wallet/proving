//! The whole striped commitment of the Cairo leg's base trace (the shapes of
//! `data/cairo_trace_shapes_pay.txt`: 2608 columns, 135.2 M coefficients, 516 MiB; random
//! coefficients), on the GPU against stwo's `CommitmentTreeProver::new_striped` (SimdBackend,
//! Blake2s): every stripe's extension and hashing, the levels above, at blowup 3 (8 stripes of
//! height 20, lifting log size 23).
//!
//! ```text
//! bench_commit [--scale N] [--blowup B] [--threads T1,T2,..] [--reps R] [--unchecked]
//!     [--no-cpu] [--no-gpu]
//! ```
//!
//! - CPU: `new_striped` on a rayon pool of each of `--threads` (default `18,36`: the physical
//!   cores, and all the hardware threads, of the i9-10980XE); the polynomials cloned and the result
//!   dropped outside the timed region, a fresh memory pool each run (as a proof's tree allocates
//!   its stripes).
//! - GPU, resident: the coefficients already on the GPU; every stripe and the levels above, one
//!   submission, waited for. The setup (buffers, every stripe's twiddles, the plans; once per tree
//!   shape) is timed apart.
//! - GPU, with upload: the coefficients' upload (`queue.write_buffer`), then the commitment.
//! - GPU, with upload and readback: then the copies of stripe 0, of the columns kept whole and of
//!   the layers into staging buffers (same submission), mapped and copied into stwo's columns: the
//!   `CommitmentTreeProver` `new_striped` returns.
//! - The split of each stripe between its extension and its hashing: from timestamp queries if the
//!   adapter has them (`TIMESTAMP_QUERY`), and from separate submissions (each pass submitted and
//!   waited for on its own: the submission's overhead included).
//!
//! The GPU's root is checked against every CPU run's, and its whole result (every layer, every
//! column's kept evaluations) against the last CPU run's.
//!
//! `--scale N` divides every component's rows by 2^N (columns of 2^4 at least; the tree 2^N
//! times lower). `--blowup B` sets the log blowup (the tree's lifting log size is the largest
//! column's plus B). `--unchecked` uses the Merkle kernels without wgpu's runtime checks
//! (native only). The adapter comes from the environment: `WGPU_BACKEND=vulkan
//! WGPU_ADAPTER_NAME=nvidia` for the RTX 3080, `WGPU_ADAPTER_NAME=llvmpipe` for lavapipe.

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
use stwo_gpu_spike::GpuContext;
use stwo_gpu_spike::commit::{
    CommitKernels, PassTimestamps, StripedCommitment, kept_whole, stripe_order, takes_striped_path,
};
use stwo_gpu_spike::shapes::{base_trace_log_sizes, cairo_pay_components};

type TreeProver = CommitmentTreeProver<SimdBackend, Blake2sM31MerkleChannel>;

struct Args {
    scale: u32,
    log_blowup: u32,
    threads: Vec<usize>,
    reps: usize,
    unchecked: bool,
    cpu: bool,
    gpu: bool,
}

fn parse_args() -> Args {
    let mut args = Args {
        scale: 0,
        log_blowup: 3,
        threads: vec![18, 36],
        reps: 3,
        unchecked: false,
        cpu: true,
        gpu: true,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut value = |name: &str| it.next().unwrap_or_else(|| panic!("{name} needs a value"));
        match arg.as_str() {
            "--scale" => args.scale = value("--scale").parse().expect("--scale N"),
            "--blowup" => args.log_blowup = value("--blowup").parse().expect("--blowup B"),
            "--threads" => {
                args.threads = value("--threads")
                    .split(',')
                    .map(|t| t.trim().parse().expect("--threads T1,T2,.."))
                    .collect()
            }
            "--reps" => args.reps = value("--reps").parse().expect("--reps R"),
            "--unchecked" => args.unchecked = true,
            "--no-cpu" => args.cpu = false,
            "--no-gpu" => args.gpu = false,
            "-h" | "--help" => {
                println!(
                    "bench_commit [--scale N] [--blowup B] [--threads T1,T2,..] [--reps R] \
                     [--unchecked] [--no-cpu] [--no-gpu]"
                );
                std::process::exit(0);
            }
            other => panic!("unknown argument {other}"),
        }
    }
    assert!(args.reps >= 1);
    args
}

fn ms(d: Duration) -> String {
    format!("{:9.2} ms", d.as_secs_f64() * 1e3)
}

fn ms_ns(ns: f64) -> String {
    format!("{:8.2} ms", ns / 1e6)
}

fn mib(bytes: u64) -> String {
    format!("{:.1} MiB", bytes as f64 / (1 << 20) as f64)
}

/// The best and the median of `reps` runs of `f`, which returns the time of its timed part.
fn time(reps: usize, mut f: impl FnMut() -> Duration) -> (Duration, Duration) {
    let mut times = (0..reps).map(|_| f()).collect_vec();
    times.sort();
    (times[0], times[times.len() / 2])
}

/// The time `f` takes.
fn timed(f: impl FnOnce()) -> Duration {
    let start = Instant::now();
    f();
    start.elapsed()
}

fn main() {
    let args = parse_args();
    let log_sizes = base_trace_log_sizes(&cairo_pay_components(), args.scale, 4);
    let log_blowup = args.log_blowup;
    let lifting_log_size = log_sizes.iter().max().unwrap() + log_blowup;
    let n_stripes = 1usize << log_blowup;
    let cells: u64 = log_sizes.iter().map(|&l| 1u64 << l).sum();
    let whole_cells: u64 =
        log_sizes.iter().filter(|&&l| kept_whole(l, log_blowup)).map(|&l| 1u64 << l).sum();
    println!(
        "Cairo base trace, rows / 2^{}: {} columns in {} sizes, {:.1} M coefficients ({}); blowup \
         {log_blowup}: {n_stripes} stripes of height {}, lifting log size {lifting_log_size}; {} \
         columns kept whole ({:.2} M coefficients)",
        args.scale,
        log_sizes.len(),
        log_sizes.iter().unique().count(),
        cells as f64 / 1e6,
        mib(cells * 4),
        lifting_log_size - log_blowup,
        log_sizes.iter().filter(|&&l| kept_whole(l, log_blowup)).count(),
        whole_cells as f64 / 1e6,
    );
    assert!(
        takes_striped_path(&log_sizes, log_blowup, lifting_log_size),
        "new_striped would not stripe these shapes"
    );

    let start = Instant::now();
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
    println!("random coefficients and twiddle tree: {}", ms(start.elapsed()).trim());

    // CPU.
    let mut cpu_tree: Option<TreeProver> = None;
    let mut cpu_roots = vec![];
    if args.cpu {
        for &threads in &args.threads {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            let commit = |polys: Vec<CircleCoefficients<SimdBackend>>| {
                let memory = BaseColumnPool::<SimdBackend>::new();
                pool.install(|| {
                    TreeProver::new_striped(polys, log_blowup, &twiddles, lifting_log_size, &memory)
                })
            };
            // Warm-up.
            drop(commit(polys.clone()));
            let mut last = None;
            let (best, median) = time(args.reps, || {
                drop(last.take());
                let polys = polys.clone();
                timed(|| last = Some(commit(polys)))
            });
            let tree = last.unwrap();
            println!(
                "CPU new_striped, {threads:2} threads:            best {}  median {}",
                ms(best),
                ms(median)
            );
            cpu_roots.push((threads, tree.commitment.root()));
            cpu_tree = Some(tree);
        }
    }

    if !args.gpu {
        return;
    }
    let gpu = GpuContext::with_optional_features(wgpu::Features::TIMESTAMP_QUERY);
    let info = &gpu.adapter_info;
    println!(
        "GPU: {} ({:?}, {:?}, driver {} {}); timestamp queries: {}",
        info.name,
        info.backend,
        info.device_type,
        info.driver,
        info.driver_info,
        gpu.device.features().contains(wgpu::Features::TIMESTAMP_QUERY),
    );
    let kernels = if args.unchecked {
        // Safety: the commitment sizes and checks every buffer it binds.
        unsafe { CommitKernels::new_unchecked(&gpu) }
    } else {
        CommitKernels::new(&gpu)
    };
    let start = Instant::now();
    let commitment =
        StripedCommitment::new(&gpu, &kernels, &log_sizes, log_blowup, lifting_log_size, &twiddles);
    gpu.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let setup = start.elapsed();
    let bytes = commitment.gpu_bytes();
    let (ext_dispatches, hash_dispatches, top_dispatches) = commitment.n_dispatches();
    println!(
        "GPU setup (buffers, every stripe's twiddles, plans; once per tree shape): {}",
        ms(setup).trim()
    );
    println!(
        "GPU layout: {} buffer(s), {} batches; per stripe {ext_dispatches} extension and \
         {hash_dispatches} hashing dispatches, {top_dispatches} above the stripes{}",
        commitment.layout().buffer_words.len(),
        commitment.layout().batches.len(),
        if args.unchecked { "; unchecked Merkle kernels" } else { "" },
    );
    println!(
        "GPU memory: {} on the device (coefficients {}, one stripe's evaluations {}, twiddles {}, \
         Merkle states {}, layers {}, columns kept whole {}) + {} of staging for the readback",
        mib(bytes.device()),
        mib(bytes.coefficients),
        mib(bytes.evaluations),
        mib(bytes.twiddles),
        mib(bytes.states),
        mib(bytes.tree),
        mib(bytes.whole),
        mib(bytes.staging),
    );

    // Warm-up: pipelines, first touch of every buffer.
    commitment.upload(&gpu, &polys);
    let _ = commitment.run_and_read(&gpu, &kernels);

    commitment.upload(&gpu, &polys);
    gpu.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let (best, median) = time(args.reps, || timed(|| commitment.run(&gpu, &kernels)));
    println!("GPU, coefficients resident:           best {}  median {}", ms(best), ms(median));

    let (best, median) = time(args.reps, || {
        timed(|| {
            commitment.upload(&gpu, &polys);
            commitment.run(&gpu, &kernels);
        })
    });
    println!("GPU, with upload:                     best {}  median {}", ms(best), ms(median));

    let mut output = None;
    let (best, median) = time(args.reps, || {
        drop(output.take());
        timed(|| {
            commitment.upload(&gpu, &polys);
            output = Some(commitment.run_and_read(&gpu, &kernels));
        })
    });
    println!("GPU, with upload and readback:        best {}  median {}", ms(best), ms(median));
    let start = Instant::now();
    let gpu_tree = commitment.tree_prover(output.take().unwrap());
    let to_tree = start.elapsed();

    // The split of each stripe: timestamps.
    if let Some(timestamps) = PassTimestamps::new(&gpu, log_blowup) {
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        commitment.encode(&kernels, &mut encoder, Some(&timestamps));
        gpu.queue.submit([encoder.finish()]);
        let times = timestamps.read(&gpu);
        println!("GPU passes, timestamp queries (coefficients resident):");
        for &(stripe, extension, hashing) in &times.stripes {
            println!(
                "  stripe {stripe}: extension {}  hashing {}",
                ms_ns(extension),
                ms_ns(hashing)
            );
        }
        let extension: f64 = times.stripes.iter().map(|s| s.1).sum();
        let hashing: f64 = times.stripes.iter().map(|s| s.2).sum();
        println!(
            "  all stripes: extension {}  hashing {}; above the stripes {}; first pass start to \
             last pass end {}",
            ms_ns(extension),
            ms_ns(hashing),
            ms_ns(times.top),
            ms_ns(times.total),
        );
    } else {
        println!("GPU passes, timestamp queries: not supported by the adapter");
    }

    // The split of each stripe: separate submissions.
    let wait = || gpu.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let submit_and_time = |record: &dyn Fn(&mut wgpu::CommandEncoder)| {
        let start = Instant::now();
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        record(&mut encoder);
        gpu.queue.submit([encoder.finish()]);
        wait();
        start.elapsed()
    };
    println!("GPU passes, separate submissions (each waited for; coefficients resident):");
    let (mut extension, mut hashing) = (Duration::ZERO, Duration::ZERO);
    for stripe in stripe_order(log_blowup) {
        let e = submit_and_time(&|encoder| {
            commitment.encode_extension(&kernels, encoder, stripe, None)
        });
        let h = submit_and_time(&|encoder| commitment.encode_hashing(encoder, stripe, None));
        println!("  stripe {stripe}: extension {}  hashing {}", ms(e), ms(h));
        extension += e;
        hashing += h;
    }
    let top = submit_and_time(&|encoder| commitment.encode_top(encoder, None));
    println!(
        "  all stripes: extension {}  hashing {}; above the stripes {}",
        ms(extension),
        ms(hashing),
        ms(top)
    );
    println!(
        "  (the readback's CommitmentTreeProver assembly, outside the GPU timings above: {})",
        ms(to_tree).trim()
    );

    // Checks.
    let gpu_root = gpu_tree.commitment.root();
    println!("GPU root {gpu_root}");
    for (threads, root) in &cpu_roots {
        assert_eq!(*root, gpu_root, "the GPU's root is not the CPU's ({threads} threads)");
    }
    if let Some(cpu) = &cpu_tree {
        assert_eq!(gpu_tree.commitment.layers, cpu.commitment.layers, "layers differ");
        for (i, (g, c)) in gpu_tree.polynomials.iter().zip(&cpu.polynomials).enumerate() {
            assert_eq!(g.evals.domain, c.evals.domain, "column {i}: domain");
            assert!(
                g.evals.values.as_slice() == c.evals.values.as_slice(),
                "column {i}: kept evaluations differ"
            );
        }
        println!(
            "check: roots equal ({} CPU runs), every layer equal, every column's kept evaluations \
             equal (stripe 0, or whole)",
            cpu_roots.len()
        );
    }
}
