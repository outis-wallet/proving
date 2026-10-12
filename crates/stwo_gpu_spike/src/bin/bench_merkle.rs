//! The Merkle hashing of the Cairo base trace's commitment, every stripe, on the GPU against
//! stwo's SIMD backend.
//!
//! The columns have the Cairo leg's shapes (`data/cairo_trace_shapes_pay.txt`: 2608 base
//! columns, 135.2 M values, 516 MiB per stripe at full size) with random values; the tree has 8
//! stripes of height 20 (blowup 3, lifting log size 23). The same random stripe is hashed as each
//! of the 8 stripes, on both sides.
//!
//! - GPU, resident: the stripe's columns already on the GPU (as the FFT would leave them), 8
//!   stripes and the levels above recorded into one compute pass, submitted, waited for.
//! - GPU, with upload: each stripe's 516 MiB written from host memory (`queue.write_buffer`) and
//!   then hashed, one stripe after the other (no overlap of the uploads with the hashing).
//! - CPU: `SimdBackend::build_layers_sparse` per stripe and `build_next_layer` above, as
//!   `CommitmentTreeProver::new_striped` hashes, on rayon's default pool and on `--threads`.
//!
//! - `--unchecked`: also the GPU, inputs resident, with kernels without wgpu's bounds checks and
//!   loop bounding (native only: browsers always keep them), to see what the checks cost.
//!
//! Usage: `bench_merkle [--scale N] [--threads T] [--reps R] [--unchecked] [--no-cpu] [--no-gpu]`;
//! `--scale N` shrinks every component's rows by 2^N (columns no smaller than 2^4, the tree 2^N
//! times lower).
//! The adapter comes from the environment (`WGPU_ADAPTER_NAME`, `WGPU_BACKEND`).
use std::time::{Duration, Instant};

use rand::SeedableRng;
use rand::rngs::SmallRng;
use rayon::prelude::*;
use stwo::core::vcs::blake2_hash::Blake2sHash;
use stwo_gpu_spike::GpuContext;
use stwo_gpu_spike::merkle::{GpuMerkle, Plan, TreeLayout, stwo_min_stored_level};
use stwo_gpu_spike::reference::{
    UploadedStripe, random_values, simd_columns, simd_stripe_layers, simd_tree_layers,
};
use stwo_gpu_spike::shapes::{
    CAIRO_LIFTING_LOG_SIZE, CAIRO_LOG_BLOWUP, base_trace_log_sizes, cairo_pay_components,
    size_batches, tree_order,
};

struct Args {
    scale: u32,
    unchecked: bool,
    threads: Option<usize>,
    reps: usize,
    cpu: bool,
    gpu: bool,
}

fn parse_args() -> Args {
    let mut args =
        Args { scale: 0, unchecked: false, threads: None, reps: 3, cpu: true, gpu: true };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut value = |name: &str| it.next().unwrap_or_else(|| panic!("{name} needs a value"));
        match arg.as_str() {
            "--scale" => args.scale = value("--scale").parse().expect("--scale N"),
            "--threads" => args.threads = Some(value("--threads").parse().expect("--threads T")),
            "--reps" => args.reps = value("--reps").parse().expect("--reps R"),
            "--no-cpu" => args.cpu = false,
            "--unchecked" => args.unchecked = true,
            "--no-gpu" => args.gpu = false,
            "-h" | "--help" => {
                println!(
                    "bench_merkle [--scale N] [--threads T] [--reps R] [--unchecked] [--no-cpu] \
                     [--no-gpu]"
                );
                std::process::exit(0);
            }
            other => panic!("unknown argument {other}"),
        }
    }
    args
}

fn ms(d: Duration) -> String {
    format!("{:9.2} ms", d.as_secs_f64() * 1e3)
}

/// The best and the median of `reps` runs of `f`.
fn time(reps: usize, mut f: impl FnMut()) -> (Duration, Duration) {
    let mut times: Vec<Duration> = (0..reps.max(1))
        .map(|_| {
            let start = Instant::now();
            f();
            start.elapsed()
        })
        .collect();
    times.sort();
    (times[0], times[times.len() / 2])
}

fn main() {
    let args = parse_args();
    let log_sizes = base_trace_log_sizes(&cairo_pay_components(), args.scale, 4);
    let sorted: Vec<u32> = tree_order(&log_sizes).into_iter().map(|i| log_sizes[i]).collect();
    let height = CAIRO_LIFTING_LOG_SIZE - CAIRO_LOG_BLOWUP - args.scale;
    let log_n_stripes = CAIRO_LOG_BLOWUP;
    let n_stripes = 1usize << log_n_stripes;
    let cells: u64 = sorted.iter().map(|&k| 1u64 << k).sum();
    let stripe_mib = cells as f64 * 4.0 / (1 << 20) as f64;
    println!(
        "Cairo base trace, scale 2^-{}: {} columns in {} sizes, {:.1} M values ({stripe_mib:.1} \
         MiB) per stripe, {n_stripes} stripes of height {height}",
        args.scale,
        sorted.len(),
        size_batches(&sorted).len(),
        cells as f64 / 1e6,
    );

    let start = Instant::now();
    let columns: Vec<Vec<u32>> = sorted
        .par_iter()
        .enumerate()
        .map(|(i, &log_size)| random_values(&mut SmallRng::seed_from_u64(i as u64), 1 << log_size))
        .collect();
    println!("random columns: {}", ms(start.elapsed()));

    let mut gpu_root: Option<Blake2sHash> = None;
    if args.gpu {
        let gpu = GpuContext::new();
        let info = &gpu.adapter_info;
        println!(
            "GPU: {} ({:?}, {:?}, driver {} {})",
            info.name, info.backend, info.device_type, info.driver, info.driver_info
        );
        let merkle = GpuMerkle::new(&gpu);
        let start = Instant::now();
        let uploaded = UploadedStripe::new(&gpu, &columns);
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        println!("first upload (buffer creation + copy): {}", ms(start.elapsed()));
        let layout = TreeLayout::new(height, log_n_stripes, stwo_min_stored_level(&sorted, height));
        let tree = merkle.tree_buffer(&gpu, &layout);
        let states = merkle.state_buffers(&gpu, &uploaded.shapes, height);
        let batches = uploaded.batches();
        let batch_words = uploaded.batch_words(&columns);
        let start = Instant::now();
        let stripe_plans: Vec<Plan> = (0..n_stripes as u32)
            .map(|stripe| merkle.plan_stripe(&gpu, &batches, &states, &tree, &layout, stripe))
            .collect();
        let top = merkle.plan_top(&gpu, &tree, &layout);
        println!(
            "plans: {} dispatches per stripe, {} above the stripes, states 2 x {:.1} MiB, layers \
             {:.1} MiB ({})",
            stripe_plans[0].n_dispatches(),
            top.n_dispatches(),
            states.words as f64 * 4.0 / (1 << 20) as f64,
            layout.total_words as f64 * 4.0 / (1 << 20) as f64,
            ms(start.elapsed()).trim(),
        );

        let resident = || {
            let mut plans: Vec<&Plan> = stripe_plans.iter().collect();
            plans.push(&top);
            gpu.run_plans(&plans);
        };
        resident(); // Warm-up: pipelines, first touch of the buffers.
        let (best, median) = time(args.reps, resident);
        println!(
            "GPU, inputs resident:  best {}  median {}  ({:.1} GB/s of columns)",
            ms(best),
            ms(median),
            (n_stripes as f64 * cells as f64 * 4.0) / best.as_secs_f64() / 1e9
        );

        let with_upload = || {
            for (stripe, plan) in stripe_plans.iter().enumerate() {
                uploaded.write(&gpu, &batch_words);
                if stripe == n_stripes - 1 {
                    gpu.run_plans(&[plan, &top]);
                } else {
                    gpu.run_plans(&[plan]);
                }
            }
        };
        let (best, median) = time(args.reps, with_upload);
        println!("GPU, with upload:      best {}  median {}", ms(best), ms(median));

        if args.unchecked {
            // Safety: the planner sizes and checks every buffer it binds.
            let unchecked = unsafe { GpuMerkle::new_unchecked(&gpu) };
            let stripe_plans: Vec<Plan> = (0..n_stripes as u32)
                .map(|stripe| {
                    unchecked.plan_stripe(&gpu, &batches, &states, &tree, &layout, stripe)
                })
                .collect();
            let top = unchecked.plan_top(&gpu, &tree, &layout);
            let resident = || {
                let mut plans: Vec<&Plan> = stripe_plans.iter().collect();
                plans.push(&top);
                gpu.run_plans(&plans);
            };
            resident();
            let (best, median) = time(args.reps, resident);
            println!(
                "GPU, unchecked kernels, inputs resident:  best {}  median {}",
                ms(best),
                ms(median)
            );
        }

        let start = Instant::now();
        let levels = merkle.read_levels(&gpu, &tree, &layout);
        println!(
            "read back the stored layers ({:.1} MiB): {}",
            layout.total_words as f64 * 4.0 / (1 << 20) as f64,
            ms(start.elapsed())
        );
        gpu_root = Some(levels.last().unwrap()[0]);
        println!("GPU root {}", gpu_root.unwrap());
    }

    if args.cpu {
        let start = Instant::now();
        let simd = simd_columns(&columns);
        println!("SIMD columns: {}", ms(start.elapsed()));
        let cpu_tree = || {
            let subtrees: Vec<_> =
                (0..n_stripes).map(|_| simd_stripe_layers(&simd, height)).collect();
            simd_tree_layers(&subtrees, height).last().unwrap()[0]
        };
        let mut pools = vec![(rayon::current_num_threads(), None)];
        if let Some(threads) = args.threads {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            pools.push((threads, Some(pool)));
        }
        for (threads, pool) in pools {
            let run = || match &pool {
                Some(pool) => pool.install(cpu_tree),
                None => cpu_tree(),
            };
            let root = run();
            let (best, median) = time(args.reps, || {
                run();
            });
            println!("CPU SIMD, {threads:2} threads:  best {}  median {}", ms(best), ms(median));
            if let Some(gpu_root) = gpu_root {
                assert_eq!(root, gpu_root, "the GPU's root is not the CPU's");
            }
            println!("CPU root {root}");
        }
        if gpu_root.is_some() {
            println!("roots equal");
        }
    }
}
