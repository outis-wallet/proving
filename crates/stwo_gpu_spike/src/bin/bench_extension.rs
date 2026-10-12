//! The polynomial extension of the Cairo leg's base trace (the shapes of
//! `data/cairo_trace_shapes_pay.txt`, random coefficients): every stripe of every column, on the
//! GPU and with stwo's SIMD backend as `CommitmentTreeProver::new_striped` evaluates them (stripe 0
//! last; no Merkle hashing on either side).
//!
//! ```text
//! cargo run --release --bin bench_extension -- [--scale S] [--threads T] [--blowup B]
//!     [--iters N] [--data FILE] [--no-cpu]
//! ```
//!
//! `--scale S` divides every component's rows by S (a power of two; columns stay at least 2^4),
//! for slow adapters. The GPU comes from the environment (`WGPU_ADAPTER_NAME=NVIDIA`,
//! `WGPU_ADAPTER_NAME=llvmpipe`, `WGPU_BACKEND=vulkan`). Timings are the best of `--iters`, after
//! one warm-up run:
//! - CPU: SimdBackend, part_twiddles and evaluate_into, rayon (`--threads`, default all cores);
//! - GPU excluding uploads: the coefficients and twiddles uploaded once per tree beforehand;
//! - GPU including uploads: the coefficients' and twiddles' uploads, then every stripe;
//! - GPU including uploads and the readback of stripe 0 (the stripe the CPU keeps).
//!
//! Stripe 0 read back from the GPU is checked against the CPU's, word for word.

use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use stwo::core::fields::m31::{BaseField, P};
use stwo::core::poly::circle::{CanonicCoset, CircleDomain};
use stwo::prover::backend::Column;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::simd::column::BaseColumn;
use stwo::prover::poly::circle::{CircleCoefficients, PolyOps, part_domain};
use stwo::prover::poly::twiddles::TwiddleTree;
use stwo_gpu_spike::GpuContext;
use stwo_gpu_spike::extension::{
    ColumnsLayout, ExtensionKernels, GpuColumns, MIN_LOG_SIZE, StripeExtension, max_buffer_words,
    stripe_twiddle_words,
};

struct Args {
    data: String,
    scale: u32,
    threads: Option<usize>,
    log_blowup: u32,
    iters: usize,
    cpu: bool,
}

fn parse_args() -> Args {
    let mut args = Args {
        data: concat!(env!("CARGO_MANIFEST_DIR"), "/data/cairo_trace_shapes_pay.txt").to_string(),
        scale: 1,
        threads: None,
        log_blowup: 3,
        iters: 3,
        cpu: true,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut value = || it.next().unwrap_or_else(|| panic!("{arg} takes a value"));
        match arg.as_str() {
            "--data" => args.data = value(),
            "--scale" => args.scale = value().parse().expect("--scale"),
            "--threads" => args.threads = Some(value().parse().expect("--threads")),
            "--blowup" => args.log_blowup = value().parse().expect("--blowup"),
            "--iters" => args.iters = value().parse().expect("--iters"),
            "--no-cpu" => args.cpu = false,
            _ => panic!("unknown argument {arg}"),
        }
    }
    assert!(args.scale.is_power_of_two(), "--scale must be a power of two");
    assert!(args.iters >= 1);
    args
}

/// The base trace's column log sizes: per component, its base columns at its log rows.
fn column_log_sizes(path: &str, scale: u32) -> Vec<u32> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut log_sizes = vec![];
    for line in text.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('#')) {
        let fields = line.split_whitespace().collect::<Vec<_>>();
        assert!(fields.len() >= 3, "bad line: {line}");
        let log_rows: u32 = fields[1].parse().expect("log rows");
        let columns: usize = fields[2].parse().expect("base columns");
        let log_size = log_rows.saturating_sub(scale.ilog2()).max(MIN_LOG_SIZE);
        log_sizes.extend(std::iter::repeat_n(log_size, columns));
    }
    log_sizes
}

fn best(times: &[Duration]) -> Duration {
    *times.iter().min().unwrap()
}

fn ms(d: Duration) -> String {
    format!("{:.1} ms", d.as_secs_f64() * 1e3)
}

/// stwo's extension as `new_striped` runs it (without the hashing): per stripe, stripe 0 last,
/// the stripe's twiddles per domain size and every column evaluated into its buffer (reused from
/// stripe to stripe). Returns the buffers holding stripe 0.
fn cpu_extension(
    polys: &[CircleCoefficients<SimdBackend>],
    log_blowup: u32,
    tree: &TwiddleTree<SimdBackend>,
    mut buffers: Vec<BaseColumn>,
) -> Vec<BaseColumn> {
    let domains = polys
        .iter()
        .map(|poly| CanonicCoset::new(poly.log_size() + log_blowup).circle_domain())
        .collect::<Vec<CircleDomain>>();
    let domain_log_sizes = domains.iter().map(|d| d.log_size()).collect::<BTreeSet<_>>();
    for stripe in (1..1 << log_blowup).chain([0]) {
        let twiddles = domain_log_sizes
            .iter()
            .map(|&log_size| {
                let domain = CanonicCoset::new(log_size).circle_domain();
                (log_size, SimdBackend::part_twiddles(domain, log_blowup, stripe, tree))
            })
            .collect::<std::collections::HashMap<_, _>>();
        buffers = polys
            .par_iter()
            .zip(domains.par_iter())
            .zip(buffers.into_par_iter())
            .map(|((poly, domain), buffer)| {
                let stripe_domain = part_domain(*domain, log_blowup, stripe);
                SimdBackend::evaluate_into(
                    poly,
                    stripe_domain,
                    &twiddles[&domain.log_size()],
                    buffer,
                )
                .values
            })
            .collect();
    }
    buffers
}

fn main() {
    let args = parse_args();
    let log_sizes = column_log_sizes(&args.data, args.scale);
    let n_cells: u64 = log_sizes.iter().map(|&l| 1u64 << l).sum();
    let n_stripes = 1usize << args.log_blowup;
    let distinct = log_sizes.iter().copied().collect::<BTreeSet<_>>();
    println!(
        "shapes: {} at 1/{} scale: {} columns, {} cells ({:.1} MiB), log sizes {:?}; blowup {}, \
         {} stripes",
        args.data,
        args.scale,
        log_sizes.len(),
        n_cells,
        n_cells as f64 * 4.0 / (1 << 20) as f64,
        distinct,
        args.log_blowup,
        n_stripes,
    );

    // Random coefficients, as stwo stores them.
    let polys = log_sizes
        .par_iter()
        .enumerate()
        .map(|(i, &log_size)| {
            let mut rng = SmallRng::seed_from_u64(i as u64);
            CircleCoefficients::<SimdBackend>::new(
                (0..1 << log_size)
                    .map(|_| BaseField::from_u32_unchecked(rng.random_range(0..P)))
                    .collect(),
            )
        })
        .collect::<Vec<_>>();
    let words = polys
        .iter()
        .map(|poly| bytemuck::cast_slice::<_, u32>(poly.coeffs.as_slice()))
        .collect::<Vec<_>>();
    let max_log_size = distinct.last().unwrap() + args.log_blowup;
    let tree = SimdBackend::precompute_twiddles(
        CanonicCoset::new(max_log_size).circle_domain().half_coset,
    );

    // CPU.
    let pool = {
        let mut builder = rayon::ThreadPoolBuilder::new();
        if let Some(threads) = args.threads {
            builder = builder.num_threads(threads);
        }
        builder.build().expect("thread pool")
    };
    let mut cpu_stripe0 = None;
    if args.cpu {
        let (times, stripe0) = pool.install(|| {
            let mut buffers =
                log_sizes.iter().map(|&l| BaseColumn::zeros(1 << l)).collect::<Vec<_>>();
            buffers = cpu_extension(&polys, args.log_blowup, &tree, buffers);
            let times = (0..args.iters)
                .map(|_| {
                    let start = Instant::now();
                    buffers =
                        cpu_extension(&polys, args.log_blowup, &tree, std::mem::take(&mut buffers));
                    start.elapsed()
                })
                .collect::<Vec<_>>();
            (times, buffers)
        });
        println!(
            "CPU SimdBackend ({} threads): {} for {} stripes ({} per stripe)",
            pool.current_num_threads(),
            ms(best(&times)),
            n_stripes,
            ms(best(&times) / n_stripes as u32),
        );
        cpu_stripe0 = Some(stripe0);
    }

    // GPU.
    let gpu = GpuContext::new();
    println!(
        "adapter: {} ({:?}, {:?})",
        gpu.adapter_info.name, gpu.adapter_info.backend, gpu.adapter_info.device_type
    );
    let kernels = ExtensionKernels::new(&gpu);
    let layout = Arc::new(ColumnsLayout::new(&log_sizes, max_buffer_words(&gpu)));
    let coeffs = GpuColumns::alloc(&gpu, layout.clone());
    let out = GpuColumns::alloc(&gpu, layout.clone());
    let staging = out.staging(&gpu);

    let start = Instant::now();
    let twiddles = (0..n_stripes)
        .into_par_iter()
        .map(|stripe| stripe_twiddle_words(&distinct, args.log_blowup, stripe, &tree))
        .collect::<Vec<_>>();
    let twiddle_prep = start.elapsed();
    let extension = StripeExtension::with_twiddle_words(
        &gpu,
        &kernels,
        &coeffs,
        &out,
        args.log_blowup,
        &twiddles,
    );
    println!(
        "GPU layout: {} buffer(s), {} batches, {} dispatches per stripe; twiddles {:.1} MiB (all \
         stripes, prepared on the host in {})",
        layout.buffer_words.len(),
        layout.batches.len(),
        extension.n_dispatches(),
        extension.twiddle_bytes() as f64 / (1 << 20) as f64,
        ms(twiddle_prep),
    );
    let wait = || {
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    };
    let upload = || {
        coeffs.write(&gpu, &words);
        extension.write_twiddles(&gpu, &twiddles);
    };
    let stripes = || {
        for stripe in (1..n_stripes).chain([0]) {
            extension.submit(&gpu, &kernels, stripe);
        }
    };

    // Warm-up.
    upload();
    stripes();
    wait();

    let mut excluding = vec![];
    let mut including = vec![];
    let mut with_readback = vec![];
    let mut stripe0 = vec![];
    for _ in 0..args.iters {
        upload();
        wait();
        let start = Instant::now();
        stripes();
        wait();
        excluding.push(start.elapsed());

        let start = Instant::now();
        upload();
        stripes();
        wait();
        including.push(start.elapsed());

        let start = Instant::now();
        upload();
        stripes();
        stripe0 = out.read_into(&gpu, &staging);
        with_readback.push(start.elapsed());
    }
    let per_stripe = |d: Duration| ms(d / n_stripes as u32);
    println!(
        "GPU excluding uploads: {} for {} stripes ({} per stripe)",
        ms(best(&excluding)),
        n_stripes,
        per_stripe(best(&excluding))
    );
    println!("GPU including uploads: {}", ms(best(&including)));
    println!("GPU including uploads and stripe 0 readback: {}", ms(best(&with_readback)));

    if let Some(cpu) = cpu_stripe0 {
        let mismatches = cpu
            .iter()
            .zip(&stripe0)
            .filter(|(cpu, gpu)| bytemuck::cast_slice::<_, u32>(cpu.as_slice()) != gpu.as_slice())
            .count();
        assert_eq!(mismatches, 0, "stripe 0: {mismatches} columns differ from the CPU's");
        println!("check: stripe 0, GPU == CPU word for word ({n_cells} words)");
    }
}
