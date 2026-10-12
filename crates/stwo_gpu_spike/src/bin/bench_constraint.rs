//! The generated constraint evaluation kernels on captured fixtures (see
//! `stwo_gpu_spike::constraint::capture`): per component and kernel configuration, the times of
//! generating the WGSL, compiling the pipelines, uploading the inputs, the kernels on resident
//! inputs, and reading the accumulator back; the result checked word for word against the SIMD
//! output the fixture holds.
//!
//! ```text
//! cargo run --release --bin bench_constraint -- <fixtures dir> [--filter S] [--reps N]
//!     [--split 0,1,2,4] [--wg 64,256] [--unchecked] [--warmup N]
//! ```
//!
//! `--split 0` (the default) splits by row nodes (`KernelOptions::default`).
//!
//! The GPU comes from the environment (`WGPU_ADAPTER_NAME=llvmpipe` for lavapipe). The kernel
//! time is the median over `--reps` dispatches (after `--warmup` warm-up dispatches, default 3) of
//! the GPU's timestamps around the passes, and of the wall-clock time from submit to idle.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use stwo_gpu_spike::GpuContext;
use stwo_gpu_spike::constraint::fixture::Fixture;
use stwo_gpu_spike::constraint::kernel::{ConstraintKernels, KernelOptions};

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort();
    v[v.len() / 2]
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn main() {
    let mut args = std::env::args().skip(1);
    let root = PathBuf::from(args.next().expect("fixtures directory"));
    let (mut filter, mut reps, mut splits, mut wgs, mut unchecked, mut warmup) =
        (String::new(), 10usize, vec![0usize], vec![128u32], false, 3usize);
    while let Some(arg) = args.next() {
        let list = |s: String| s.split(',').map(|x| x.parse().unwrap()).collect::<Vec<_>>();
        match arg.as_str() {
            "--filter" => filter = args.next().unwrap(),
            "--reps" => reps = args.next().unwrap().parse().unwrap(),
            "--split" => splits = list(args.next().unwrap()),
            "--wg" => wgs = list(args.next().unwrap()).into_iter().map(|x| x as u32).collect(),
            "--unchecked" => unchecked = true,
            "--warmup" => warmup = args.next().unwrap().parse().unwrap(),
            other => panic!("unknown argument {other}"),
        }
    }
    let gpu = GpuContext::with_optional_features(wgpu::Features::TIMESTAMP_QUERY);
    println!("adapter: {} ({:?})", gpu.adapter_info.name, gpu.adapter_info.backend);
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(&root)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.join("meta.txt").exists() && p.to_string_lossy().contains(&filter))
        .collect();
    dirs.sort();
    println!(
        "{:<28} {:>5} {:>4} {:>5} {:>8} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>8}",
        "component",
        "split",
        "wg",
        "rows",
        "MiB in",
        "emit ms",
        "compile",
        "upload",
        "gpu ms",
        "wall ms",
        "readback",
        "identical"
    );
    for dir in dirs {
        let fixture = Fixture::load(&dir).unwrap();
        let name = dir.file_name().unwrap().to_string_lossy().to_string();
        let slices = fixture.column_slices();
        let uniforms = fixture.program.bind(&fixture.params).unwrap();
        let before: [&[u32]; 4] = std::array::from_fn(|c| fixture.before[c].as_slice());
        for &split in &splits {
            for &workgroup_size in &wgs {
                let options =
                    KernelOptions { workgroup_size, split, unchecked, ..Default::default() };
                let start = Instant::now();
                let kernels = ConstraintKernels::new(&gpu, &fixture.program, options, None);
                gpu.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
                let compile = start.elapsed();
                let (trace_log, eval_log) = (fixture.trace_log_size, fixture.eval_log_size);
                let start = Instant::now();
                let inputs = kernels.upload(
                    &gpu,
                    &slices,
                    &uniforms,
                    &fixture.coeffs,
                    &fixture.denom_inv,
                    &before,
                    trace_log,
                    eval_log,
                );
                let upload = start.elapsed();
                let bind_group = kernels.bind_inputs(&gpu, &inputs);
                // The result, from the fixture's accumulator.
                kernels.run(&gpu, &bind_group, eval_log);
                let start = Instant::now();
                let out = kernels.read_out(&gpu, &inputs);
                let readback = start.elapsed();
                let identical = (0..4).all(|c| out[c] == fixture.after[c]);
                for _ in 0..warmup {
                    kernels.run(&gpu, &bind_group, eval_log);
                }
                let (mut gpu_times, mut walls) = (vec![], vec![]);
                for _ in 0..reps {
                    let (gpu_time, wall) = kernels.run(&gpu, &bind_group, eval_log);
                    gpu_times.push(gpu_time.unwrap_or_default());
                    walls.push(wall);
                }
                println!(
                    "{:<28} {:>5} {:>4} {:>5} {:>8.0} {:>9.2} {:>9.1} {:>9.1} {:>9.3} {:>9.3} \
                     {:>9.1} {:>8}",
                    name,
                    kernels.kernels.len(),
                    workgroup_size,
                    format!("2^{}", fixture.eval_log_size),
                    inputs.bytes() as f64 / (1 << 20) as f64,
                    ms(kernels.emit_time),
                    ms(compile),
                    ms(upload),
                    ms(median(gpu_times)),
                    ms(median(walls)),
                    ms(readback),
                    if identical { "yes" } else { "NO" },
                );
            }
        }
    }
}
