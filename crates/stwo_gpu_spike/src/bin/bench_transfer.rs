//! Where the time of moving a tree's columns between the host and the GPU goes, and what
//! overlaps with what (the Cairo base trace's coefficients: 2608 columns, 516 MiB).
//!
//! ```text
//! bench_transfer [--reps R] [--hog-bar MiB]
//! ```
//!
//! - Upload: `queue.write_buffer_with` per size batch (the first version's upload: a staging buffer
//!   allocated per call), against the [`UploadRing`] (persistent mapped chunks, filled by every
//!   thread), at several chunk sizes and depths, and on one thread.
//! - Readback: the [`ReadbackRing`] into fresh stwo columns, at several chunk sizes and depths.
//! - Overlap, with the commitment's compute (every stripe, coefficients resident): a copy from a
//!   buffer it does not use to a `MAP_READ` buffer, recorded before or after it, in one submission
//!   or two, or before the extension alone (no copies of its own, so no barrier against the copy);
//!   and the CPU's writes into a mapped chunk while it runs.
//!
//! `--hog-bar MiB` first takes that much `MAP_WRITE` memory in 8 MiB buffers, so that the rings'
//! chunks land in host memory rather than in the BAR (wgpu's allocator prefers device-local
//! mapped memory for `MAP_WRITE` buffers, 256 MiB of it on a card without resizable BAR).
//!
//! What it showed on the RTX 3080 (PCIe 3.0 x16, i9-10980XE, Vulkan):
//! - `write_buffer_with` per batch: 225-233 ms (2.3 GB/s), 160 ms of it on the CPU (the allocations
//!   of mapped memory, the copies), 66 ms on the GPU.
//! - The ring: 53-58 ms (9.4-10.2 GB/s) for chunks of 8 to 64 MiB, two of them (32 MiB best); 58 ms
//!   on one thread. In the BAR, the CPU's writes cross PCIe themselves (filling is the whole time,
//!   the GPU's copy 1.9 ms); with the BAR full, the chunks are in host memory (filling 27 ms) and
//!   the GPU's copy crosses PCIe (44 ms): the same total.
//! - Readback ring: 48 ms (11.3 GB/s) with 16 or 32 MiB chunks; one staging buffer of the whole, 62
//!   ms. The device-to-host copy alone: 42 ms (13 GB/s).
//! - Copies and compute do not overlap on wgpu's one queue: 42 ms of copy and 62 ms of compute take
//!   105 ms in either order, in one submission or two, and the copy and the extension alone (50 ms)
//!   92 ms. So reading stripe 0 back while the other stripes compute gains nothing.
//! - The CPU's writes into a mapped chunk do overlap the GPU's compute: 516 MiB written into the
//!   BAR while the commitment computes take 65 ms, the compute's time.

use std::time::{Duration, Instant};

use itertools::Itertools;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use stwo::core::fields::m31::{BaseField, P};
use stwo::core::poly::circle::CanonicCoset;
use stwo::prover::backend::Column;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::simd::column::BaseColumn;
use stwo::prover::poly::circle::{CircleCoefficients, PolyOps};
use stwo_gpu_spike::GpuContext;
use stwo_gpu_spike::commit::{CommitKernels, StripedCommitment};
use stwo_gpu_spike::extension::{ColumnsLayout, GpuColumns, max_buffer_words};
use stwo_gpu_spike::shapes::{base_trace_log_sizes, cairo_pay_components};
use stwo_gpu_spike::transfer::{Readback, ReadbackRing, Upload, UploadRing};

fn ms(d: Duration) -> String {
    format!("{:8.2} ms", d.as_secs_f64() * 1e3)
}

fn gbs(bytes: u64, d: Duration) -> String {
    format!("{:5.2} GB/s", bytes as f64 / d.as_secs_f64() / 1e9)
}

fn wait(gpu: &GpuContext) {
    gpu.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
}

/// The best of `reps` runs of `f`, which returns the time of its timed part.
fn best(reps: usize, mut f: impl FnMut() -> Duration) -> Duration {
    (0..reps).map(|_| f()).min().unwrap()
}

/// The upload of every column into its place in `dst` (the layout's one buffer).
fn uploads<'a>(
    layout: &ColumnsLayout,
    dst: &'a wgpu::Buffer,
    polys: &'a [CircleCoefficients<SimdBackend>],
) -> Vec<Upload<'a>> {
    layout
        .batches
        .iter()
        .flat_map(|batch| {
            batch.columns.iter().enumerate().map(move |(k, &i)| Upload {
                dst,
                dst_offset: (batch.offset + ((k as u64) << batch.log_size)) * 4,
                src: bytemuck::cast_slice(polys[i].coeffs.as_slice()),
            })
        })
        .collect()
}

/// Every column of `src` (laid out as `layout`, one buffer) read back into fresh stwo columns.
fn read_columns(
    gpu: &GpuContext,
    ring: &mut ReadbackRing,
    src: &wgpu::Buffer,
    layout: &ColumnsLayout,
) -> Vec<BaseColumn> {
    // Safety: every value is read back below.
    let mut columns = layout
        .slots
        .iter()
        .map(|slot| unsafe { BaseColumn::uninitialized(1 << slot.log_size) })
        .collect_vec();
    let mut pieces = columns
        .iter_mut()
        .zip(&layout.slots)
        .map(|(column, slot)| Readback {
            src,
            src_offset: slot.offset * 4,
            dst: bytemuck::cast_slice_mut::<BaseField, u8>(column.as_mut_slice()),
        })
        .collect_vec();
    pieces.sort_by_key(|piece| piece.src_offset);
    ring.read(gpu, pieces);
    columns
}

/// A mapped chunk's address, for rayon's threads, which write disjoint ranges of it.
#[derive(Clone, Copy)]
struct ChunkPtr(*mut u8);
// Safety: the threads write disjoint ranges, within the mapping, while it lives.
unsafe impl Send for ChunkPtr {}
unsafe impl Sync for ChunkPtr {}

impl ChunkPtr {
    fn at(&self, offset: usize) -> *mut u8 {
        self.0.wrapping_add(offset)
    }
}

/// Writes `total` bytes of `src` into the mapped `view` (over and over), by every thread.
fn fill(view: &mut wgpu::BufferViewMut, src: &[u8], total: u64) {
    let mut whole = view.slice(..);
    let len = whole.len();
    let base = ChunkPtr(whole.as_raw_ptr().as_ptr() as *mut u8);
    let part = 1 << 18;
    let parts = total.div_ceil(part as u64) as usize;
    (0..parts).into_par_iter().for_each(|k| {
        let offset = (k * part) % len;
        let n = part.min(len - offset).min(src.len());
        // Safety: within the mapping, which `view` keeps; the threads' ranges may coincide, but
        // nothing reads them (a measure of the writes' speed only).
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), base.at(offset), n) };
    });
}

fn main() {
    let mut reps = 3;
    let mut hog_bar = 0u64;
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--reps" => reps = it.next().expect("--reps R").parse().expect("--reps R"),
            "--hog-bar" => hog_bar = it.next().expect("--hog-bar MiB").parse().expect("MiB"),
            other => panic!("unknown argument {other}"),
        }
    }
    let log_sizes = base_trace_log_sizes(&cairo_pay_components(), 0, 4);
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
    let gpu = GpuContext::new();
    println!("GPU: {} ({:?})", gpu.adapter_info.name, gpu.adapter_info.backend);
    let _hog = (0..hog_bar / 8)
        .map(|_| {
            gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("hog"),
                size: 8 << 20,
                usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        })
        .collect_vec();
    if hog_bar > 0 {
        println!("{hog_bar} MiB of MAP_WRITE memory taken first (the BAR, if any)");
    }
    let layout = std::sync::Arc::new(ColumnsLayout::new(&log_sizes, max_buffer_words(&gpu)));
    assert_eq!(layout.buffer_words.len(), 1, "one buffer of columns");
    let total = layout.buffer_words[0] * 4;
    let coeffs = GpuColumns::alloc(&gpu, layout.clone());
    let dst = &coeffs.buffers[0];
    let words = polys
        .iter()
        .map(|p| bytemuck::cast_slice::<BaseField, u32>(p.coeffs.as_slice()))
        .collect_vec();
    let pieces = uploads(&layout, dst, &polys);
    println!("{} MiB of coefficients", total >> 20);

    // Upload: write_buffer_with per batch.
    coeffs.write(&gpu, &words);
    gpu.queue.submit([]);
    wait(&gpu);
    for _ in 0..reps {
        let start = Instant::now();
        coeffs.write(&gpu, &words);
        let queued = start.elapsed();
        gpu.queue.submit([]);
        wait(&gpu);
        let all = start.elapsed();
        println!(
            "upload, write_buffer_with per batch: {} ({}): CPU {}, then GPU {}",
            ms(all),
            gbs(total, all),
            ms(queued),
            ms(all - queued),
        );
    }

    // Upload: rings.
    for (chunk_mib, depth) in [(8u64, 2usize), (16, 2), (32, 2), (32, 3), (64, 2)] {
        let mut ring = UploadRing::new(&gpu, chunk_mib << 20, depth);
        let first = Instant::now();
        ring.upload(&gpu, &pieces);
        wait(&gpu);
        let first = first.elapsed();
        let time = best(reps, || {
            let start = Instant::now();
            ring.upload(&gpu, &pieces);
            wait(&gpu);
            start.elapsed()
        });
        println!(
            "upload, ring of {depth} x {chunk_mib:2} MiB: {} ({}); first use {}",
            ms(time),
            gbs(total, time),
            ms(first)
        );
    }
    {
        let mut ring = UploadRing::new(&gpu, 32 << 20, 2);
        let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        let time = pool.install(|| {
            best(reps, || {
                let start = Instant::now();
                ring.upload(&gpu, &pieces);
                wait(&gpu);
                start.elapsed()
            })
        });
        println!("upload, ring of 2 x 32 MiB, one thread: {} ({})", ms(time), gbs(total, time));
    }
    // The upload's check.
    let check = read_columns(&gpu, &mut ReadbackRing::with_defaults(&gpu), dst, &layout);
    for (i, column) in check.iter().enumerate() {
        assert!(column.as_slice() == polys[i].coeffs.as_slice(), "column {i} uploaded wrong");
    }
    drop(check);

    // Readback: rings.
    for (chunk_mib, depth) in [(16u64, 2usize), (32, 2), (32, 3), (64, 2), (128, 2)] {
        let mut ring = ReadbackRing::new(&gpu, chunk_mib << 20, depth);
        drop(read_columns(&gpu, &mut ring, dst, &layout));
        let time = best(reps, || {
            let start = Instant::now();
            let columns = read_columns(&gpu, &mut ring, dst, &layout);
            let time = start.elapsed();
            drop(columns);
            time
        });
        println!(
            "readback, ring of {depth} x {chunk_mib:3} MiB, into fresh columns: {} ({})",
            ms(time),
            gbs(total, time)
        );
    }

    // Overlap with the commitment's compute.
    let lifting = log_sizes.iter().max().unwrap() + 3;
    let twiddles =
        SimdBackend::precompute_twiddles(CanonicCoset::new(lifting).circle_domain().half_coset);
    let kernels = CommitKernels::new(&gpu);
    let commitment = StripedCommitment::new(&gpu, &kernels, &log_sizes, 3, lifting, &twiddles);
    commitment.upload_through(&gpu, &mut UploadRing::with_defaults(&gpu), &polys);
    commitment.run(&gpu, &kernels);
    let down = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("down"),
        size: total,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let submit_and_time = |record: &dyn Fn(&mut wgpu::CommandEncoder)| {
        best(reps + 1, || {
            let start = Instant::now();
            let mut encoder = gpu.device.create_command_encoder(&Default::default());
            record(&mut encoder);
            gpu.queue.submit([encoder.finish()]);
            wait(&gpu);
            start.elapsed()
        })
    };
    // `dst` is not the commitment's: the copy shares nothing with its compute.
    let copy = |e: &mut wgpu::CommandEncoder| e.copy_buffer_to_buffer(dst, 0, &down, 0, total);
    let extension = |e: &mut wgpu::CommandEncoder| {
        for stripe in 0..8 {
            commitment.encode_extension(&kernels, e, stripe, None);
        }
    };
    let compute = submit_and_time(&|e| commitment.encode(&kernels, e, None));
    let copy_down = submit_and_time(&copy);
    let extension_alone = submit_and_time(&extension);
    println!(
        "compute {}; the extension alone {}; a copy to the host {} ({})",
        ms(compute),
        ms(extension_alone),
        ms(copy_down),
        gbs(total, copy_down)
    );
    let copy_then_compute = submit_and_time(&|e| {
        copy(e);
        commitment.encode(&kernels, e, None);
    });
    let compute_then_copy = submit_and_time(&|e| {
        commitment.encode(&kernels, e, None);
        copy(e);
    });
    let copy_then_extension = submit_and_time(&|e| {
        copy(e);
        extension(e);
    });
    let two_submissions = best(reps + 1, || {
        let start = Instant::now();
        let mut e = gpu.device.create_command_encoder(&Default::default());
        copy(&mut e);
        gpu.queue.submit([e.finish()]);
        let mut e = gpu.device.create_command_encoder(&Default::default());
        commitment.encode(&kernels, &mut e, None);
        gpu.queue.submit([e.finish()]);
        wait(&gpu);
        start.elapsed()
    });
    println!(
        "copy, then compute: {}; compute, then copy: {}; copy, then the extension alone: {}; copy \
         and compute in two submissions: {}",
        ms(copy_then_compute),
        ms(compute_then_copy),
        ms(copy_then_extension),
        ms(two_submissions)
    );

    // The CPU's writes into a mapped chunk while the GPU computes (no copies from it: those
    // would queue behind the compute).
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("written while computing"),
        size: 32 << 20,
        usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: true,
    });
    let src = bytemuck::cast_slice::<BaseField, u8>(polys[0].coeffs.as_slice());
    let mut view = staging.slice(..).get_mapped_range_mut().expect("mapped");
    let written_alone = best(reps, || {
        let start = Instant::now();
        fill(&mut view, src, total);
        start.elapsed()
    });
    let written_while_computing = best(reps, || {
        let start = Instant::now();
        let mut e = gpu.device.create_command_encoder(&Default::default());
        commitment.encode(&kernels, &mut e, None);
        gpu.queue.submit([e.finish()]);
        fill(&mut view, src, total);
        wait(&gpu);
        start.elapsed()
    });
    println!(
        "{} MiB written into a mapped chunk: alone {}; while the GPU computes, until both are \
         done: {}",
        total >> 20,
        ms(written_alone),
        ms(written_while_computing)
    );
}
