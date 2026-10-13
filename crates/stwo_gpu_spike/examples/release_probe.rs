//! EXPERIMENT (lean): when does a released buffer's memory leave the process? A 512 MiB storage
//! buffer, filled by the GPU, destroyed: the footprint before, right after (with a poll) and when
//! it is back down. Then the heap: 400 MiB of 512 KiB blocks freed (the system allocator keeps
//! them). Apple only (the footprint from `proc_pid_rusage`).
//!
//! On an Apple M5 Pro (macOS 26.5): the buffer's 512 MiB leave about 180-230 ms after the
//! destroy (making the Metal buffer purgeable-empty first changes nothing); the freed blocks stay.
use std::time::{Duration, Instant};

use stwo_gpu_spike::GpuContext;

#[cfg(target_vendor = "apple")]
fn footprint_mib() -> u64 {
    unsafe extern "C" {
        fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *mut u8) -> i32;
        fn getpid() -> i32;
    }
    let mut buffer = [0u8; 512];
    // Safety: RUSAGE_INFO_V4 fills fewer than 512 bytes.
    if unsafe { proc_pid_rusage(getpid(), 4, buffer.as_mut_ptr()) } == 0 {
        return u64::from_le_bytes(buffer[72..80].try_into().unwrap()) >> 20;
    }
    0
}

/// Elsewhere the probe builds (for `cargo test`, which builds the examples) but measures nothing.
#[cfg(not(target_vendor = "apple"))]
fn footprint_mib() -> u64 {
    0
}

fn main() {
    let gpu = GpuContext::new();
    for _ in 0..3 {
        let base = footprint_mib();
        let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("probe"),
            size: 512 << 20,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        encoder.clear_buffer(&buffer, 0, None);
        gpu.queue.submit([encoder.finish()]);
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let filled = footprint_mib();
        let t = Instant::now();
        buffer.destroy();
        drop(buffer);
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let released = footprint_mib();
        let mut later = released;
        while t.elapsed() < Duration::from_secs(1) && later > base + 64 {
            std::thread::sleep(Duration::from_millis(5));
            later = footprint_mib();
        }
        println!(
            "buffer: base {base} MiB, filled {filled}, destroyed {released}, back to {later} \
             after {:.0} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
    }
    let base = footprint_mib();
    let blocks: Vec<Vec<u8>> = (0..800).map(|_| vec![1u8; 512 << 10]).collect();
    let held = footprint_mib();
    drop(blocks);
    std::thread::sleep(Duration::from_millis(300));
    println!("heap: base {base} MiB, held {held}, freed (300 ms later) {}", footprint_mib());
}
