//! The device's memory as `stwo_gpu_spike::memory` reads it, to check what the driver's budget
//! reflects:
//!
//! ```text
//! vram_probe query <seconds>        prints the device-local heap, the driver's budget and this
//!                                   process's usage every half second
//! vram_probe hog <MiB> <seconds>    holds a buffer of that size (written) for that long
//! ```
use std::time::{Duration, Instant};

use stwo_gpu_spike::GpuContext;
use stwo_gpu_spike::memory::{DeviceMemory, process_reserved};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let gpu = GpuContext::new();
    match args.get(1).map(String::as_str) {
        Some("query") => {
            let seconds: f64 = args[2].parse().unwrap();
            let start = Instant::now();
            while start.elapsed().as_secs_f64() < seconds {
                let m = DeviceMemory::query(&gpu);
                println!(
                    "{:5.1} s: {:?} MiB local, budget {:?} MiB, usage {:?} MiB, reserved {:?} MiB",
                    start.elapsed().as_secs_f64(),
                    m.map(|m| m.local_heap >> 20),
                    m.and_then(|m| m.budget).map(|b| b >> 20),
                    m.and_then(|m| m.usage).map(|u| u >> 20),
                    process_reserved(&gpu).map(|r| r >> 20),
                );
                std::thread::sleep(Duration::from_millis(500));
            }
        }
        Some("hog") => {
            let mib: u64 = args[2].parse().unwrap();
            let seconds: f64 = args[3].parse().unwrap();
            let buffer = gpu.storage("hog", (mib << 20) as usize / 4);
            let mut encoder = gpu.device.create_command_encoder(&Default::default());
            encoder.clear_buffer(&buffer, 0, None);
            gpu.queue.submit([encoder.finish()]);
            gpu.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            println!("holding {mib} MiB");
            std::thread::sleep(Duration::from_secs_f64(seconds));
            drop(buffer);
        }
        _ => panic!("usage: vram_probe query <seconds> | hog <MiB> <seconds>"),
    }
}
