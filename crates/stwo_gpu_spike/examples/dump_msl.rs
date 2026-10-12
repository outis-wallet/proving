//! Prints the MSL wgpu generates (through naga) for the extension kernels and the Merkle kernels on
//! this device: wgpu-hal logs each Metal library it compiles at debug level ("Naga generated
//! shader"); this example catches those records.
//!
//! `cargo run --release --example dump_msl [unchecked]`
use stwo_gpu_spike::GpuContext;
use stwo_gpu_spike::extension::ExtensionKernels;

struct Catch;
impl log::Log for Catch {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.target().starts_with("wgpu_hal")
    }
    fn log(&self, record: &log::Record) {
        let text = format!("{}", record.args());
        if text.contains("Naga generated shader") {
            println!("//////// {text}\n");
        }
    }
    fn flush(&self) {}
}

fn main() {
    log::set_logger(&Catch).unwrap();
    log::set_max_level(log::LevelFilter::Debug);
    let gpu = GpuContext::new();
    eprintln!("adapter: {:?}", gpu.adapter_info.name);
    let _extension = ExtensionKernels::new(&gpu);
    let _merkle = stwo_gpu_spike::merkle::GpuMerkle::new(&gpu);
}
