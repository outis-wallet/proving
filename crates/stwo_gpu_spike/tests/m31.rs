//! The WGSL M31 arithmetic against stwo's, on random canonical values and the edge cases.
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use stwo::core::fields::m31::{M31, P};
use stwo_gpu_spike::{GpuContext, M31_WGSL};

const KERNEL: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read> b: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = arrayLength(&a);
    let i = id.x;
    if (i >= n) { return; }
    out[i] = m31_mul(a[i], b[i]);
    out[n + i] = m31_add(a[i], b[i]);
    out[2u * n + i] = m31_sub(a[i], b[i]);
}
"#;

#[test]
fn m31_arithmetic_matches_stwo() {
    let gpu = GpuContext::new();
    eprintln!("adapter: {:?}", gpu.adapter_info);
    let mut rng = SmallRng::seed_from_u64(31);
    let edges = [0, 1, 2, P - 1, P - 2, 1 << 30, (1 << 30) - 1, 1 << 16, (1 << 16) - 1];
    let n = 1 << 18;
    let mut a: Vec<u32> = (0..n).map(|_| rng.random_range(0..P)).collect();
    let mut b: Vec<u32> = (0..n).map(|_| rng.random_range(0..P)).collect();
    for (i, &x) in edges.iter().enumerate() {
        for (j, &y) in edges.iter().enumerate() {
            a[i * edges.len() + j] = x;
            b[i * edges.len() + j] = y;
        }
    }
    let source = format!("{M31_WGSL}\n{KERNEL}");
    let pipeline = gpu.pipeline("m31-test", &source, "main");
    let (a_buffer, b_buffer) = (gpu.upload("a", &a), gpu.upload("b", &b));
    let out = gpu.storage("out", 3 * n);
    gpu.run(&pipeline, &[&a_buffer, &b_buffer, &out], ((n / 256) as u32, 1, 1));
    let out = gpu.read(&out, 3 * n);
    for i in 0..n {
        let (x, y) = (M31::from_u32_unchecked(a[i]), M31::from_u32_unchecked(b[i]));
        assert_eq!(out[i], (x * y).0, "mul {} {}", a[i], b[i]);
        assert_eq!(out[n + i], (x + y).0, "add {} {}", a[i], b[i]);
        assert_eq!(out[2 * n + i], (x - y).0, "sub {} {}", a[i], b[i]);
    }
}
