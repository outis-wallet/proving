//! The kernels are core WGSL: they validate without any optional capability (no 64-bit integers,
//! no subgroups, no binding arrays), and naga translates them to MSL (Metal) and HLSL (DX12), the
//! way wgpu does natively. Browsers compile WGSL with their own compilers (Tint, WebKit's), which
//! this does not cover.
use naga::valid::{Capabilities, ValidationFlags, Validator};
use stwo_gpu_spike::extension::ExtensionKernels;
use stwo_gpu_spike::merkle::{absorb_shader, nodes_shader};

fn check(name: &str, source: &str) {
    let module = naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(source)));
    let info = Validator::new(ValidationFlags::all(), Capabilities::empty())
        .validate(&module)
        .unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(source)));

    let (msl, _) = naga::back::msl::write_string(
        &module,
        &info,
        &naga::back::msl::Options::default(),
        &naga::back::msl::PipelineOptions::default(),
    )
    .unwrap_or_else(|e| panic!("{name}: MSL: {e}"));
    assert!(msl.contains("kernel"), "{name}: no kernel in the MSL");

    let mut hlsl = String::new();
    let options = naga::back::hlsl::Options::default();
    let pipeline_options = naga::back::hlsl::PipelineOptions::default();
    naga::back::hlsl::Writer::new(&mut hlsl, &options, &pipeline_options)
        .write(&module, &info, None)
        .unwrap_or_else(|e| panic!("{name}: HLSL: {e}"));
    assert!(hlsl.contains("numthreads"), "{name}: no compute entry point in the HLSL");
    eprintln!(
        "{name}: {} lines of WGSL, {} of MSL, {} of HLSL",
        source.lines().count(),
        msl.lines().count(),
        hlsl.lines().count()
    );
}

#[test]
fn kernels_are_portable() {
    check("absorb", &absorb_shader());
    check("nodes", &nodes_shader());
    let (out_of_place, in_place) = ExtensionKernels::sources();
    check("extension out of place", &out_of_place);
    check("extension in place", &in_place);
    let (out_of_place, in_place) = ExtensionKernels::merged_sources();
    check("extension out of place, merged", &out_of_place);
    check("extension in place, merged", &in_place);
}
