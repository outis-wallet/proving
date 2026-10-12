#!/bin/sh
# Builds the bridge demo's two wasm modules into pkg/ (serve this directory with COOP/COEP):
#   pkg/gpu/   the GPU worker's module: plain wasm32, wgpu 30 (WebGPU backend), wasm-bindgen 0.2.128
#              (its CLI is installed into .tools/ on first use: wgpu 30 needs >= 0.2.127);
#   pkg/demo/  the prover-side demo: threaded like the Outis prover (atomics, shared memory up to
#              4 GiB, -Z build-std, wasm-bindgen-rayon), wasm-bindgen 0.2.126 (the installed CLI).
set -e
cd "$(dirname "$0")"
unset RUSTC_WRAPPER
COMMON='--cfg getrandom_backend="wasm_js"'

if [ ! -x .tools/bin/wasm-bindgen ] || ! .tools/bin/wasm-bindgen --version | grep -q '0.2.128'; then
  cargo install wasm-bindgen-cli --version =0.2.128 --locked --root .tools
fi
case "$(wasm-bindgen --version)" in
  *0.2.126*) ;;
  *) echo "the demo needs wasm-bindgen-cli 0.2.126 on PATH (the prover's pin)" >&2; exit 1 ;;
esac

if [ "${1:-all}" != demo ]; then
  RUSTFLAGS="$COMMON" CARGO_TARGET_DIR=target/gpu \
    cargo build --release --target wasm32-unknown-unknown --manifest-path gpu/Cargo.toml
  .tools/bin/wasm-bindgen --target web --out-dir pkg/gpu \
    target/gpu/wasm32-unknown-unknown/release/stwo_gpu_web_worker.wasm
fi

if [ "${1:-all}" != gpu ]; then
  # The prover's threaded flags (outis-wallet/prover/scripts/build-wasm.sh web).
  THREADS='-C target-feature=+atomics,+bulk-memory,+mutable-globals
    -C link-arg=--shared-memory -C link-arg=--import-memory -C link-arg=--max-memory=4294967296
    -C link-arg=--export=__wasm_init_tls -C link-arg=--export=__tls_size
    -C link-arg=--export=__tls_align -C link-arg=--export=__tls_base'
  RUSTFLAGS="$COMMON -C target-feature=+simd128 -C link-arg=-zstack-size=16777216 $(echo $THREADS)" \
    CARGO_TARGET_DIR=target/demo \
    cargo build --release --target wasm32-unknown-unknown -Z build-std=panic_abort,std \
    --manifest-path demo/Cargo.toml
  wasm-bindgen --target web --out-dir pkg/demo \
    target/demo/wasm32-unknown-unknown/release/stwo_gpu_web_demo.wasm
fi
ls -l pkg/gpu/*.wasm pkg/demo/*.wasm 2>/dev/null
