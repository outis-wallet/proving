# A synchronous GPU bridge for the threaded wasm prover

The Outis prover proves in one long synchronous wasm call (`session.prove()`, 15–30 s) on a
worker and rayon's Web Workers. WebGPU only resolves on a JS event loop (`mapAsync`,
`onSubmittedWorkDone`). This directory is a bridge between the two, a demo of it, and its
measurements:

- a prover thread calls a plain synchronous Rust function;
- a GPU worker runs the job and writes the result back into the prover's memory.

The demo function is `gpu_m31_mul(a: &[u32], b: &[u32]) -> Result<Vec<u32>, BridgeError>`
(demo/src/lib.rs). It is called from rayon's threads and checked against stwo's M31 multiply.

## Architecture

```
 page (main thread) ── creates once ──► GPU worker (gpu.worker.mjs + pkg/gpu: wgpu 30, WebGPU)
   │                                      ▲  owns the GPUDevice and the pipelines, for the page's life
   │ MessageChannel per run               │  {attach: {memory, base}} / {detach}
   ▼                                      │
 prover worker (prover.worker.mjs) ───────┘
   └─ threaded module (pkg/demo; the prover's flags), rayon on Web Workers
        shared WebAssembly.Memory
        ┌─────────────────────────────────────────────────────────────────┐
        │ job table (static, 16 KiB): header + 64 slots × 256 B            │
        │   header: magic, version, doorbell, server state                │
        │   slot: state, op, args[16], times[6], error text               │
        │ inputs / outputs: the prover's own Vecs                         │
        └─────────────────────────────────────────────────────────────────┘
```

**A call, as the prover thread sees it** (`bridge::call`):

1. It claims a free slot with a CAS.
2. It writes the op and its args: pointers and lengths into its own memory.
3. It marks the slot SUBMITTED, bumps the doorbell and runs `memory.atomic.notify`.
4. It waits on the slot with `memory.atomic.wait32`, in 100 ms steps. Each step rechecks the
   server state, so a dead GPU worker can't hang `prove()`.

**What the GPU worker does with it** (`gpu.worker.mjs`):

1. It waits on the doorbell with `Atomics.waitAsync`, then CASes the slot from SUBMITTED to
   RUNNING.
2. It calls `Gpu.execute(op, args, memory.buffer)`, which runs the op in Rust with wgpu. It
   re-reads `memory.buffer` for each job, because a grown shared memory has a new, longer
   `SharedArrayBuffer`.
3. It marks the slot DONE or ERROR (with a message) and runs `Atomics.notify`.

**How failures are handled:**

- A WebGPU validation or out-of-memory error, caught by error scopes around each job, makes that
  job fail.
- A lost device, or a trap in the GPU module (caught by the worker's `error` and
  `unhandledrejection` handlers), sets the server state to LOST. Every waiting and future call
  then gets `Err`.
- On `Err`, the caller uses the CPU, whose outputs are the same.

**Why this shape:**

- **The GPU worker is a separate, plain wasm32 module running wgpu 30.** Two other options were
  considered:
  - *JS-driven kernels* would duplicate the kernels' host code in JS: passes, bind groups, the
    batching, the splits at binding limits.
  - *wgpu inside the prover's module* needs:
    - wasm-bindgen ≥ 0.2.127 (wgpu 30's web backend requires it; the prover pins =0.2.126);
    - a dedicated wasm thread with its own event loop, because wgpu's WebGPU handles aren't
      `Send` in atomics builds ([wgpu#6267](https://github.com/gfx-rs/wgpu/issues/6267)).

  The separate module keeps the prover's build as it is and compiles the kernel crates' Rust
  unchanged. Its data path never enters its own memory:
  - uploads are `writeBuffer` from views of the prover's `SharedArrayBuffer`;
  - readbacks are mapped ranges `.set()` into such views.

  So there are no extra copies, and no load on the prover's 4 GiB wasm32 address space. The
  module is 304 KB of wasm (wgpu's web backend plus the ops), and needs a second wasm-bindgen CLI
  (0.2.128, installed into `.tools/` by build.sh).
- **The page owns the GPU worker, not the prover's worker.** Three reasons:
  - The device and compiled pipelines outlive each proof.
  - Nothing has to start a worker while its parent is blocked. Starting a nested worker from a
    blocked parent is unspecified, and worker.mjs already works around the same issue.
  - The page never holds the memory, because the prover's worker sends it over a MessageChannel
    port.

  The GPU worker waits with `waitAsync`, never in a wasm atomic wait, so ending it is never a
  problem (unlike Safari's rayon workers, see `stop_threads`).
- **Jobs run one at a time.** A commitment is one job; concurrent callers queue.

## writeBuffer and SharedArrayBuffer

`writeBuffer` takes views of a `SharedArrayBuffer`, so no copy is needed in JS.

- **The spec** says `AllowSharedBufferSource` ([gpuweb#4188](https://github.com/gpuweb/gpuweb/pull/4188),
  2023). It has been `[AllowShared]` since [#819](https://github.com/gpuweb/gpuweb/pull/819), in 2020.
- **Tested here** (probe.html, and every upload of the demo up to 512 MiB):
  - Chrome 154: `Uint8Array` and `Uint32Array` views of a shared `WebAssembly.Memory` are
    accepted, and the results are bit-exact.
  - Firefox 157: the same is accepted. The fix landed in Firefox 139
    ([bug 1938257](https://bugzilla.mozilla.org/show_bug.cgi?id=1938257)).
- **Safari:** `[AllowShared]` since 2022 (WebKit bug 249632), so Safari 26 should accept them.
  This is from the source; there's no device here.
- **Caveats:**
  - A view larger than 2 GB throws in Firefox
    ([bug 2022805](https://bugzilla.mozilla.org/show_bug.cgi?id=2022805)), so keep views
    chunk-sized. The bridge's views are 16 MiB or less.
  - WebKit rejects growable SABs, so don't use `memory.toResizableBuffer()` views.
- **The browsers still copy**, on their side:
  - Chrome makes a shared-memory region per write of 4 MiB or more.
  - Safari copies writes over 16 MiB into the WebContent process.
  - Firefox makes a new segment and staging buffer per call
    ([bug 1968102](https://bugzilla.mozilla.org/show_bug.cgi?id=1968102)), and here it uploads at
    a fifth of Chrome's rate.
- **What the bridge does about it:**
  - Uploads go in 16 MiB `writeBuffer`s, with a queue wait after each 64 MiB.
  - Readbacks go through two 16 MiB mappable buffers: chunk k + 1 is copied on the GPU and mapped
    while chunk k is copied out.

  The sizes are configurable with `upload_chunk_mib`, `upload_window_mib` and
  `readback_chunk_mib`. See the memory numbers below.

## Measurements

**These are software-GPU numbers.** Chrome 154 ran headless on SwiftShader (Vulkan on the CPU),
because this machine's NVIDIA driver is broken. Firefox 157 ran headless with the WebGPU prefs
forced on (Linux doesn't ship it), on its fallback adapter. Both used 8 rayon threads on a
36-thread x86 machine. The raw reports are in results/.

What carries over to real hardware:

- **Bridge latency:** yes. It is CPU-side (wake and return of the waiters).
- **Transfer rates:** roughly, for phones' unified memory. Browsers' uploads and readbacks are
  memcpy-bound. They are an upper bound for a discrete GPU, which adds PCIe.
- **Compute:** no. SwiftShader computes the 512 MiB product in 630 ms; 8 wasm threads check it in
  34 ms.

**Correctness.** All outputs were equal to stwo's M31 multiply:

- 256 jobs of 81–300 000 words, sent concurrently from all 8 rayon threads (39 M products);
- the product at 64, 128, 256 and 512 MiB per buffer, which crosses the dispatch-chunk boundary
  (16 776 960 words), 3 reps each, in both browsers.

Failures come back as errors, and the bridge keeps serving after them: a range outside the
memory, an unknown op.

**Latency** (ms). Medians, with p90 in parentheses, over 200 calls from one rayon thread:

| | Chrome / SwiftShader | Firefox / fallback adapter |
|---|---|---|
| ping (bridge only, no GPU) | 0.015 (0.020) | 0.02 (0.02) |
| … wake: doorbell → GPU worker | 0.010 | 0.02 |
| … return: notify → prover thread | 0.005 | ~0 (below timer resolution) |
| empty submit + `onSubmittedWorkDone` | 0.09 (0.11) | 2.4 (2.7) with kicks; 100 without |
| M31 product, 1024 words, 3 syncs | 3.3 (5.8) | 6.7 (7.1) |
| same, pipelined (1 sync) | 2.9 (3.3) | 2.5 (2.8) with kicks; 100 without |
| pings from 8 threads at once | 165 000 jobs/s | 121 000 jobs/s |

**Firefox's 100 ms.** Firefox polls WebGPU for completions on a 100 ms timer and at each submit
([bug 1870699](https://bugzilla.mozilla.org/show_bug.cgi?id=1870699),
[bug 2008103](https://bugzilla.mozilla.org/show_bug.cgi?id=2008103)). Every GPU→CPU sync can
therefore take up to 100 ms. While a job waits, the GPU worker submits an empty command buffer
every 2 ms (`kick_ms`, on by default in Firefox only), which takes a sync to about 2 ms. The 256
correctness jobs took 77 s without it and 1.8 s with it.

**Throughput** at 512 MiB per buffer. Warm reps are the best of 2–3; the first rep also creates
the GPU buffers.

| | Chrome / SwiftShader | Firefox |
|---|---|---|
| upload (`writeBuffer` from the prover's memory, until the queue is done) | 120 ms, **4.2 GiB/s** | 580 ms, 0.9 GiB/s |
| readback (GPU → prover's memory) | 137 ms, **3.7 GiB/s** | 145 ms, 3.6 GiB/s |
| product, phases timed (upload 1 GiB, compute, readback 512 MiB) | 1.01 s (240 + 630 + 140 ms) | 1.33 s (1133 + 50 + 146 ms) |
| product, pipelined | 1.0 s | 1.33 s |

Rates at 64–256 MiB were within 8 % of these (results/). At Chrome's warm rate, the 0.5 GiB of
coefficients of one Cairo trace upload in about 0.12 s.

**Transfer sizes** in Chrome at 512 MiB. Throughput in GiB/s:

- upload, by `writeBuffer` size:
  - 1 MiB: 1.5
  - 4 MiB: 3.8
  - 16 MiB: 4.3
  - 64 MiB: 4.4
  - 512 MiB: 4.4
  - A queue wait every 16 MiB costs about 10 %.
- readback, by staging-buffer size:
  - 1 MiB: 3.1
  - 4 MiB: 3.3
  - 16 MiB: 3.8
  - 64 MiB: 3.7
  - one 512 MiB buffer: 2.4. Its first readback also took 0.7 s rather than 0.2 s.

**Memory.** Peak RSS of Chrome's processes in the 512 MiB run. The wasm memory was 1.57 GiB, and
the renderer's baseline with a tiny run was 0.6 GiB.

| transfers | renderer (the tab) | GPU process |
|---|---|---|
| bounded: 16 MiB writes, a wait every 64 MiB, 2 × 16 MiB staging | 2.50 GiB | 1.83 GiB |
| bounded: 4 MiB / 16 MiB / 2 × 4 MiB | 2.38 GiB | 1.69 GiB |
| unbounded: 512 MiB writes, one 512 MiB staging buffer | **3.88 GiB** | **3.69 GiB** |

Unbounded transfers add 1.4 GiB to the tab's own process. Chrome backs writes and mappable
buffers with shared memory mapped into the renderer, which is why the bridge bounds them.

## How the FFT and Merkle kernels plug in

This is based on their branches: `gpu-fft` at 85dfe57e (`GpuColumns`,
`StripeExtension::encode/submit`) and `gpu-merkle` at dfca87b7 (`GpuMerkle::plan_*`,
`Plan::record`, `run_plans`, `read_levels`).

1. **The encoding code compiles into the GPU module unchanged.** `gpu/` depends on
   `stwo_gpu_spike` by path, and the pipelines, plans, records, submits and splits at binding
   limits are the same as natively.
2. **Only the I/O edges need web versions.** On the web, `device.poll(wait)` returns
   immediately, and a map's callback runs only from the event loop. So:
   - **Waits become awaits.** Split the "encode and submit" steps (already separate in both
     crates) from the waits. Natively, a wait is `poll(wait)`; in `gpu/`, it is
     `queue_done().await` or the map's future.
   - **Uploads.** `GpuColumns::write` uses `write_buffer_with`, a staging slice in the calling
     module's memory, which on the web would copy the whole trace into the GPU module. In `gpu/`,
     it becomes `upload()` from views of the prover's memory, at a buffer offset, per column or
     batch.
   - **Readbacks.** `read_into` and `read_levels` become the ring readback (`read_into` here,
     extended with a source offset), straight into the prover's Vecs.
3. **One bridge op per commitment.** Its args point to a descriptor in the prover's memory: the
   columns (pointer, log size), the blowup, and where the extended evaluations and the Merkle
   layers go. The GPU worker uploads once, runs the extension and the Merkle tree in a few large
   submits, and reads back. To the prover, a commitment is one blocking call that fills Vecs it
   allocated, in SimdBackend's layout.
4. **The prover's side.** stwo's commitment (the tree prover's extend, then its Merkle commit)
   calls the op when `bridge::ready()`, and the CPU path otherwise or on `Err`.
5. **Check outputs on every browser.** Browsers compile WGSL with their own compilers: Tint in
   Chrome, naga in Firefox, WebKit's own in Safari. The kernels' bit-exactness has only been
   shown natively; here, M31 was exact on Tint→SwiftShader and on Firefox's adapter.
   - Add cheap CPU spot checks: a few random rows of the extension, a few hashes of the tree.
     They also catch the silent failure suspected below for backgrounded iOS tabs.
   - A wrong commitment fails the proof's final verification. That is a lost proof, not a
     soundness problem.
6. **Later: residency.** Keep the extended evaluations on the GPU for quotients and FRI, rather
   than reading back twice the trace.

## What changes in the prover's wasm build

- **os_wasm_probe:**
  - It depends on `stwo-gpu-bridge` by path. The crate has no dependencies, and uses nightly's
    `stdarch_wasm_atomic_wait`, which the prover's toolchain already has.
  - It exports the region's address (`bridge_region()`).
  - The wasm-bindgen pin, the flags and `-Z build-std` stay as they are.
- **stwo (Outis fork):** the commitment hook above, behind a feature, for wasm32 with atomics
  only.
- **worker.mjs:**
  - It gets a port from the page.
  - After `initThreadPool`, it posts `{attach: {memory: wasm.memory, base}}` and awaits
    `{attached}` before `session.prove()`.
  - At the end, it posts `{detach}` before `endChildren()`.
- **The page:** it creates the GPU worker once (gpu.worker.mjs and pkg/gpu, 304 KB of wasm), and
  a MessageChannel per run.
- **The build:** build-wasm.sh gains the GPU module's step, as build.sh does here. It is a plain
  `cargo build --target wasm32-unknown-unknown`, with no threads and no build-std, and
  wasm-bindgen 0.2.128 from `.tools/`. Bumping the prover's pin to ≥ 0.2.127 would let one CLI
  serve both, but nothing requires it.

## Risks

1. **iPhone memory is the biggest risk.** WebKit charges WebGPU's Metal buffers to the tab's
   WebContent process, through `MTLResource setOwnerWithIdentity`
   ([WebKit bug 254219](https://bugs.webkit.org/show_bug.cgi?id=254219), 2024). An Apple engineer
   in [bug 256703](https://bugs.webkit.org/show_bug.cgi?id=256703): "WebKit assigns the cost GPU
   process memory for a tab to the tab itself".
   - **So GPU buffers count against the same ~2.25 GB limit as the prover's ~2.3 GiB wasm heap.**
     A drop-in GPU commitment adds its buffers on top. The coefficients alone are 0.5 GiB, and
     the extension and the tree more. On an iPhone, the GPU path pays only with a residency
     design that frees the CPU copies, and it won't fit as a drop-in.
   - **Transfers add their own copies there too.** Safari copies each write over 16 MiB into
     WebContent, and `getMappedRange` returns a copy into WebContent. Use chunks of 16 MiB or
     less (8 MiB to be safe) and the ring readback.
   - **Keep buffers at 256 MiB or less on iOS.** Safari's `maxBufferSize` is
     max(256 MiB, min(1 GiB, maxBufferLength / 3)), and an Apple engineer wrote "iOS is too likely
     to jetsam at 1GB" ([bug 275958](https://bugs.webkit.org/show_bug.cgi?id=275958)).
   - **The GPU process can die too**, from unattributed staging or a leak (bug 303203). When it
     does, all of WebGPU is gone, and `requestAdapter` returns null afterwards.
2. **Safari / iOS support.**
   - WebGPU is on by default since Safari 26 (iOS 26, September 2025), in workers and in
     WKWebView, and `Atomics.waitAsync` since 16.4.
   - Several compute command buffers in flight can hang the queue on iOS
     ([bug 311598](https://bugs.webkit.org/show_bug.cgi?id=311598)). WebKit's advice is about one
     at a time, each with several ms of work. The ring readback keeps 2 copies in flight; iOS may
     need a depth of 1.
   - From WebKit's source (inferred, not tested): a backgrounded app's command buffers fail with
     `NotPermitted` without losing the device, and the outputs may then be silently stale. Spot
     checks and the final verification cover this.
   - Not tested on a device. probe.html is ready for one.
3. **Firefox.**
   - WebGPU ships on Windows (141) and Apple-silicon macOS (145/147), but not on Linux or
     Android.
   - `Atomics.waitAsync` is there since 145.
   - The 100 ms poll timer needs the kicks above.
   - Uploads are slow: 0.9 GiB/s here.
   - Views over 2 GB throw (bug 2022805).
   - This adapter's `maxStorageBufferBindingSize` is 128 MiB. The bridge binds in chunks.
4. **Chrome.**
   - Android needs Android 12+ and a Qualcomm or ARM GPU.
   - Limits are tiered. Storage binding: 128 MiB on every Android adapter, ≥ 256 MiB on 75 %,
     ≥ 1 GiB on 35 % (web3dsurvey). Kernels must split at the binding limit, as the Merkle branch
     does.
   - The renderer pays for staging, as measured above.
5. **Worker and device limits.**
   - **Workers:** the bridge needs one more worker. There's no hard cap: Firefox queues past 512
     per domain. The table's 64 slots cover any rayon pool.
   - **Dispatch:** `maxComputeWorkgroupsPerDimension` is 65 535, so one dispatch at workgroup
     size 256 covers less than 64 MiB of u32 (16 776 960 words). The product is dispatched in
     chunks.
   - **Bindings:** offsets are 256-byte aligned. WebGPU's defaults are 128 MiB per binding and
     256 MiB per buffer, unless the device requests the adapter's limits, as `Gpu.create` does.
   - **Long dispatches** can trip GPU watchdogs (Windows' 2 s TDR, mobile drivers). Keep each
     submit to tens of ms of work. No WebKit timeout was found.
   - **A lost device** fails the jobs in flight, and the prover falls back to the CPU.

## Layout, build, run

- `bridge/`: the protocol (the layout, the ops) and the client. No dependencies, so both
  wasm-bindgen versions build it.
- `gpu/`, `gpu.worker.mjs`: the GPU worker.
  - The ops are ping, empty submit, upload, readback and the M31 product, with chunked bindings,
    error scopes and bounded transfers.
  - The worker's JS owns the slot protocol and failure handling.
- `demo/`, `prover.worker.mjs`, `index.html`, `main.mjs`: the demo.
  - It is threaded like the prover (wasm-bindgen =0.2.126, wasm-bindgen-rayon with `no-bundler`).
  - Its `demo_correctness`, `demo_errors`, `demo_latency` and `demo_throughput` are each one long
    synchronous call.
- `probe.html`: a browser's capabilities for the bridge, with no build.

```sh
./build.sh                                   # both modules into pkg/ (installs wasm-bindgen 0.2.128 into .tools/)
./run-headless.sh 'threads=8'                # headless Chrome on SwiftShader; prints the reports
SAMPLE_RSS=1 ./run-headless.sh 'phases=throughput&sizes=512'   # with Chrome's per-process peak RSS
PAGE=probe.html ./run-headless.sh
```

Chrome needs `--enable-unsafe-webgpu --use-webgpu-adapter=swiftshader --enable-features=Vulkan`,
and not `--disable-gpu`. The page needs COOP/COEP; run-headless.sh serves this directory with
agent-tools/server.py on port 8795. Page parameters:

- `threads`, `phases`, `jobs`, `iters`, `sizes`, `reps`;
- `upload_chunk_mib`, `upload_window_mib`, `readback_chunk_mib` (0: unbounded);
- `kick_ms`.

On any other device, serve this directory (HTTPS for a phone) and open `/` or `/probe.html`.
