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
   server state and, with `call_with_deadline`, the caller's deadline, so a dead GPU worker
   can't hang `prove()`.

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
  then gets `Err`, and detaching doesn't wait for a job the trap left unsettled.
- A GPU worker that ends without marking anything (the page terminated it) is caught by the
  caller's deadline (`call_with_deadline`; the prover's hook uses 60 s): the call returns
  `TimedOut` and marks the server state LOST, so later calls fail at once.
- With no GPU worker attached (no WebGPU, no adapter), every call returns `Err` at once.
- On `Err`, the caller uses the CPU, whose outputs are the same. After `Lost` or `TimedOut`
  (`job_may_be_running`) the worker may still write the job's outputs, so the caller leaks them
  rather than free memory it could write into.

The last three were run, not just written: `phases=trap` sends `op::TEST_TRAP`, which panics in
the GPU module, and Firefox without its WebGPU prefs has no WebGPU. In both cases the calls
returned errors and the run ended cleanly. A recovered page would need a new GPU worker.

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
    chunk-sized. The bridge's views are 8 MiB by default.
  - WebKit rejects growable SABs, so don't use `memory.toResizableBuffer()` views.
- **The browsers still copy**, on their side:
  - Chrome makes a shared-memory region per write of 4 MiB or more.
  - Safari copies writes over 16 MiB into the WebContent process.
  - Firefox makes a new segment and staging buffer per call
    ([bug 1968102](https://bugzilla.mozilla.org/show_bug.cgi?id=1968102)), and here it uploads at
    a fifth of Chrome's rate.
- **What the bridge does about it:**
  - Uploads go in 8 MiB `writeBuffer`s, with a queue wait after each 64 MiB.
  - Readbacks go through two 8 MiB mappable buffers: chunk k + 1 is copied on the GPU and mapped
    while chunk k is copied out.

  The sizes are configurable with `upload_chunk_mib`, `upload_window_mib` and
  `readback_chunk_mib`. See the memory numbers below.

## Measurements

**These are the bridge demo's software-GPU numbers.** Chrome 154 ran headless on SwiftShader
(Vulkan on the CPU), because Chrome couldn't reach this machine's NVIDIA GPU then (it can now,
with the flags in **The striped commitment in Chrome on an RTX 3080** below). Firefox 157 ran headless with the WebGPU prefs
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
the GPU buffers. These runs used 16 MiB transfer chunks. The default is now 8 MiB (see
**Transfer sizes**).

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
  - 8 MiB (the default): 3.8
  - 16 MiB: 4.3
  - 64 MiB: 4.4
  - 512 MiB: 4.4
  - A queue wait every 16 MiB costs about 10 %.
- readback, by staging-buffer size:
  - 1 MiB: 3.1
  - 4 MiB: 3.3
  - 8 MiB (the default): 3.6
  - 16 MiB: 3.8
  - 64 MiB: 3.7
  - one 512 MiB buffer: 2.4. Its first readback also took 0.7 s rather than 0.2 s.
- the product, pipelined: 1.04 s with 8 MiB chunks, 1.00 s with 16 MiB.

The default is 8 MiB to stay clear of Safari's 16 MiB copy threshold. In Chrome that costs 14 %
of the upload rate and 4 % of the product's time.

**Memory.** Peak RSS of Chrome's processes in the 512 MiB run. The wasm memory was 1.57 GiB, and
the renderer's baseline with a tiny run was 0.6 GiB.

| transfers | renderer (the tab) | GPU process |
|---|---|---|
| bounded: 16 MiB writes, a wait every 64 MiB, 2 × 16 MiB staging | 2.50 GiB | 1.83 GiB |
| bounded: 8 MiB writes, a wait every 64 MiB, 2 × 8 MiB staging (the default) | 2.49 GiB | 1.80 GiB |
| bounded: 4 MiB / 16 MiB / 2 × 4 MiB | 2.38 GiB | 1.69 GiB |
| unbounded: 512 MiB writes, one 512 MiB staging buffer | **3.88 GiB** | **3.69 GiB** |

Unbounded transfers add 1.4 GiB to the tab's own process. Chrome backs writes and mappable
buffers with shared memory mapped into the renderer, which is why the bridge bounds them.

## The striped commitment in Chrome on an RTX 3080

The prover's threaded build with feature `gpu` (see **The prover's side, as built**) proving the
four captures in headless Chrome 154 on this machine's RTX 3080 (18 cores, 36 threads; NVIDIA
595.99, Vulkan), 16 rayon threads. Chrome reaches the GPU through Vulkan with
`--enable-unsafe-webgpu --enable-features=Vulkan --use-angle=vulkan --disable-gpu-compositing`;
the adapter is `nvidia`/`ampere`, and the device gets its limits (4 GiB buffers, 2 GiB bindings).
GPU on (`gpu=1`), GPU off (the same build, no port) and the default build (feature `threads`
only) ran interleaved, 3 reps each, with the same flags; other agents' jobs put the machine's
load at 3 to 15.

**Correctness.** Every proof was byte-identical to the native prover's and verified: 12 GPU-on
proofs with all 8 striped commitments on the GPU each, 12 GPU-off, 12 default-build. So were:

- a proof with `max_gpu_mib=600`: 4 commitments on the GPU, the 4 larger ones refused for the
  CPU;
- a proof where the page terminated the GPU worker 6 s in (hook deadline 3 s): 3 on the GPU, the
  4th timed out, the rest on the CPU;
- 10 GPU-on proofs in Firefox (below).

**Prove time** (s; best and median of 3):

| capture | GPU on | GPU off | default build | GPU on vs off |
|---|---|---|---|---|
| register | 11.15 / 11.17 | 13.57 / 13.64 | 13.67 / 13.71 | 1.22x |
| deposit | 12.08 / 12.20 | 14.55 / 14.65 | 14.55 / 14.59 | 1.20x |
| pay | 11.24 / 11.32 | 13.71 / 13.78 | 14.06 / 14.20 | 1.22x |
| withdraw | 11.05 / 11.32 | 13.56 / 13.57 | 13.76 / 13.77 | 1.23x |

GPU off is the default build's time: the feature costs nothing without a GPU worker.

**The commitments** (ms, medians): the 8 striped commitments take 2.0 to 2.3 s through the hook,
where `new_striped` takes 4.6 to 5.4 s on the CPU (`cpu_times=1`, a tracing layer on its span).
Per commitment, register, in proof order (the Cairo proof's four trees, log blowup 3, then
the circuit proof's four, log blowup 2):

| columns | 156 | 2608 | 1616 | 8 | 45 | 114 | 152 | 8 |
|---|---|---|---|---|---|---|---|---|
| coefficients (MiB) | 39 | 490 | 467 | 32 | 131 | 386 | 432 | 64 |
| GPU, through the hook | 189 | 524 | 338 | 111 | 257 | 252 | 278 | 87 |
| CPU, `new_striped` | 234 | 1309 | 1249 | 154 | 249 | 591 | 643 | 160 |

The two large Cairo trees gain the most (2.5x and 3.7x); on the small ones the transfers and
setup cost about what the CPU's FFTs do.

**Where a proof's 2.2 s of GPU commitment goes** (sums over the 8, medians of 12 proofs): upload
0.75 s (2.03 GiB at 2.7 GiB/s, 0.36 s of it waiting for the ring's buffers), compute 0.62 s,
readback 0.49 s (2.16 GiB at 4.4 GiB/s), setup (buffers, plans, stripe twiddles) 0.18 s, the
hook's allocation of the outputs 0.07 s, the twiddles 0.03 s. Transfers are 57 % of it: on a
discrete GPU the next gains are in keeping data resident, not in the kernels.

**Memory** (MiB; the range of the four captures' medians; RSS peaks; video memory is the
device's used memory over its idle level):

| | wasm memory | renderer (the tab) | GPU process | video memory |
|---|---|---|---|---|
| GPU on | 2374 to 2497 | 2908 to 3034 | 456 to 458 | +1235 |
| GPU off | 2390 to 2495 | 2712 to 2829 | 213 to 214 | +53 |

The wasm memory doesn't change: the hook allocates what `new_striped` would. The tab grows by
about 200 MiB (the GPU worker, and the transfers' mappable buffers, which Chrome maps into it),
the GPU process by about 245 MiB. The largest commitment holds 1148 MiB of device memory. The GPU worker starts in about 0.3 s (0.5 s from the page), compiling the
commitment's pipelines.

**Firefox 157** (headless, `dom.webgpu.enabled`, `gfx.webgpu.force-enabled`,
`gfx.webgpu.ignore-blocklist`; Linux doesn't ship WebGPU) gets the RTX 3080 with the adapter's
own limits (2 GiB buffers and bindings). GPU on: register 11.71, deposit 12.45, pay 11.41,
withdraw 11.55 s (best; 1 to 5 runs), every proof identical and verified with 8 of 8
commitments on the GPU; pay with the GPU off: 14.68 s (best of 2). The poll kicks (`kick_ms` 2)
are on. Its WebGPU runs in the parent process: 649 to 769 MiB of RSS with the GPU, 512 to 518
without; the content process 2921 to 3040 MiB, against 2882 to 2903. One of 11 GPU-on runs had
a commitment refused with an error the report couldn't carry, which failed that run (fixed:
the reasons are now escaped, and a report that doesn't parse no longer fails a proof); the 5
runs after the fix had none.

## How the FFT and Merkle kernels plug in

This is based on their branches: `gpu-fft` at 85dfe57e (`GpuColumns`,
`StripeExtension::encode/submit`) and `gpu-merkle` at dfca87b7 (`GpuMerkle::plan_*`,
`Plan::record`, `run_plans`, `read_levels`).

1. **The encoding code compiles into the GPU module unchanged.** `gpu/` depends on
   `stwo_gpu_spike` by path, and the pipelines, plans, records, submits and splits at binding
   limits are the same as natively.
   - The crate's library must therefore keep building for `wasm32-unknown-unknown`. Native-only
     dependencies belong in bins and tests, or behind `cfg(not(target_arch = "wasm32"))`.
     Examples are a benchmark harness, threads, `std::time::Instant` and `pollster` used outside
     `new()`.
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

## The prover's side, as built

The prover (outis-wallet/prover) builds this with feature `gpu` (`scripts/build-wasm.sh web-gpu`,
into `pkg-mt-gpu/`); its default web build is unchanged.

- **stwo (Outis fork):** `set_striped_commit_hook::<B, MC>(Some(hook))`.
  `CommitmentTreeProver::new_striped` calls the hook once its striped path applies; the hook
  returns the tree, or gives the polynomials back and `new_striped` commits them itself. Without
  a hook nothing changes. The hook is in stwo for every target; only the prover registers one.
- **os_wasm_probe, `src/gpu.rs` (feature `gpu`):**
  - `gpu_bridge_region()`: the job table's address, its header written.
  - `gpu_enable()` / `gpu_disable()`: registers and removes the hook (SimdBackend, Blake2s M31).
  - The hook allocates the outputs as `new_striped` leaves them (each column's stripe 0, or its
    whole evaluation for the small ones; the layers it stores, root first), faults their pages
    in on all threads, writes a descriptor (`bridge::commit`) and calls `op::COMMIT_STRIPED`.
    The calling thread blocks; the GPU worker fills those Vecs, and the hook builds the
    `CommitmentTreeProver` from them. On any `Err` (no worker, a lost device, a shape or a size
    the worker refuses) it returns the polynomials, for the CPU.
  - `gpu_report()`: per commitment, its shape, where it ran (or why not) and its times.
  - `gpu_time_cpu_commitments()`: a tracing layer that times `new_striped`'s own span, for the
    CPU's reference in a run without the GPU.
- **js/worker.mjs:** given a `gpuPort`, after `initThreadPool` it posts
  `{attach: {memory, base: gpu_bridge_region()}}`, awaits `{attached}` and calls `gpu_enable()`;
  after the proof (and on failure) it calls `gpu_disable()` and posts `{detach}`, before
  `endChildren()`. Both requests time out after 10 s; a proof whose attach timed out runs on
  the CPU.
- **js/gpu.mjs:** `startGpu()` for the page: starts `gpu/gpu.worker.mjs` once, resolves with
  `{ready, adapter, info}` (or `{ready: false, error}`, also after 20 s without a device), and
  `port()` gives a port for one proof.
- **scripts/build-web-engine.sh:** `PKG=pkg-mt-gpu GPU_WEB=<this directory>` adds the GPU
  worker (`gpu.worker.mjs`, `pkg/gpu/`) to the site as `gpu/`, and `engine/gpu.mjs`.

The GPU module's side of `op::COMMIT_STRIPED` is `gpu/src/commit.rs`:

1. It reads the descriptor and checks every length against `new_striped`'s layout (the striped
   path, the kept-whole columns, the stored levels) and the device's binding limits. A
   mismatch is an error, so that commitment goes to the CPU; a panic would trap and end the
   worker.
2. The twiddles: the tail of the prover's twiddle tree the largest domain reads, copied from
   its memory (or computed once and kept, `configure_commit`).
3. `StripedCommitment::with_options(.., without_staging)`: the native commitment's buffers,
   plans and stripe twiddles.
4. Upload: the coefficients, packed into a ring of 4 mappable 8 MiB buffers kept for the
   page's life (`mapAsync(WRITE)`, copy from the prover's memory, unmap,
   `copyBufferToBuffer`).
5. One submission for every stripe's extension and hashing and the levels above
   (`StripedCommitment::encode`), then `onSubmittedWorkDone`.
6. Readback: a ring of 4 mappable 8 MiB buffers, chunks packed from the outputs' segments and
   copied straight into the prover's columns and layers.
7. Every buffer of the commitment is destroyed (wgpu's WebGPU backend doesn't on drop), and the
   stats are written into the descriptor.

`Gpu.create` compiles the commitment's pipelines up front, so a proof's first commitment doesn't
wait for the browser's shader compiler.

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
   - **Measured on the RTX 3080:** one commitment holds up to 1148 MiB of device memory, on top
     of the unchanged 2.3 to 2.4 GiB wasm heap. `max_gpu_mib` caps it (a larger commitment goes
     to the CPU), but the commitments that fit under a phone's budget are the small ones, which
     gain the least. Leave the GPU off on iOS until the commitment streams its stripes through
     bounded buffers or stays resident.
   - **Transfers add their own copies there too.** Safari copies each write over 16 MiB into
     WebContent, and `getMappedRange` returns a copy into WebContent. The bridge's defaults stay
     clear of that: 8 MiB writes and an 8 MiB ring readback.
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
    error scopes and bounded transfers, and the striped commitment (`gpu/src/commit.rs`).
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
