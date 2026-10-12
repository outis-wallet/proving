// The GPU worker: owns the WebGPU device (through the GPU module, wgpu 30) and serves the bridge's
// job table in a prover's shared wasm memory. It lives as long as the page, so the device and the
// compiled pipelines outlive each proof; each proof attaches its memory and detaches it after.
//
// The protocol (the layout comes from the bridge crate, through the module's layout()):
//   a prover thread writes a job into a free slot, marks it SUBMITTED, bumps the doorbell and
//   notifies it, then waits on the slot's state (memory.atomic.wait32);
//   this worker waits on the doorbell with Atomics.waitAsync, takes SUBMITTED slots (-> RUNNING),
//   runs them one at a time on the GPU module, marks them DONE or ERROR and Atomics.notify()s.
// Messages: from the page, { configure: { upload_chunk, upload_window, readback_chunk, kick_ms,
// compute_twiddles, upload: "mapped" | "gather" | "write", upload_ring, readback_ring,
// max_gpu_mib } } (each optional), { connect: port } (a port to a prover's worker); on that port,
// { attach: { memory, base } } -> { attached, info } or { attached: false, error }, and
// { detach: true } -> { detached: true }.
import init, { Gpu, layout } from "./pkg/gpu/stwo_gpu_web_worker.js";

const absNow = () => performance.timeOrigin + performance.now();
let gpu = null;
let L = null;
let io = null; // transfer sizes from the page's { configure }, before start() finishes
// Interval of empty submits while a job waits, in ms (0: none); Firefox only by default.
let kickMs = navigator.userAgent.includes("Firefox") ? 2 : 0;
const sessions = new Set();

/** The device, or why there's none. */
async function start() {
  if (!navigator.gpu) throw new Error("no navigator.gpu in this worker");
  if (typeof Atomics.waitAsync !== "function") throw new Error("no Atomics.waitAsync");
  // Asked here first: the module's own request can't fail without a panic-free path to JS.
  // ?fallback=1 on this script's URL asks for the browser's fallback (software) adapter.
  const forceFallbackAdapter = new URL(self.location.href).searchParams.get("fallback") === "1";
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance", forceFallbackAdapter });
  if (!adapter) throw new Error("no WebGPU adapter");
  const { vendor, architecture, device, description } = adapter.info || {};
  const started = performance.now();
  await init({ module_or_path: new URL("./pkg/gpu/stwo_gpu_web_worker_bg.wasm", import.meta.url) });
  L = layout();
  gpu = await Gpu.create(forceFallbackAdapter);
  applyIo();
  const adapterInfo = { vendor, architecture, device, description, fallback: adapter.info?.isFallbackAdapter };
  return { adapter: adapterInfo, info: { ...JSON.parse(gpu.info()), kick_ms: kickMs }, start_ms: performance.now() - started };
}

/** The page's transfer sizes, over the module's defaults. */
function applyIo() {
  if (io && io.kick_ms !== undefined) kickMs = io.kick_ms;
  if (!gpu || !io) return;
  const current = JSON.parse(gpu.info());
  const pick = (key) => io[key] ?? current[key];
  gpu.configure(pick("upload_chunk"), pick("upload_window"), pick("readback_chunk"));
  // The striped commitment's twiddles: the prover's (default), or computed here and kept.
  gpu.configure_commit(!!io.compute_twiddles, io.upload ?? "mapped", io.upload_ring ?? 0, io.readback_ring ?? 0, io.max_gpu_mib ?? 0);
}

const slotBase = (s) => L.HEADER_WORDS + s * L.SLOT_WORDS;

function writeError(session, s, message) {
  const bytes = new TextEncoder().encode(String(message)).subarray(0, L.ERROR_BYTES);
  const at = session.base + 4 * (slotBase(s) + L.S_ERROR);
  new Uint8Array(session.memory.buffer, at, bytes.length).set(bytes);
  Atomics.store(session.h, slotBase(s) + L.S_ERROR_LEN, bytes.length);
}

/** The device is gone (or this worker failed): fail every job, now and later. */
function markLost(session, message) {
  const { h } = session;
  Atomics.store(h, L.H_SERVER, L.SERVER_LOST);
  Atomics.notify(h, L.H_SERVER);
  for (let s = 0; s < L.NSLOTS; s++) {
    const state = slotBase(s) + L.S_STATE;
    const now = Atomics.load(h, state);
    if (now === L.SUBMITTED || now === L.RUNNING) {
      try {
        writeError(session, s, message);
      } catch {}
      Atomics.store(h, state, L.ERROR);
      Atomics.notify(h, state);
    }
  }
}

async function runSlot(session, s) {
  const { memory, base, h } = session;
  const slot = slotBase(s);
  // The memory's current buffer: a grown shared memory has a new, longer SharedArrayBuffer.
  const buffer = memory.buffer;
  const times = new Float64Array(buffer, base + 4 * (slot + L.S_TIMES), 6);
  times[L.T_WAKE] = absNow();
  const op = h[slot + L.S_OP];
  const args = new Uint32Array(Math.min(h[slot + L.S_NARGS], L.MAX_ARGS));
  for (let i = 0; i < args.length; i++) args[i] = h[slot + L.S_ARGS + i];
  let state = L.DONE;
  // Firefox: empty submits while the job waits make it poll for completions (see Gpu.kick).
  const kicks = kickMs > 0 && op !== 0 ? setInterval(() => gpu.kick(), kickMs) : null;
  try {
    const phases = await gpu.execute(op, args, buffer);
    times[L.T_UPLOAD] = phases[0];
    times[L.T_UPLOAD + 1] = phases[1];
    times[L.T_UPLOAD + 2] = phases[2];
  } catch (error) {
    state = L.ERROR;
    writeError(session, s, error && error.message ? error.message : error);
    // A trap leaves the module unusable: every job fails from now on.
    if (error instanceof WebAssembly.RuntimeError) fail(error.message);
  } finally {
    if (kicks !== null) clearInterval(kicks);
  }
  times[L.T_DONE] = absNow();
  Atomics.add(h, L.H_COMPLETED, 1);
  Atomics.store(h, slot + L.S_STATE, state);
  Atomics.notify(h, slot + L.S_STATE);
  if (gpu.lost()) markLost(session, "WebGPU device lost");
}

async function serve(session) {
  const { h } = session;
  Atomics.store(h, L.H_SERVER, L.SERVER_READY);
  Atomics.notify(h, L.H_SERVER);
  while (!session.detached && Atomics.load(h, L.H_SERVER) === L.SERVER_READY) {
    const seen = Atomics.load(h, L.H_DOORBELL);
    for (let s = 0; s < L.NSLOTS; s++) {
      const state = slotBase(s) + L.S_STATE;
      if (Atomics.compareExchange(h, state, L.SUBMITTED, L.RUNNING) === L.SUBMITTED) await runSlot(session, s);
    }
    if (session.detached || Atomics.load(h, L.H_DOORBELL) !== seen) continue;
    // { async: false, value: "not-equal" } when the doorbell moved since `seen`: scan again.
    const wait = Atomics.waitAsync(h, L.H_DOORBELL, seen);
    if (wait.async) await wait.value;
  }
}

function attach(port, { memory, base }) {
  const h = new Int32Array(memory.buffer, base, L.REGION_WORDS);
  const header = [L.H_MAGIC, L.H_VERSION, L.H_NSLOTS, L.H_SLOT_WORDS].map((i) => Atomics.load(h, i));
  const expected = [L.MAGIC, L.VERSION, L.NSLOTS, L.SLOT_WORDS];
  if (header.some((value, i) => value !== expected[i])) {
    port.postMessage({ attached: false, error: `bridge header ${header} != ${expected}` });
    return;
  }
  const session = { memory, base, h, detached: false };
  sessions.add(session);
  port.onmessage = async ({ data }) => {
    if (!data.detach) return;
    session.detached = true;
    const lost = Atomics.load(h, L.H_SERVER) === L.SERVER_LOST;
    if (!lost) Atomics.store(h, L.H_SERVER, L.SERVER_NONE);
    Atomics.notify(h, L.H_DOORBELL); // wakes serve()'s waitAsync
    // After a trap, serve() may be stuck on a job whose promise never settles: don't wait then.
    if (!lost) await session.done;
    sessions.delete(session);
    if (!lost) gpu.release();
    port.postMessage({ detached: true, lost, completed: Atomics.load(h, L.H_COMPLETED) });
    port.close();
  };
  session.done = serve(session).catch((error) => markLost(session, String(error)));
  port.postMessage({ attached: true });
}

// A trap in the module (a panic) leaves its promise pending: fail the jobs rather than hang them.
function fail(message) {
  for (const session of sessions) markLost(session, `GPU worker failed: ${message}`);
  self.postMessage({ failed: String(message) });
}
self.addEventListener("error", (event) => fail(event.message));
self.addEventListener("unhandledrejection", (event) => fail(event.reason));

const started = start().then(
  (ready) => (self.postMessage({ ready: true, ...ready }), true),
  (error) => (self.postMessage({ ready: false, error: String(error) }), false),
);

self.onmessage = async ({ data: { connect, configure } }) => {
  if (configure) {
    io = configure;
    applyIo();
  }
  if (!connect) return;
  const ok = await started;
  // Until an attach succeeds (attach() then takes the port over), a detach has nothing to end.
  connect.onmessage = ({ data }) => {
    if (data.detach) connect.postMessage({ detached: true, attached: false });
    if (!data.attach) return;
    if (!ok) connect.postMessage({ attached: false, error: "no GPU" });
    else attach(connect, data.attach);
  };
};
