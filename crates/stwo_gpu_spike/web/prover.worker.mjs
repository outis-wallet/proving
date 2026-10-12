// The demo's prover worker, standing in for the Outis prover's worker (prover/js/worker.mjs): it
// loads the threaded module, starts rayon's pool, attaches the GPU worker to the module's memory
// (through the port the page sent), then makes long synchronous calls whose rayon threads use the
// GPU through the bridge. Posts each call's report, then { done }.
import init, * as demo from "./pkg/demo/stwo_gpu_web_demo.js";

/** Posts `message` on `port` and resolves with the reply. */
const request = (port, message) =>
  new Promise((resolve) => {
    port.onmessage = ({ data }) => resolve(data);
    port.postMessage(message);
  });

self.onmessage = async ({ data: { threads, gpuPort, plan } }) => {
  const post = (message) => self.postMessage(message);
  try {
    const wasm = await init({ module_or_path: new URL("./pkg/demo/stwo_gpu_web_demo_bg.wasm", import.meta.url) });
    await demo.initThreadPool(threads);
    // The GPU worker attaches while this worker's event loop still runs: once the synchronous
    // calls start, this worker can't receive its reply (nor start a worker of its own).
    const attached = await request(gpuPort, { attach: { memory: wasm.memory, base: demo.bridge_region() } });
    post({ event: "attached", ...attached, gpu_ready: demo.gpu_ready() });
    for (const { fn, args } of plan) {
      const started = performance.now();
      const report = JSON.parse(demo[fn](...args));
      post({ event: "report", call: fn, args, call_ms: performance.now() - started, report });
      // Between calls (not needed by the bridge): lets the posts go out as they're made.
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
    const detached = await request(gpuPort, { detach: true });
    post({ event: "done", ...detached, wasm_memory_mib: demo.memory_mib() });
  } catch (error) {
    post({ event: "error", error: `${error}\n${error && error.stack}` });
  }
};
