// The demo page: starts the GPU worker (page-owned, so it outlives each proof), then a prover
// worker, and connects them with a MessageChannel: the prover's worker sends its memory straight
// to the GPU worker, and the page never holds it. Logs each report and POSTs it to /result
// (agent-tools/server.py appends it to results.log), ending with { event: "end" }.
const params = new URLSearchParams(location.search);
const number = (key, fallback) => Number(params.get(key)) || fallback;
const threads = number("threads", Math.min(navigator.hardwareConcurrency || 4, 8));
const jobs = number("jobs", 256);
const iters = number("iters", 200);
const reps = number("reps", 3);
const phases = (params.get("phases") || "correctness,errors,latency,throughput").split(",");
const sizes = (params.get("sizes") || "64,128,256,512").split(",").map(Number).filter(Boolean);
// The GPU worker's transfer sizes in MiB (0 allowed: see Gpu.configure); unset: its defaults.
const mib = (key) => (params.has(key) ? Math.round(Number(params.get(key)) * 2 ** 20) : undefined);
const io = {
  upload_chunk: mib("upload_chunk_mib"),
  upload_window: mib("upload_window_mib"),
  readback_chunk: mib("readback_chunk_mib"),
  kick_ms: params.has("kick_ms") ? Number(params.get("kick_ms")) : undefined,
};

const log = document.getElementById("log");
function report(message) {
  log.textContent += `${JSON.stringify(message, null, 1)}\n`;
  fetch("/result", { method: "POST", body: JSON.stringify(message) }).catch(() => {});
}

const once = (worker) =>
  new Promise((resolve) => {
    worker.addEventListener("message", ({ data }) => resolve(data), { once: true });
    worker.addEventListener("error", (event) => resolve({ ready: false, error: event.message }), { once: true });
  });

async function main() {
  report({
    event: "page",
    userAgent: navigator.userAgent,
    crossOriginIsolated: self.crossOriginIsolated,
    webgpu: !!navigator.gpu,
    hardwareConcurrency: navigator.hardwareConcurrency,
    threads,
  });
  const gpuWorker = new Worker("./gpu.worker.mjs", { type: "module" });
  gpuWorker.postMessage({ configure: io });
  const gpu = await once(gpuWorker);
  report({ event: "gpu", ...gpu });
  gpuWorker.addEventListener("message", ({ data }) => data.failed && report({ event: "gpu-failed", ...data }));

  const plan = [
    ...(phases.includes("correctness") ? [{ fn: "demo_correctness", args: [jobs] }] : []),
    ...(phases.includes("errors") ? [{ fn: "demo_errors", args: [] }] : []),
    ...(phases.includes("latency") ? [{ fn: "demo_latency", args: [iters] }] : []),
    ...(phases.includes("throughput") ? sizes.map((mib) => ({ fn: "demo_throughput", args: [mib, reps] })) : []),
    // Not by default: it ends the GPU worker's usefulness for the page's life.
    ...(phases.includes("trap") ? [{ fn: "demo_trap", args: [] }] : []),
  ];
  const prover = new Worker("./prover.worker.mjs", { type: "module" });
  const channel = new MessageChannel();
  gpuWorker.postMessage({ connect: channel.port1 }, [channel.port1]);
  prover.postMessage({ threads, gpuPort: channel.port2, plan }, [channel.port2]);
  await new Promise((resolve) => {
    prover.onmessage = ({ data }) => {
      report(data);
      if (data.event === "done" || data.event === "error") resolve();
    };
    prover.onerror = (event) => {
      report({ event: "error", error: event.message });
      resolve();
    };
  });
  prover.terminate();
  report({ event: "end" });
}

main().catch((error) => report({ event: "end", error: String(error) }));
