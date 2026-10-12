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
const sizes = (params.get("sizes") || "64,128,256,512").split(",").map(Number).filter(Boolean);

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
  const gpu = await once(gpuWorker);
  report({ event: "gpu", ...gpu });
  gpuWorker.addEventListener("message", ({ data }) => data.failed && report({ event: "gpu-failed", ...data }));

  const plan = [
    { fn: "demo_correctness", args: [jobs] },
    { fn: "demo_errors", args: [] },
    { fn: "demo_latency", args: [iters] },
    ...sizes.map((mib) => ({ fn: "demo_throughput", args: [mib, reps] })),
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
