// probe.html's worker: the bridge's requirements, checked where the GPU worker runs.
const out = {
  crossOriginIsolated: self.crossOriginIsolated,
  webgpu_in_worker: !!navigator.gpu,
  atomics_wait_async: typeof Atomics.waitAsync === "function",
  shared_array_buffer: typeof SharedArrayBuffer === "function",
};
const LIMITS = [
  "maxBufferSize",
  "maxStorageBufferBindingSize",
  "maxComputeWorkgroupsPerDimension",
  "maxComputeInvocationsPerWorkgroup",
  "maxComputeWorkgroupSizeX",
  "maxComputeWorkgroupStorageSize",
  "maxStorageBuffersPerShaderStage",
  "minStorageBufferOffsetAlignment",
];

async function probe() {
  if (!navigator.gpu) return;
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) {
    out.adapter = null;
    return;
  }
  const { vendor, architecture, device: name, description } = adapter.info || {};
  out.adapter = { vendor, architecture, device: name, description, fallback: adapter.info?.isFallbackAdapter };
  out.adapter_limits = Object.fromEntries(LIMITS.map((key) => [key, adapter.limits[key]]));
  out.features = [...adapter.features].sort();
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    },
  });
  const errors = [];
  device.addEventListener("uncapturederror", (event) => errors.push(String(event.error.message)));

  // A shared memory like the prover's (threaded wasm imports one).
  const n = 1 << 20;
  const memory = new WebAssembly.Memory({ initial: 64, maximum: 1024, shared: true });
  out.wasm_memory_buffer = Object.prototype.toString.call(memory.buffer);
  const source = new Uint8Array(memory.buffer, 4096, n);
  for (let i = 0; i < n; i++) source[i] = (i * 7 + 3) & 255;
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const buffer = device.createBuffer({ size: n, usage });
  out.write_buffer = {};
  for (const [name, data] of [
    ["shared_u8_view", source],
    ["shared_u32_view", new Uint32Array(memory.buffer, 4096, n / 4)],
  ]) {
    try {
      device.queue.writeBuffer(buffer, 0, data);
      out.write_buffer[name] = "accepted";
    } catch (error) {
      out.write_buffer[name] = `throws ${error}`;
    }
  }
  // What landed on the GPU, read back into the shared memory.
  try {
    device.queue.writeBuffer(buffer, 0, source);
    const staging = device.createBuffer({ size: n, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, 0, staging, 0, n);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const dest = new Uint8Array(memory.buffer, 4096 + n, n);
    dest.set(new Uint8Array(staging.getMappedRange()));
    staging.unmap();
    let equal = true;
    for (let i = 0; i < n && equal; i++) equal = dest[i] === source[i];
    out.round_trip_equal = equal;
  } catch (error) {
    out.round_trip_error = String(error);
  }
  // waitAsync wakes on a notify from another agent (here: this one, after a tick).
  if (out.atomics_wait_async) {
    const word = new Int32Array(memory.buffer, 0, 1);
    const wait = Atomics.waitAsync(word, 0, 0, 1000);
    setTimeout(() => {
      Atomics.store(word, 0, 1);
      Atomics.notify(word, 0);
    }, 1);
    out.wait_async_result = wait.async ? await wait.value : wait.value;
  }
  await device.queue.onSubmittedWorkDone();
  out.uncaptured_errors = errors;
}

probe()
  .catch((error) => (out.error = `${error}`))
  .finally(() => postMessage(out));
