// The decommitment's evaluations of coefficient columns (see decommit.rs), after m31.wgsl.
//
// A column of 2^n coefficients, in stwo's storage order, holds vector h (16 coefficients) at
// words 16h..16h+16. For a group g of points sharing their mappings past the first four, the
// weight of vector h is W_g(h) = lo[g][h mod 2^L] * hi[g][h >> L]. `fold` computes, per column c,
// chunk u (the vectors u 2^L .. (u+1) 2^L) and group g, the 16 sums
//   partial[c][u][g][t] = hi[g][u] * sum_v coeffs[16 (u 2^L + v) + t] * lo[g][v];
// `reduce` sums them over the chunks: out[c][g][t] = sum_u partial[c][u][g][t].

struct Params {
    lo_log: u32,
    n_chunks: u32,
    n_groups: u32,
    n_columns: u32,
    wg_x: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<storage, read> coeffs: array<u32>;
@group(0) @binding(1) var<storage, read> columns: array<u32>;
@group(0) @binding(2) var<storage, read> lo: array<u32>;
@group(0) @binding(3) var<storage, read> hi: array<u32>;
@group(0) @binding(4) var<storage, read_write> partial: array<u32>;
@group(0) @binding(5) var<uniform> params: Params;
@group(0) @binding(6) var<storage, read_write> out: array<u32>;

// A chunk's coefficients: at most 2^8 vectors of 16.
var<workgroup> tile: array<u32, 4096>;

@compute @workgroup_size(256)
fn fold(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let n_blocks = (params.n_groups + 15u) / 16u;
    let id = wg.x + wg.y * params.wg_x;
    if (id >= params.n_columns * params.n_chunks * n_blocks) {
        return;
    }
    let block = id % n_blocks;
    let u = (id / n_blocks) % params.n_chunks;
    let c = id / (n_blocks * params.n_chunks);
    let n_vectors = 1u << params.lo_log;
    let chunk_words = 16u * n_vectors;
    let base = columns[c] + u * chunk_words;
    for (var i = li; i < chunk_words; i += 256u) {
        tile[i] = coeffs[base + i];
    }
    workgroupBarrier();
    let g = block * 16u + li / 16u;
    let t = li % 16u;
    if (g < params.n_groups) {
        var acc = 0u;
        let lo_base = g * n_vectors;
        for (var v = 0u; v < n_vectors; v++) {
            acc = m31_add(acc, m31_mul(tile[v * 16u + t], lo[lo_base + v]));
        }
        acc = m31_mul(acc, hi[g * params.n_chunks + u]);
        partial[((c * params.n_chunks + u) * params.n_groups + g) * 16u + t] = acc;
    }
}

@compute @workgroup_size(256)
fn reduce(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.x + gid.y * nwg.x * 256u;
    let per_column = params.n_groups * 16u;
    if (i >= params.n_columns * per_column) {
        return;
    }
    let c = i / per_column;
    let r = i % per_column;
    var acc = 0u;
    for (var u = 0u; u < params.n_chunks; u++) {
        acc = m31_add(acc, partial[(c * params.n_chunks + u) * per_column + r]);
    }
    out[i] = acc;
}
