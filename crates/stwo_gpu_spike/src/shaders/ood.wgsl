// The out-of-domain samples of coefficient columns (see ood.rs), after m31.wgsl and qm31.wgsl.
//
// A column of 2^n coefficients, in stwo's storage order, holds vector h (16 coefficients) at
// words 16h..16h+16. At a point, the weight of vector h is W(h) = lo[h mod 2^L] * hi[h >> L], in
// QM31. `fold` computes, per column c and span s (`span` chunks u of 2^L vectors each), the 16 sums
//   partial[c][s][t] = sum_{u in s} hi[u] * sum_v coeffs[16 (u 2^L + v) + t] * lo[v]
// (QM31, stored as 4 planes of 16 words); `reduce` sums them over the spans:
//   out[c][k][t] = sum_s partial[c][s][k][t].

struct Params {
    lo_log: u32,
    span: u32,
    n_spans: u32,
    n_columns: u32,
    wg_x: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<storage, read> coeffs: array<u32>;
@group(0) @binding(1) var<storage, read> columns: array<u32>;
@group(0) @binding(2) var<storage, read> lo: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> hi: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read_write> partial: array<u32>;
@group(0) @binding(5) var<uniform> params: Params;
@group(0) @binding(6) var<storage, read_write> out: array<u32>;

// A chunk's coefficients: at most 2^8 vectors of 16.
var<workgroup> tile: array<u32, 4096>;
// Every invocation's sums, for the reduction over the 16 slices.
var<workgroup> sums: array<vec4<u32>, 256>;

@compute @workgroup_size(256)
fn fold(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let id = wg.x + wg.y * params.wg_x;
    // Uniform within the workgroup (the barriers below stay in uniform control flow).
    if (id >= params.n_columns * params.n_spans) {
        return;
    }
    let s = id % params.n_spans;
    let c = id / params.n_spans;
    let n_vectors = 1u << params.lo_log;
    let chunk_words = 16u * n_vectors;
    // Lane t of the vectors, slice `slice` of them (the vectors v = slice mod 16).
    let t = li % 16u;
    let slice = li / 16u;
    var acc = vec4<u32>(0u);
    for (var k = 0u; k < params.span; k++) {
        let u = s * params.span + k;
        let base = columns[c] + u * chunk_words;
        // The previous chunk's reads are done before the tile is overwritten.
        workgroupBarrier();
        for (var i = li; i < chunk_words; i += 256u) {
            tile[i] = coeffs[base + i];
        }
        workgroupBarrier();
        var chunk = vec4<u32>(0u);
        for (var v = slice; v < n_vectors; v += 16u) {
            chunk = m31_add4(chunk, m31_mul4(vec4<u32>(tile[v * 16u + t]), lo[v]));
        }
        acc = m31_add4(acc, qm31_mul(chunk, hi[u]));
    }
    sums[li] = acc;
    workgroupBarrier();
    if (li < 16u) {
        var total = vec4<u32>(0u);
        for (var j = 0u; j < 16u; j++) {
            total = m31_add4(total, sums[j * 16u + li]);
        }
        let at = (c * params.n_spans + s) * 64u + li;
        partial[at] = total.x;
        partial[at + 16u] = total.y;
        partial[at + 32u] = total.z;
        partial[at + 48u] = total.w;
    }
}

@compute @workgroup_size(256)
fn reduce(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.x + gid.y * nwg.x * 256u;
    if (i >= params.n_columns * 64u) {
        return;
    }
    let c = i / 64u;
    let r = i % 64u;
    var acc = 0u;
    for (var s = 0u; s < params.n_spans; s++) {
        acc = m31_add(acc, partial[(c * params.n_spans + s) * 64u + r]);
    }
    out[i] = acc;
}
