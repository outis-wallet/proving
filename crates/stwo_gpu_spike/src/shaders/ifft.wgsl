// The inverse circle FFT of stwo's SIMD backend (`SimdBackend::interpolate`), over a batch of
// columns of 2^L evaluations each (bit-reversed order, on a circle domain), laid out one after the
// other, in place; then the scaling by 1/2^L into the coefficients' buffer, in stwo's storage
// order. Prepended: M31_WGSL. A spike's kernels, plain radix 2:
// - `low`: the layers of the bits [0, min(L, 12)), a tile of 4096 words per workgroup, in
//   workgroup memory (a barrier between the layers);
// - `layer`: one layer of bit k >= 12, one butterfly per invocation;
// - `scale`: every word times 1/2^L, written where stwo stores that coefficient.
//
// The inverse butterfly of bit k on the pair (i, i + 2^k), bit k of i clear: (a, b) ->
// (a + b, (a - b) t), as stwo's `ibutterfly`, the layers from bit 0 up, with the inverse
// twiddles of the domain:
// - k >= 1: t = inverse line layer k-1 at i >> (k+1);
// - k = 0, the circle layer: from inverse line layer 0 as stwo's `circle_twiddles_from_line_twiddles`
//   (which "works also for inverse of the twiddles"): pairs 4j..4j+3 take [t1, -t1, -t0, t0], with
//   t0, t1 inverse line layer 0 at 2j, 2j + 1.

struct Params {
    // L: the columns' log size.
    log_n: u32,
    // `layer`: the bit k.
    k: u32,
    // The words of the batch: its columns times 2^L.
    total: u32,
    // Word offsets of the batch in `buf` and (`scale`) in `dst`.
    base: u32,
    dst_base: u32,
    // 1 / 2^L (`scale`).
    inv: u32,
    // 1 if the coefficients are stored with their vectors transposed (L > CACHED_FFT_LOG_SIZE).
    transposed: u32,
    // The workgroups of a row of the dispatch grid.
    n_wg_x: u32,
}

@group(0) @binding(0) var<storage, read_write> buf: array<u32>;
@group(0) @binding(1) var<storage, read> itw: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> dst: array<u32>;

var<workgroup> tile: array<u32, 4096>;

// Inverse line layer k-1 at idx (k >= 1): layer j of 2^(L-2-j) twiddles at 2^(L-1) - 2^(L-1-j).
fn tw_line(k: u32, idx: u32) -> u32 {
    let l = params.log_n;
    return itw[(1u << (l - 1u)) - (1u << (l - k)) + idx];
}

// The circle layer's twiddle of pair h (= i >> 1).
fn tw_circle(h: u32) -> u32 {
    let r = h & 3u;
    let v = itw[((h >> 2u) << 1u) + select(0u, 1u, r < 2u)];
    return select(v, m31_neg(v), r == 1u || r == 2u);
}

fn twiddle(k: u32, i: u32) -> u32 {
    if (k == 0u) {
        return tw_circle(i >> 1u);
    }
    return tw_line(k, i >> (k + 1u));
}

@compute @workgroup_size(256)
fn low(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
) {
    let w = wid.y * params.n_wg_x + wid.x;
    let n_tiles = (params.total + 4095u) >> 12u;
    if (w >= n_tiles) {
        return;
    }
    let first = w << 12u;
    let col_mask = (1u << params.log_n) - 1u;
    for (var i = tid; i < 4096u; i += 256u) {
        if (first + i < params.total) {
            tile[i] = buf[params.base + first + i];
        }
    }
    workgroupBarrier();
    let layers = min(params.log_n, 12u);
    for (var k = 0u; k < layers; k++) {
        for (var q = tid; q < 2048u; q += 256u) {
            let i = ((q >> k) << (k + 1u)) | (q & ((1u << k) - 1u));
            let j = i + (1u << k);
            // i and j are in the same column (k < L), the first in it if i is.
            if (first + i < params.total) {
                let t = twiddle(k, (first + i) & col_mask);
                let a = tile[i];
                let b = tile[j];
                tile[i] = m31_add(a, b);
                tile[j] = m31_mul(m31_sub(a, b), t);
            }
        }
        workgroupBarrier();
    }
    for (var i = tid; i < 4096u; i += 256u) {
        if (first + i < params.total) {
            buf[params.base + first + i] = tile[i];
        }
    }
}

@compute @workgroup_size(256)
fn layer(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
) {
    let p = (wid.y * params.n_wg_x + wid.x) * 256u + tid;
    if (p >= params.total >> 1u) {
        return;
    }
    let l = params.log_n;
    let k = params.k;
    let col = p >> (l - 1u);
    let q = p & ((1u << (l - 1u)) - 1u);
    let i = ((q >> k) << (k + 1u)) | (q & ((1u << k) - 1u));
    let j = i + (1u << k);
    let at = params.base + (col << l);
    let a = buf[at + i];
    let b = buf[at + j];
    buf[at + i] = m31_add(a, b);
    buf[at + j] = m31_mul(m31_sub(a, b), twiddle(k, i));
}

// Where stwo stores coefficient g of the batch: in place, or, for a column larger than
// 2^CACHED_FFT_LOG_SIZE, with its vectors of 16 words transposed, vector index (a, b, c) <->
// (c, b, a) with |a| = |c| (`transpose_vecs`), as extension.wgsl's `src_index` reads them.
fn dst_index(g: u32) -> u32 {
    if (params.transposed == 0u) {
        return params.dst_base + g;
    }
    let l = params.log_n;
    let gi = g & ((1u << l) - 1u);
    let v = gi >> 4u;
    let n_vecs_log = l - 4u;
    let half = n_vecs_log >> 1u;
    let mid = n_vecs_log & 1u;
    let a = v & ((1u << half) - 1u);
    let b = (v >> half) & ((1u << mid) - 1u);
    let c = v >> (half + mid);
    let m = (a << (half + mid)) | (b << half) | c;
    return params.dst_base + ((g >> l) << l) + (m << 4u) + (gi & 15u);
}

@compute @workgroup_size(256)
fn scale(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
) {
    let g = (wid.y * params.n_wg_x + wid.x) * 256u + tid;
    if (g >= params.total) {
        return;
    }
    dst[dst_index(g)] = m31_mul(buf[params.base + g], params.inv);
}
