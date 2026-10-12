// One pass of the forward circle FFT of stwo's SIMD backend (`rfft::fft`), over a batch of
// columns of 2^L coefficients each, laid out one after the other. Prepended: M31_WGSL. Filled in
// by `extension.rs` (at the markers in block comments): the bindings and the load of an input
// word, for the two variants, out of place (the first pass, from the coefficients) and in place
// (the others), and the radix-2^r butterfly units.
//
// The FFT. In a column of n = 2^L values, for the bits k = L-1 down to 0, every pair (i, i + 2^k)
// with bit k of i clear goes through the butterfly (a, b) -> (a + t b, a - t b), whose twiddle t
// depends on the bits of i above k:
// - k >= 1: t = line layer k-1 at i >> (k+1), one of stwo's line twiddles (x coordinates);
// - k = 0, the circle layer: t is a y coordinate, derived from line layer 0 as stwo's
//   `compute_first_twiddles` does.
// The input is the coefficients in stwo's storage order (see `src_index`), the output the
// evaluation in bit-reversed order, as `SimdBackend::evaluate_into` writes it.
//
// A pass applies the layers of the bits [lo, hi) of the column index. A workgroup loads a tile of
// 2^TILE_LOG words into workgroup memory: the words whose batch index (column << L | index) is
// fixed outside the bits [0, c) and [lo, lo + TILE_LOG - c). The low bits [0, c) come along so
// that a load or store covers runs of 2^c consecutive words (c >= 4: 64 bytes at least). The
// layers are then applied in rounds of up to 4 layers: each invocation takes a unit of 2^r words,
// differing in the round's bits, through r layers in registers. A barrier separates the rounds.

struct Params {
    // L: the columns' log size.
    log_n: u32,
    // The pass applies the layers of the bits [lo, hi).
    lo: u32,
    hi: u32,
    // The tile's low bits [0, c), c <= lo; its other bits are [lo, lo + TILE_LOG - c).
    c: u32,
    // The number of words of the batch: its columns times 2^L.
    total: u32,
    // Word offsets of the batch in the input and output buffers.
    src_base: u32,
    dst_base: u32,
    // Word offset of the stripe's line twiddles for log size L in `tw`: line layer j, of
    // 2^(L-2-j) twiddles, at tw_base + 2^(L-1) - 2^(L-1-j).
    tw_base: u32,
    // 1 if the input is in stwo's vec-transposed order (L > CACHED_FFT_LOG_SIZE).
    transposed: u32,
    // The workgroups: n_wg in all, laid out n_wg_x per row of the dispatch grid.
    n_wg_x: u32,
    n_wg: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    pad3: u32,
    pad4: u32,
}

const TILE_LOG: u32 = 12u;
const TILE: u32 = 4096u;
const THREADS: u32 = 256u;

/*BINDINGS*/
@group(0) @binding(2) var<storage, read> tw: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> tile: array<u32, TILE>;

// Workgroup memory index of tile word i: rows of 32 words, each row's words permuted by an XOR
// with the row number, so that the strided accesses of the rounds touch distinct banks.
fn sw(i: u32) -> u32 {
    return i ^ ((i >> 5u) & 31u);
}

// The twiddle of the butterflies of bit k >= 1 of block idx (= i >> (k + 1)).
fn tw_line(k: u32, idx: u32) -> u32 {
    let l = params.log_n;
    return tw[params.tw_base + (1u << (l - 1u)) - (1u << (l - k)) + idx];
}

// The twiddle of the butterfly of the circle layer (bit 0) of pair h (= i >> 1): a y coordinate,
// which the domain's bit-reversed order makes a line twiddle up to sign. As stwo's
// `compute_first_twiddles` has it, the circle twiddles of pairs 4j..4j+3 are [t1, -t1, -t0, t0],
// with t0, t1 the twiddles of line layer 0 at 2j, 2j + 1.
fn tw_circle(h: u32) -> u32 {
    let r = h & 3u;
    let v = tw[params.tw_base + ((h >> 2u) << 1u) + select(0u, 1u, r < 2u)];
    return select(v, m31_neg(v), r == 1u || r == 2u);
}

// The twiddle of the butterflies of bit kb of block idx, the circle layer if kb = 0.
fn tw_low(kb: u32, idx: u32) -> u32 {
    if (kb == 0u) {
        return tw_circle(idx);
    }
    return tw_line(kb, idx);
}

/*RADIX*/

// The batch index of tile word i: its bits [0, c) and [lo, lo + TILE_LOG - c) come from i, the
// others (`mid`) from the workgroup.
fn batch_index(i: u32, mid: u32) -> u32 {
    let c = params.c;
    return (i & ((1u << c) - 1u)) | ((i >> c) << params.lo) | mid;
}

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
) {
    let w = wid.x + wid.y * params.n_wg_x;
    if (w >= params.n_wg) {
        return;
    }
    let c = params.c;
    let lo = params.lo;
    let n_mid = lo - c;
    let mid = ((w & ((1u << n_mid) - 1u)) << c) | ((w >> n_mid) << (lo + TILE_LOG - c));
    let col_mask = (1u << params.log_n) - 1u;

    for (var i = tid; i < TILE; i += THREADS) {
        let g = batch_index(i, mid);
        if (g < params.total) {
            tile[sw(i)] = /*LOAD*/;
        }
    }
    workgroupBarrier();

    // The rounds, from the top layer down.
    var top = params.hi;
    loop {
        if (top <= lo) {
            break;
        }
        let r = min(4u, top - lo);
        // The round's lowest bit, in the column index and in the tile.
        let kb = top - r;
        let lb = kb - lo + c;
        let low_mask = (1u << lb) - 1u;
        for (var u = tid; u < (TILE >> r); u += THREADS) {
            // The unit's first word: u with r zero bits inserted at lb.
            let base = (u & low_mask) | ((u >> lb) << (lb + r));
            let gi = batch_index(base, mid) & col_mask;
            switch r {
                case 4u: {
                    radix16(base, lb, kb, gi);
                }
                case 3u: {
                    radix8(base, lb, kb, gi);
                }
                case 2u: {
                    radix4(base, lb, kb, gi);
                }
                default: {
                    radix2(base, lb, kb, gi);
                }
            }
        }
        workgroupBarrier();
        top = kb;
    }

    for (var i = tid; i < TILE; i += THREADS) {
        let g = batch_index(i, mid);
        if (g < params.total) {
            dst[params.dst_base + g] = tile[sw(i)];
        }
    }
}
