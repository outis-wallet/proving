// The FRI quotients (see fri.rs), after m31.wgsl and qm31.wgsl.
//
// `combine`: a batch's columns (one log size, one sample point) combined with their `c`
// coefficients, coordinate by coordinate: for every word t of the batch's columns,
//   out[k][t] += sum_m c_m[k] * src[offset_m + t],   k = 0..4,
// the members m of one source buffer. On coefficients (the batch's numerator polynomial, before
// its evaluation on stripe 0) or on stripe-0 values (the columns not kept as coefficients).
//
// `quotients`: per row i of the lifting domain's first stripe (bit-reversed), the sum over the
// sample points p of
//   (sum_{batches g of p} E_g[lift(i, N - n_g)] - B_p - A_p y_i) / ((Re p.x - x_i) Im p.y - (Re p.y - y_i) Im p.x)
// with E_g the batch's numerator on its stripe 0, A_p and B_p the sums of the `a` and `b` line
// coefficients of p's batches, (x_i, y_i) the row's point.

struct CombineParams {
    n_words: u32,
    first_member: u32,
    n_members: u32,
    wg_x: u32,
    out0: u32,
    out1: u32,
    out2: u32,
    out3: u32,
}

@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read> member_offsets: array<u32>;
@group(0) @binding(2) var<storage, read> member_coeffs: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> out: array<u32>;
@group(0) @binding(4) var<uniform> combine_params: CombineParams;

@compute @workgroup_size(256)
fn combine(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let t = gid.x + gid.y * combine_params.wg_x * 256u;
    if (t >= combine_params.n_words) {
        return;
    }
    var acc = vec4<u32>(0u);
    let end = combine_params.first_member + combine_params.n_members;
    for (var m = combine_params.first_member; m < end; m++) {
        let value = src[member_offsets[m] + t];
        acc = m31_add4(acc, m31_mul4(member_coeffs[m], vec4<u32>(value)));
    }
    let p = combine_params;
    out[p.out0 + t] = m31_add(out[p.out0 + t], acc.x);
    out[p.out1 + t] = m31_add(out[p.out1 + t], acc.y);
    out[p.out2 + t] = m31_add(out[p.out2 + t], acc.z);
    out[p.out3 + t] = m31_add(out[p.out3 + t], acc.w);
}

// The quotients' parameters: the domain, the points and the batches.
struct QuotientParams {
    // The rows: 2^log_n (the lifting domain's first stripe).
    log_n: u32,
    n_points: u32,
    wg_x: u32,
    // The row's output words: plane k at out_base[k] (the first in `q_out`'s order).
    out0: u32,
    out1: u32,
    out2: u32,
    out3: u32,
    pad: u32,
}

// (Bindings 10 and up: the module's other kernel has 0 to 4.)
// Per point (16 words): Re p.x (2), Re p.y (2), Im p.x (2), Im p.y (2), A_p (4), B_p (4); per
// point too, its batches: first and count (in `points_batches`).
@group(0) @binding(10) var<storage, read> evaluations: array<u32>;
@group(0) @binding(11) var<storage, read> points: array<u32>;
@group(0) @binding(12) var<storage, read> points_batches: array<vec2<u32>>;
// Per batch (8 words): its 4 planes' offsets in `evaluations`, its log size, pad.
@group(0) @binding(13) var<storage, read> batches: array<u32>;
// The domain: its half coset's initial point (x, y), then step * 2^b (x, y) for b < log_n - 1.
@group(0) @binding(14) var<storage, read> domain: array<u32>;
@group(0) @binding(15) var<storage, read_write> q_out: array<u32>;
@group(0) @binding(16) var<uniform> quotient_params: QuotientParams;

fn point_add(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let p = m31_mul4(vec4<u32>(a.x, a.y, a.x, a.y), vec4<u32>(b.x, b.y, b.y, b.x));
    return vec2<u32>(m31_sub(p.x, p.y), m31_add(p.z, p.w));
}

// 1 / (a + b i) = (a - b i) / (a^2 + b^2).
fn cm31_inv(x: vec2<u32>) -> vec2<u32> {
    let norm = m31_add(m31_mul(x.x, x.x), m31_mul(x.y, x.y));
    let inv = m31_inv(norm);
    return vec2<u32>(m31_mul(x.x, inv), m31_mul(m31_neg(x.y), inv));
}

@compute @workgroup_size(256)
fn quotients(@builtin(global_invocation_id) gid: vec3<u32>) {
    let qp = quotient_params;
    let i = gid.x + gid.y * qp.wg_x * 256u;
    let n = qp.log_n;
    if (i >= (1u << n)) {
        return;
    }
    // The point at bit-reversed position i: the half coset's point j = rev_{n-1}(i >> 1),
    // conjugated if i is odd.
    var j = 0u;
    if (n > 1u) {
        j = reverse_bits32(i >> 1u) >> (33u - n);
    }
    var pt = vec2<u32>(domain[0], domain[1]);
    for (var b = 0u; b + 1u < n; b++) {
        if (((j >> b) & 1u) == 1u) {
            pt = point_add(pt, vec2<u32>(domain[2u + 2u * b], domain[3u + 2u * b]));
        }
    }
    let x = pt.x;
    var y = pt.y;
    if ((i & 1u) == 1u) {
        y = m31_neg(y);
    }
    var q = vec4<u32>(0u);
    for (var p = 0u; p < qp.n_points; p++) {
        let base = 16u * p;
        let prx = vec2<u32>(points[base], points[base + 1u]);
        let pry = vec2<u32>(points[base + 2u], points[base + 3u]);
        let pix = vec2<u32>(points[base + 4u], points[base + 5u]);
        let piy = vec2<u32>(points[base + 6u], points[base + 7u]);
        let a = vec4<u32>(points[base + 8u], points[base + 9u], points[base + 10u], points[base + 11u]);
        let bsum = vec4<u32>(points[base + 12u], points[base + 13u], points[base + 14u], points[base + 15u]);
        // The numerator.
        var num = vec4<u32>(0u);
        let range = points_batches[p];
        for (var g = range.x; g < range.x + range.y; g++) {
            let gb = 8u * g;
            let log_g = batches[gb + 4u];
            // The lift: row ((i >> (r + 1)) << 1) + (i & 1) of the batch, r = n - log_g.
            var row = i;
            if (log_g < n) {
                row = ((i >> (n - log_g + 1u)) << 1u) + (i & 1u);
            }
            num = m31_add4(num, vec4<u32>(
                evaluations[batches[gb] + row],
                evaluations[batches[gb + 1u] + row],
                evaluations[batches[gb + 2u] + row],
                evaluations[batches[gb + 3u] + row],
            ));
        }
        num = m31_sub4(num, bsum);
        num = m31_sub4(num, m31_mul4(a, vec4<u32>(y)));
        // The denominator (Re p.x - x) Im p.y - (Re p.y - y) Im p.x, in CM31.
        let dx = vec2<u32>(m31_sub(prx.x, x), prx.y);
        let dy = vec2<u32>(m31_sub(pry.x, y), pry.y);
        let den = cm31_sub(cm31_mul(dx, piy), cm31_mul(dy, pix));
        let inv = cm31_inv(den);
        let quotient = vec4<u32>(cm31_mul(num.xy, inv), cm31_mul(num.zw, inv));
        q = m31_add4(q, quotient);
    }
    q_out[qp.out0 + i] = q.x;
    q_out[qp.out1 + i] = q.y;
    q_out[qp.out2 + i] = q.z;
    q_out[qp.out3 + i] = q.w;
}

fn cm31_sub(x: vec2<u32>, y: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(m31_sub(x.x, y.x), m31_sub(x.y, y.y));
}
