// CM31 and QM31 arithmetic on canonical M31 values (append to m31.wgsl). A QM31 value
// (a + bi) + (c + di)u, with i^2 = -1 and u^2 = 2 + i (stwo's `R`), is the vec4<u32> (a, b, c, d).

fn m31_add4(a: vec4<u32>, b: vec4<u32>) -> vec4<u32> {
    let c = a + b;
    return select(c, c - vec4<u32>(P), c >= vec4<u32>(P));
}

fn m31_sub4(a: vec4<u32>, b: vec4<u32>) -> vec4<u32> {
    let c = a + vec4<u32>(P) - b;
    return select(c, c - vec4<u32>(P), c >= vec4<u32>(P));
}

fn m31_neg4(a: vec4<u32>) -> vec4<u32> {
    let c = vec4<u32>(P) - a;
    return select(c, c - vec4<u32>(P), c >= vec4<u32>(P));
}

// Lane-wise products, as m31_mul.
fn m31_mul4(a: vec4<u32>, b: vec4<u32>) -> vec4<u32> {
    let a_lo = a & vec4<u32>(0xffffu);
    let a_hi = a >> vec4<u32>(16u);
    let b_lo = b & vec4<u32>(0xffffu);
    let b_hi = b >> vec4<u32>(16u);
    let lo_lo = a_lo * b_lo;
    let hi_lo = a_hi * b_lo;
    let lo_hi = a_lo * b_hi;
    let hi_hi = a_hi * b_hi;
    // a, b < 2^31: each cross term is below 2^31, so mid fits in 32 bits.
    let mid = hi_lo + lo_hi;
    let lo = lo_lo + (mid << vec4<u32>(16u));
    let lo_carry = select(vec4<u32>(0u), vec4<u32>(1u), lo < lo_lo);
    let hi = hi_hi + (mid >> vec4<u32>(16u)) + lo_carry;
    // a * b = hi * 2^32 + lo, hi < 2^30; 2^31 = 1 mod p.
    let s = (lo & vec4<u32>(P)) + (lo >> vec4<u32>(31u)) + (hi << vec4<u32>(1u));
    let t = select(s, s - vec4<u32>(P), s >= vec4<u32>(P));
    return select(t, t - vec4<u32>(P), t >= vec4<u32>(P));
}

// a^(p-2) = 1/a (0 for 0). p - 2 = 2^31 - 3 = (2^29 - 1)·4 + 1: 29 ones, then 0b01.
fn m31_inv(a: u32) -> u32 {
    let r2 = m31_mul(m31_mul(a, a), a);              // a^(2^2 - 1)
    let r4 = m31_mul(m31_pow2k(r2, 2u), r2);         // a^(2^4 - 1)
    let r8 = m31_mul(m31_pow2k(r4, 4u), r4);         // a^(2^8 - 1)
    let r16 = m31_mul(m31_pow2k(r8, 8u), r8);        // a^(2^16 - 1)
    let r24 = m31_mul(m31_pow2k(r16, 8u), r8);       // a^(2^24 - 1)
    let r28 = m31_mul(m31_pow2k(r24, 4u), r4);       // a^(2^28 - 1)
    let r29 = m31_mul(m31_pow2k(r28, 1u), a);        // a^(2^29 - 1)
    return m31_mul(m31_pow2k(r29, 2u), a);
}

// a^(2^k).
fn m31_pow2k(a: u32, k: u32) -> u32 {
    var x = a;
    for (var i = 0u; i < k; i++) {
        x = m31_mul(x, x);
    }
    return x;
}

// CM31 product (a + bi)(c + di) = (ac - bd) + (ad + bc)i, of x = (a, b) and y = (c, d).
fn cm31_mul(x: vec2<u32>, y: vec2<u32>) -> vec2<u32> {
    let p = m31_mul4(vec4<u32>(x.x, x.y, x.x, x.y), vec4<u32>(y.x, y.y, y.y, y.x));
    return vec2<u32>(m31_sub(p.x, p.y), m31_add(p.z, p.w));
}

// (2 + i)·(a + bi) = (2a - b) + (a + 2b)i.
fn cm31_mul_r(x: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(m31_sub(m31_add(x.x, x.x), x.y), m31_add(x.x, m31_add(x.y, x.y)));
}

fn cm31_add(x: vec2<u32>, y: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(m31_add(x.x, y.x), m31_add(x.y, y.y));
}

// (a0 + a1 u)(b0 + b1 u) = (a0 b0 + R a1 b1) + (a0 b1 + a1 b0) u.
fn qm31_mul(a: vec4<u32>, b: vec4<u32>) -> vec4<u32> {
    let a0 = a.xy;
    let a1 = a.zw;
    let b0 = b.xy;
    let b1 = b.zw;
    let c0 = cm31_add(cm31_mul(a0, b0), cm31_mul_r(cm31_mul(a1, b1)));
    let c1 = cm31_add(cm31_mul(a0, b1), cm31_mul(a1, b0));
    return vec4<u32>(c0, c1);
}

// A QM31 value times an M31 value.
fn qm31_mul_m31(a: vec4<u32>, b: u32) -> vec4<u32> {
    return m31_mul4(a, vec4<u32>(b));
}

// A QM31 value plus an M31 value.
fn qm31_add_m31(a: vec4<u32>, b: u32) -> vec4<u32> {
    return vec4<u32>(m31_add(a.x, b), a.yzw);
}

// An M31 value minus a QM31 value.
fn m31_sub_qm31(a: u32, b: vec4<u32>) -> vec4<u32> {
    let n = m31_neg4(b);
    return vec4<u32>(m31_add(a, n.x), n.yzw);
}

// A QM31 value minus an M31 value.
fn qm31_sub_m31(a: vec4<u32>, b: u32) -> vec4<u32> {
    return vec4<u32>(m31_sub(a.x, b), a.yzw);
}

// The index in a bit-reversed evaluation of log size `eval_log` of the row `offset` steps of the
// trace domain (log size `trace_log`) from row `i`: stwo's
// `offset_bit_reversed_circle_domain_index`.
fn offset_row(i: u32, trace_log: u32, eval_log: u32, offset: i32) -> u32 {
    let prev = reverse_bits32(i) >> (32u - eval_log);
    let half = 1u << (eval_log - 1u);
    let step = u32(abs(offset)) << (eval_log - trace_log - 1u);
    // (x + s) mod half and (x - s) mod half, with s reduced mod half first.
    let s = step & (half - 1u);
    let forward = offset >= 0;
    var moved: u32;
    if (prev < half) {
        moved = select((prev + half - s) & (half - 1u), (prev + s) & (half - 1u), forward);
    } else {
        let x = prev - half;
        moved = select((x + s) & (half - 1u), (x + half - s) & (half - 1u), forward) + half;
    }
    return reverse_bits32(moved) >> (32u - eval_log);
}

// The bits of x in reverse order (WGSL's reverseBits, which naga's default MSL target lacks).
fn reverse_bits32(x: u32) -> u32 {
    var v = x;
    v = ((v >> 1u) & 0x55555555u) | ((v & 0x55555555u) << 1u);
    v = ((v >> 2u) & 0x33333333u) | ((v & 0x33333333u) << 2u);
    v = ((v >> 4u) & 0x0f0f0f0fu) | ((v & 0x0f0f0f0fu) << 4u);
    v = ((v >> 8u) & 0x00ff00ffu) | ((v & 0x00ff00ffu) << 8u);
    return (v >> 16u) | (v << 16u);
}
