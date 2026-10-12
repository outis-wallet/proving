// M31 arithmetic, p = 2^31 - 1, on canonical values in [0, p). WGSL has no 64-bit integers, so
// products are formed from 16-bit halves.
const P: u32 = 0x7fffffffu;

fn m31_reduce_once(x: u32) -> u32 {
    // x < 2p: one conditional subtraction.
    return select(x, x - P, x >= P);
}

fn m31_add(a: u32, b: u32) -> u32 {
    return m31_reduce_once(a + b);
}

fn m31_sub(a: u32, b: u32) -> u32 {
    return m31_reduce_once(a + P - b);
}

fn m31_neg(a: u32) -> u32 {
    return m31_reduce_once(P - a);
}

// The 62-bit product a * b as (hi, lo) 32-bit words.
fn mul_wide(a: u32, b: u32) -> vec2<u32> {
    let a_lo = a & 0xffffu;
    let a_hi = a >> 16u;
    let b_lo = b & 0xffffu;
    let b_hi = b >> 16u;
    let lo_lo = a_lo * b_lo;
    let hi_lo = a_hi * b_lo;
    let lo_hi = a_lo * b_hi;
    let hi_hi = a_hi * b_hi;
    // Each cross term is below 2^31 (a, b < 2^31), so mid fits; the carry is kept for clarity.
    let mid = hi_lo + lo_hi;
    let mid_carry = select(0u, 1u, mid < hi_lo);
    let lo = lo_lo + (mid << 16u);
    let lo_carry = select(0u, 1u, lo < lo_lo);
    let hi = hi_hi + (mid >> 16u) + (mid_carry << 16u) + lo_carry;
    return vec2<u32>(hi, lo);
}

fn m31_mul(a: u32, b: u32) -> u32 {
    // a * b = hi * 2^32 + lo with hi < 2^30. As 2^31 = 1 mod p: a * b = (lo & p) + (lo >> 31) + 2 hi,
    // a sum s <= 2p, made canonical by two conditional subtractions.
    let w = mul_wide(a, b);
    let s = (w.y & P) + (w.y >> 31u) + (w.x << 1u);
    return m31_reduce_once(m31_reduce_once(s));
}
