//! Evaluation of a polynomial, from its coefficients, at a few points of a circle domain.
//!
//! A commitment tree that keeps only a prefix of its columns' evaluations (see
//! [`crate::prover::CommitmentTreeProver::retain_prefixes`]) opens its queried rows from the
//! coefficients. The coefficient at (storage) index `n` multiplies the product of the point's
//! mappings `m_j` over the set bits `j` of `n`; the mappings are `y`, `x` and the images of `x`
//! under `π`, in an order that depends on the backend's storage (see
//! [`super::PolyOps::split_at_mid`]). The mappings past the first four are images of `π³(x)`
//! under `π`, so the points with the same `π³(x)` share them. For such a group, one pass over the
//! coefficients folds the shared mappings into 16 sums, one per combination of the first four
//! mappings, and every point of the group is then a sum of 16 products. An aligned block of 32
//! bit-reversed positions holds two such groups, so the openings of a query cost two passes over
//! each column.

use std::simd::cmp::SimdPartialOrd;
use std::simd::u64x8;

use itertools::Itertools;
use num_traits::{One, Zero};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::core::circle::CirclePoint;
use crate::core::fields::m31::{BaseField, P};
use crate::core::poly::circle::CircleDomain;
use crate::core::utils::bit_reverse_index;
use crate::prover::backend::simd::m31::{LOG_N_LANES, N_LANES, PackedBaseField};
use crate::prover::poly::circle::{CircleCoefficients, PolyOps};
use crate::prover::poly::twiddles::TwiddleTree;

/// The number of leading mappings that may differ between the points of a group.
pub const LOG_GROUP_COEFFS: u32 = LOG_N_LANES;

/// The products of `factors` over the set bits of every index below `1 << factors.len()`.
fn tensor_products(factors: &[BaseField]) -> Vec<BaseField> {
    let mut products = vec![BaseField::one(); 1 << factors.len()];
    for (j, &factor) in factors.iter().enumerate() {
        let half = 1 << j;
        for i in 0..half {
            products[i + half] = products[i] * factor;
        }
    }
    products
}

/// The log number of coefficient vectors that a task folds with a table of weights of its own.
const LOG_CHUNK_VECS: u32 = 8;
/// The number of groups whose sums a pass over a chunk accumulates together.
const GROUP_BLOCK: usize = 2;

/// Evaluates, at the points of `domain` at the bit-reversed positions `indices`, the polynomial of
/// `1 << log_size` coefficients `coeffs` (16 per vector, in storage order), whose basis at a point
/// is given by `mappings(point)` (in the same order). Requires `log_size > LOG_GROUP_COEFFS`.
pub fn eval_at_domain_indices(
    log_size: u32,
    coeffs: &[PackedBaseField],
    mappings: impl Fn(CirclePoint<BaseField>) -> Vec<BaseField>,
    domain: CircleDomain,
    indices: &[usize],
) -> Vec<BaseField> {
    assert!(log_size > LOG_GROUP_COEFFS);
    assert_eq!(coeffs.len(), 1 << (log_size - LOG_GROUP_COEFFS));
    let low = LOG_GROUP_COEFFS as usize;
    let point_mappings = indices
        .iter()
        .map(|&index| mappings(domain.at(bit_reverse_index(index, domain.log_size()))))
        .collect_vec();

    // Group the points by their shared mappings.
    let mut groups: Vec<&[BaseField]> = vec![];
    let group_of_point = point_mappings
        .iter()
        .map(|point| {
            let shared = &point[low..];
            groups.iter().position(|&group| group == shared).unwrap_or_else(|| {
                groups.push(shared);
                groups.len() - 1
            })
        })
        .collect_vec();
    let n_groups = groups.len();

    // The weight of a coefficient vector in a group is the product of the group's mappings over
    // the set bits of the vector's index. The vectors are folded by chunks: the weights within a
    // chunk (the low bits) are a small table shared by the chunks, and each chunk's sums are then
    // multiplied by the weight of the chunk (the high bits).
    let log_chunk = LOG_CHUNK_VECS.min(log_size - LOG_GROUP_COEFFS) as usize;
    // Entry `g << log_chunk | t` is the weight of the chunk's vector `t` in group `g`.
    let chunk_weights: Vec<u32> = groups
        .iter()
        .flat_map(|shared| tensor_products(&shared[..log_chunk]).into_iter().map(|w| w.0))
        .collect();
    let high_weights =
        groups.iter().map(|shared| tensor_products(&shared[log_chunk..])).collect_vec();

    let fold_chunk = |(u, chunk): (usize, &[PackedBaseField])| -> Vec<PackedBaseField> {
        let mut sums = vec![PackedBaseField::zero(); n_groups];
        for first in (0..n_groups).step_by(GROUP_BLOCK) {
            // A last, partial block repeats its last group, whose sums it then ignores.
            let weights: [&[u32]; GROUP_BLOCK] = std::array::from_fn(|b| {
                let g = (first + b).min(n_groups - 1);
                &chunk_weights[g << log_chunk..(g + 1) << log_chunk]
            });
            let block_sums = fold_block(chunk, weights);
            for (b, sum) in block_sums.into_iter().enumerate().take(n_groups - first) {
                sums[first + b] = sum * PackedBaseField::broadcast(high_weights[first + b][u]);
            }
        }
        sums
    };
    let add = |mut a: Vec<PackedBaseField>, b: Vec<PackedBaseField>| {
        a.iter_mut().zip(b).for_each(|(a, b)| *a += b);
        a
    };
    #[cfg(feature = "parallel")]
    let sums = coeffs
        .par_chunks(1 << log_chunk)
        .enumerate()
        .map(fold_chunk)
        .reduce(|| vec![PackedBaseField::zero(); n_groups], add);
    #[cfg(not(feature = "parallel"))]
    let sums = coeffs
        .chunks(1 << log_chunk)
        .enumerate()
        .map(fold_chunk)
        .fold(vec![PackedBaseField::zero(); n_groups], add);
    let sums: Vec<[BaseField; N_LANES]> = sums.into_iter().map(|sum| sum.to_array()).collect();

    // Every point: the products of its leading mappings against its group's sums.
    point_mappings
        .iter()
        .zip(group_of_point)
        .map(|(point, g)| {
            tensor_products(&point[..low])
                .into_iter()
                .zip(sums[g])
                .map(|(product, sum)| product * sum)
                .sum()
        })
        .collect()
}

/// The sums `sum_t chunk[t] * weights[b][t]`, for each group `b` of a block, reduced.
#[inline(always)]
fn fold_block(
    chunk: &[PackedBaseField],
    weights: [&[u32]; GROUP_BLOCK],
) -> [PackedBaseField; GROUP_BLOCK] {
    let mut acc = [[u64x8::splat(0); 2]; GROUP_BLOCK];
    // Products are below 2^62: four of them fit on a partly reduced sum (below 2^34).
    for (quad, vectors) in chunk.chunks(4).enumerate() {
        for (i, &vector) in vectors.iter().enumerate() {
            let t = quad * 4 + i;
            // The even and the odd lanes of the vector, as 64-bit lanes.
            let words = u64x8::from_array(bytemuck::cast::<PackedBaseField, [u64; 8]>(vector));
            let even = words & u64x8::splat(0xFFFF_FFFF);
            let odd = words >> 32;
            for b in 0..GROUP_BLOCK {
                let weight = u64x8::splat(weights[b][t] as u64);
                acc[b][0] += even * weight;
                acc[b][1] += odd * weight;
            }
        }
        for acc in &mut acc {
            acc[0] = partial_reduce(acc[0]);
            acc[1] = partial_reduce(acc[1]);
        }
    }
    acc.map(|acc| {
        let [even, odd] = acc.map(reduce);
        let words = (even | (odd << 32)).to_array();
        bytemuck::cast::<[u64; 8], PackedBaseField>(words)
    })
}

/// `x mod P` up to a multiple of `P`, below `2^34` for any `x`.
#[inline(always)]
fn partial_reduce(x: u64x8) -> u64x8 {
    (x & u64x8::splat(P as u64)) + (x >> 31)
}

/// `x mod P`, in `[0, P)`.
#[inline(always)]
fn reduce(x: u64x8) -> u64x8 {
    let x = partial_reduce(partial_reduce(x));
    // Now below `P + 8`.
    let p = u64x8::splat(P as u64);
    x.simd_ge(p).select(x - p, x)
}

/// [`PolyOps::eval_at_domain_indices`] for any backend and size: one [`PolyOps::eval_at_point`] per
/// position.
pub fn eval_at_domain_indices_by_points<B: PolyOps>(
    poly: &CircleCoefficients<B>,
    domain: CircleDomain,
    indices: &[usize],
) -> Vec<BaseField> {
    indices
        .iter()
        .map(|&index| {
            let point = domain.at(bit_reverse_index(index, domain.log_size()));
            B::eval_at_point(poly, point.into_ef()).to_m31_array()[0]
        })
        .collect()
}

/// The part `part` of `domain` split into `1 << log_parts` parts: the domain whose evaluations, in
/// bit-reversed order, are the evaluations on `domain` at the bit-reversed positions
/// `part << (domain.log_size() - log_parts)..(part + 1) << (domain.log_size() - log_parts)`.
pub fn part_domain(domain: CircleDomain, log_parts: u32, part: usize) -> CircleDomain {
    let (subdomain, shifts) = domain.split(log_parts);
    subdomain.shift(shifts[bit_reverse_index(part, log_parts)])
}

/// [`PolyOps::part_twiddles`] for any backend: the twiddles of the part, computed anew.
pub fn part_twiddles_by_precompute<B: PolyOps>(
    domain: CircleDomain,
    log_parts: u32,
    part: usize,
) -> TwiddleTree<B> {
    B::precompute_twiddles(part_domain(domain, log_parts, part).half_coset)
}
