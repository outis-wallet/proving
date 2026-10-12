//! A hook for the part extension of constraint evaluation: a column committed by
//! `CommitmentTreeProver::new_striped` holds only its first stripe, and constraint evaluation,
//! which reads it on more of its domain, evaluates it there part by part
//! (`component_prover::trace_part`): it interpolates the column from its first stripe and
//! evaluates it on the part, on the CPU. A registered [`TracePartHook`] is asked for those
//! evaluations first, e.g. a GPU commitment that computed them already, and the CPU evaluates
//! the columns the hook doesn't provide.
//!
//! The hook is only an optimization: without one, or for every column it declines, the CPU's
//! evaluation runs unchanged.

use std::sync::{Arc, RwLock};

use stwo::core::fields::m31::BaseField;
use stwo::core::poly::circle::CircleDomain;
use stwo::prover::backend::Col;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::poly::BitReversedOrder;
use stwo::prover::poly::circle::CircleEvaluation;

/// A column constraint evaluation reads on a part of its domain it doesn't hold.
pub struct TracePartColumn<'a> {
    /// The index of the column's tree (`PREPROCESSED_TRACE_IDX` for a preprocessed column, which
    /// several components may read).
    pub tree: usize,
    /// The committed column: its domain is the committed domain, and it holds its evaluations on
    /// the first `values.len()` points of it (in bit-reversed order), its first stripe.
    pub column: &'a CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>,
}

/// A request to a [`TracePartHook`]: the evaluations of `columns` on part `part` of their
/// committed domain split into `2^log_parts` parts, `domain`.
pub struct TracePartRequest<'a> {
    /// The part, from 1 (part 0 is every column's first stripe, which it holds).
    pub part: usize,
    /// The committed domain is split into `2^log_parts` parts (its log blowup factor).
    pub log_parts: u32,
    /// The part's domain: `part_domain(committed_domain, log_parts, part)`.
    pub domain: CircleDomain,
    /// The columns of one component that hold less than the part, in the component's order.
    pub columns: Vec<TracePartColumn<'a>>,
}

/// Given a [`TracePartRequest`], returns for each of its columns, in order, either the
/// column's evaluations on the part's domain or `None` (the CPU evaluates it).
///
/// A column a hook returns must be exactly what the CPU computes for it, word for word:
/// `Poly::stripe_coefficients` (interpolated from the first stripe) evaluated with
/// `evaluate_with_twiddles` on the part's domain. The composition polynomial, and so every
/// later step of the proof, depends on it; stwo does not check it (a wrong column makes the
/// prover's out-of-domain sanity check fail). A returned vector of the wrong length, or a column
/// of the wrong length, is ignored (the CPU evaluates the columns).
pub type TracePartHook =
    dyn Fn(&TracePartRequest<'_>) -> Vec<Option<Col<SimdBackend, BaseField>>> + Send + Sync;

/// The registered hook.
static TRACE_PART_HOOK: RwLock<Option<Arc<TracePartHook>>> = RwLock::new(None);

/// Sets the hook constraint evaluation asks for a part's columns first, or removes it (`None`).
/// It applies to the components evaluated from then on, in every thread.
pub fn set_trace_part_hook(hook: Option<Arc<TracePartHook>>) {
    *TRACE_PART_HOOK.write().unwrap_or_else(|poisoned| poisoned.into_inner()) = hook;
}

/// The registered hook, if any.
pub(crate) fn trace_part_hook() -> Option<Arc<TracePartHook>> {
    TRACE_PART_HOOK.read().unwrap_or_else(|poisoned| poisoned.into_inner()).clone()
}
