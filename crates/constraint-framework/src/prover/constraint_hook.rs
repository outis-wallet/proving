//! A hook on a [`crate::FrameworkComponent`]'s constraint evaluation on the SIMD backend, for
//! evaluating the constraints elsewhere (a GPU) or for observing the SIMD evaluation's inputs and
//! output. Off unless registered ([`set_constraint_eval_hook`]): without a hook, the evaluation is
//! exactly as before.
//!
//! The hook is called once per component of the composition, after the small-trace CPU fallback
//! and before the part extension of the columns (see `evaluate_constraint_quotients_on_domain`),
//! with a [`ConstraintEvalJob`] describing the component's evaluation and the accumulator of its
//! evaluation domain's size. The accumulator may already hold other components' evaluations: a
//! hook that evaluates must add into it, not overwrite it.

use std::borrow::Cow;
use std::sync::{Arc, RwLock};

use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::core::pcs::TreeVec;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::poly::BitReversedOrder;
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::secure_column::SecureColumnByCoords;
use stwo::prover::Poly;

use crate::expr::ExprEvaluator;

/// A component's constraint evaluation: everything the SIMD evaluation uses, besides the
/// component's `FrameworkEval`, which [`Self::constraints`] traces.
///
/// The evaluation adds, at each row `r` of the evaluation domain (bit-reversed order),
/// `sum_i random_coeff_powers[i] * constraint_i(r) * denom_inv[r >> trace_log_size]` to the
/// accumulator.
pub struct ConstraintEvalJob<'a> {
    /// The component's `FrameworkEval` type (`core::any::type_name`).
    pub component: &'static str,
    /// The log size of the component's trace domain (`FrameworkEval::log_size`).
    pub trace_log_size: u32,
    /// The log size of the evaluation domain (the accumulator's column size).
    pub eval_log_size: u32,
    /// The number of constraints of the component (its `InfoEvaluator` count).
    pub n_constraints: usize,
    /// `random_coeff_powers[i]` multiplies constraint `i`, in the order the component adds them.
    pub random_coeff_powers: &'a [SecureField],
    /// The inverses of the vanishing polynomial on the evaluation domain's parts of the trace
    /// domain's size: row `r` uses `denom_inv[r >> trace_log_size]`.
    pub denom_inv: &'a [BaseField],
    /// The component's claimed logup sum.
    pub claimed_sum: SecureField,
    /// The component's preprocessed columns' indices in the preprocessed tree, in the order the
    /// component asks for them: its preprocessed column `k` is the tree's column
    /// `preprocessed_column_indices[k]` (already applied in `polys` and `columns`).
    pub preprocessed_column_indices: &'a [usize],
    /// The component's committed polynomials, per interaction, the preprocessed ones remapped.
    pub polys: &'a TreeVec<Vec<&'a Poly<SimdBackend>>>,
    pub(super) evaluations:
        &'a TreeVec<Vec<Cow<'a, CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>>>,
    pub(super) trace_constraints: &'a dyn Fn() -> ExprEvaluator,
}

impl ConstraintEvalJob<'_> {
    /// The component's constraints, traced symbolically.
    pub fn constraints(&self) -> ExprEvaluator {
        (self.trace_constraints)()
    }
}

/// What the hook did with a job.
pub enum ConstraintEvalOutcome {
    /// The hook added the component's evaluations into the accumulator; the SIMD evaluation does
    /// not run.
    Evaluated,
    /// The SIMD evaluation runs; then the observer, if any, is called with the accumulator.
    Declined(Option<Box<dyn FnOnce(&SecureColumnByCoords<SimdBackend>) + Send>>),
}

/// A constraint evaluation hook: see the module docs.
pub type ConstraintEvalHook = dyn Fn(&ConstraintEvalJob<'_>, &mut SecureColumnByCoords<SimdBackend>) -> ConstraintEvalOutcome
    + Send
    + Sync;

static CONSTRAINT_EVAL_HOOK: RwLock<Option<Arc<ConstraintEvalHook>>> = RwLock::new(None);

/// Sets the hook the SIMD constraint evaluation of every `FrameworkComponent` tries first, or
/// removes it (`None`). It applies to the evaluations started from then on, in every thread.
pub fn set_constraint_eval_hook(hook: Option<Arc<ConstraintEvalHook>>) {
    *CONSTRAINT_EVAL_HOOK.write().unwrap_or_else(|poisoned| poisoned.into_inner()) = hook;
}

/// The registered hook, if any.
pub(super) fn constraint_eval_hook() -> Option<Arc<ConstraintEvalHook>> {
    CONSTRAINT_EVAL_HOOK.read().unwrap_or_else(|poisoned| poisoned.into_inner()).clone()
}
