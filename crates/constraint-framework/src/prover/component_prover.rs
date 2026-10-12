use std::borrow::Cow;

use itertools::Itertools;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use stwo::core::air::Component;
use stwo::core::constraints::coset_vanishing;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::core::pcs::TreeVec;
use stwo::core::poly::circle::{CanonicCoset, CircleDomain};
use stwo::core::utils::bit_reverse;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::simd::column::{BaseColumn, VeryPackedSecureColumnByCoords};
use stwo::prover::backend::simd::m31::LOG_N_LANES;
use stwo::prover::backend::simd::very_packed_m31::{LOG_N_VERY_PACKED_ELEMS, VeryPackedBaseField};
use stwo::prover::backend::{Backend, Column, CpuBackend};
use stwo::prover::poly::BitReversedOrder;
use stwo::prover::poly::circle::{CircleEvaluation, PolyOps, part_domain};
use stwo::prover::secure_column::SecureColumnByCoords;
use stwo::prover::{ComponentProver, DomainEvaluationAccumulator, EvaluationMode, Poly, Trace};
use tracing::{Level, span};

use super::constraint_hook::{ConstraintEvalJob, ConstraintEvalOutcome, constraint_eval_hook};
use super::{CpuDomainEvaluator, SimdDomainEvaluator};
use crate::expr::ExprEvaluator;
use crate::{FrameworkComponent, FrameworkEval, PREPROCESSED_TRACE_IDX};

const CHUNK_SIZE: usize = 1;

/// Common inputs for constraint quotient evaluation, shared between the SIMD and CPU backends.
struct ConstraintQuotientInputs<'a, B: Backend> {
    eval_domain: CircleDomain,
    trace_domain: CanonicCoset,
    trace: TreeVec<Vec<Cow<'a, CircleEvaluation<B, BaseField, BitReversedOrder>>>>,
    denom_inv: Vec<BaseField>,
    /// The component's committed columns, in the order of `trace`.
    polys: TreeVec<Vec<&'a Poly<B>>>,
}

/// Prepares trace evaluations: borrows directly (subdomain) or extends to eval domain.
fn get_trace_columns<'a, B: Backend>(
    component_polys: TreeVec<Vec<&'a &Poly<B>>>,
    eval_domain: CircleDomain,
    mode: EvaluationMode,
) -> TreeVec<Vec<Cow<'a, CircleEvaluation<B, BaseField, BitReversedOrder>>>> {
    match mode {
        EvaluationMode::SubDomain { .. } => {
            // Borrow committed evaluations directly. Only the first
            // 2^max_constraint_log_degree_bound indices are going to be used for the
            // constraint quotient evaluation (in bit-reversed order these form the
            // subdomain coset).
            //
            // Ideally we'd slice to just those indices, but the type system requires
            // borrowing the entire evaluation.
            //
            // A committed column may hold only a prefix of those (see
            // `CommitmentTreeProver::retain_prefixes`); the SIMD evaluation then evaluates the rest
            // part by part, see `evaluate_in_parts`.
            component_polys.map_cols(|c| Cow::Borrowed(&c.evals))
        }
        EvaluationMode::ExtendToEvalDomain => {
            let _span = span!(Level::INFO, "Constraint Extension").entered();
            let twiddles = B::precompute_twiddles(eval_domain.half_coset);
            #[cfg(not(feature = "parallel"))]
            {
                component_polys.as_cols_ref().map_cols(|col| {
                    Cow::Owned(col.get_evaluation_on_domain(eval_domain, &twiddles))
                })
            }
            #[cfg(feature = "parallel")]
            {
                component_polys.as_cols_ref().par_map_cols(|col| {
                    Cow::Owned(col.get_evaluation_on_domain(eval_domain, &twiddles))
                })
            }
        }
    }
}

/// Constructs the inputs needed for constraint quotient evaluation from a component and trace.
/// Computes the eval/trace domains, prepares trace columns (borrowing or extending as needed),
/// and precomputes denominator inverses.
fn get_constraint_quotient_inputs<'a, E: FrameworkEval, B: Backend>(
    component: &FrameworkComponent<E>,
    trace: &'a Trace<'a, B>,
    mode: EvaluationMode,
) -> ConstraintQuotientInputs<'a, B> {
    let max_constraint_log_degree_bound = component.max_constraint_log_degree_bound();
    let trace_domain = CanonicCoset::new(component.eval.log_size());

    let mut component_polys = trace.polys.sub_tree(&component.trace_locations);
    component_polys[PREPROCESSED_TRACE_IDX] = component
        .preprocessed_column_indices
        .iter()
        .map(|idx| &trace.polys[PREPROCESSED_TRACE_IDX][*idx])
        .collect();
    let polys = component_polys.as_cols_ref().map_cols(|poly| **poly);

    let eval_domain = match mode {
        EvaluationMode::SubDomain { log_expansion } => {
            subdomain_eval_domain(max_constraint_log_degree_bound, log_expansion)
        }
        EvaluationMode::ExtendToEvalDomain => {
            CanonicCoset::new(max_constraint_log_degree_bound).circle_domain()
        }
    };
    let trace = get_trace_columns(component_polys, eval_domain, mode);

    // Denom inverses.
    let log_expand = eval_domain.log_size() - trace_domain.log_size();
    let mut denom_inv = (0..1 << log_expand)
        .map(|i| coset_vanishing(trace_domain.coset(), eval_domain.at(i)).inverse())
        .collect_vec();
    bit_reverse(&mut denom_inv);

    ConstraintQuotientInputs { eval_domain, trace_domain, trace, denom_inv, polys }
}

/// The columns of a component on part `part` of its constraint evaluation domain split into parts
/// of the size of its trace domain, for columns that hold only a prefix of the evaluation domain
/// (see `CommitmentTreeProver::retain_prefixes`): part 0 is borrowed (every committed column holds
/// at least its own size), a further part a column holds is copied out, and the others are
/// evaluated from the column's coefficients. In bit-reversed order, part `part` of the evaluation
/// domain, the first `2^log_n_parts` parts of the committed domain, is the part `part` of the
/// committed domain split into parts of the column's size.
fn trace_part<'a>(
    trace: &'a TreeVec<Vec<Cow<'a, CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>>>,
    polys: &TreeVec<Vec<&Poly<SimdBackend>>>,
    trace_domain: CanonicCoset,
    part: usize,
) -> TreeVec<Vec<Cow<'a, CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>>> {
    let part_size = trace_domain.size();
    let committed_log_size = polys
        .iter()
        .flatten()
        .map(|poly| poly.evals.domain.log_size())
        .dedup()
        .exactly_one()
        .unwrap_or_else(|_| panic!("a component's columns have the size of its trace domain"));
    let committed_domain = CanonicCoset::new(committed_log_size).circle_domain();
    let domain = part_domain(committed_domain, committed_log_size - trace_domain.log_size(), part);
    let twiddles = SimdBackend::precompute_twiddles(domain.half_coset);
    // The interpolation twiddles of the first part, from which a column that holds only its first
    // stripe (see `CommitmentTreeProver::new_striped`) is interpolated again.
    let first_part = part_domain(committed_domain, committed_log_size - trace_domain.log_size(), 0);
    let stripe_twiddles = (part > 0).then(|| SimdBackend::precompute_twiddles(first_part.half_coset));
    let evaluate = |(column, poly): (
        &'a Cow<'a, CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
        &&Poly<SimdBackend>,
    )| {
        let held = column.values.len();
        if part == 0 {
            assert!(held >= part_size, "a committed column holds at least its own size");
            return Cow::Borrowed(column.as_ref());
        }
        if held >= (part + 1) * part_size {
            let values = column.values.to_cpu()[part * part_size..(part + 1) * part_size].to_vec();
            return Cow::Owned(CircleEvaluation::new(domain, values.into_iter().collect()));
        }
        let evaluation = match &poly.coeffs {
            Some(coeffs) => coeffs.evaluate_with_twiddles(domain, &twiddles),
            None => poly
                .stripe_coefficients(stripe_twiddles.as_ref().unwrap())
                .evaluate_with_twiddles(domain, &twiddles),
        };
        Cow::Owned(evaluation)
    };
    TreeVec(
        trace
            .iter()
            .zip(polys.iter())
            .map(|(columns, polys)| {
                #[cfg(feature = "parallel")]
                {
                    columns.par_iter().zip(polys.par_iter()).map(evaluate).collect()
                }
                #[cfg(not(feature = "parallel"))]
                {
                    columns.iter().zip(polys.iter()).map(evaluate).collect()
                }
            })
            .collect(),
    )
}

impl<E: FrameworkEval + Sync> ComponentProver<SimdBackend> for FrameworkComponent<E> {
    fn evaluate_constraint_quotients_on_domain(
        &self,
        trace: &Trace<'_, SimdBackend>,
        evaluation_accumulator: &mut DomainEvaluationAccumulator<SimdBackend>,
    ) {
        if self.n_constraints() == 0 {
            return;
        }

        let ConstraintQuotientInputs { eval_domain, trace_domain, trace, denom_inv, polys } =
            get_constraint_quotient_inputs(self, trace, evaluation_accumulator.evaluation_mode());

        let [mut accum] =
            evaluation_accumulator.columns([(eval_domain.log_size(), self.n_constraints())]);
        accum.random_coeff_powers.reverse();

        let _span = span!(
            Level::INFO,
            "Constraint point-wise eval",
            class = "ConstraintEval",
            component = core::any::type_name::<E>()
        )
        .entered();

        let holds_eval_domain = trace.iter().flatten().all(|c| c.values.len() >= eval_domain.size());
        // Fall back to CPU if the trace is too small.
        if trace_domain.log_size() < LOG_N_LANES + LOG_N_VERY_PACKED_ELEMS {
            assert!(holds_eval_domain, "a small committed column holds the evaluation domain");
            let trace_cols = trace.as_cols_ref().map_cols(|c| c.to_cpu());
            let trace_cols = trace_cols.as_cols_ref();
            *accum.col = SecureColumnByCoords::from_cpu(accumulate_pointwise_cpu(
                self,
                trace_cols,
                eval_domain.log_size(),
                trace_domain.log_size(),
                denom_inv,
                &accum.random_coeff_powers,
                &accum.col.to_cpu(),
            ));
            return;
        }

        // A registered hook (see `constraint_hook`) evaluates the component, or declines and
        // possibly observes the evaluation below. Without one, nothing changes.
        let observer = match constraint_eval_hook() {
            None => None,
            Some(hook) => {
                let trace_constraints = || self.eval.evaluate(ExprEvaluator::new());
                let job = ConstraintEvalJob {
                    component: core::any::type_name::<E>(),
                    trace_log_size: trace_domain.log_size(),
                    eval_log_size: eval_domain.log_size(),
                    n_constraints: self.n_constraints(),
                    random_coeff_powers: &accum.random_coeff_powers,
                    denom_inv: &denom_inv,
                    claimed_sum: self.claimed_sum,
                    preprocessed_column_indices: &self.preprocessed_column_indices,
                    polys: &polys,
                    evaluations: &trace,
                    trace_constraints: &trace_constraints,
                };
                match hook(&job, accum.col) {
                    ConstraintEvalOutcome::Evaluated => return,
                    ConstraintEvalOutcome::Declined(observer) => observer,
                }
            }
        };

        let col = unsafe { VeryPackedSecureColumnByCoords::transform_under_mut(accum.col) };

        // Define any `self` values outside the loop to prevent the compiler thinking there is a
        // `Sync` requirement on `Self`.
        let self_eval = &self.eval;
        let self_claimed_sum = self.claimed_sum;
        let random_coeff_powers = &accum.random_coeff_powers;

        // Evaluates the `n_vec_rows` vector rows of the evaluation domain from `vec_start` on, which
        // the columns of `trace_cols` hold from their first row on; `parts` are the columns on
        // every part of the evaluation domain if it is evaluated in parts.
        type Columns<'c> =
            TreeVec<Vec<&'c CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>>;
        let mut evaluate_rows =
            |trace_cols: &Columns<'_>, vec_start: usize, n_vec_rows: usize, parts: &[Columns<'_>]| {
                let range = vec_start..vec_start + n_vec_rows;
                let row_base = vec_start << (LOG_N_LANES + LOG_N_VERY_PACKED_ELEMS);

                #[cfg(not(feature = "parallel"))]
                let iter = range
                    .step_by(CHUNK_SIZE)
                    .zip(col.chunks_mut(CHUNK_SIZE).skip(vec_start / CHUNK_SIZE));

                #[cfg(feature = "parallel")]
                let iter = range
                    .into_par_iter()
                    .step_by(CHUNK_SIZE)
                    .zip(col.par_chunks_mut(CHUNK_SIZE).skip(vec_start / CHUNK_SIZE));

                iter.for_each(|(chunk_idx, mut chunk)| {
                    for idx_in_chunk in 0..CHUNK_SIZE {
                        let vec_row = chunk_idx * CHUNK_SIZE + idx_in_chunk;
                        // Evaluate constrains at row.
                        let eval = SimdDomainEvaluator::new(
                            trace_cols,
                            vec_row - vec_start,
                            random_coeff_powers,
                            trace_domain.log_size(),
                            eval_domain.log_size(),
                            self_eval.log_size(),
                            self_claimed_sum,
                        )
                        .in_part(row_base, parts);
                        let row_res = self_eval.evaluate(eval).row_res;

                        // Finalize row.
                        unsafe {
                            let row_denom_inv = VeryPackedBaseField::broadcast(
                                denom_inv[vec_row
                                    >> (trace_domain.log_size()
                                        - LOG_N_LANES
                                        - LOG_N_VERY_PACKED_ELEMS)],
                            );
                            chunk.set_packed(
                                idx_in_chunk,
                                chunk.packed_at(idx_in_chunk) + row_res * row_denom_inv,
                            )
                        }
                    }
                });
            };

        if holds_eval_domain {
            // The column references are the same for every row; build them once.
            let trace_cols = trace.as_cols_ref().map_cols(|c| c.as_ref());
            evaluate_rows(
                &trace_cols,
                0,
                1 << (eval_domain.log_size() - LOG_N_LANES - LOG_N_VERY_PACKED_ELEMS),
                &[],
            );
        } else {
            // Columns that hold only a prefix of the evaluation domain: their evaluations on the
            // other parts of the trace domain's size are evaluated, the first part being the prefix
            // every column holds, and the domain is evaluated part by part.
            let n_parts = 1 << (eval_domain.log_size() - trace_domain.log_size());
            let part_traces = {
                let _span = span!(Level::INFO, "Constraint part extension").entered();
                (0..n_parts).map(|part| trace_part(&trace, &polys, trace_domain, part)).collect_vec()
            };
            let parts = part_traces
                .iter()
                .map(|part_trace| part_trace.as_cols_ref().map_cols(|c| c.as_ref()))
                .collect_vec();
            let n_vec_rows = 1 << (trace_domain.log_size() - LOG_N_LANES - LOG_N_VERY_PACKED_ELEMS);
            for (part, trace_cols) in parts.iter().enumerate() {
                evaluate_rows(trace_cols, part * n_vec_rows, n_vec_rows, &parts);
            }
        }
        if let Some(observer) = observer {
            observer(accum.col);
        }
    }
}

impl ConstraintEvalJob<'_> {
    /// The component's columns on the whole evaluation domain, per interaction (the preprocessed
    /// ones remapped), as the SIMD evaluation reads them: a column's values on part 0 of the
    /// domain (the trace domain's size), then on part 1, and so on, each part a column does not
    /// hold evaluated on the CPU as the SIMD evaluation evaluates it (`trace_part`). The values
    /// are the columns' words as stored.
    pub fn columns_on_eval_domain(&self) -> TreeVec<Vec<BaseColumn>> {
        let trace_domain = CanonicCoset::new(self.trace_log_size);
        let part_size = trace_domain.size();
        let eval_size = 1usize << self.eval_log_size;
        let n_parts = eval_size / part_size;
        if self.evaluations.iter().flatten().all(|c| c.values.len() >= eval_size) {
            return self
                .evaluations
                .as_cols_ref()
                .map_cols(|c| c.values.as_slice()[..eval_size].iter().copied().collect());
        }
        let parts = (0..n_parts)
            .map(|part| trace_part(self.evaluations, self.polys, trace_domain, part))
            .collect_vec();
        TreeVec(
            self.evaluations
                .iter()
                .enumerate()
                .map(|(interaction, columns)| {
                    (0..columns.len())
                        .map(|index| {
                            parts
                                .iter()
                                .flat_map(|part| {
                                    part[interaction][index].values.as_slice()[..part_size]
                                        .iter()
                                        .copied()
                                })
                                .collect()
                        })
                        .collect()
                })
                .collect(),
        )
    }
}

impl<E: FrameworkEval + Sync> ComponentProver<CpuBackend> for FrameworkComponent<E> {
    fn evaluate_constraint_quotients_on_domain(
        &self,
        trace: &Trace<'_, CpuBackend>,
        evaluation_accumulator: &mut DomainEvaluationAccumulator<CpuBackend>,
    ) {
        if self.n_constraints() == 0 {
            return;
        }

        let ConstraintQuotientInputs { eval_domain, trace_domain, trace, denom_inv, .. } =
            get_constraint_quotient_inputs(self, trace, evaluation_accumulator.evaluation_mode());
        assert!(
            trace.iter().flatten().all(|c| c.values.len() >= eval_domain.size()),
            "a committed column does not hold the constraint evaluation subdomain"
        );

        let [mut accum] =
            evaluation_accumulator.columns([(eval_domain.log_size(), self.n_constraints())]);
        accum.random_coeff_powers.reverse();

        let _span = span!(
            Level::INFO,
            "Constraint point-wise eval",
            class = "ConstraintEval",
            component = core::any::type_name::<E>()
        )
        .entered();
        let trace_cols = trace.as_cols_ref().map_cols(|c| c.as_ref());

        *accum.col = accumulate_pointwise_cpu(
            self,
            trace_cols,
            eval_domain.log_size(),
            trace_domain.log_size(),
            denom_inv,
            &accum.random_coeff_powers,
            accum.col,
        );
    }
}

/// Computes the evaluation subdomain for a component given its constraint degree bound
/// and the log_expansion from `EvaluationMode::SubDomain`.
///
/// When `log_expansion == 0`, returns the canonical domain.
/// When `log_expansion > 0`, returns the first subdomain obtained by splitting the
/// committed domain `log_expansion` times.
fn subdomain_eval_domain(max_constraint_log_degree_bound: u32, log_expansion: u32) -> CircleDomain {
    let committed_domain =
        CanonicCoset::new(max_constraint_log_degree_bound + log_expansion).circle_domain();
    committed_domain.split(log_expansion).0
}

fn accumulate_pointwise_cpu<E: FrameworkEval>(
    component: &FrameworkComponent<E>,
    trace_cols: TreeVec<Vec<&CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>>>,
    eval_log_size: u32,
    trace_log_size: u32,
    denom_inv: Vec<BaseField>,
    random_coeff_powers: &[SecureField],
    accum: &SecureColumnByCoords<CpuBackend>,
) -> SecureColumnByCoords<CpuBackend> {
    let mut res = SecureColumnByCoords::zeros(1 << eval_log_size);
    for row in 0..(1 << eval_log_size) {
        // Evaluate constrains at row.
        let eval = CpuDomainEvaluator::new(
            &trace_cols,
            row,
            random_coeff_powers,
            trace_log_size,
            eval_log_size,
            component.eval.log_size(),
            component.claimed_sum,
        );
        let row_res = component.eval.evaluate(eval).row_res;

        // Finalize row.
        let row_denom_inv = denom_inv[row >> trace_log_size];
        res.set(row, accum.at(row) + row_res * row_denom_inv)
    }
    res
}
