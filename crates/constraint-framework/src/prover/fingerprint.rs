//! A fingerprint of a component's constraints, cheap to compute (one evaluation of the
//! component's `evaluate` over the secure field), for caching what is derived from its traced
//! constraints (see [`crate::expr::ExprEvaluator`]) by more than the component's type: components
//! of one type may differ in their constraints (e.g. an offset in an instance's constants).
//!
//! The fingerprint is a random linear combination of the constraints evaluated at a
//! pseudo-random point: every column (by interaction, index among the interaction's columns as
//! the domain evaluators count them, and mask offset), every relation's formal elements (by the
//! relation's name and power, as `ExprEvaluator` names them), the claimed sum and the column size
//! are fixed pseudo-random values, the same in every process. Two components whose constraints
//! are different polynomials of these variables get the same value with probability about
//! `degree / |QM31|` (Schwartz–Zippel). It also records the relations' actual elements, as
//! `ExprEvaluator::relation_values` does.

use core::ops::Mul;

use num_traits::Zero;
use stwo::core::Fraction;
use stwo::core::fields::FieldExpOps;
use stwo::core::fields::m31::{M31, P};
use stwo::core::fields::qm31::{QM31, SECURE_EXTENSION_DEGREE, SecureField};

use crate::{
    EvalAtRow, FrameworkEval, INTERACTION_TRACE_IDX, MAX_N_INTERACTIONS, Relation, RelationEntry,
};

/// See the module docs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConstraintsFingerprint {
    /// The random linear combination of the constraints at the pseudo-random point.
    pub value: SecureField,
    /// The number of constraints.
    pub n_constraints: usize,
    /// The number of columns per interaction, as the domain evaluators count them.
    pub n_columns: [usize; MAX_N_INTERACTIONS],
    /// The relations' actual elements, by relation name in the order first added to: `(name, z,
    /// [alpha_0, alpha_1, ...])`, as `ExprEvaluator::relation_values`.
    pub relation_values: Vec<(String, SecureField, Vec<SecureField>)>,
}

impl ConstraintsFingerprint {
    /// The fingerprint of `eval`'s constraints.
    pub fn of<E: FrameworkEval>(eval: &E) -> Self {
        let evaluator = eval.evaluate(FingerprintEvaluator::default());
        Self {
            value: evaluator.value,
            n_constraints: evaluator.n_constraints,
            n_columns: evaluator.n_columns,
            relation_values: evaluator.relation_values,
        }
    }
}

const TAG_COLUMN: u64 = 1;
const TAG_COEFF: u64 = 2;
const TAG_ALPHA: u64 = 3;
const TAG_Z: u64 = 4;
const TAG_CLAIMED_SUM: u64 = 5;
const TAG_COLUMN_SIZE: u64 = 6;

const fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// A fixed pseudo-random secure field element for `(tag, a, b, c)`.
fn pseudo_random(tag: u64, a: u64, b: u64, c: u64) -> SecureField {
    let mut state = splitmix64(tag ^ splitmix64(a ^ splitmix64(b ^ splitmix64(c))));
    let mut next = || {
        state = splitmix64(state);
        M31((state >> 33) as u32 % P)
    };
    QM31::from_m31(next(), next(), next(), next())
}

/// FNV-1a of a name.
fn name_hash(name: &str) -> u64 {
    name.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3))
}

/// The evaluator of [`ConstraintsFingerprint::of`].
#[derive(Default)]
struct FingerprintEvaluator {
    n_columns: [usize; MAX_N_INTERACTIONS],
    n_constraints: usize,
    value: SecureField,
    fracs: Vec<Fraction<SecureField, SecureField>>,
    relation_values: Vec<(String, SecureField, Vec<SecureField>)>,
}

impl FingerprintEvaluator {
    fn record_relation_values<R: Relation<SecureField, SecureField>>(&mut self, relation: &R) {
        let name = relation.get_name();
        if self.relation_values.iter().any(|(recorded, ..)| recorded == name) {
            return;
        }
        let combine = |values: &[SecureField]| -> SecureField { relation.combine(values) };
        let z = -combine(&[SecureField::zero()]);
        let alphas = (0..relation.get_size())
            .map(|k| {
                let mut unit = vec![SecureField::zero(); k + 1];
                unit[k] = SecureField::from(M31(1));
                combine(&unit) + z
            })
            .collect();
        self.relation_values.push((name.to_owned(), z, alphas));
    }
}

impl EvalAtRow for FingerprintEvaluator {
    type F = SecureField;
    type EF = SecureField;

    fn next_interaction_mask<const N: usize>(
        &mut self,
        interaction: usize,
        offsets: [isize; N],
    ) -> [Self::F; N] {
        let index = self.n_columns[interaction];
        self.n_columns[interaction] += 1;
        offsets.map(|offset| {
            pseudo_random(TAG_COLUMN, interaction as u64, index as u64, offset as i64 as u64)
        })
    }

    fn add_constraint<G>(&mut self, constraint: G)
    where
        Self::EF: Mul<G, Output = Self::EF> + From<G>,
    {
        let coeff = pseudo_random(TAG_COEFF, self.n_constraints as u64, 0, 0);
        let constraint = Self::EF::from(constraint);
        self.value += <SecureField as Mul<SecureField>>::mul(coeff, constraint);
        self.n_constraints += 1;
    }

    fn combine_ef(values: [Self::F; SECURE_EXTENSION_DEGREE]) -> Self::EF {
        SecureField::from_partial_evals(values)
    }

    fn add_to_relation<R: Relation<Self::F, Self::EF>>(
        &mut self,
        entry: RelationEntry<'_, Self::F, Self::EF, R>,
    ) {
        self.record_relation_values(entry.relation);
        // The relation's formal elements, as `ExprEvaluator` names them: `<name>_alpha<k>` and
        // `<name>_z`.
        let name = name_hash(entry.relation.get_name());
        let combined = entry
            .values
            .iter()
            .enumerate()
            .map(|(k, value)| pseudo_random(TAG_ALPHA, name, k as u64, 0) * *value)
            .sum::<SecureField>()
            - pseudo_random(TAG_Z, name, 0, 0);
        self.write_logup_frac(Fraction::new(entry.multiplicity, combined));
    }

    fn write_logup_frac(&mut self, fraction: Fraction<Self::EF, Self::EF>) {
        self.fracs.push(fraction);
    }

    /// As `logup_proxy`'s (`ExprEvaluator`'s): the fractions summed in batches, one cumulative
    /// sum column per batch, the last at offsets -1 and 0 and shifted by the claimed sum over
    /// the column size.
    fn finalize_logup_batched(&mut self, batch_size: usize) {
        assert!(batch_size > 0, "Batch size must be positive");
        let fracs = std::mem::take(&mut self.fracs);
        let mut batched: Vec<Fraction<SecureField, SecureField>> =
            fracs.chunks(batch_size).map(|chunk| chunk.iter().copied().sum()).collect();
        let last = batched.pop().expect("No fractions to finalize");
        let mut prev_col_cumsum = SecureField::zero();
        for frac in batched {
            let [cur_cumsum] = self.next_extension_interaction_mask(INTERACTION_TRACE_IDX, [0]);
            let diff = cur_cumsum - prev_col_cumsum;
            prev_col_cumsum = cur_cumsum;
            self.add_constraint(diff * frac.denominator - frac.numerator);
        }
        let [prev_row_cumsum, cur_cumsum] =
            self.next_extension_interaction_mask(INTERACTION_TRACE_IDX, [-1, 0]);
        let cumsum_shift = pseudo_random(TAG_CLAIMED_SUM, 0, 0, 0)
            * pseudo_random(TAG_COLUMN_SIZE, 0, 0, 0).inverse();
        let diff = cur_cumsum - prev_row_cumsum - prev_col_cumsum + cumsum_shift;
        self.add_constraint(diff * last.denominator - last.numerator);
    }

    fn finalize_logup(&mut self) {
        self.finalize_logup_batched(1)
    }

    fn finalize_logup_in_pairs(&mut self) {
        self.finalize_logup_batched(2)
    }
}

#[cfg(test)]
mod tests {
    use num_traits::One;

    use super::ConstraintsFingerprint;
    use crate::preprocessed_columns::PreProcessedColumnId;
    use crate::{EvalAtRow, FrameworkEval, RelationEntry, relation};

    relation!(TestRelation, 3);

    struct Component {
        relation: TestRelation,
        constant: u32,
        swap: bool,
    }

    impl FrameworkEval for Component {
        fn log_size(&self) -> u32 {
            4
        }
        fn max_constraint_log_degree_bound(&self) -> u32 {
            5
        }
        fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
            let p = eval.get_preprocessed_column(PreProcessedColumnId { id: "p".to_owned() });
            let a = eval.next_trace_mask();
            let b = eval.next_trace_mask();
            let (x, y) = if self.swap { (b.clone(), a.clone()) } else { (a.clone(), b.clone()) };
            eval.add_constraint(
                x.clone() * y.clone() - p.clone() - E::F::from(self.constant.into()),
            );
            eval.add_to_relation(RelationEntry::new(&self.relation, E::EF::one(), &[a, b]));
            eval.finalize_logup_in_pairs();
            eval
        }
    }

    #[test]
    fn fingerprint_tells_constraints_apart() {
        let of = |relation: TestRelation, constant, swap| {
            ConstraintsFingerprint::of(&Component { relation, constant, swap })
        };
        let base = of(TestRelation::dummy(), 7, false);
        // The relation's actual elements don't change it (only the recorded values)...
        let mut channel = stwo::core::channel::Blake2sChannel::default();
        let drawn = of(TestRelation::draw(&mut channel), 7, false);
        assert_eq!(drawn.value, base.value);
        assert_ne!(drawn.relation_values, base.relation_values);
        // ... the constraints do, the same polynomial written otherwise doesn't.
        assert_ne!(of(TestRelation::dummy(), 8, false).value, base.value);
        assert_eq!(of(TestRelation::dummy(), 7, true).value, base.value);
        assert_eq!(base.n_constraints, 2);
        assert_eq!(base.n_columns[..3], [1, 2, 4]);
        // The same in every process: a fixed value.
        assert_eq!(base, of(TestRelation::dummy(), 7, false));
    }
}
