use std::array;

use hashbrown::HashMap;
use itertools::Itertools;
#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use tracing::{Level, info, span};

use crate::core::ColumnVec;
use crate::core::channel::{Channel, MerkleChannel};
use crate::core::circle::CirclePoint;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::pcs::quotients::{
    CommitmentSchemeProof, CommitmentSchemeProofAux, ExtendedCommitmentSchemeProof, PointSample,
};
use crate::core::pcs::utils::prepare_preprocessed_query_positions;
use crate::core::pcs::{PcsConfig, TreeSubspan, TreeVec};
use crate::core::poly::circle::{CanonicCoset, CircleDomain};
use crate::core::utils::MaybeOwned;
use crate::core::vcs_lifted::merkle_hasher::MerkleHasherLifted;
use crate::core::vcs_lifted::verifier::ExtendedMerkleDecommitmentLifted;
use crate::prover::air::component_prover::{Poly, Trace, WeightsHashMap};
use crate::prover::backend::{BackendForChannel, Col, Column};
use crate::prover::fri::{FriDecommitResult, FriProver};
use crate::prover::mempool::BaseColumnPool;
use crate::prover::pcs::quotient_ops::compute_fri_quotients;
use crate::prover::poly::BitReversedOrder;
use crate::prover::poly::circle::{CircleCoefficients, CircleEvaluation, part_domain};
use crate::prover::poly::twiddles::TwiddleTree;
use crate::prover::secure_column::SecureColumnByCoords;
use crate::prover::vcs_lifted::prover::MerkleProverLifted;

pub mod quotient_ops;

/// The prover side of a FRI polynomial commitment scheme. See [super].
pub struct CommitmentSchemeProver<'a, B: BackendForChannel<MC>, MC: MerkleChannel> {
    pub trees: TreeVec<MaybeOwned<'a, CommitmentTreeProver<B, MC>>>,
    pub config: PcsConfig,
    pub twiddles: &'a TwiddleTree<B>,
    pub store_polynomials_coefficients: bool,
    /// Pre-allocated base field column pool for polynomial evaluation during commit.
    pub base_column_pool: MaybeOwned<'a, BaseColumnPool<B>>,
    /// Whether the trees committed from here on keep only the first part of every column's
    /// evaluations; see [`Self::set_stripe_residency`].
    pub stripe_residency: bool,
}
impl<'a, B: BackendForChannel<MC>, MC: MerkleChannel> CommitmentSchemeProver<'a, B, MC> {
    /// Creates a new empty commitment scheme prover with the given configuration and twiddles. The
    /// commitment scheme does not store the polynomials coefficients by default.
    pub fn new(config: PcsConfig, twiddles: &'a TwiddleTree<B>) -> Self {
        CommitmentSchemeProver {
            trees: TreeVec::default(),
            config,
            twiddles,
            store_polynomials_coefficients: false,
            base_column_pool: MaybeOwned::Owned(BaseColumnPool::new()),
            stripe_residency: false,
        }
    }

    pub fn with_memory_pool(
        config: PcsConfig,
        twiddles: &'a TwiddleTree<B>,
        base_column_pool: &'a BaseColumnPool<B>,
    ) -> Self {
        base_column_pool.release_small_idle();
        CommitmentSchemeProver {
            trees: TreeVec::default(),
            config,
            twiddles,
            store_polynomials_coefficients: false,
            base_column_pool: MaybeOwned::Borrowed(base_column_pool),
            stripe_residency: false,
        }
    }

    /// Sets the commitment scheme to store the polynomials coefficients starting from the next
    /// commit.
    pub const fn set_store_polynomials_coefficients(&mut self) {
        self.store_polynomials_coefficients = true;
    }

    /// Sets the trees committed from the next commit on to keep, of every column, only the
    /// evaluations on the first of the `2^log_blowup_factor` parts of its domain (its first
    /// stripe), as [`CommitmentTreeProver::new_striped`] commits them. The out-of-domain samples
    /// and the FRI quotients read only that stripe; constraint evaluation and the decommitment
    /// evaluate what else they read from it. Takes effect only without stored coefficients.
    pub const fn set_stripe_residency(&mut self) {
        self.stripe_residency = true;
    }

    /// Evaluates the given polynomials, commits them into a Merkle tree, mixes the root into
    /// the channel, and appends the resulting tree to the scheme.
    ///
    /// [`PcsConfig::lifting_log_size`] gives the tree its height by index: the first tree
    /// committed is the preprocessed one, every later one a trace tree.
    fn commit(&mut self, polynomials: ColumnVec<CircleCoefficients<B>>, channel: &mut MC::C) {
        let _span = span!(Level::INFO, "Commitment").entered();
        let lifting_log_size = self.config.lifting_log_size(self.trees.len());
        let mut tree = if self.stripe_residency && !self.store_polynomials_coefficients {
            CommitmentTreeProver::new_striped(
                polynomials,
                self.config.fri_config.log_blowup_factor,
                self.twiddles,
                lifting_log_size,
                &self.base_column_pool,
            )
        } else {
            CommitmentTreeProver::new(
                polynomials,
                self.config.fri_config.log_blowup_factor,
                self.twiddles,
                self.store_polynomials_coefficients,
                lifting_log_size,
                &self.base_column_pool,
            )
        };
        MC::mix_root(channel, tree.commitment.root());
        // Only newly owned ordinary commitments are considered. Externally supplied or borrowed
        // preprocessed trees use commit_tree and retain their existing storage behavior.
        for poly in &mut tree.polynomials {
            poly.evals.values.release_zeroed_pages();
        }
        self.trees.push(MaybeOwned::Owned(tree));
    }

    /// Appends an externally constructed [`CommitmentTreeProver`] to the scheme and mixes its
    /// Merkle root into the channel. Accepts both owned and borrowed trees.
    pub fn commit_tree(
        &mut self,
        tree: MaybeOwned<'a, CommitmentTreeProver<B, MC>>,
        channel: &mut MC::C,
    ) {
        MC::mix_root(channel, tree.commitment.root());
        self.trees.push(tree);
    }

    pub fn tree_builder(&mut self) -> TreeBuilder<'_, 'a, B, MC> {
        TreeBuilder { tree_index: self.trees.len(), commitment_scheme: self, polys: Vec::default() }
    }

    pub fn roots(&self) -> TreeVec<<MC::H as MerkleHasherLifted>::Hash> {
        self.trees.as_ref().map(|tree| tree.commitment.root())
    }

    pub fn polynomials(&self) -> TreeVec<ColumnVec<&Poly<B>>> {
        self.trees.as_ref().map(|tree| tree.polynomials.iter().collect())
    }

    pub fn evaluations(
        &self,
    ) -> TreeVec<ColumnVec<&CircleEvaluation<B, BaseField, BitReversedOrder>>> {
        self.trees.as_ref().map(|tree| tree.polynomials.iter().map(|poly| &poly.evals).collect())
    }

    pub fn trace(&self) -> Trace<'_, B> {
        let polys = self.polynomials();
        Trace { polys }
    }

    /// Computes the barycentric weights for every (column size, sampled point) pair, on buffers
    /// taken from [`Self::base_column_pool`]. The buffers are returned to the pool by
    /// [`Self::compute_samples()`].
    pub fn build_weights_hash_map(
        &self,
        sampled_points: &TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>>,
        max_log_size: u32,
    ) -> WeightsHashMap<B> {
        let weights_dashmap = WeightsHashMap::<B>::new();

        self.polynomials().zip_cols(sampled_points).map_cols(|(poly, points)| {
            let compute_weights = |(log_size, point): (u32, CirclePoint<SecureField>)| {
                weights_dashmap.entry((log_size, point)).or_insert_with(|| {
                    let weights_log_size = B::barycentric_log_size(
                        log_size,
                        self.config.fri_config.log_blowup_factor,
                    );
                    let buffer = SecureColumnByCoords {
                        columns: array::from_fn(|_| {
                            self.base_column_pool.take_or_alloc(weights_log_size)
                        }),
                    };
                    CircleEvaluation::<B, BaseField, BitReversedOrder>::barycentric_weights_into(
                        CanonicCoset::new(log_size),
                        point,
                        buffer,
                    )
                });
            };

            let log_size = poly.evals.domain.log_size();
            // For each sample point, compute the weights needed to evaluate the polynomial at
            // the folded sample point.
            // TODO(Leo): the computation `point.repeated_double(max_log_size - log_size)` is
            // likely repeated a bunch of times in a typical flat air. Consider moving it
            // outside the loop.
            #[cfg(not(feature = "parallel"))]
            points.iter().for_each(|&point| {
                compute_weights((log_size, point.repeated_double(max_log_size - log_size)))
            });

            #[cfg(feature = "parallel")]
            points.par_iter().for_each(|&point| {
                compute_weights((log_size, point.repeated_double(max_log_size - log_size)))
            });
        });

        weights_dashmap
    }

    /// Groups coefficient-free columns by domain size and projected sample point, so a backend
    /// can generate a small weight block once and reuse it across every column in the group.
    fn compute_grouped_samples(
        &self,
        sampled_points: &TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>>,
        lifting_log_size: u32,
    ) -> TreeVec<Vec<Vec<PointSample>>> {
        let polynomials = self.polynomials();
        let mut samples = sampled_points.as_cols_ref().map_cols(|points| {
            points
                .iter()
                .map(|&point| PointSample { point, value: SecureField::default() })
                .collect_vec()
        });
        let mut groups: HashMap<(u32, CirclePoint<SecureField>), Vec<(usize, usize, usize)>> =
            HashMap::new();
        assert_eq!(polynomials.len(), sampled_points.len());
        for (tree_index, columns) in polynomials.iter().enumerate() {
            assert_eq!(columns.len(), sampled_points[tree_index].len());
            for (column_index, poly) in columns.iter().enumerate() {
                let log_size = poly.evals.domain.log_size();
                for (sample_index, &point) in
                    sampled_points[tree_index][column_index].iter().enumerate()
                {
                    let projected_point = point.repeated_double(lifting_log_size - log_size);
                    if let Some(coefficients) = &poly.coeffs {
                        samples[tree_index][column_index][sample_index].value =
                            coefficients.eval_at_point(projected_point);
                    } else {
                        groups.entry((log_size, projected_point)).or_default().push((
                            tree_index,
                            column_index,
                            sample_index,
                        ));
                    }
                }
            }
        }
        let log_blowup = self.config.fri_config.log_blowup_factor;
        for ((log_size, point), members) in groups {
            let evaluations = members
                .iter()
                .map(|&(tree, column, _)| &polynomials[tree][column].evals)
                .collect_vec();
            let values = B::subdomain_eval_group(
                CanonicCoset::new(log_size),
                log_blowup,
                point,
                &evaluations,
            );
            assert_eq!(members.len(), values.len());
            for ((tree, column, sample), value) in members.into_iter().zip(values) {
                samples[tree][column][sample].value = value;
            }
        }
        samples
    }

    /// Evaluates the committed polynomials on the sampled points.
    fn compute_samples(
        &self,
        sampled_points: &TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>>,
        lifting_log_size: u32,
    ) -> TreeVec<Vec<Vec<PointSample>>> {
        let _span =
            span!(Level::INFO, "Evaluate columns out of domain", class = "EvaluateOutOfDomain")
                .entered();

        if !self.store_polynomials_coefficients {
            return self.compute_grouped_samples(sampled_points, lifting_log_size);
        }

        let weights_hash_map = if self.store_polynomials_coefficients {
            None
        } else {
            Some(self.build_weights_hash_map(sampled_points, lifting_log_size))
        };

        // Lambda that evaluates a polynomial on a collection of circle points and returns a vector
        // of point samples.
        let eval_at_points = |(poly, points): (&Poly<B>, &Vec<CirclePoint<SecureField>>)| {
            points
                .iter()
                .map(|&point| PointSample {
                    point,
                    value: poly.eval_at_point(
                        point.repeated_double(lifting_log_size - poly.evals.domain.log_size()),
                        weights_hash_map.as_ref(),
                    ),
                })
                .collect_vec()
        };

        #[cfg(not(feature = "parallel"))]
        let samples = self.polynomials().zip_cols(sampled_points).map_cols(eval_at_points);
        #[cfg(feature = "parallel")]
        let samples = self.polynomials().zip_cols(sampled_points).par_map_cols(eval_at_points);

        // Return the weights buffers to the memory pool for reuse.
        if let Some(weights_hash_map) = weights_hash_map {
            for ((log_size, _), weights) in weights_hash_map {
                let weights_log_size =
                    B::barycentric_log_size(log_size, self.config.fri_config.log_blowup_factor);
                for column in weights.columns {
                    self.base_column_pool.give_back(weights_log_size, column);
                }
            }
        }

        samples
    }

    pub fn prove_values(
        mut self,
        sampled_points: TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>>,
        channel: &mut MC::C,
    ) -> ExtendedCommitmentSchemeProof<MC::H> {
        let lifting_log_size = self.trees.last().unwrap().commitment.layers.len() as u32 - 1;

        // Evaluate polynomials on open points.
        let samples = self.compute_samples(&sampled_points, lifting_log_size);

        // Interpolation weights are complete; quotient and opening paths own their buffers.
        self.base_column_pool.release_all_idle();

        // Point samples are complete; subsequent quotient and opening paths use evaluations.
        for tree in &mut self.trees.0 {
            if let MaybeOwned::Owned(tree) = tree {
                for poly in &mut tree.polynomials {
                    drop(poly.coeffs.take());
                }
            }
        }

        let sampled_values =
            samples.as_cols_ref().map_cols(|x| x.iter().map(|o| o.value).collect());
        channel.mix_felts(&sampled_values.clone().flatten_cols());

        let columns = self.evaluations();
        print_column_size_histogram::<B, MC>(&columns);
        // Compute oods quotients for boundary constraints on the sampled points.
        let quotients = compute_fri_quotients(
            &columns,
            &samples,
            channel.draw_secure_felt(),
            lifting_log_size,
            self.twiddles,
            self.config.fri_config.log_blowup_factor,
        );

        // Run FRI commitment phase on the oods quotients.
        let fri_prover =
            FriProver::<B, MC>::commit(channel, self.config.fri_config, &quotients, self.twiddles);

        // Proof of work.
        let span1 = span!(Level::INFO, "Grind", class = "Queries POW").entered();
        let proof_of_work = B::grind(channel, self.config.fri_config.pow_bits);
        span1.exit();
        channel.mix_u64(proof_of_work);

        // FRI decommitment phase.
        let FriDecommitResult { fri_proof, query_positions, unsorted_query_locations } =
            fri_prover.decommit(channel);
        // Build the query position tree.
        let preprocessed_query_positions = prepare_preprocessed_query_positions(
            &query_positions,
            lifting_log_size,
            self.trees[0].commitment.layers.len() as u32 - 1,
        );
        let query_positions_tree = TreeVec::new(
            self.trees
                .iter()
                .enumerate()
                .map(|(i, _)| {
                    if i == 0 {
                        preprocessed_query_positions.as_slice()
                    } else {
                        query_positions.as_slice()
                    }
                })
                .collect::<Vec<_>>(),
        );
        let commitments = self.roots();
        let (queried_values, decommitments, aux): (Vec<_>, Vec<_>, Vec<_>) = self
            .trees
            .as_ref()
            .zip_eq(query_positions_tree)
            .map(|(tree, query_positions)| tree.decommit(query_positions))
            .0
            .into_iter()
            .map(|(v, x)| (v, x.decommitment, x.aux))
            .multiunzip();

        // The FRI input column is no longer needed: its four coordinate buffers go back to the
        // pool, where the next proof's evaluations of the same size reuse them.
        for column in quotients.values.columns {
            self.base_column_pool.give_back(lifting_log_size, column);
        }

        // Return evaluation buffers to the memory pool for reuse (owned trees only).
        for tree in &mut self.trees.0 {
            if let MaybeOwned::Owned(tree) = tree {
                for poly in tree.polynomials.drain(..) {
                    // A column of a striped tree holds only its first stripe.
                    let log_size = poly.evals.values.len().ilog2();
                    self.base_column_pool.give_back(log_size, poly.evals.values);
                }
            }
        }

        self.base_column_pool.release_all_idle();

        ExtendedCommitmentSchemeProof {
            proof: CommitmentSchemeProof {
                commitments,
                sampled_values,
                decommitments: TreeVec(decommitments),
                queried_values: TreeVec(queried_values),
                proof_of_work,
                fri_proof: fri_proof.proof,
                config: self.config,
            },
            aux: CommitmentSchemeProofAux {
                unsorted_query_locations,
                trace_decommitment: TreeVec(aux),
                fri: fri_proof.aux,
            },
        }
    }
}

/// Helper struct for aggregating polynomials and evaluations for a commitment tree.
pub struct TreeBuilder<'a, 'b, B: BackendForChannel<MC>, MC: MerkleChannel> {
    tree_index: usize,
    commitment_scheme: &'a mut CommitmentSchemeProver<'b, B, MC>,
    polys: ColumnVec<CircleCoefficients<B>>,
}
impl<B: BackendForChannel<MC>, MC: MerkleChannel> TreeBuilder<'_, '_, B, MC> {
    pub fn extend_evals(
        &mut self,
        columns: Vec<CircleEvaluation<B, BaseField, BitReversedOrder>>,
    ) -> TreeSubspan {
        let span = span!(Level::INFO, "Interpolation for commitment").entered();
        let polys = B::interpolate_columns_pooled(
            columns,
            self.commitment_scheme.twiddles,
            &self.commitment_scheme.base_column_pool,
        );
        span.exit();

        self.extend_polys(polys)
    }

    pub fn extend_polys(
        &mut self,
        columns: impl IntoIterator<Item = CircleCoefficients<B>>,
    ) -> TreeSubspan {
        let col_start = self.polys.len();
        self.polys.extend(columns);
        let col_end = self.polys.len();
        TreeSubspan { tree_index: self.tree_index, col_start, col_end }
    }

    pub fn commit(self, channel: &mut MC::C) {
        let _span = span!(Level::INFO, "Commitment").entered();
        self.commitment_scheme.commit(self.polys, channel);
    }
}

/// Prover data for a single commitment tree in a commitment scheme. The commitment scheme allows to
/// commit on a set of polynomials at a time. This corresponds to such a set.
pub struct CommitmentTreeProver<B: BackendForChannel<MC>, MC: MerkleChannel> {
    pub polynomials: ColumnVec<Poly<B>>,
    pub commitment: MerkleProverLifted<B, MC::H>,
}

impl<B: BackendForChannel<MC>, MC: MerkleChannel> CommitmentTreeProver<B, MC> {
    pub fn new(
        polynomials: ColumnVec<CircleCoefficients<B>>,
        log_blowup_factor: u32,
        twiddles: &TwiddleTree<B>,
        store_polynomials_coefficients: bool,
        lifting_log_size: u32,
        base_column_pool: &BaseColumnPool<B>,
    ) -> Self {
        let span = span!(Level::INFO, "Extension").entered();
        let polynomials = B::evaluate_polynomials(
            polynomials,
            log_blowup_factor,
            twiddles,
            store_polynomials_coefficients,
            base_column_pool,
        );
        span.exit();

        let _span = span!(Level::INFO, "Merkle").entered();
        let tree = MerkleProverLifted::commit(
            polynomials.iter().map(|poly: &Poly<B>| &poly.evals.values).collect(),
            lifting_log_size,
            0,
        );

        CommitmentTreeProver { polynomials, commitment: tree }
    }

    /// [`Self::new`] without stored coefficients, keeping of every column only its evaluations on
    /// the first of the `2^log_blowup_factor` parts of its domain (its first stripe), and without
    /// ever holding a column's whole evaluation: the stripes are computed and hashed one at a
    /// time. In bit-reversed order a column's stripe is a contiguous block of its evaluation, the
    /// block that the rows of the same part of the lifted domain read, so each stripe is a
    /// complete subtree of the lifted tree: the layers below the subtrees' roots are theirs side
    /// by side, and the layers above are hashed from their roots. Every layer is the one
    /// [`Self::new`] builds. Small columns are kept whole.
    pub fn new_striped(
        polynomials: ColumnVec<CircleCoefficients<B>>,
        log_blowup_factor: u32,
        twiddles: &TwiddleTree<B>,
        lifting_log_size: u32,
        base_column_pool: &BaseColumnPool<B>,
    ) -> Self {
        /// The smallest coefficients log size a striped tree takes, so that every column's
        /// stripe fills a SIMD vector.
        const MIN_COEFFS_LOG_SIZE: u32 = 4;
        /// The smallest height of the subtree of a stripe.
        const MIN_STRIPE_HEIGHT: u32 = 10;
        /// Columns with fewer evaluations are kept whole.
        const MIN_STRIPED_LOG_SIZE: u32 = 14;
        let max_log_size = polynomials.iter().map(|poly| poly.log_size()).max().unwrap_or(0);
        if polynomials.is_empty()
            || log_blowup_factor == 0
            || lifting_log_size < log_blowup_factor + MIN_STRIPE_HEIGHT
            || lifting_log_size < max_log_size + log_blowup_factor
            || polynomials.iter().any(|poly| poly.log_size() < MIN_COEFFS_LOG_SIZE)
        {
            return Self::new(
                polynomials,
                log_blowup_factor,
                twiddles,
                false,
                lifting_log_size,
                base_column_pool,
            );
        }
        let _span = span!(Level::INFO, "Striped commitment").entered();

        let domains = polynomials
            .iter()
            .map(|poly| CanonicCoset::new(poly.log_size() + log_blowup_factor).circle_domain())
            .collect_vec();
        let whole =
            domains.iter().map(|domain| domain.log_size() < MIN_STRIPED_LOG_SIZE).collect_vec();
        // The order in which the leaves hash the columns.
        let order = (0..polynomials.len()).sorted_by_key(|&i| domains[i].log_size()).collect_vec();

        // The evaluations of every column on one stripe of its domain.
        let evaluate_stripe = |polynomials: &[CircleCoefficients<B>],
                               stripe: usize,
                               reused: Vec<Option<Col<B, BaseField>>>|
         -> Vec<Col<B, BaseField>> {
            let stripe_twiddles: HashMap<u32, TwiddleTree<B>> = domains
                .iter()
                .map(|domain| domain.log_size())
                .sorted()
                .dedup()
                .map(|log_size| {
                    let domain = CanonicCoset::new(log_size).circle_domain();
                    (log_size, B::part_twiddles(domain, log_blowup_factor, stripe, twiddles))
                })
                .collect();
            let buffers = domains
                .iter()
                .zip(reused)
                .map(|(domain, reused)| {
                    reused.unwrap_or_else(|| {
                        base_column_pool.take_or_alloc(domain.log_size() - log_blowup_factor)
                    })
                })
                .collect_vec();
            let evaluate =
                |((poly, domain), buffer): ((&CircleCoefficients<B>, &CircleDomain), _)| {
                    let stripe_domain = part_domain(*domain, log_blowup_factor, stripe);
                    let twiddles = &stripe_twiddles[&domain.log_size()];
                    B::evaluate_into(poly, stripe_domain, twiddles, buffer).values
                };
            #[cfg(feature = "parallel")]
            {
                use rayon::iter::{IndexedParallelIterator, IntoParallelIterator};
                polynomials
                    .par_iter()
                    .zip(domains.par_iter())
                    .zip(buffers.into_par_iter())
                    .map(evaluate)
                    .collect()
            }
            #[cfg(not(feature = "parallel"))]
            polynomials.iter().zip(&domains).zip(buffers).map(evaluate).collect()
        };
        let stripe_height = lifting_log_size - log_blowup_factor;
        let hash = |columns: &[Col<B, BaseField>]| {
            let sorted = order.iter().map(|&i| &columns[i]).collect_vec();
            B::build_layers_sparse(&sorted, stripe_height)
        };

        let n_columns = polynomials.len();
        let n_stripes = 1 << log_blowup_factor;
        let mut first_stripe = vec![];
        let mut subtrees: Vec<Option<Vec<_>>> = (0..n_stripes).map(|_| None).collect();
        let mut whole_stripes: Vec<Vec<Col<B, BaseField>>> = vec![vec![]; n_columns];
        // The first stripe, the one that is kept, goes last: the coefficients go as soon as it is
        // evaluated, so that a column never holds more than its coefficients and one stripe, its
        // buffer passed on from stripe to stripe through the pool.
        // Each column's buffer goes straight on to the column's next stripe. Going through the
        // pool instead let columns of other sizes take (and grow) them in between, so every
        // stripe allocated some buffers afresh while the pool filled with idle ones.
        let no_buffers = || (0..n_columns).map(|_| None).collect_vec();
        let mut reused = no_buffers();
        let mut polynomials = Some(polynomials);
        for stripe in (1..n_stripes).chain([0]) {
            let _span = span!(Level::INFO, "Extension and Merkle", stripe).entered();
            let values = evaluate_stripe(
                polynomials.as_ref().unwrap(),
                stripe,
                std::mem::replace(&mut reused, no_buffers()),
            );
            if stripe == 0 {
                drop(polynomials.take());
            }
            subtrees[stripe] = Some(hash(&values));
            if stripe == 0 {
                first_stripe = values;
                continue;
            }
            for (i, column) in values.into_iter().enumerate() {
                if whole[i] {
                    whole_stripes[i].push(column);
                } else {
                    reused[i] = Some(column);
                }
            }
        }
        let subtrees = subtrees.into_iter().map(Option::unwrap).collect_vec();

        // The stripes' buffers, idle in the pool now, are of no use to the next tree, whose
        // columns have other sizes: they go.
        base_column_pool.release_all_idle();

        // The layers below the subtrees' roots, side by side in the order of the rows (a layer the
        // subtrees leave empty stays empty), and then the layers above.
        let mut layers: Vec<Col<B, <MC::H as MerkleHasherLifted>::Hash>> = (0..=stripe_height
            as usize)
            .map(|level| {
                if subtrees.iter().any(|subtree| subtree[level].is_empty()) {
                    return std::iter::empty().collect();
                }
                subtrees.iter().flat_map(|subtree| subtree[level].to_cpu()).collect()
            })
            .collect();
        drop(subtrees);
        for _ in stripe_height..lifting_log_size {
            let next = B::build_next_layer(layers.last().unwrap());
            layers.push(next);
        }
        layers.reverse();
        let commitment = MerkleProverLifted { layers };

        let polynomials = first_stripe
            .into_iter()
            .zip(domains)
            .zip(whole_stripes)
            .map(|((values, domain), stripes)| {
                let values = if stripes.is_empty() {
                    values
                } else {
                    let whole_values = values
                        .to_cpu()
                        .into_iter()
                        .chain(stripes.iter().flat_map(|stripe| stripe.to_cpu()))
                        .collect();
                    base_column_pool.give_back(values.len().ilog2(), values);
                    for stripe in stripes {
                        base_column_pool.give_back(stripe.len().ilog2(), stripe);
                    }
                    whole_values
                };
                let mut evals = CircleEvaluation::new(
                    part_domain(domain, domain.log_size() - values.len().ilog2(), 0),
                    values,
                );
                evals.domain = domain;
                Poly::new(None, evals)
            })
            .collect();
        CommitmentTreeProver { polynomials, commitment }
    }

    /// Decommits the merkle tree on the given query positions.
    /// Returns the values at the queried positions and the decommitment.
    /// The queries are given as a mapping from the log size of the layer size to the queried
    /// positions on each column of that size.
    fn decommit(
        &self,
        queries: &[usize],
    ) -> (ColumnVec<Vec<BaseField>>, ExtendedMerkleDecommitmentLifted<MC::H>) {
        if self.polynomials.iter().any(|poly| poly.evals.values.len() < poly.evals.domain.size()) {
            return self.decommit_from_stripes(queries);
        }
        let eval_vec = self.polynomials.iter().map(|poly| &poly.evals.values).collect_vec();
        self.commitment.decommit(queries, eval_vec)
    }

    /// [`Self::decommit`] for a tree some of whose columns hold only their first stripe (see
    /// [`Self::new_striped`]): the rows the decommitment reads elsewhere are evaluated from the
    /// coefficients, interpolated again from the first stripe one column at a time.
    fn decommit_from_stripes(
        &self,
        queries: &[usize],
    ) -> (ColumnVec<Vec<BaseField>>, ExtendedMerkleDecommitmentLifted<MC::H>) {
        let _span = span!(Level::INFO, "Decommit from stripes").entered();
        let log_sizes =
            self.polynomials.iter().map(|poly| poly.evals.domain.log_size()).collect_vec();
        // The interpolation twiddles of every stripe domain.
        let stripe_twiddles: HashMap<(u32, u32), TwiddleTree<B>> = self
            .polynomials
            .iter()
            .filter(|poly| poly.evals.values.len() < poly.evals.domain.size())
            .map(|poly| (poly.evals.domain.log_size(), poly.evals.values.len().ilog2()))
            .sorted()
            .dedup()
            .map(|key| {
                let poly = self
                    .polynomials
                    .iter()
                    .find(|poly| {
                        (poly.evals.domain.log_size(), poly.evals.values.len().ilog2()) == key
                    })
                    .unwrap();
                (key, B::precompute_twiddles(poly.stripe_domain().half_coset))
            })
            .collect();
        let opened_values = |poly: &Poly<B>| -> HashMap<usize, BaseField> {
            let domain = poly.evals.domain;
            let indices = self.commitment.decommit_indices(queries, domain.log_size());
            // The rows the column holds are read; the others are evaluated.
            let held = poly.evals.values.len();
            let n_held = indices.partition_point(|&index| index < held);
            let mut values =
                indices[..n_held].iter().map(|&index| poly.evals.values.at(index)).collect_vec();
            if n_held < indices.len() {
                let twiddles = &stripe_twiddles[&(domain.log_size(), held.ilog2())];
                let coeffs = poly.stripe_coefficients(twiddles);
                values.extend(B::eval_at_domain_indices(&coeffs, domain, &indices[n_held..]));
            }
            indices.into_iter().zip(values).collect()
        };
        #[cfg(feature = "parallel")]
        let opened: Vec<HashMap<usize, BaseField>> =
            self.polynomials.par_iter().map(opened_values).collect();
        #[cfg(not(feature = "parallel"))]
        let opened: Vec<HashMap<usize, BaseField>> =
            self.polynomials.iter().map(opened_values).collect();
        self.commitment
            .decommit_from_values(queries, &log_sizes, |column, index| opened[column][&index])
    }
}

fn print_column_size_histogram<B: BackendForChannel<MC>, MC: MerkleChannel>(
    columns_per_tree: &TreeVec<ColumnVec<&CircleEvaluation<B, BaseField, BitReversedOrder>>>,
) {
    let mut log_size_histogram = HashMap::new();
    for columns in columns_per_tree.iter() {
        for column in columns {
            *log_size_histogram.entry(column.domain.log_size()).or_insert(0) += 1;
        }
    }
    for (log_size, count) in log_size_histogram {
        info!("Log size {log_size}: {count}");
    }
}
