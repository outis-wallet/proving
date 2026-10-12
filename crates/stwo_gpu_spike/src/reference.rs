//! What the GPU results are checked against, and the plumbing the tests and the benchmark share:
//! stwo's SIMD backend on the same columns, random columns, their upload as per-size batches.

use itertools::Itertools;
use rand::Rng;
use stwo::core::fields::m31::{BaseField, P};
use stwo::core::vcs::blake2_hash::Blake2sHash;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleHasher;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::simd::column::BaseColumn;
use stwo::prover::vcs_lifted::ops::MerkleOpsLifted;

use crate::GpuContext;
use crate::merkle::ColumnBatch;
use crate::shapes::size_batches;

/// `n` random canonical M31 values.
pub fn random_values(rng: &mut impl Rng, n: usize) -> Vec<u32> {
    (0..n).map(|_| rng.random_range(0..P)).collect()
}

/// Random columns of these log sizes.
pub fn random_columns(rng: &mut impl Rng, log_sizes: &[u32]) -> Vec<Vec<u32>> {
    log_sizes.iter().map(|&log_size| random_values(rng, 1 << log_size)).collect()
}

/// The columns as the SIMD backend's.
pub fn simd_columns(columns: &[Vec<u32>]) -> Vec<BaseColumn> {
    columns
        .iter()
        .map(|values| values.iter().map(|&v| BaseField::from_u32_unchecked(v)).collect())
        .collect()
}

/// The layers of a stripe's subtree as stwo builds them: `build_layers_sparse` on the columns, in
/// tree order, leaves first.
pub fn simd_stripe_layers(sorted_columns: &[BaseColumn], height: u32) -> Vec<Vec<Blake2sHash>> {
    <SimdBackend as MerkleOpsLifted<Blake2sMerkleHasher>>::build_layers_sparse(
        &sorted_columns.iter().collect_vec(),
        height,
    )
}

/// The layers of a whole tree from its stripes' subtrees, as `CommitmentTreeProver::new_striped`
/// assembles them (leaves first): the subtrees' levels side by side (a level a subtree leaves
/// empty stays empty), then `build_next_layer` up to the root.
pub fn simd_tree_layers(subtrees: &[Vec<Vec<Blake2sHash>>], stripe_height: u32) -> Vec<Vec<Blake2sHash>> {
    let mut layers: Vec<Vec<Blake2sHash>> = (0..=stripe_height as usize)
        .map(|level| {
            if subtrees.iter().any(|subtree| subtree[level].is_empty()) {
                return vec![];
            }
            subtrees.iter().flat_map(|subtree| subtree[level].iter().copied()).collect()
        })
        .collect();
    let log_n_stripes = subtrees.len().ilog2();
    for _ in 0..log_n_stripes {
        let next = <SimdBackend as MerkleOpsLifted<Blake2sMerkleHasher>>::build_next_layer(
            layers.last().unwrap(),
        );
        layers.push(next);
    }
    layers
}

/// A stripe's columns on the GPU: one buffer per size batch.
pub struct UploadedStripe {
    pub buffers: Vec<wgpu::Buffer>,
    /// `(log_size, n_columns)` of each buffer.
    pub shapes: Vec<(u32, u32)>,
}

impl UploadedStripe {
    /// Uploads columns given in tree order (nondecreasing sizes): a buffer per size, or several
    /// for a size whose columns exceed the device's largest buffer.
    pub fn new(gpu: &GpuContext, sorted_columns: &[Vec<u32>]) -> Self {
        let log_sizes = sorted_columns.iter().map(|c| c.len().ilog2()).collect_vec();
        let max_buffer = gpu.device.limits().max_buffer_size;
        let mut shapes = vec![];
        for (log_size, n) in size_batches(&log_sizes) {
            let per_buffer = (max_buffer / (4 << log_size)).min(n as u64) as u32;
            assert!(per_buffer > 0, "a column of 2^{log_size} values exceeds a buffer");
            let mut first = 0;
            while first < n {
                shapes.push((log_size, per_buffer.min(n - first)));
                first += per_buffer.min(n - first);
            }
        }
        let mut columns = sorted_columns.iter();
        let buffers = shapes
            .iter()
            .map(|&(log_size, n)| {
                let words: Vec<u32> = columns.by_ref().take(n as usize).flatten().copied().collect();
                gpu.upload(&format!("columns 2^{log_size}"), &words)
            })
            .collect();
        Self { buffers, shapes }
    }

    /// The columns, given in tree order, as the batches' words: one vector per buffer.
    pub fn batch_words(&self, sorted_columns: &[Vec<u32>]) -> Vec<Vec<u32>> {
        let mut columns = sorted_columns.iter();
        self.shapes
            .iter()
            .map(|&(_, n)| columns.by_ref().take(n as usize).flatten().copied().collect())
            .collect()
    }

    /// Writes new values into the buffers (`queue.write_buffer`, one per batch), as
    /// [`Self::batch_words`] lays them out.
    pub fn write(&self, gpu: &GpuContext, batch_words: &[Vec<u32>]) {
        for (buffer, words) in self.buffers.iter().zip(batch_words) {
            gpu.queue.write_buffer(buffer, 0, bytemuck::cast_slice(words));
        }
    }

    pub fn batches(&self) -> Vec<ColumnBatch<'_>> {
        self.buffers
            .iter()
            .zip(&self.shapes)
            .map(|(buffer, &(log_size, n_columns))| ColumnBatch { buffer, word_offset: 0, log_size, n_columns })
            .collect()
    }
}
