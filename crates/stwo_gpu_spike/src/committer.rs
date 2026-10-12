//! The GPU striped commitment as a prover uses it: one long-lived [`GpuCommitter`] per process
//! (the device and the compiled kernels, and whatever else can be reused between commitments),
//! whose [`GpuCommitter::commit_striped`] either returns exactly what
//! `CommitmentTreeProver::new_striped` returns for the same arguments, or gives the polynomials
//! back with the reason, for the caller to commit them on the CPU.
//!
//! This is the API the prover's hook calls; what is behind it (buffer reuse, faster transfers)
//! can change without the hook changing.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Mutex;

use itertools::Itertools;
use stwo::core::ColumnVec;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sM31MerkleChannel;
use stwo::prover::CommitmentTreeProver;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::poly::circle::CircleCoefficients;
use stwo::prover::poly::twiddles::TwiddleTree;

use crate::GpuContext;
use crate::commit::{CommitKernels, StripedCommitment, takes_striped_path};

/// The tree `new_striped` returns.
pub type StripedTree = CommitmentTreeProver<SimdBackend, Blake2sM31MerkleChannel>;

/// Why a commitment wasn't made on the GPU: the caller commits on the CPU instead.
#[derive(Debug)]
pub enum Declined {
    /// `new_striped` wouldn't take its striped path for these arguments.
    NotStriped,
    /// The GPU failed (a validation error, a lost device, ...).
    Gpu(String),
}

/// What [`GpuCommitter::commit_striped`] returns when it declines: the polynomials, untouched.
pub type DeclinedCommit = (ColumnVec<CircleCoefficients<SimdBackend>>, Declined);

/// A device and its compiled kernels, for a process's commitments, one at a time.
pub struct GpuCommitter {
    gpu: GpuContext,
    kernels: CommitKernels,
    /// One commitment at a time: the device's queue is shared, and so will be its buffers.
    busy: Mutex<()>,
}

impl GpuCommitter {
    /// The committer on the adapter the environment picks (see the crate docs), or why there is
    /// none.
    pub fn new() -> Result<Self, String> {
        let gpu = GpuContext::try_new()?;
        let kernels = catch_unwind(AssertUnwindSafe(|| CommitKernels::new(&gpu)))
            .map_err(|panic| format!("kernels: {}", panic_message(&panic)))?;
        Ok(Self { gpu, kernels, busy: Mutex::new(()) })
    }

    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.gpu.adapter_info
    }

    /// `CommitmentTreeProver::new_striped(polynomials, log_blowup_factor, twiddles,
    /// lifting_log_size, pool)` on the GPU: the same tree (every layer, every column's kept
    /// evaluations), or the polynomials back with the reason.
    pub fn commit_striped(
        &self,
        polynomials: ColumnVec<CircleCoefficients<SimdBackend>>,
        log_blowup_factor: u32,
        twiddles: &TwiddleTree<SimdBackend>,
        lifting_log_size: u32,
    ) -> Result<StripedTree, DeclinedCommit> {
        let log_sizes = polynomials.iter().map(|poly| poly.log_size()).collect_vec();
        if !takes_striped_path(&log_sizes, log_blowup_factor, lifting_log_size) {
            return Err((polynomials, Declined::NotStriped));
        }
        let _busy = self.busy.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        // The polynomials are kept until the tree is read back, so that a failure can still
        // give them back.
        let tree = catch_unwind(AssertUnwindSafe(|| {
            let commitment = StripedCommitment::new(
                &self.gpu,
                &self.kernels,
                &log_sizes,
                log_blowup_factor,
                lifting_log_size,
                twiddles,
            );
            commitment.upload(&self.gpu, &polynomials);
            let output = commitment.run_and_read(&self.gpu, &self.kernels);
            commitment.tree_prover(output)
        }));
        match tree {
            Ok(tree) => Ok(tree),
            Err(panic) => Err((polynomials, Declined::Gpu(panic_message(&panic)))),
        }
    }
}

fn panic_message(panic: &Box<dyn std::any::Any + Send>) -> String {
    panic
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| panic.downcast_ref::<&str>().map(|message| message.to_string()))
        .unwrap_or_else(|| "a panic".to_string())
}
