//! A component's constraint evaluation (the composition's point-wise part) on the GPU, generated
//! from the constraints a constraint-framework `ExprEvaluator` traces, checked bit for bit against
//! stwo's SIMD evaluation:
//!
//! - [`ir`]: the traced expressions lowered to a hash-consed SSA program over M31 and QM31.
//! - [`interp`]: a CPU interpreter of the program, the first oracle.
//! - [`wgsl`]: the program as a WGSL kernel, one invocation per row.
//! - [`kernel`]: the kernels' pipelines, buffers and dispatches.
//! - [`capture`]: a constraint evaluation hook that observes a proof's components (shapes,
//!   checks) and writes [`fixture`]s of the SIMD evaluation's inputs and output.
pub mod capture;
pub mod fixture;
pub mod interp;
pub mod ir;
pub mod kernel;
pub mod wgsl;
