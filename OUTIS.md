# Outis's branch of StarkWare's prover

This is `starkware-libs/proving` at `e0035ffc`, the revision the sequencer's `main-v0.14.4` pins
(since `148de445`, 2026-10-28) and whose proofs Sepolia's gateway verifies, plus Outis's commits.
Until then it was `2b495a36`, the revision of `PRIVACY-0.14.4-RC.0`, whose recursion circuit
Sepolia no longer knows ("Unknown preprocessed root"). The changes make the whole Starknet transaction
proof small enough in memory for phones and browsers (wasm32, 4 GiB), and let it run on a GPU
(native, browsers, phones) with every GPU path off unless the prover turns it on. They are used by
[outis-wallet/prover](https://github.com/outis-wallet/prover), and run on outis.cash/prover and
in the Outis Prover apps.

Proofs are byte-identical to the upstream prover's for the same input.

## The commits

| Commit | What | Whose |
|---|---|---|
| The provably.fast frontier 754321f3, on 2b495a36 | SIMD backend (FFT, Blake2s, grinding, circle), memory pool, witness generation, packed-leaf commitment path | The contributors to the [provably.fast](https://provably.fast) challenge (Starknet Innovation), as its promoted frontier `754321f3` |
| Striped commitment: stripe-0 residency on both legs (s2) | Each committed column keeps only its first stripe; the rest is re-evaluated when read. About 5.9 GiB down to 2.5 GiB on the provably.fast benchmark proof | Outis, after ryun1's public description of the approach on the provably.fast research graph |
| Build and run on wasm32-unknown-unknown | No file system, no sonic-rs, 32-bit `usize`, real memory for reserved capacity | Outis |
| Android: the large-block cache maps with Linux's MAP_ANONYMOUS | `target_os = "android"` took macOS's flag | Outis |
| Witness: no heap Vec per row in the logup combines | `verify_bitwise_xor_12` (Cairo and circuit) and `memory_id_to_big` combined each row's values through a fresh `Vec`. On wasm32 every allocation takes one global spin lock, so the parallel loop serialized on it | Outis |
| privacy_recursive_prove_once | The recursive proof for a process that proves once: each leg's precomputes are built when it starts and freed when it ends (about -85 MiB at the peak), and none outlive the proof | Outis |
| privacy_recursive_prove_once_stoppable | The proof for a process that proves once, stoppable: it reads a stop flag between steps (each leg, and the stages of the Cairo proof through `prove_cairo_with_precompute_stoppable`) and returns `Cancelled` once the flag is set, so an app can cancel a proof. The functions that can't be stopped call these with a flag never set; proofs are byte-identical | Outis |

| The GPU (stwo-gpu-spike, a new crate, wgpu) | The striped commitments on a GPU (Vulkan, Metal, DX12; WebGPU in a page's worker), streamed; composition (constraint kernels generated from the components' expressions), decommitment, out-of-domain samples and FRI quotients on the GPU with the trace's coefficients resident; lean commitments for unified memory (a stripe and a column group at a time, the host's coefficients read in place on Metal, the CPU path on any GPU failure); fixes for Metal (no workgroup zero-initialization) and Adreno (no indexed uniform table) | Outis |
| The GPU's hooks in stwo and the constraint framework | `CommitmentTreeProver::new_striped`, the decommitment, the out-of-domain samples and the FRI quotients each try a registered hook first; the constraint evaluation and the trace parts likewise (with ExprEvaluator's side tables). Every hook is off unless registered, its answer is shape-checked and the CPU computes whatever it declines: with no hook the computation is the one before | Outis |
| CI gates passing again | fmt, clippy (CI's flags), stwo-examples, the fft benchmark and Cargo.lock, which the frontier, striped-commitment and wasm commits had broken; part 0 of a trace part computes no twiddles it doesn't use | Outis |
| The large-block cache from 64 KiB on Apple | macOS and iOS keep freed blocks between 64 KiB and 1 MiB: mapping them here returns them (the CPU path's peak footprint 2576 to 2358 MiB on an M5 Pro). 1 MiB elsewhere, as before | Outis |
| Lookup data narrowed from the base trace to the interaction trace | Each Cairo component's lookup tuples live from its `write_trace` until its interaction trace, across the base trace commitment. The bitwise-XOR tuples are now stored as one `u32` (4x smaller) and `memory_id_to_big`'s as 11 `u32` words (about 2.7x), in the bits the AIR range-checks them to, and widened back for the interaction trace. -350 MiB of wasm memory on every capture, no time cost. Hand-edits 34 AIR-generated witness files | ryun1, in the provably.fast frontier `66670df7` (whole-proof job f2498dc7), taken as is |
| The verifier context freed before the circuit's base trace commitment | `prove_once` takes the padded context's values out (`FinalizedContext::into_values`, freeing the gates) and hands them to `prove_circuit_with_precompute_owned`, which frees them once the base trace is written. About -200 MiB through the circuit proof | Outis, after realadii's change in the provably.fast frontier `55720f34` |
| Upstream `e0035ffc` merged in | Upstream's 167 commits from `2b495a36`, among them more zk-blinding rows in the circuit (`e0035ffc`), one Blake2s channel whose output is not reduced to M31 (#445) and `FriConfig` renamed `FriParams` (#296, #297). Upstream's semantics, the fork's mechanisms: the preprocessed trace is upstream's, with the compact and generated columns and their parallel build; the SIMD grind keeps its precomputed round 0 but no M31 reduction; the once paths prove through upstream's single-argument `prepare_cairo_proof_for_circuit_verifier`. Tested: proofs byte-identical to upstream's (below), and `privacy_recursive_prove_once` gives `privacy_recursive_prove`'s bytes | Outis |

The prover's releases pin this branch at a tag: `outis-v0.1.0` (16f4a93a, the CPU prover),
`outis-v0.2.0` (58503232, the GPU modes) and `outis-v0.2.1` (the lookup-data narrowing and the freed
verifier context: about -350 MiB of wasm memory and -200 to -320 MiB natively, the same time). Their measurements are in outis-wallet/prover's
BASELINES.md.

Of the non-trivial lines these commits add, about 2,600 come from the frontier, about 830 are
ours (the striped prover and its port to `2b495a36`), about 50 are the wasm and Android fixes, and
about 25 remove the per-row allocations. The lookup-data narrowing adds about 1,000 more lines from the frontier.

## Following upstream

Rebase this branch on a newer upstream revision, or merge it in (as for `e0035ffc`: 9 conflicted
files once, against 108 commits replayed), then check that proofs are still byte-identical to the
upstream prover's. For `e0035ffc` that was the sequencer's transaction prover at `148de445`, built
as it is and built with this branch patched in for every proving crate it uses, proving the same
recorded Sepolia request pinned to one block: the unpatched prover twice, then the patched one,
all three byte-identical. outis-wallet/prover's captures carry the reference proofs.

## License

Apache-2.0, as upstream (`LICENSE`). The files changed from `e0035ffc`:

- `Cargo.lock`
- `OUTIS.md`
- `crates/cairo-air/src/components/verify_bitwise_xor_12.rs`
- `crates/cairo-program-runner-lib/Cargo.toml`
- `crates/cairo-program-runner-lib/src/hints/mod.rs`
- `crates/cairo-program-runner-lib/src/hints/privacy_simple_bootloader_hints.rs`
- `crates/cairo-program-runner-lib/src/utils.rs`
- `crates/circuit_common/Cargo.toml`
- `crates/circuit_common/src/preprocessed.rs`
- `crates/circuit_common/src/preprocessed_test.rs`
- `crates/circuit_prover/src/circuit_air/components/verify_bitwise_xor_12.rs`
- `crates/circuit_prover/src/prover.rs`
- `crates/circuit_prover/src/witness/components/blake_g_gate.rs`
- `crates/circuit_prover/src/witness/components/eq.rs`
- `crates/circuit_prover/src/witness/components/m_31_to_u_32.rs`
- `crates/circuit_prover/src/witness/components/prelude.rs`
- `crates/circuit_prover/src/witness/components/qm_31_ops.rs`
- `crates/circuit_prover/src/witness/components/triple_xor.rs`
- `crates/circuit_prover/src/witness/components/verify_bitwise_xor_12.rs`
- `crates/circuit_prover/src/witness/components/verify_bitwise_xor_4.rs`
- `crates/circuit_prover/src/witness/components/verify_bitwise_xor_7.rs`
- `crates/circuit_prover/src/witness/components/verify_bitwise_xor_8.rs`
- `crates/circuit_prover/src/witness/components/verify_bitwise_xor_9.rs`
- `crates/circuits/Cargo.toml`
- `crates/circuits/src/circuit.rs`
- `crates/circuits/src/context.rs`
- `crates/constraint-framework/src/component.rs`
- `crates/constraint-framework/src/expr/evaluator.rs`
- `crates/constraint-framework/src/expr/mod.rs`
- `crates/constraint-framework/src/lib.rs`
- `crates/constraint-framework/src/logup.rs`
- `crates/constraint-framework/src/prover/component_prover.rs`
- `crates/constraint-framework/src/prover/constraint_hook.rs`
- `crates/constraint-framework/src/prover/fingerprint.rs`
- `crates/constraint-framework/src/prover/logup.rs`
- `crates/constraint-framework/src/prover/mod.rs`
- `crates/constraint-framework/src/prover/simd_domain.rs`
- `crates/constraint-framework/src/prover/trace_part_hook.rs`
- `crates/constraint-framework/tests/trace_part_hook.rs`
- `crates/examples/src/xor/gkr_lookups/mle_eval.rs`
- `crates/privacy_prove/Cargo.toml`
- `crates/privacy_prove/src/consts.rs`
- `crates/privacy_prove/src/lib.rs`
- `crates/privacy_prove/src/tests.rs`
- `crates/prover/src/prover.rs`
- `crates/prover/src/witness/components/add_ap_opcode.rs`
- `crates/prover/src/witness/components/add_mod_builtin.rs`
- `crates/prover/src/witness/components/add_opcode.rs`
- `crates/prover/src/witness/components/add_opcode_small.rs`
- `crates/prover/src/witness/components/assert_eq_opcode_double_deref.rs`
- `crates/prover/src/witness/components/bitwise_builtin.rs`
- `crates/prover/src/witness/components/blake_compress_opcode.rs`
- `crates/prover/src/witness/components/blake_g.rs`
- `crates/prover/src/witness/components/blake_round.rs`
- `crates/prover/src/witness/components/blake_round_sigma.rs`
- `crates/prover/src/witness/components/call_opcode_abs.rs`
- `crates/prover/src/witness/components/call_opcode_rel_imm.rs`
- `crates/prover/src/witness/components/cube_252.rs`
- `crates/prover/src/witness/components/ec_op_builtin.rs`
- `crates/prover/src/witness/components/jnz_opcode_non_taken.rs`
- `crates/prover/src/witness/components/jnz_opcode_taken.rs`
- `crates/prover/src/witness/components/jump_opcode_abs.rs`
- `crates/prover/src/witness/components/jump_opcode_double_deref.rs`
- `crates/prover/src/witness/components/jump_opcode_rel.rs`
- `crates/prover/src/witness/components/jump_opcode_rel_imm.rs`
- `crates/prover/src/witness/components/memory_id_to_big.rs`
- `crates/prover/src/witness/components/mul_mod_builtin.rs`
- `crates/prover/src/witness/components/mul_opcode.rs`
- `crates/prover/src/witness/components/mul_opcode_small.rs`
- `crates/prover/src/witness/components/partial_ec_mul_window_bits_18.rs`
- `crates/prover/src/witness/components/partial_ec_mul_window_bits_9.rs`
- `crates/prover/src/witness/components/pedersen_aggregator_window_bits_18.rs`
- `crates/prover/src/witness/components/pedersen_aggregator_window_bits_9.rs`
- `crates/prover/src/witness/components/pedersen_points_table_window_bits_18.rs`
- `crates/prover/src/witness/components/pedersen_points_table_window_bits_9.rs`
- `crates/prover/src/witness/components/poseidon_aggregator.rs`
- `crates/prover/src/witness/components/poseidon_round_keys.rs`
- `crates/prover/src/witness/components/qm_31_add_mul_opcode.rs`
- `crates/prover/src/witness/components/range_check96_builtin.rs`
- `crates/prover/src/witness/components/range_check_3_3_3_3_3.rs`
- `crates/prover/src/witness/components/range_check_3_6_6_3.rs`
- `crates/prover/src/witness/components/range_check_4_3.rs`
- `crates/prover/src/witness/components/range_check_4_4.rs`
- `crates/prover/src/witness/components/range_check_4_4_4_4.rs`
- `crates/prover/src/witness/components/range_check_7_2_5.rs`
- `crates/prover/src/witness/components/range_check_9_9.rs`
- `crates/prover/src/witness/components/range_check_builtin.rs`
- `crates/prover/src/witness/components/ret_opcode.rs`
- `crates/prover/src/witness/components/triple_xor_32.rs`
- `crates/prover/src/witness/components/verify_bitwise_xor_12.rs`
- `crates/prover/src/witness/components/verify_bitwise_xor_4.rs`
- `crates/prover/src/witness/components/verify_bitwise_xor_7.rs`
- `crates/prover/src/witness/components/verify_bitwise_xor_8.rs`
- `crates/prover/src/witness/components/verify_bitwise_xor_9.rs`
- `crates/prover/src/witness/components/verify_instruction.rs`
- `crates/prover/src/witness/prelude.rs`
- `crates/prover/src/witness/utils.rs`
- `crates/stwo/benches/fft.rs`
- `crates/stwo/src/core/channel/mod.rs`
- `crates/stwo/src/prover/air/accumulation.rs`
- `crates/stwo/src/prover/air/component_prover.rs`
- `crates/stwo/src/prover/backend/mod.rs`
- `crates/stwo/src/prover/backend/simd/accumulation.rs`
- `crates/stwo/src/prover/backend/simd/blake2s.rs`
- `crates/stwo/src/prover/backend/simd/blake2s_lifted.rs`
- `crates/stwo/src/prover/backend/simd/circle.rs`
- `crates/stwo/src/prover/backend/simd/cm31.rs`
- `crates/stwo/src/prover/backend/simd/column.rs`
- `crates/stwo/src/prover/backend/simd/fft/ifft.rs`
- `crates/stwo/src/prover/backend/simd/fft/rfft.rs`
- `crates/stwo/src/prover/backend/simd/grind.rs`
- `crates/stwo/src/prover/backend/simd/m31.rs`
- `crates/stwo/src/prover/backend/simd/qm31.rs`
- `crates/stwo/src/prover/backend/simd/quotients.rs`
- `crates/stwo/src/prover/backend/simd/very_packed_m31.rs`
- `crates/stwo/src/prover/fri.rs`
- `crates/stwo/src/prover/mempool.rs`
- `crates/stwo/src/prover/mod.rs`
- `crates/stwo/src/prover/pcs/mod.rs`
- `crates/stwo/src/prover/pcs/quotient_ops.rs`
- `crates/stwo/src/prover/poly/circle/domain_eval.rs`
- `crates/stwo/src/prover/poly/circle/mod.rs`
- `crates/stwo/src/prover/poly/circle/ops.rs`
- `crates/stwo/src/prover/vcs_lifted/ops.rs`
- `crates/stwo/src/prover/vcs_lifted/prover.rs`

and the new crate `crates/stwo_gpu_spike/` (Outis).
