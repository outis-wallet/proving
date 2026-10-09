# Outis's branch of StarkWare's prover

This is `starkware-libs/proving` at `2b495a36`, the revision the sequencer's
`PRIVACY-0.14.4-RC.0` pins, plus six commits. The changes make the whole Starknet transaction
proof small enough in memory for phones and browsers (wasm32, 4 GiB). They are used by
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

Of the non-trivial lines these commits add, about 2,600 come from the frontier, about 830 are
ours (the striped prover and its port to `2b495a36`), about 50 are the wasm and Android fixes, and
about 25 remove the per-row allocations.

## Following upstream

Rebase this branch on a newer upstream revision, then check that proofs are still byte-identical
to the upstream prover's (outis-wallet/prover's captures carry the reference proofs).

## License

Apache-2.0, as upstream (`LICENSE`). The files changed from `2b495a36`:

- `Cargo.lock`
- `crates/cairo-air/src/components/verify_bitwise_xor_12.rs`
- `crates/cairo-program-runner-lib/Cargo.toml`
- `crates/cairo-program-runner-lib/src/hints/mod.rs`
- `crates/cairo-program-runner-lib/src/hints/privacy_simple_bootloader_hints.rs`
- `crates/cairo-program-runner-lib/src/utils.rs`
- `crates/circuit_common/Cargo.toml`
- `crates/circuit_common/src/preprocessed.rs`
- `crates/circuit_prover/src/circuit_air/components/verify_bitwise_xor_12.rs`
- `crates/circuit_prover/src/prover.rs`
- `crates/circuit_prover/src/witness/components/blake_g_gate.rs`
- `crates/circuit_prover/src/witness/components/eq.rs`
- `crates/circuit_prover/src/witness/components/m_31_to_u_32.rs`
- `crates/circuit_prover/src/witness/components/prelude.rs`
- `crates/circuit_prover/src/witness/components/qm31_ops.rs`
- `crates/circuit_prover/src/witness/components/triple_xor.rs`
- `crates/circuit_prover/src/witness/components/verify_bitwise_xor_12.rs`
- `crates/circuit_prover/src/witness/components/verify_bitwise_xor_4.rs`
- `crates/circuit_prover/src/witness/components/verify_bitwise_xor_7.rs`
- `crates/circuit_prover/src/witness/components/verify_bitwise_xor_8.rs`
- `crates/circuit_prover/src/witness/components/verify_bitwise_xor_9.rs`
- `crates/circuits/Cargo.toml`
- `crates/circuits/src/circuit.rs`
- `crates/constraint-framework/src/component.rs`
- `crates/constraint-framework/src/logup.rs`
- `crates/constraint-framework/src/prover/component_prover.rs`
- `crates/constraint-framework/src/prover/logup.rs`
- `crates/constraint-framework/src/prover/simd_domain.rs`
- `crates/privacy_prove/Cargo.toml`
- `crates/privacy_prove/src/consts.rs`
- `crates/privacy_prove/src/lib.rs`
- `crates/prover/src/prover.rs`
- `crates/prover/src/witness/components/blake_round_sigma.rs`
- `crates/prover/src/witness/components/cube_252.rs`
- `crates/prover/src/witness/components/memory_id_to_big.rs`
- `crates/prover/src/witness/components/partial_ec_mul_window_bits_18.rs`
- `crates/prover/src/witness/components/partial_ec_mul_window_bits_9.rs`
- `crates/prover/src/witness/components/pedersen_aggregator_window_bits_18.rs`
- `crates/prover/src/witness/components/pedersen_aggregator_window_bits_9.rs`
- `crates/prover/src/witness/components/pedersen_points_table_window_bits_18.rs`
- `crates/prover/src/witness/components/pedersen_points_table_window_bits_9.rs`
- `crates/prover/src/witness/components/poseidon_aggregator.rs`
- `crates/prover/src/witness/components/poseidon_round_keys.rs`
- `crates/prover/src/witness/components/range_check_3_3_3_3_3.rs`
- `crates/prover/src/witness/components/range_check_3_6_6_3.rs`
- `crates/prover/src/witness/components/range_check_4_3.rs`
- `crates/prover/src/witness/components/range_check_4_4.rs`
- `crates/prover/src/witness/components/range_check_4_4_4_4.rs`
- `crates/prover/src/witness/components/range_check_7_2_5.rs`
- `crates/prover/src/witness/components/range_check_9_9.rs`
- `crates/prover/src/witness/components/verify_bitwise_xor_12.rs`
- `crates/prover/src/witness/components/verify_bitwise_xor_4.rs`
- `crates/prover/src/witness/components/verify_bitwise_xor_7.rs`
- `crates/prover/src/witness/components/verify_bitwise_xor_8.rs`
- `crates/prover/src/witness/components/verify_bitwise_xor_9.rs`
- `crates/prover/src/witness/components/verify_instruction.rs`
- `crates/prover/src/witness/prelude.rs`
- `crates/prover/src/witness/utils.rs`
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
