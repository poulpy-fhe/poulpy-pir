# Sequential Key-Switch Mask Precompute Plan

## Current Progress

Completed so far:

- Step 1 is implemented and validated.
  - Added `GGLWEPreparedVmpPMatRef` in `poulpy-core`.
  - Forwarded immutable VMP access through prepared switching-key and
    automorphism-key wrappers.
  - Added `prepared_vmp_pmat_is_immutable_and_dimensioned`.
- Step 2 is implemented and validated.
  - Added PIR-local `fixed_mask_1x1_vmp_body_addend`.
  - The helper mirrors the core key-switch VMP pipeline while specializing the
    product to `1 x 1` with `dsize = 1`.
  - Added `fixed_mask_1x1_vmp_addend_matches_reference`.
- Step 3 is implemented as a storage/scratch skeleton.
  - Added `SequentialCollapseMaskPrecompute`.
  - Added `sequential_collapse_mask_precompute_alloc`.
  - Added `fixed_mask_1x1_vmp_body_addend_tmp_bytes`.
- Step 4 is implemented and validated against the split baseline.
  - Added `fixed_mask_1x1_vmp_keyswitch_body`, which adds the input body before
    normalization to match the generic key-switch body rounding point.
  - Added `sequential_keyswitch_collapse_aggregate_mask_split`, which takes the
    body and mask prepared VMPs separately and is bit-identical to the full
    `1 x 2` collapse.
  - Added `precompute_sequential_keyswitch_collapse_aggregate_mask`.
  - Added `precompute_sequential_keyswitch_collapse_aggregate_mask_tmp_bytes`.
  - Added `sequential_keyswitch_collapse_split_matches_full`.
  - Added `precompute_mask_state_matches_split_baseline`.
- Step 5 is implemented and validated against the split baseline.
  - `SequentialCollapseMaskPrecompute` now stores per-step fixed mask inputs
    for the online body-side VMPs.
  - Added `sequential_keyswitch_collapse_aggregate_mask_precomputed`.
  - Added `sequential_keyswitch_collapse_aggregate_mask_precomputed_tmp_bytes`.
  - Added `precomputed_sequential_collapse_matches_split_baseline`.
- Step 6 is implemented and validated.
  - Added `precomputed_sequential_collapse_decrypts`.
  - The test runs the DB-style path end to end: encrypt, expand, aggregate,
    precompute fixed mask collapse, online precomputed collapse, decrypt.
- Step 7 is implemented and validated.
  - Added the `strict_collapse` bench target.
  - The benchmark reports baseline online collapse, offline fixed-mask
    precompute, online precomputed collapse, and core GLWE packing separately.
  - The bench performs a bit-equality sanity check before timing.
- Step 8 is implemented as an explicit non-bit-identical variant.
  - Added `fixed_mask_1x1_vmp_dft_product`.
  - Added `precompute_sequential_keyswitch_collapse_aggregate_mask_dft_accumulated`.
  - Added `dft_accumulated_precompute_decrypts`.
  - The benchmark now reports the DFT-accumulated precompute separately.

Validation already run:

```bash
cargo check -p poulpy-core --lib
cargo check --manifest-path Cargo.toml
cargo test --manifest-path Cargo.toml prepared_vmp_pmat_is_immutable_and_dimensioned
cargo test --manifest-path Cargo.toml fixed_mask_1x1_vmp_addend_matches_reference
cargo test --manifest-path Cargo.toml sequential_keyswitch_collapse_split_matches_full
cargo test --manifest-path Cargo.toml precompute_mask_state_matches_split_baseline
cargo test --manifest-path Cargo.toml precomputed_sequential_collapse_matches_split_baseline
cargo test --manifest-path Cargo.toml precomputed_sequential_collapse_decrypts
cargo check --manifest-path Cargo.toml --benches
cargo bench --manifest-path Cargo.toml --bench strict_collapse
cargo test --manifest-path Cargo.toml dft_accumulated_precompute_decrypts
cargo test --manifest-path Cargo.toml
```

Current full `poulpy-pir` result: `10 passed`.

Latest benchmark sample:

```text
strict collapse benchmark
  n: 1024
  iterations: 5
  baseline online full 1x2 collapse:          45.085 ms
  offline fixed-mask precompute:              12.362 ms
  offline DFT-accumulated precompute:         12.639 ms
  online precomputed body-only collapse:      11.043 ms
  core GLWE pack of n GLWEs (log n keys):     50.395 ms
```

Next gate: benchmark and profiling follow-up if we want to chase a faster
DFT-accumulated schedule beyond the current terminal accumulation.

## Context

The current sequential collapse uses two prepared key-switching keys:

```text
Kg: switches a tau_g share back to the base share
Kh: switches the final tau_h share back to the result secret
```

Each baseline collapse step acts like a rank-1 GLWE key-switch. For a term:

```text
(b_j, a_j)
```

the baseline computes a `1 x 2` VMP:

```text
a_j x K = (a_j x K_body, a_j x K_mask)
```

then updates:

```text
body <- normalize(body_part + b_j)
mask <- mask_part added/copied into the target mask column
```

The collapse order is:

```text
first half:  reverse collapse with Kg
second half: reverse collapse with Kg
final:       collapse tau_h share with Kh
```

For degree `n`, the strict baseline is the same sequence of `n` logical
collapse products using `1 x 2` VMPs.

## Target

In the DB query path, both of these are fixed:

```text
GLWE mask column a_j
key-switch mask column K_mask
```

Therefore `a_j x K_mask` can be precomputed and applied to the fixed mask state
ahead of time. Online collapse only needs:

```text
a_j x K_body
```

which is a `1 x 1` VMP performed online. The key body is query-dependent, so no
body-side addend is stored in the DB precompute.

Strict mode must be bit-identical to the baseline:

```text
baseline:
  n collapse steps using 1 x 2 VMPs

optimized:
  precompute all fixed GLWE-mask x key-mask updates
  n online collapse steps using query-dependent 1 x 1 VMP body products

requirement:
  final GLWE bytes are identical
```

This is stronger than decrypting to the same message. Strict-mode tests must
compare final GLWE body and mask buffers directly.

## Invariants

- The DB owns the fixed mask seed and derives the aggregate mask internally.
- `GLWEAutomorphismKeyCompressed` is not decompressed or prepared for this
  precompute.
- `VmpPMat` is immutable. Only read access is exposed.
- Prepared matrices/keys are never automorphed.
- Operand automorphisms stay on GLWE body/mask values, matching the current
  collapse.
- Strict mode preserves the baseline operation order and normalization points.
- The PIR implementation mirrors the relevant `poulpy-core`
  key-switch/automorphism pipeline:

```text
operand automorphism
VecZnx -> VecZnxDft preparation
VMP application
IDFT
VecZnxBig / normalization path
body or mask copy/add
```

## Step 1: Expose Immutable VMP Access

Add a read-only helper in `poulpy-core` to access the immutable `VmpPMat` needed
by the PIR-side `1 x 1` VMP.

Candidate API:

```rust
pub trait GGLWEPreparedVmpPMatRef<B: Backend> {
    fn vmp_pmat_backend_ref(&self) -> VmpPMatBackendRef<'_, B>;
}
```

Implement it only for the prepared layouts needed by the collapse path:

```text
GGLWEPrepared<_, B>
GLWESwitchingKeyPrepared<_, B> if used as the carrier for the prepared matrix
GLWEAutomorphismKeyPrepared<_, B> only if the prepared 1x1 matrix is stored there
```

Do not add a mutable accessor. Do not add a high-level core precompute API.

Checklist:

- [x] Immutable `VmpPMat` backend-ref helper exists in `poulpy-core`.
- [x] No mutable `VmpPMat` accessor is exposed.
- [x] No compressed automorphism-key preparation path is added.
- [x] `poulpy-core` compiles.
- [x] `poulpy-pir` still compiles.

Required validation before Step 2:

```bash
cargo check -p poulpy-core --lib
cargo check --manifest-path Cargo.toml
```

Required test:

```text
prepared_vmp_pmat_is_immutable_and_dimensioned
```

The test must access the immutable prepared `VmpPMat` and check the expected
`1 x 1` dimensions used by the collapse precompute.

## Step 2: Add PIR 1x1 VMP Helper

Implement the fixed-mask body-addend product locally in `poulpy-pir`.

Suggested file:

```text
src/circuit/collapse_precompute.rs
```

The helper computes:

```text
fixed mask column
  -> VecZnxDft
  -> vmp_apply_dft_to_dft(mask_dft, mask_resampling_vmp, ...)
  -> IDFT
  -> VecZnx body contribution
```

It uses a `1 x 1` prepared matrix shape:

```text
rows = dnum
cols_in = 1
cols_out = 1
dsize = 1
```

For now it assumes `dsize = 1`, so the helper only needs the single-product
path:

```text
vmp_apply_dft_to_dft(product, mask_dft, mask_resampling_vmp, 0)
```

Checklist:

- [x] Helper mirrors the relevant core key-switch VMP pipeline.
- [x] Helper performs only `1 x 1` VMP, not full `1 x 2`.
- [x] `dsize == 1` is asserted and handled.
- [x] Output is a body addend, stored with the same rounding schedule needed by
      strict mode.
- [x] No online mask-output product is introduced.

Required validation before Step 3:

```bash
cargo test --manifest-path Cargo.toml fixed_mask_1x1_vmp_addend_matches_reference
```

Required test:

```text
fixed_mask_1x1_vmp_addend_matches_reference
```

The test compares the PIR `1 x 1` VMP addend with a direct/reference
mask-resampling contribution.

## Step 3: Add Precompute Layout

Add a PIR-local layout for the collapse precompute.

It stores:

```text
SequentialCollapseMaskPrecompute
  - per-step fixed mask inputs for online body-side VMPs
  - final result mask
  - parameter metadata: n, base2k, key_base2k, size, rank
```

Prefer:

```text
VecZnx for the per-step fixed mask inputs
VecZnx for the final precomputed mask
```

Checklist:

- [x] Layout carries enough metadata to assert parameter compatibility.
- [x] Layout stores the fixed per-step mask inputs needed by online body VMPs.
- [x] Layout stores the final fixed mask value copied into the online result.
- [x] Layout does not store unused full `1 x 2` products.
- [x] Layout does not store query-dependent body addends.
- [x] Allocation helpers exist.
- [x] Scratch byte helper skeletons exist.

Required validation before Step 4:

```bash
cargo check --manifest-path Cargo.toml
```

## Step 4: Implement Strict Precompute

Add:

```rust
precompute_sequential_keyswitch_collapse_aggregate_mask(...)
```

Inputs:

```text
module
fixed aggregate mask
prepared 1x1 mask VMP data for Kg
prepared 1x1 mask VMP data for Kh/final
scratch
```

Processing:

1. Split the aggregate mask into the same two halves as the current collapse.
2. For each `Kg` step, apply the same operand automorphism to the fixed mask
   column as the online collapse would apply.
3. Store the automorphed fixed mask input for the matching online body-side
   `1 x 1` VMP.
4. Run the PIR-local `1 x 1` VMP helper with the fixed key mask.
5. Apply the precomputed fixed `GLWE mask x key-mask` update to the fixed mask
   state so the next step sees the same mask as the baseline.
6. Repeat for both halves.
7. Run the final `Kh`/tau-h mask-side `1 x 1` VMP.
8. Store the final result mask.

Checklist:

- [x] Loop order matches current collapse exactly.
- [x] Operand automorphism indices match current collapse exactly.
- [x] Fixed mask state after each step matches the split baseline mask state.
- [x] Per-step fixed mask inputs for online body VMPs are stored.
- [x] No query-dependent body addends are stored in the DB precompute.
- [x] Final precomputed mask matches the split baseline final mask.

Required validation before Step 5:

```bash
cargo test --manifest-path Cargo.toml sequential_keyswitch_collapse_split_matches_full
cargo test --manifest-path Cargo.toml precompute_mask_state_matches_split_baseline
```

Required test:

```text
sequential_keyswitch_collapse_split_matches_full
precompute_mask_state_matches_split_baseline
```

The split test first proves the separated body/mask `1 x 1` VMP collapse is
bit-identical to the full `1 x 2` key-switch collapse. The precompute test then
runs the split collapse and the precompute mask updates with the same aggregate
mask and asserts bit-equality of the final fixed mask state.

## Step 5: Implement Online Precomputed Collapse

Add:

```rust
sequential_keyswitch_collapse_aggregate_mask_precomputed(...)
```

Inputs:

```text
module
result GLWE
dynamic body
precomputed collapse data
scratch
```

Processing:

1. Apply the same body automorphisms as the current collapse.
2. At each step, read the stored fixed mask input from the precompute.
3. Compute the query-dependent body-side `1 x 1` VMP online.
4. Normalize at the same points as the strict baseline.
5. Copy the precomputed final mask state.
6. Write the final body and final mask to `result`.

The precompute stores mask inputs, not body addends. The body side remains
query-dependent because it uses the online key body.

No online mask-side `vmp_apply_dft_to_dft` should remain in this path.

Checklist:

- [x] Body automorphism schedule matches baseline.
- [x] Query-dependent body-side VMP schedule matches baseline.
- [x] Normalization points match baseline.
- [x] Final mask is copied from precompute.
- [x] Online routine performs no mask-side VMP.

Required validation before Step 6:

```bash
cargo test --manifest-path Cargo.toml precomputed_sequential_collapse_matches_split_baseline
```

Required test:

```text
precomputed_sequential_collapse_matches_split_baseline
```

The test runs the split `1 x 2` collapse and precomputed online collapse with
the same body/mask and asserts final GLWE body and mask are bit-identical.

## Step 6: End-To-End Decryption Test

Add:

```text
precomputed_sequential_collapse_decrypts
```

The test should:

1. Encrypt a GLWE.
2. Expand it to an LWE matrix.
3. Aggregate the mask.
4. Precompute the collapse data.
5. Run online precomputed collapse.
6. Decrypt and compare to plaintext.

Checklist:

- [x] Test core is backend-generic.
- [x] Only `#[test]` wrappers pick concrete backends.
- [x] Test lives under `tests/cases/` and is called from `tests/mod.rs`.
- [x] Decryption matches plaintext.

Required validation before Step 7:

```bash
cargo test --manifest-path Cargo.toml precomputed_sequential_collapse_decrypts
cargo test --manifest-path Cargo.toml
```

## Step 7: Benchmark Strict Mode

Benchmark:

```text
baseline current collapse
strict precompute construction
strict online precomputed collapse
core GLWE packing of n GLWEs with log(n) automorphism keys
```

Checklist:

- [x] Benchmark separates one-time precompute from online query cost.
- [x] Benchmark reports no online mask-side VMP in strict online path.
- [x] Benchmark reports core GLWE packing of `n` GLWEs for comparison.
- [x] Full tests still pass.

Required validation before optional Step 8:

```bash
cargo check --manifest-path Cargo.toml --benches
cargo bench --manifest-path Cargo.toml --bench strict_collapse
cargo test --manifest-path Cargo.toml
```

## Step 8: Optional DFT-Domain Accumulation

After strict mode passes, add an explicit optimized variant that keeps compatible
fixed additions in the DFT domain during precompute and materializes them with
fewer IDFTs.

This may change final bits because normalization is delayed. It should not use
bit-equality as its primary correctness oracle.

Validation mode:

```text
strict mode:
  baseline normalization schedule
  final GLWE bit-equality required

DFT-accumulation mode:
  delayed compatible normalizations
  decryption equality and noise sanity required
  final GLWE bit-equality not required
```

Checklist:

- [x] DFT accumulation is a separate explicit variant.
- [x] Tests do not require bit-equality for this variant.
- [x] Decryption equality is checked for this variant.
- [x] Strict-mode tests still pass unchanged.

Required tests:

```text
dft_accumulated_precompute_decrypts
```

Required validation:

```bash
cargo test --manifest-path Cargo.toml dft_accumulated_precompute_decrypts
cargo test --manifest-path Cargo.toml
```
