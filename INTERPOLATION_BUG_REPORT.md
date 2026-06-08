# InsPIRe (interpolation) MISMATCH — root-cause report

**Status:** root cause localized to the **plaintext interpolation**, specifically the
**axis (row vs column) on which the local `CoeffMatrix` is interpolated**. The entire
homomorphic pipeline is proven bit-exact correct. InsPIRe² (recursion) is unaffected.

**Regression introduced by:** `cd76cdc` *"Add local coeffmatrix"* (the removal of the
HAL-level homomorphic coefficient-matrix product, replaced by the local `i16`
`CoeffMatrix` + `f64` GEMM).

---

## Symptom

`cargo run --release --example pir -- interpolation` returns a **clean** ciphertext
that decrypts to the **wrong** payload:

```
NOISE log2(max) : -1.0     (this is NOT noise — see below)
got  != want
retrieved payload : MISMATCH
```

The reported `-1.0` is `client.noise(selected, true_payload)`, i.e. the distance
between a *clean* encryption and the *true* payload. It conflates "wrong value" with
"noise". The actual encryption noise is healthy everywhere.

InsPIRe² (recursion), which shares the `U·(A,b)` first step, works (`log2(std) ≈ -28`).

---

## Bisection (instrumentation added to the example/client/server)

All checks below were added as debug hooks (`Client::debug_*`,
`Server::debug_interpolation_first_step`) and run on the failing query
(`item_index = 5_000_000` → matrix 1, block_col 9, col_in_block 832, row_offset 544).

| stage | what was checked | result |
|---|---|---|
| **first step `U·(A,b)`** | decrypt `(U·A, U·b)` under `sk`, compare values to the plaintext interpolated-`U` column | **0 error / 0 mismatches**, all panels |
| **packed panels** | decrypt `packed[j]` under `sk_pack`, compare values to the same `U` column | **0 error / 0 mismatches**, all panels |
| **packed self-noise** | residual to the mod-`p` lattice | ≈ `-22 … -24` (clean) |
| **`selected` self-noise** | residual to the mod-`p` lattice | ≈ `-20.8` (clean) |
| **ciphertext vs plaintext Horner** | `decrypt(selected)` vs a plaintext Horner of the `U` columns at `X^{matrix·2n/t}` | **0 mismatches / 2048** |

### What this proves

1. The shared **`U·(A,b)` first step is exactly correct** (GEMM, query one-hot,
   interpolated-`U` values). This *clears* the matmul-removal GEMM.
2. **Packing is value-correct** — each packed panel is a bit-exact encryption of the
   correct `U` column.
3. The **GGSW×GLWE Horner `reduce` is exactly correct** — `decrypt(selected)` equals,
   coefficient-for-coefficient, the *plaintext* Horner of the same `U` columns.

Therefore **no homomorphic operation is at fault.** `selected` is a clean encryption
of exactly what the plaintext interpolation prescribes — and that plaintext value is
wrong.

> Conclusion: the bug is entirely in the **plaintext interpolation**
> (`Interpolation::prepare` / `interpolate_into` in
> `src/interpolation/strategy.rs`), which builds the `U` panels from the raw DB.

---

## Root cause: the interpolation is performed on the wrong axis of the `CoeffMatrix`

The local `CoeffMatrix` is row-major `rows[out][in]`, addressed as `row(R)[C]`.

Three places fix what `R` (row) and `C` (column) mean, and they **disagree**:

1. **Storage / readout** (`src/database/storage.rs`) — a record runs *down rows* at a
   fixed column:
   ```rust
   sub.row_mut(row_out_base + row_offset + k)[col_in_block] = digit;   // (R = coeff, C = column)
   ```
   So the payload digits live along the **row (`R`) axis**.

2. **Pack + Horner** — the GEMM selects a column and the packed GLWE encrypts
   `U_j[·][col_in_block]` indexed by **row `R`** (that becomes the packed polynomial's
   coefficient axis). The Horner `selected = Σ_j packed[j]·X^{j·matrix·(2n/t)}` rotates
   the **`R` axis**. (Confirmed: `decrypt(selected)` == plaintext Horner rotating `R`.)

3. **Interpolation** (`prepare` / `interpolate_into`) — loads a *row* into a working
   column and interpolates over it:
   ```rust
   let src = &db.matrices()[m * kb + bc];
   for col in 0..n {
       w.at_mut(col, 0) /* = working[m] column `col` */ .copy_from(src.row(col));
   }
   for col in 0..n { module.monomial_interpolate(&mut working, col, ..); }
   ```
   `monomial_interpolate` treats each `working[m].at(col,0) = src.row(col)` as a
   **polynomial over the column (`C`) axis**; its IDFT twiddles are monomial rotations
   of the **`C` axis**. So it reconstructs `v_m` by rotating **`C`**, not `R`.

**The interpolation reconstructs along `C`; pack+Horner reconstruct along `R`.** These
must be the same axis. The IDFT (`monomial_interpolate`) is correct — it is simply fed
(and writes back) the matrix **transposed** relative to the axis the packing later
extracts and the Horner later rotates.

### Why the switch to the local `CoeffMatrix` triggered it

Before `cd76cdc`, the operand was a poulpy base2k `CoeffMatrix` whose `data().at(col,0)`
orientation lined these axes up (the `prepare` comment still reads *"no transpose,
because the stored orientation is already the matmul U orientation"*). The diff replaced
`src.data().at(col,0)` / `dst.data_mut().at_mut(col,0)` with `src.row(col)` /
`dst.row_mut(col)` on the new row-major `i16` `CoeffMatrix`. The storage write was
changed the same way, so storage↔interpolation stay self-consistent — **but both now
disagree with the pack/Horner axis**, which was not changed. The lost orientation is
effectively a missing transpose between "interpolation axis" and "packed axis".

---

## Fix direction

Make the interpolation operate on the **same axis that the pack extracts and the Horner
rotates** (the `R`/row axis that carries the payload). Concretely, one of:

- transpose the load/store in `prepare` / `interpolate_into` so `monomial_interpolate`
  runs over the column (`C`) that is later packed — i.e. interpolate `src` columns, not
  `src` rows; **or**
- equivalently, restore the transpose the old base2k `CoeffMatrix` orientation provided.

Whichever is chosen, the invariant to enforce is:

> the axis `monomial_interpolate` rotates (its IDFT twiddle / monomial axis) **must
> equal** the axis the packed GLWE uses as its polynomial coefficient axis (the
> `R`/row axis along which `storage` lays out a record).

## Verification

1. **Existing instrumentation:** re-run `cargo run --release --example pir -- interpolation`.
   Expect `retrieved payload : OK` and `NOISE log2(std) ≈ -28` (like recursion). The
   debug lines `first_step[*] 0/0`, `packed[*] 0/0`, and
   `plaintext-horner vs decrypt(selected) 0/2048` should stay green (they already are);
   the new green is `got == want`.
2. **Point/axis probe (optional):** for each interpolation point `m`, plaintext-Horner
   the produced `U` columns at `X^{m·2n/t}` and check which raw DB matrix it
   reconstructs. After the fix, point `m` reconstructs raw matrix `m` (diagonal).
3. **Add a unit test** that round-trips `interpolate` → plaintext-Horner for the
   `nb_matrices == interpolation_t` natural case and asserts it recovers the raw DB
   column. The current `interpolate_*` unit tests pass despite the bug, so they do not
   cover the row-vs-column axis distinction that the pack/Horner imposes.

---

## Appendix: debug hooks added (remove before merge)

- `examples/pir.rs` — first-step / packed-value / plaintext-Horner / point-probe prints.
- `src/client/core.rs` — `debug_packed_noise`, `debug_decrypt_first_step`,
  `debug_decrypt_packed_values`, `glwe_self_noise`.
- `src/server/interpolation/online.rs` — `debug_interpolation_first_step`.
- `src/interpolation/strategy.rs` — `InterpolationResponse` carries the per-panel
  `packed` GLWEs for inspection.
- `poulpy-core/src/default/noise/glwe.rs` — removed a stray non-compiling
  `println!("res_backend …")` that blocked the build.
