# Single-Server PIR Walkthrough

End-to-end specification of one PIR retrieval against a `poulpy-pir` `Database`,
broken into the steps requested:

1. CRS generation
2. DB encoding + first-dim preprocessing
3. Client query generation
4. Server-side query evaluation
5. Client-side decryption
6. Correctness check against the known DB content
7. Final noise measurement

It is meant as a build/test recipe: every step names the concrete API in
`poulpy-pir` / `poulpy-core` that implements it, and flags the spots where
glue code is still needed.

## 0. Dimensions and parameters

Notation used throughout:

| Symbol | Meaning |
|---|---|
| `n` | Ring degree of `R = Z[X]/(X^n + 1)`; power of two. |
| `base2k`, `k` | Limb width and total torus precision. Limb count `size = ⌈k / base2k⌉`. |
| `rank` | GLWE rank used for the first-dim query (paper's `d` mask components). |
| `cols = C = k_blocks · n` | First-dim query width; **must be a multiple of `n`**. |
| `k_blocks` | Number of first-dim column blocks (`C / n`). |
| `D` | Number of matrices on the interpolation axis (`⌈db_entries / (n · C)⌉`). |
| `t` | Padded matrix count for the IDFT, `D.next_power_of_two()`. |
| `p` | Plaintext modulus (odd, > 1) used by `ModPEncoder`. |
| `Δ` | BFV-style scaling factor, `Δ ≈ 2^k / p`. |

Database content is conceptually the 3-tensor

```text
DB[m][i][c]    with    0 ≤ m < D,   0 ≤ i < n,   0 ≤ c < C
```

The retrieval indexes a single triple `(m_target, i_target, c_target)`:

* `c_target` is selected by the encrypted first-dim query (`C`-wide, in
  `k_blocks` blocks of `n`).
* `m_target` is selected by an encrypted Horner evaluation against an
  RGSW-encrypted root of unity, after server-side polynomial interpolation
  across the `D` axis.
* `i_target` is just the coefficient index that the client reads out after
  decrypting the final GLWE.

## 1. Generate the CRS

The Common Reference Seed is a single uniformly random 32-byte string:

```rust
use rand::RngCore;
let mut crs = [0u8; 32];
rand::rngs::OsRng.fill_bytes(&mut crs);
```

Both sides expand it identically via

```rust
let seeds: Vec<[u8; 32]> = poulpy_pir::database::derive_block_seeds(crs, k_blocks);
```

`seeds[b]` is the mask seed for first-dim block `b`. The CRS is public; only
the GLWE secret key is private to the client.

## 2. Encode the DB and preprocess

### 2.1 Allocate

```rust
let mut db = Database::<BE>::new(&module, db_entries, base2k, cols);
```

Internally this allocates `D` per-matrix interpolation slots and
`D · k_blocks` `n × n` coefficient sub-matrices (one per column block), plus
the `t`-slot interpolation buffer.

### 2.2 Encode

Lay the entries out **row-major within each matrix**:

```text
shard[m · (n · C) + i · C + c] = DB[m][i][c]
```

Then

```rust
db.encode_shard(&module, 0, &shard);
```

`encode_shard` splits each `(row_out=i, col=c)` into
`(block = c/n, row_in = c%n)` and writes into the corresponding `n × n`
sub-matrix.

### 2.3 (Optional) Eager first-dim preprocessing

The first-dim mask depends only on the CRS, not on the message-carrying
bodies, so it can be amortized:

```rust
db.preprocess_query_mask(&module, crs, &query_layout, &mut scratch);
```

where `query_layout: GLWEInfos` matches the blocks the client will send.
This step runs `k_blocks · D` mask matmuls and aggregations and caches the
result keyed on the CRS. `query` and `query_interpolate` will skip the
preprocess on subsequent calls with the same CRS.

## 3. Generate a query

### 3.1 Index decomposition

```rust
let block_target  = c_target / n;   // which first-dim block carries the 1.
let coeff_within  = c_target % n;   // monomial degree inside that block.
```

### 3.2 Client keys (one-time)

```rust
let mut sk: GLWESecret<_> = module.glwe_secret_alloc(rank);
sk.fill_ternary_prob(0.5, &mut source_xs);
let mut sk_prepared = module.glwe_secret_prepared_alloc(rank);
module.glwe_secret_prepare(&mut sk_prepared, &sk);
```

`source_xs`, `source_xe`, `source_xa` are independent CSPRNG streams for the
secret, error, and (only for non-compressed encryption) mask.

### 3.3 First-dim blocks — `k_blocks` compressed GLWE ciphertexts

For each block `b`, the plaintext is

```text
P_b(X)  =  Δ · X^{coeff_within}    if b == block_target
        =  0                         otherwise
```

with `Δ` taken from `ModPEncoder::new(p, k)`.

```rust
let block_seeds = derive_block_seeds(crs, k_blocks);
let mut blocks: Vec<GLWECompressed<Vec<u8>>> = Vec::with_capacity(k_blocks);

for b in 0..k_blocks {
    let mut block = module.glwe_compressed_alloc_from_infos(&query_layout);
    *block.seed_mut() = block_seeds[b];

    // Build plaintext: Δ at coefficient `coeff_within` of column 0 for the
    // selected block; all-zero for the others.
    let mut pt = module.glwe_plaintext_alloc_from_infos(&query_layout);
    if b == block_target {
        let mut values = vec![0i64; n];
        values[coeff_within] = 1;
        encoder.encode_vec_i64(&mut pt.data, base2k, 0, &values);
    }

    // Standard seed-derived GLWE encryption.
    module.glwe_compressed_encrypt_sk(
        &mut block,
        &pt,
        &sk_prepared,
        &query_layout,
        &mut source_xe,
        &mut scratch.borrow(),
    );

    blocks.push(block);
}
```

The mask of block `b` is *not* freshly sampled — it is the deterministic
expansion of `block_seeds[b]`, exactly what the server will reconstruct in
`preprocess_query_mask`.

### 3.4 Second-dim root — one GGSW ciphertext

The Horner step needs `ω^{m_target}` encrypted as a GGSW, where the
primitive `t`-th root of unity is `ω = X^{2n/t}`. The signed monomial:

```text
exponent  e  =  (2n · m_target / t) mod 2n
root(X)   =  X^e        if e < n
          =  -X^{e-n}   if e ≥ n
```

```rust
let exponent = (2 * n * m_target) / t;        // 0 ≤ exponent < 2n
let mut root = module.scalar_znx_alloc(1);
if exponent < n {
    root.at_mut(0, 0)[exponent] = 1;
} else {
    root.at_mut(0, 0)[exponent - n] = -1;
}

let mut root_ct = module.ggsw_alloc_from_infos(&ggsw_layout);
module.ggsw_encrypt_sk(
    &mut root_ct, &root, &sk_prepared, &ggsw_layout,
    &mut source_xe, &mut source_xa, &mut scratch.borrow(),
);
let mut root_prepared = module.ggsw_prepared_alloc_from_infos(&root_ct);
module.ggsw_prepare(&mut root_prepared, &root_ct, &mut scratch.borrow());
```

The query the client ships to the server is `(crs, blocks, root_ct)`. The
GGSW dimensions (`dnum`, `dsize`, `k`) trade noise growth in the Horner
step against bandwidth; see the existing test
[`encrypted_horner_rgsw_selects_root_point`](../tests/cases/encrypted_horner_rgsw_selects_root_point.rs)
for working numbers at `n = 8`.

## 4. Evaluate the query (server)

### 4.1 First-dim + interpolation — already wired

```rust
let interp = db.query_interpolate(&module, crs, &blocks, &mut scratch);
// interp.len() == t (= D.next_power_of_two()).
```

For every padded matrix slot `j ∈ [0, t)`, `interp[j]` is an
[`LWEMatrix`] holding the unnormalised IDFT coefficient `t · c_j`.

The IDFT relation is

```text
y_m  =  Σ_{j=0}^{t-1}  c_j · ω^{jm}
```

so the per-matrix payload `y_m` is recovered by Horner evaluation of the
`c_j` at `ω^m`. Slots `j ≥ D` (padding) carry `0`.

### 4.2 Collapse each `interp[j]` to a single GLWE

`LWEMatrix` is *not* a GLWE: it packs `n` LWE ciphertexts under the
intermediate "secret view" structure of the paper. The
sequential-keyswitch / pack routines in
[`circuit/collapse_precompute.rs`](../src/circuit/collapse_precompute.rs)
and [`circuit/pack.rs`](../src/circuit/pack.rs)
(`SequentialCollapseMaskPrecompute*`, `InspirePackLWEToGLWE`, …) convert
that into a standard rank-`rank` GLWE under `sk`:

```text
glwe_j  =  collapse(interp[j])     for j ∈ [0, t)
```

The collapse uses preprocessed automorphism / key-switch material; it is
*independent of the message* (only of the CRS-derived masks), so the
mask-side of the collapse should be precomputed once per CRS just like the
first-dim mask preprocess.

> **Glue needed.** There is no single `Database::query_interpolate_to_glwe`
> helper today. The drivers are exercised by the
> `precomputed_sequential_collapse_*` and `dft_hot_body_collapse_decrypts`
> tests; a thin wrapper over them is what this step needs.

### 4.3 Apply the `1/t` normalisation

`interpolate` deliberately omits the `1/t` factor (see the rustdoc
on [`interpolate`](../src/circuit/interpolate.rs)). Two equivalent
ways to absorb it:

* **Encoder side:** divide `Δ` by `t` when encoding the query plaintext in
  3.3, i.e. encode the unit at coefficient `coeff_within` as `Δ/t` instead
  of `Δ`. Cheapest, recommended.
* **Server side:** right-shift `glwe_j` by `log2(t)` limb bits before
  Horner. Costs `t` extra rescales.

### 4.4 Encrypted Horner with the GGSW root

The shape mirrors
[`encrypted_horner_at_root`](../tests/cases/encrypted_horner_rgsw_selects_root_point.rs):

```rust
let mut acc     = module.glwe_alloc_from_infos(&glwe_layout);
let mut product = module.glwe_alloc_from_infos(&glwe_layout);

module.glwe_copy(&mut acc, &glwe_j[t - 1]);
for j in (0..t-1).rev() {
    module.glwe_external_product(&mut product, &acc, &root_prepared,
                                 root_prepared.size(), &mut scratch.borrow());
    module.glwe_add_into(&mut acc, &product, &glwe_j[j]);
}
```

After the loop, `acc` encrypts

```text
M(X) = Σ_{j=0}^{t-1} c_j · ω^{j · m_target}  =  y_{m_target}(X)
     = ( Σ_i  DB[m_target][i][c_target] · X^i )   (up to noise)
```

The server returns `acc` to the client.

## 5. Decrypt the response

```rust
let mut got_pt = module.glwe_plaintext_alloc_from_infos(&glwe_layout);
module.glwe_decrypt(&acc, &mut got_pt, &sk_prepared, &mut scratch.borrow());

let mut decoded = vec![0i64; n];
encoder.decode_vec_i64(&got_pt.data, base2k, 0, &mut decoded);

let recovered = decoded[i_target];
```

`recovered ∈ (-p/2, p/2]` is the requested DB entry reduced mod `p` and
centered.

## 6. Correctness check

Reduce the known DB value the same way and compare:

```rust
let expected = encoder.normalize(DB[m_target][i_target][c_target] as i64);
assert_eq!(recovered, expected, "PIR retrieved the wrong entry");
```

For a sanity sweep, repeat steps 3–6 for several `(m_target, i_target,
c_target)` triples (cycling through `m_target ∈ [0, D)`,
`i_target ∈ [0, n)`, `c_target ∈ [0, C)`). The first-dim mask preprocess
in 2.3 amortises over all of them as long as the CRS is held fixed.

## 7. Final noise

Build the noiseless plaintext that the response *should* carry and ask
`glwe_noise` for the residual:

```rust
let mut want_pt = module.glwe_plaintext_alloc_from_infos(&glwe_layout);
let mut want_values = vec![0i64; n];
for i in 0..n {
    want_values[i] = encoder.normalize(DB[m_target][i][c_target] as i64);
}
encoder.encode_vec_i64(&mut want_pt.data, base2k, 0, &want_values);

let noise = module.glwe_noise(&acc, &want_pt, &sk_prepared, &mut scratch.borrow());
let log2_sigma = noise.std().log2();
println!("final noise: log2(σ) = {log2_sigma:.2}");
```

A correct retrieval requires the rounded centered noise to stay inside
`(-Δ/2, Δ/2]`, i.e.

```text
log2(σ) + safety  <  log2(Δ) - 1
log2(Δ)            =  k - ⌈log2 p⌉   (BFV scaling)
```

The noise budget is consumed by, roughly in order of magnitude:

* the first-dim matmul (linear in `cols`),
* the IRCtx aggregation (linear in `rank`),
* the LWE-matrix → GLWE collapse (linear in the collapse depth),
* the encrypted Horner — `t-1` GLWE × GGSW external products, each adding
  roughly `log2(σ_extprod)` bits; this is the term that depends on
  `(dnum, dsize, k_ggsw)` in 3.4.

If the assertion in 6 fails, the noise number from this step tells you
which budget was exceeded.

---

## Whole-flow at a glance

```text
 client                                                    server
 ──────                                                    ──────
 sample crs ────────────────────────────────────────────►  store crs

 sk, sk_prepared
 derive_block_seeds(crs, k_blocks)
 for b in 0..k_blocks:                                    Database::new
     encrypt P_b(X) at seed[b]  ─── blocks ──────────►    encode_shard
 encrypt X^{2n·m/t} as GGSW ─── root ────────────►        preprocess_query_mask(crs)
                                                          query_interpolate(crs, blocks)
                                                          collapse each interp[j] → glwe_j
                                                          Horner(root, glwe_*)
 decrypt(acc) ◄──────────────────── acc ──────────────    return acc
 decoded[i_target]  ==  DB[m][i][c]  ?
 glwe_noise(acc, expected)   →   log2(σ)
```

## What is already in the crate vs. what to wire

Available today:

* `Database::new`, `encode_shard`, `preprocess_query_mask`, `query`,
  `query_interpolate` — steps 2 and 4.1.
* `derive_block_seeds` — steps 1 and 3.
* `circuit::interpolate*` — used inside `query_interpolate`.
* `circuit::SequentialCollapseMaskPrecompute*`, `InspirePackLWEToGLWE`,
  `circuit::sequential_collapse_*` — the building blocks for step 4.2.
* `ModPEncoder` — the BFV-style encoder/decoder for steps 3.3, 5, 6.
* `module.glwe_encrypt_sk`, `glwe_compressed_encrypt_sk`,
  `ggsw_encrypt_sk`, `ggsw_prepare`, `glwe_external_product`, `glwe_add_into`,
  `glwe_decrypt`, `glwe_noise` — steps 3–7, used in the existing
  `encrypted_horner_rgsw_selects_root_point` test.

Glue still to write:

* A client helper that turns `(crs, sk, m_target, i_target, c_target)` into
  `(Vec<GLWECompressed>, GGSWPrepared)` (step 3).
* A server helper that drives "interpolate → per-slot collapse → Horner"
  in one call (step 4.2 + 4.4). Today each piece exists in isolation; the
  pipeline is what the end-to-end test will assemble.
* A full integration test under `tests/cases/` that runs the seven steps
  end-to-end at small parameters (`n = 16`, `D = 3`, `cols = 2n`, `rank = 1`,
  small `p`) and asserts both correctness (step 6) and a noise bound
  (step 7).
