# Packing With Plain Automorphism Keys

## Purpose

This note records how the packing path consumes plain
`GLWEAutomorphismKeyCompressed` material signed with the client's raw LWE
secret `sk_base`, without requiring any hand-crafted shifted secret on the
client side.

## Current State

The packing front-end accepts compressed automorphism keys directly, and the
test [`bsgs_dft_hot_collapse_decrypts`](../src/packing/tests/bsgs_dft_hot_collapse.rs)
builds them in the most natural form:

- `key_g` is generated with `(p = g^{-1}, sk = sk_base)`.
- `key_h` is generated with `(p = -1, sk = sk_base)`.

Both keys are signed with the same secret the client already owns and the
same public packing mask seed. The server reads that seed through
`GGLWECompressedSeed`; there is no separate seed wrapper in the packing API.
The client never has to construct rotated secret-key shares like `sk_g` or
`sk_h` to satisfy the API. Decryption of the packed result uses `sk_base`
directly, matching the historical switching-key flow.

The collapse algorithm itself
([`precompute_collapse_half`](../src/packing/collapse_precompute.rs),
[`sequential_collapse_bsgs_dft_build`](../src/packing/collapse_precompute.rs),
[`bsgs_pack`](../src/packing/bsgs_pack.rs))
is unchanged from the switching-key path. The body logic lives in
[`pack_keys_precompute`](../src/packing/key_precompute.rs), while the
seed-derived mask logic is folded into
[`pack_precompute`](../src/packing/default.rs).

## The Convention Mismatch

The packing collapse is written around a key-switch direction `sk_g -> sk_base`
(and `sk_h -> sk_base` for the final step). With switching keys this is a
direct property of the encryption: pass `sk_in = sk_g`, `sk_out = sk_base`
and the key has the right shape.

`GLWEAutomorphismKeyCompressed` does not expose that direction directly.
Given `(p, sk)` the API encrypts:

```text
sk_in  = sk
sk_out = automorphism(inv(p), sk)
```

If the client signs `key_g` with `(p = g^{-1}, sk = sk_base)` they get:

```text
sk_in  = sk_base
sk_out = automorphism(g, sk_base) = sk_{g^1}
```

That is a perfectly valid key-switch, but it goes `sk_base -> sk_{g^1}` —
*from* `sk_base` instead of *to* it, and into `sk_{g^1}` rather than the
`sk_g = sk_{g^{-1}}` the collapse expects. So a natural key cannot be
plugged directly into the historical collapse.

## How The Server Realigns The Key

Rotating the whole key (mask plus body) by a Galois automorphism `tau_phi`
turns a key that encrypts `sk_in` under `sk_out` into one that encrypts
`tau_phi(sk_in)` under `tau_phi(sk_out)`. Picking `phi = g^{-1}` for `key_g`:

- `tau_{g^{-1}}(sk_base)  = sk_{g^{-1}} = sk_g`
- `tau_{g^{-1}}(sk_{g^1}) = sk_{g^0}    = sk_base`

So the rotated `key_g` switches `sk_g -> sk_base`. The same trick with
`phi = -1` realigns `key_h` to `sk_h -> sk_base`.

The body rotation is implemented in
[`pack_keys_precompute`](../src/packing/key_precompute.rs), and the mask
rotation is implemented from the public packing seed during
[`pack_precompute`](../src/packing/default.rs):

```rust
let key_g_rotation = module.galois_element_inv(module.galois_element(1));
let key_h_rotation = -1i64;

let key_g_body  = split_body_key_from_compressed(module, key_g, key_g_rotation);
let key_h_body  = prepare_body_key_from_compressed(module, key_h, key_h_rotation, scratch);

let key_g_mask = prepare_mask_key_from_seed(module, key_mask_source, key_g_rotation, scratch);
let key_h_mask = prepare_mask_key_from_seed(module, key_mask_source, key_h_rotation, scratch);
```

`split_output_key_plain` and `split_mask_key_from_seed` take a `rotation`
argument: when it is `1` they fall back to the original copy path, otherwise
they thread the body column / expanded mask seed through
`vec_znx_automorphism_backend(rotation, ...)`. The server checks that both
compressed keys expose the same seed vector, then derives the fixed masks
from a `GGLWECompressedSeed` source instead of reading client-specific key
bodies.
The rotation cost is one automorphism per `(dnum, rank_in)` cell of each
body/mask projection, paid once at precompute time.

From there every downstream operation
(`split_output_key_plain`, `baby_keys_from_split`, the offline mask
collapse in `precompute_sequential_keyswitch_collapse_aggregate_mask`, the
online BSGS DFT-hot loop in `pack_with_precomputations`, and the final
`key_h` step) sees a key that already has the historical direction, so no
algorithm logic had to change.

## Why A Pure-Algorithm Rewrite Did Not Land

An earlier attempt tried to reverse the half-collapse recurrence so the
natural key direction `sk_base -> sk_g` matched the schedule directly:

- walk `target_col` low to high with `source_col = target_col - 1`,
- redefine `tau_g_j = galois_element(source_col)` instead of
  `galois_element(target_col)`,
- flip the BSGS baby-step factor from `g^{+baby_idx}` to `g^{-baby_idx}` on
  both the stored DFT mask cache and the prepared baby keys,
- recompute the giant-step plan from the first `source_col` of the group
  instead of the maximum `target_col`,
- add an explicit post-key-switch automorphism plan for the final `key_h`
  step (matching the per-step pattern with `source_col = half - 1`).

That variant decrypted correctly under
`sk_end = sk_{g^{-(half-1)}}`: each step still preserved the message and
the per-step view shift was `-1` per column instead of `+1`, so the final
aggregate column landed at `sk_end` rather than `sk_base`. With the
existing test contract (`decoded == data` decrypting under `sk_base`) the
rewrite cannot pass directly, because there is no message-preserving
automorphism that maps `sk_end` to `sk_base` — any automorphism that
realigns the view also permutes the polynomial slots.

Rotating the keys at precompute time is the architecturally cheaper option:
one extra automorphism per key projection during precompute keeps the
collapse, the client API, and the destination key shape simple.

## Verification Snapshot

Current passing state:

```text
cargo test --manifest-path Cargo.toml bsgs_dft_hot_collapse_decrypts
cargo check --manifest-path Cargo.toml --tests
```

All pass. The only warnings observed are unrelated unused helper warnings
in `interpolation/tests/interpolate_columns_independent.rs`.
