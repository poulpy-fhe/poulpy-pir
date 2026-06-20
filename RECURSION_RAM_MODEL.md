# InsPIRe² — RAM Model

Resident memory of the recursive (InsPIRe²) pipeline as a function of the database
layout and parameterization.

## Symbols

| Symbol | Meaning |
|---|---|
| `n` | ring degree (e.g. 2048) |
| `R`, `C` | database `DB_ROWS`, `DB_COLS` (`R·C` = DB size in coefficients) |
| `size` | `⌈k/base2k⌉` limbs (e.g. 3) |
| `γ0`, `γ1` | recursion packing widths |
| `T` | thread-pool width |
| `S_pack` | per-worker pack scratch, DB-size-independent (~145 MiB) |

Storage widths: i16 = 2 B, f64 = 8 B, VecZnx coefficient = 8 B/limb.

## Resident terms

| Term | What it is | Bytes | Scales as |
|---|---|---|---|
| **Plaintext DB** | the canonical i16 store | `2·R·C` | DB size (`R·C`) |
| **`q0_masks`** | fixed CRS mask A₀ (f64) | `8·n·C` | `C` |
| **`l1_precompute`** | level-1 partial-γ0 precomputes | `16·n·size·R` | `R` |
| **Scratch pool** | `T` persistent per-worker pack arenas | `T·S_pack` | `T` |
| Fixed remainder | second-level precomputes/masks, transients | `O(n, γ)` | DB-independent |

## Estimate

```
RAM ≈ 2·R·C  +  8·n·C  +  16·n·size·R  +  T·S_pack  +  O(n, γ)
       └ DB ┘   └ A₀  ┘   └ level-1   ┘   └ threads┘
```

Three terms scale with the database — the plaintext DB (`R·C`), `q0_masks` (`C`), and
`l1_precompute` (`R`); everything else is fixed in DB size. A given DB size fixes `R·C`
but not the split: the `R` and `C` terms move in opposite directions, so the layout
choice shifts the non-DB footprint (a C-heavy split grows `q0_masks` and shrinks
`l1_precompute`, and vice versa).

### Worked example — 32 GiB DB, `R=2¹⁵`, `C=2¹⁹`, `n=2048`, `size=3`, `γ0=32`

| Term | Bytes |
|---|---|
| `2·R·C` (DB) | 32.0 GiB |
| `8·n·C` (`q0_masks`) | 8.0 GiB |
| `16·n·size·R` (`l1_precompute`) | 3.0 GiB |
| `T·S_pack` | 0.14 GiB (T=1) … 3.4 GiB (T=24) |
| remainder | ~0.8 GiB |
| **Total** | **~44 GiB (T=1) … ~48 GiB (T=24)** |
