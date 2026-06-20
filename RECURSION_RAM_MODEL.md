# Recursive PIR (InsPIRe²) — RAM Model — 32 GiB DB

RAM model for the recursive (InsPIRe²) collapse pipeline, with exact scaling formulas.
Instantiated below for a **32 GiB plaintext database** (`2·R·C = 2³⁵`).

## Symbols

| Symbol | Meaning | Value @ 32 GiB DB |
|---|---|---|
| `n` | ring degree | 2048 |
| `R` | `DB_ROWS` | 32768 (`2¹⁵`) |
| `C` | `DB_COLS` | 524288 (`2¹⁹`) |
| `size` | `⌈k/base2k⌉` limbs | 3 |
| `γ0`, `γ1` | recursion gammas | 32, 1024 |
| `t` | level-1 batches = `R/γ0` | 1024 |
| `br`, `bc` | block rows `R/n`, block cols `C/n` | 16, 256 |
| `T` | thread count (pool width) | 1 … 24 |
| `S_pack` | per-worker pack scratch (DB-size-independent) | ~145 MiB |

Storage widths: i16 = 2 B, f64 = 8 B, VecZnx coefficient = 8 B/limb.

> **Config choice.** A 32 GiB DB fixes `R·C = 2³⁴` but not the split. The **C-scaled**
> split `R=2¹⁵, C=2¹⁹` is used here because it reproduces the measured 48 GB RSS. An
> R-scaled split (e.g. `R=2¹⁷, C=2¹⁷`) holds the same DB but grows `l1_precompute`
> (∝R) to ~12 GiB and shrinks `q0_masks` (∝C) to ~2 GiB → ~51 GiB RSS instead.

## Persistent state (resident after `offline()`, during serving)

| Aspect | What it is | Bytes (formula) | Scales as | @ R=2¹⁵, C=2¹⁹ |
|---|---|---|---|---|
| **Plaintext DB** (i16) | the one canonical store (`CoeffMatrix`) | `2·R·C` = `2·n²·br·bc` | **R·C** (DB size) | 32.0 GiB |
| ~~`db_prep`~~ | **eliminated** — zero-copy views | `0` (was `2·R·C`) | — | 0 (was 32.0) |
| **`q0_masks`** (f64) | fixed CRS mask A₀ (`bc` panels of n×n) | `8·n·C` (= `bc · n²·8`) | **C** (⊥R) | ~8.0 GiB |
| **`l1_precompute`** | `t = R/γ0` partial-γ0 BSGS precomputes | `≈ 16·n·size·R` (= `t · 2·γ0·n·size·8`) | **R** (γ0 cancels; ⊥C) | ~3.0 GiB |
| `resp1_precompute` | partial-γ1 BSGS precomputes (mask digits) | `≈ 16·n·size·γ1·n_b` | γ1 (DB-⊥) | ~0.4 GiB |
| `resp1_prep` (i16) | digit D₁ panels | `≈ 2·n·γ0·τ·…` (small) | digit DB (DB-⊥) | ~16 MiB |
| `q1_masks` (f64) | fixed CRS mask A₁ | `≈ n²·8 · (small)` | n² (DB-⊥) | ~32 MiB |

## Scratch + transients

| Aspect | What it is | Bytes (formula) | Scales as | @ R=2¹⁵, C=2¹⁹ |
|---|---|---|---|---|
| **Scratch pool** (M2′) | `T` persistent per-worker pack arenas | `T · S_pack` | **T** (threads) | 0.14 GiB (T=1) … 3.4 GiB (T=24) |
| Offline transient | per-worker f64 widen buffer + mask `out` | `≈ T · n²·8` | T·n² (DB-⊥) | ~0.4 GiB |
| Online transient | `t` level-1 bodies + response | `≈ t · n·size·8` | R/γ0 · n (∝R) | small |

## Reading the model

- **Three terms scale with the database:** the plaintext DB (**R·C**), `l1_precompute` (**R**), `q0_masks` (**C**). Everything else is fixed in DB size (depends on `n`, `γ`, `T`).
- **Total ≈ `2·R·C + 8·n·C + 16·n·size·R + T·S_pack + O(n,γ)`.**
  At R=2¹⁵, C=2¹⁹: `32.0 + 8.0 + 3.0 + 0.4 + pool + ~0.4`
  → **~44 GiB at T=1**, **~47–48 GiB at T=24** (`T·S_pack` = 3.4 GiB) — matches the measured 48 GB RSS.
- **Dominant resident cost** is `2·R·C` (the DB, 32 GiB) + `8·n·C` (the f64 query masks, 8 GiB) + `16·n·size·R` (the level-1 BSGS precomputes, 3 GiB). At this C-scaled config `q0_masks` overtakes `l1_precompute` and is the single largest *non-DB* term.

## Validation against measurement

| DB | `2·R·C` | `l1_precompute` | `q0_masks` | rest | model total | measured |
|---|---|---|---|---|---|---|
| R=2¹⁴, C=2¹⁵ (1 GiB) | 1.0 | 1.5 | 0.5 | ~1.0 | ~4.0 GiB | 3.97 GiB, T=1 (4.97 − 1.0 `db_prep`) |
| R=2¹⁵, C=2¹⁷ (8 GiB) | 8.0 | 3.0 | 2.0 | ~1.6 | ~14.6 GiB | 14.76 GiB, T=1 |
| R=2¹⁵, C=2¹⁹ (32 GiB) | 32.0 | 3.0 | 8.0 | ~0.8 | ~44 GiB (T=1) / ~47 (T=24) | 48 GB RSS |

## Remaining levers (by formula)

1. **`q0_masks`** (`8·n·C`, **8.0 GiB here** — the largest non-DB term): offline-only **and** seed-derived (`expand_crs_masks(server_seed.recursion_a0(), …)`). Cannot be stored at lower precision — the f64 mantissa (53 bits) is what keeps the clear-text `U·A` dot (over `n` terms, `U ~ 2¹⁵`) below the ~2⁻²¹ noise floor; i32 would push the error to ~2⁻¹¹·⁵ and break correctness. The safe win is to **not store it** — regenerate from the seed inside `offline()` and drop it, freeing the full `8·n·C` (here **8 GiB**, ~17% of RSS) during serving.
2. **Scratch pool** (`T · S_pack`): cap `T` to physical cores and/or shrink `S_pack` to the online-pack need; this is the only term that grows with thread count (up to 3.4 GiB at T=24).
3. The two big terms (`2·R·C`, `16·n·size·R`) are intrinsic — the DB itself and one BSGS precompute per batch; reducing them needs an algorithmic change, not a representation one.
