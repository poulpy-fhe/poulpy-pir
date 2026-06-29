# Table 2 Database Parameters

Source: `2025-1352.pdf`, Table 2, cross-referenced with Appendix Table 8
for the labeled InsPIRe^(2) `gamma0` configurations.

Assumptions:

- Each database coefficient is one `u16`.
- The table database sizes are interpreted as binary sizes:
  - `1 GB = 2^33` bits = `2^29` `u16` coefficients.
  - `8 GB = 2^36` bits = `2^32` `u16` coefficients.
  - `32 GB = 2^38` bits = `2^34` `u16` coefficients.
- `cols` is the first database dimension, `N/t`.
- `rows = total_u16_coeffs / cols`.
- For InsPIRe^(2), `gamma0` is not deduced from Table 2 alone; it is
  assigned by matching the Table 2 query sizes to Appendix Table 8.

## Default 32-byte Payload Parameters

The following defaults are encoded by `DefaultPirParameters32B` and resolve to
typed parameter bundles that carry the database size, layout, config, and (for
InsPIRe^(2)) `gamma0`, `gamma1`, and `gamma2`. They provide one database layout
per power-of-two database size from 1 GiB to 32 GiB and use the crate's default
cryptographic constants:

- InsPIRe: `n=2048`, `base2k=18`, `k=54`, `Collapse::Interpolation`,
  `U256P65535`.
- InsPIRe^(2): `n=2048`, `base2k=18`, `k=54`, `U256P65536`, with
  `Collapse::Recursion { gamma0, gamma1: 1024, gamma2 }` for
  `(gamma0,gamma2) = (16,16)`, `(32,32)`, or `(64,64)`.

The default layout rule keeps the second dimension fixed at the measured
Table 2 / Table 8 sweet spot and scales the first dimension:

```text
rows = 2^16
cols = 2^13 * db_gib
```

For InsPIRe, this fixes `interpolation_t = rows / n = 32` and uses
`query_KiB = 84 + 7 * cols / 1024`. For InsPIRe^(2), this fixes
`t = rows / gamma0`, so the same DB layout gives `t=4096` for `gamma0=16`,
`t=2048` for `gamma0=32`, and `t=1024` for `gamma0=64`. The 1, 8, and 32 GiB
entries are anchored where the paper tables contain the matching fixed-row
shape; 2, 4, and 16 GiB are extrapolated with the same curves. The 32 GiB
`gamma0=64` entry is also extrapolated because Appendix Table 8 stops one width
earlier for that gamma family.

| DB | rows | cols | InsPIRe query | Rec `16,1024,16` query | Rec `32,1024,32` query | Rec `64,1024,64` query |
|---:|---:|---:|---:|---:|---:|---:|
| 1 GiB | 65,536 | 8,192 | 140 KiB | 80 KiB | 66 KiB | 60 KiB |
| 2 GiB | 65,536 | 16,384 | 196 KiB | 132 KiB | 119 KiB | 113 KiB |
| 4 GiB | 65,536 | 32,768 | 308 KiB | 238 KiB | 225 KiB | 219 KiB |
| 8 GiB | 65,536 | 65,536 | 532 KiB | 450 KiB | 437 KiB | 431 KiB |
| 16 GiB | 65,536 | 131,072 | 980 KiB | 874 KiB | 861 KiB | 855 KiB |
| 32 GiB | 65,536 | 262,144 | 1,876 KiB | 1,722 KiB | 1,709 KiB | 1,703 KiB |

Note: this preserves the paper's binary database-size convention used below.
For InsPIRe with `U256P65535`, each 32-byte payload occupies 17 `u16`
coefficients, so the realized 32-byte payload capacity is slightly below the
coefficient-byte size. InsPIRe^(2) with `U256P65536` uses 16 `u16` coefficients
per 32-byte payload and matches the listed GiB capacity exactly.

| Scheme | DB | Table 2 query | Variant | cols | rows |
|---|---:|---:|---|---:|---:|
| InsPIRe^(2) | 1 GB | 40 KB | `gamma0=64` | 4,096 | 131,072 |
| InsPIRe^(2) | 1 GB | 60 KB | `gamma0=64` | 8,192 | 65,536 |
| InsPIRe^(2) | 1 GB | 109 KB | `gamma0=64` | 16,384 | 32,768 |
| InsPIRe^(2) | 1 GB | 113 KB | `gamma0=32` | 16,384 | 32,768 |
| InsPIRe^(2) | 1 GB | 214 KB | `gamma0=64` | 32,768 | 16,384 |
| InsPIRe^(2) | 1 GB | 215 KB | `gamma0=32` | 32,768 | 16,384 |
| InsPIRe^(2) | 8 GB | 106 KB | `gamma0=64` | 8,192 | 524,288 |
| InsPIRe^(2) | 8 GB | 132 KB | `gamma0=64` | 16,384 | 262,144 |
| InsPIRe^(2) | 8 GB | 225 KB | `gamma0=64` | 32,768 | 131,072 |
| InsPIRe^(2) | 8 GB | 431 KB | `gamma0=64` | 65,536 | 65,536 |
| InsPIRe^(2) | 8 GB | 855 KB | `gamma0=32` | 131,072 | 32,768 |
| InsPIRe^(2) | 32 GB | 265 KB | `gamma0=64` | 32,768 | 524,288 |
| InsPIRe^(2) | 32 GB | 450 KB | `gamma0=64` | 65,536 | 262,144 |
| InsPIRe^(2) | 32 GB | 477 KB | `gamma0=32` | 65,536 | 262,144 |
| InsPIRe^(2) | 32 GB | 861 KB | `gamma0=64` | 131,072 | 131,072 |
| InsPIRe^(2) | 32 GB | 874 KB | `gamma0=32` | 131,072 | 131,072 |
| InsPIRe^(2) | 32 GB | 1709 KB | `gamma0=32` | 262,144 | 65,536 |
| InsPIRe | 1 GB | 140 KB | - | 8,192 | 65,536 |
| InsPIRe | 1 GB | 196 KB | - | 16,384 | 32,768 |
| InsPIRe | 1 GB | 308 KB | - | 32,768 | 16,384 |
| InsPIRe | 8 GB | 532 KB | - | 65,536 | 65,536 |
| InsPIRe | 8 GB | 532 KB | - | 65,536 | 65,536 |
| InsPIRe | 8 GB | 980 KB | - | 131,072 | 32,768 |
| InsPIRe | 32 GB | 980 KB | - | 131,072 | 131,072 |
| InsPIRe | 32 GB | 1876 KB | - | 262,144 | 65,536 |
| InsPIRe | 32 GB | 1876 KB | - | 262,144 | 65,536 |

For InsPIRe, the query-size inversion is:

```text
cols = (query_KiB - 84 KiB) * 1024 / 7
```

This uses Table 1's `log2(q)=56`, so each encrypted indicator scalar is
7 bytes, and the fixed RGSW component contributes 84 KiB.
