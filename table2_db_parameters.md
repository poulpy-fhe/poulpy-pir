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
