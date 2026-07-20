# `pir` example

End-to-end PIR round trip: setup, database fill, offline preprocessing,
query, answer, decrypt, with per-phase timings, wire sizes, and payload
verification.

## Dependencies

- An AVX-512F host (`FFT64Avx512` backend, compile-time guarded — build with
  `RUSTFLAGS="-C target-feature=+avx512f,..."`).
- `libopenblas-pthread-dev` (Ubuntu), required by the `cblas-gemm` feature.
  Drop `cblas-gemm` from `--features` to build without it.

## Usage

```sh
cargo run --release --features avx512-fhe --example pir -- <preset> [batch]
```

- `<preset>` — a `DefaultPirParameters32B` name; run without arguments to
  list them all.
- `[batch]` — queries answered together per online batch (default 1).

## Configuration

- `batch` = 1: build with `--features numa-db-interleave`.
- `batch` > 1: build without `numa-db-interleave`.
- Thread counts default to the logical-CPU count; override per phase with
  `PIR_THREADS` (base), `PIR_SETUP_THREADS`, `PIR_OFFLINE_THREADS`,
  `PIR_ONLINE_THREADS`.

## Commands (InsPIRe², γ0 = 32, batch = 1)

```sh
RUSTFLAGS="-C target-feature=+avx512f,+avx512dq" \
cargo run --release --features "avx512-fhe,cblas-gemm,numa-db-interleave" --example pir -- InsPIRe2-g32-32GiB-c262144

RUSTFLAGS="-C target-feature=+avx512f,+avx512dq" \
cargo run --release --features "avx512-fhe,cblas-gemm,numa-db-interleave" --example pir -- InsPIRe2-g32-16GiB-c262144

RUSTFLAGS="-C target-feature=+avx512f,+avx512dq" \
cargo run --release --features "avx512-fhe,cblas-gemm,numa-db-interleave" --example pir -- InsPIRe2-g32-8GiB-c131072

RUSTFLAGS="-C target-feature=+avx512f,+avx512dq" \
cargo run --release --features "avx512-fhe,cblas-gemm,numa-db-interleave" --example pir -- InsPIRe2-g32-4GiB-c131072

RUSTFLAGS="-C target-feature=+avx512f,+avx512dq" \
cargo run --release --features "avx512-fhe,cblas-gemm,numa-db-interleave" --example pir -- InsPIRe2-g32-2GiB-c65536

RUSTFLAGS="-C target-feature=+avx512f,+avx512dq" \
cargo run --release --features "avx512-fhe,cblas-gemm,numa-db-interleave" --example pir -- InsPIRe2-g32-1GiB-c32768
```

Batched (drop `numa-db-interleave`, pass the batch size):

```sh
RUSTFLAGS="-C target-feature=+avx512f,+avx512dq" \
cargo run --release --features "avx512-fhe,cblas-gemm" --example pir -- InsPIRe2-g32-32GiB-c262144 256
```
