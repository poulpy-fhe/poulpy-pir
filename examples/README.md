# Examples

The live-update ETH token-balance demo has moved to the adjacent
`../eth-pir` repository.

# `key_word_pir` example

Keyword PIR round trip: 16 M ETH addresses, each with a 64-byte record — the
full address (zero-padded to 32 B) plus its full 256-bit token balance
(little-endian) — retrieved with a **single PIR query** per lookup. The
backend is `FFT64Avx` (AVX2/FMA) by default; add `--features avx512-fhe`
(with `RUSTFLAGS="-C target-feature=+avx512f"`) to run it on `FFT64Avx512`.

The server builds a minimal perfect hash function (MPHF, `poulpy_pir::keyword`
— a pure `[u8; N] key ↔ index` layer) over the address set, and each record is
**one** `U512P65536` payload (`Payload::Block = [u8; 64]`): address in bytes
`0..32`, little-endian balance in bytes `32..64`. The MPHF is minimal, so its
index is the payload index — `DB[MPHF(address)] = record`, no placement math —
and `Client::decode` returns the whole 64 B block from one response. The
client holds only the ~4 MiB MPHF parameters, never the key set. The address
half is the in-set check: an out-of-set address lands on some other account's
record, whose address cannot match, so the lookup is rejected instead of
leaking a balance.

```sh
cargo run --release --example key_word_pir [-- <preset> [batch]]
```

- `<preset>` — an InsPIRe² `DefaultPirParameters32B` preset with `γ0 ≥ 32`
  (default `InsPIRe2-g32-1GiB-c32768`); the preset supplies geometry and
  collapse, the 64 B payload type is the example's. The byte capacity is
  unchanged — the same coefficients grouped as half as many, twice-as-large
  records.
- `[batch]` — address lookups (one query each) answered together per online
  batch (default 1).
- Thread counts: same `PIR_THREADS` / `PIR_SETUP_THREADS` /
  `PIR_OFFLINE_THREADS` / `PIR_ONLINE_THREADS` overrides as `pir`.

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
cargo run --release --features "avx512-fhe,cblas-gemm,numa-db-interleave" --example pir -- InsPIRe2-g32-4GiB-c65536

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
