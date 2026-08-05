# Contributing

Thanks for helping improve `poulpy-pir`.

## Development Checks

Run these before opening a pull request:

```sh
cargo fmt --check
cargo check --all-targets
cargo clippy --all-targets -- -D warnings
cargo test api_errors --lib
cargo test wire_bounds --lib
```

For changes to protocol logic, serialization, database layout, packing,
interpolation, recursion, or batching, also run:

```sh
RUSTFLAGS="-C target-feature=+avx2,+fma" cargo test --release --lib --features avx2-fhe
```

The full suite exercises FHE round trips; run it in release mode.

## Public API Changes

Prefer adding fallible `try_*` APIs for service-facing behavior. Panic-style
helpers may remain as convenience wrappers, but network and user-input paths
should be able to return typed errors.

## Security-Sensitive Changes

Changes to cryptographic parameters, serialization, randomness, keyword-index
validation, or untrusted-input handling should call out the security impact in
the pull request description.
