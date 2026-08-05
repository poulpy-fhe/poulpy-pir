# Release Checklist

Before publishing an official release:

1. Update `CHANGELOG.md` with the release date and notable changes.
2. Run the fast CI checks locally:

   ```sh
   cargo fmt --check
   cargo check --all-targets
cargo clippy --all-targets -- -D warnings
cargo test api_errors --lib
cargo test wire_bounds --lib
cargo doc --no-deps
```

3. Run the full AVX2 library test suite on an AVX2/FMA host:

   ```sh
   RUSTFLAGS="-C target-feature=+avx2,+fma" cargo test --release --lib --features avx2-fhe
   ```

4. Inspect the package contents:

   ```sh
   cargo package --list
   ```

5. Dry-run package creation:

   ```sh
   cargo package
   ```

6. Tag the release only after CI and the release-check workflow pass.
