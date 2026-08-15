# ptr_hash 2.0.2, forked

Upstream: <https://github.com/RagnarGrootKoerkamp/ptrhash> (MIT), vendored from
the crates.io 2.0.2 release. Applied via `[patch.crates-io]`.

## Why

The MPHF is built on the server and evaluated on the client, so its serialized
form crosses architectures. `epserde` writes `usize` at the host's width and
refuses to read a blob written at another, which makes an x86-64 server's MPHF
unreadable by a 32-bit `wasm32` client:

```
The file was serialized on an architecture where a usize has size 8,
but on the current architecture it has size 4.
```

## Change 1: the hash is architecture-dependent

`Xxh3_128::hash` hashes keys through `std::hash::Hash`. The `Hash` impl for
slices prefixes the length with `write_usize`, whose default writes
`size_of::<usize>()` native-endian bytes. A 64-bit server and a 32-bit client
therefore hash the *same key* to *different values*, so every index disagrees —
measured, not theorised: 0/40 probes matched before this fix, 40/40 after.

`src/hash.rs` now wraps the inner hasher in `FixedWidth`, which forwards
everything but writes `usize`/`isize` at a fixed 64 bits. This changes the hash
function, so any MPHF built before it must be rebuilt.

## Change 2: the serialized form is architecture-dependent

One added file, `src/portable.rs`, plus its `mod portable;` line.

It implements `write_portable` / `read_portable` on the concrete instantiation
keyword PIR uses (`F = Vec<u32>`, `V = Vec<u8>`), writing every field at a fixed
width. Values that do not fit the reading target's `usize` are an error, never a
truncation.

Derived fields (`parts`, `slots`, `buckets`, …) are written rather than
recomputed on read. `PtrHash::init` derives them with floating-point `ln` and
`floor`; libm's last ulp is not guaranteed identical across targets, and one ulp
either side of a `floor` boundary would change the geometry and silently yield a
different hash function.

The `Cargo.toml` also takes `epserde` with `default-features = false` and uses
`cacheline-ef?/epserde` rather than `cacheline-ef/epserde`. Both keep `mmap-rs`,
which has no wasm backend, out of the graph. They are inert while the `epserde`
feature is off — poulpy-pir no longer enables it — but are kept so the crate
stays wasm-buildable if it is turned back on.

## Upstreaming

The `cacheline-ef?/epserde` change and the `FixedWidth` hasher are both plain bug
fixes and should go upstream. The portable format is more poulpy-pir-specific;
upstream would more likely want it as a general `epserde` capability.
