# poulpy-pir

A Rust implementation of **single-server, communication-efficient Private Information Retrieval (PIR)** with server-side preprocessing, built on the [Poulpy](https://github.com/poulpy-fhe/poulpy) FHE library.

PIR lets a client fetch record `i` from a server-held database **without revealing `i`** to the server. This crate implements the two constructions of [*InsPIRe: Communication-Efficient PIR with Server-side Preprocessing*](https://eprint.iacr.org/2025/1352) (Akhavan Mahdavi, Patel, Seo, Yeo) — the first PIR family to obtain **high throughput and low query communication simultaneously** using only server-side preprocessing in the **Common Reference String (CRS) model**, i.e. with *no* offline client–server communication.

## How the protocol works

Client and server share one public random string (the CRS). Because all the protocol's public randomness comes from this shared string, the server can do the bulk of the work *before* seeing any query, and the client's query stays small — there is no offline communication between the two parties.

A query proceeds in four phases:

- **Setup** — client and server instantiate the same public parameters; the server materializes its fixed query material from the CRS. Independent of any query.
- **Offline preprocessing** *(server, query-independent)* — the heavy, database-sized work, done once before any query and reused across all of them. Re-run only when the database changes.
- **Query** *(client)* — the client samples a fresh secret locally, encrypts the index `i`, and sends a small query. The secret never leaves the client.
- **Answer** *(server)* — the server combines the query with its preprocessed state and returns a single encrypted response.

The client decrypts the response to recover its record. The server learns nothing about `i`.

The protocol is **doubly stateless**: the server keeps no per-client or per-session state — every query carries everything needed to answer it — and because each query is built from a freshly sampled client secret, queries are unlinkable. Given two queries, the server cannot tell whether they came from the same client or from two different clients.

## The two constructions

Both share the same cryptosystem and the same server and answer through the same response type; they differ only in how the second retrieval dimension is collapsed into the final answer.

- **InsPIRe** — collapses the second dimension by polynomial interpolation. Smaller query, single-ciphertext response.
- **InsPIRe²** — adds a second, recursive level of PIR. Trades a slightly larger query for a smaller response and less per-query work, which pays off on large databases.

Switching between them is a single configuration change.

## Parameterization

A run is described by one configuration bundling the construction, the database layout, and the payload type.

- **Construction** — InsPIRe (interpolation) or InsPIRe² (recursion).
- **InsPIRe² packing widths** — `γ0` (records per group), `γ1` and `γ2` (the two second-level packing widths). These trade query size against response size and per-query work.
- **Database shape** — total `rows × cols`; any split holding the same database is valid and trades query size against response size.
- **Payload** — the record type (the examples retrieve a 32-byte / 256-bit payload).

For full 32-byte defaults, use `DefaultPirParameters32B`: it enumerates
InsPIRe and InsPIRe² layouts for 1, 2, 4, 8, 16, and 32 GiB databases and
resolves each preset to a parameter bundle containing the DB size, matching
`Config`, `DatabaseLayout`, and for InsPIRe² the `gamma0`, `gamma1`, and
`gamma2` packing widths. InsPIRe² presets include the `gamma0/gamma2` families
`16/16`, `32/32`, and `64/64` with `gamma1=1024`.

The reconstructed database splits from the paper's Table 2 are tabulated in [`table2_db_parameters.md`](table2_db_parameters.md); [`RECURSION_RAM_MODEL.md`](RECURSION_RAM_MODEL.md) gives memory-scaling estimates for InsPIRe² up to a 32 GiB database.

## Running the example

The driver [`examples/pir.rs`](examples/pir.rs) runs a full round trip — setup, database fill, offline preprocessing, query, answer, and decrypt — checks the recovered payload against ground truth, and prints a phase-by-phase timing and noise breakdown.

```sh
cargo run --release --example pir
```

> **Always build with `--release`** — a debug build is orders of magnitude slower.

The example fixes the backend to the AVX2/FMA-accelerated `FFT64Avx` (`type BE = FFT64Avx`). The crate is generic over the backend, so for a portable, dependency-free build you can swap that alias for the scalar reference backend `poulpy_cpu_ref::FFT64Ref` — at a performance cost.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
