//! Batch query processing round-trips. The batched body product (i16×f64 GEMM)
//! must produce results identical to answering each query individually.

use std::marker::PhantomData;

use poulpy_cpu_avx::FFT64Avx;

use crate::{
    client::{Client, Response},
    config::{Collapse, Config, DefaultPirParameters32B, DEFAULT_BASE2K, DEFAULT_K},
    database::DatabaseLayout,
    payload::{U256P65535, U256P65536},
    server::{Query, Server},
};

type BE = FFT64Avx;
type Layout = DatabaseLayout<U256P65535>;

/// A batch of interpolation queries answered via [`Server::respond_batch`] must
/// (a) decode to the correct payloads and (b) be bit-identical to answering each
/// query individually with [`Server::respond`]. The batch spans both matrices so
/// the Horner reduction is genuinely exercised per query.
// Full n=2048 FHE end-to-end: run with `--release`.
#[test]
fn batch_interpolation_matches_per_query() {
    let config = DefaultPirParameters32B::InspireInt1GiB
        .interpolation()
        .expect("InspireInt1GiB must resolve to interpolation params")
        .config;
    let n = config.n();
    let layout = Layout::new(2 * n, n);

    let mut server = Server::<BE, U256P65535>::new(config, layout);
    let capacity = layout.num_payloads(n);
    let mut payloads = vec![[0u8; 32]; capacity];
    for (idx, payload) in payloads.iter_mut().enumerate() {
        for (byte_idx, b) in payload.iter_mut().enumerate() {
            *b = (idx as u8)
                .wrapping_mul(31)
                .wrapping_add((byte_idx as u8).wrapping_mul(17))
                .wrapping_add(7);
        }
    }
    server.update_shard(0, &payloads);
    server.offline();

    // Items chosen to land in different matrices / columns (and one repeat).
    let items = [12_345usize, 300_000, 12_345, capacity - 1, 1];

    let mut client = Client::<BE, U256P65535>::new(config, layout);
    let mut queries: Vec<Query<BE>> = Vec::with_capacity(items.len());
    let mut states = Vec::with_capacity(items.len());
    for &item in &items {
        let (query, state) = client.query(item);
        queries.push(query);
        states.push(state);
    }

    // Per-query reference responses (decoded payloads).
    let per_query: Vec<[u8; 32]> = queries
        .iter()
        .zip(&states)
        .map(|(q, st)| {
            let response: Response<BE> = server.respond(q);
            client.decode(&response, st)
        })
        .collect();

    // Batched responses.
    let batch_responses = server.respond_batch(&queries);
    assert_eq!(batch_responses.len(), items.len());

    for (i, ((response, state), &item)) in
        batch_responses.iter().zip(&states).zip(&items).enumerate()
    {
        let got = client.decode(response, state);
        assert_eq!(
            got, payloads[item],
            "batch item {i} (index {item}) decoded wrong"
        );
        assert_eq!(
            got, per_query[i],
            "batch item {i} (index {item}) differs from per-query answer"
        );
    }
}

/// A single-element batch must behave exactly like `respond` (the `nq == 1` path
/// delegates the GEMM to the memory-bound GEMV).
#[test]
fn batch_interpolation_single_element() {
    let config = DefaultPirParameters32B::InspireInt1GiB
        .interpolation()
        .expect("InspireInt1GiB must resolve to interpolation params")
        .config;
    let n = config.n();
    let layout = Layout::new(2 * n, n);
    let item = 300_000usize;

    let mut payload = [0u8; 32];
    for (i, b) in payload.iter_mut().enumerate() {
        *b = (i as u8).wrapping_mul(7).wrapping_add(3);
    }

    let mut server = Server::<BE, U256P65535>::new(config, layout);
    server.update_shard(item, &[payload]);
    server.offline();

    let mut client = Client::<BE, U256P65535>::new(config, layout);
    let (query, state) = client.query(item);
    let responses = server.respond_batch(std::slice::from_ref(&query));
    assert_eq!(responses.len(), 1);
    let got = client.decode(&responses[0], &state);
    assert_eq!(got, payload, "single-element batch round-trip mismatch");
}

/// Recursion uses the per-query fallback (no batched fast path yet); the batch
/// API must still return correct, per-query-identical responses.
#[test]
fn batch_recursion_fallback_matches_per_query() {
    let config = Config::<[u8; 32], U256P65536> {
        n: 64,
        base2k: DEFAULT_BASE2K,
        k: DEFAULT_K,
        collapse: Collapse::Recursion {
            gamma0: 32,
            gamma1: 32,
            gamma2: 16,
        },
        _phantom: PhantomData,
    };
    let Collapse::Recursion { gamma0, .. } = config.collapse() else {
        panic!("test config must use Recursion");
    };
    let layout = DatabaseLayout::<U256P65536>::new(70 * gamma0, 70);
    let capacity = layout.num_payloads(gamma0);

    let mut server = Server::<BE, U256P65536>::new(config, layout);
    let items = [capacity - 1, 0usize, 137];
    let mut payloads = std::collections::HashMap::new();
    for &item in &items {
        let mut payload = [0u8; 32];
        for (i, b) in payload.iter_mut().enumerate() {
            *b = (i as u8)
                .wrapping_mul(19)
                .wrapping_add(item as u8)
                .wrapping_add(11);
        }
        server.update_shard(item, &[payload]);
        payloads.insert(item, payload);
    }
    server.offline();

    let mut client = Client::<BE, U256P65536>::new(config, layout);
    let mut queries: Vec<Query<BE>> = Vec::new();
    let mut states = Vec::new();
    for &item in &items {
        let (query, state) = client.query(item);
        queries.push(query);
        states.push(state);
    }

    let responses = server.respond_batch(&queries);
    assert_eq!(responses.len(), items.len());
    for ((response, state), &item) in responses.iter().zip(&states).zip(&items) {
        let got = client.decode(response, state);
        assert_eq!(
            got, payloads[&item],
            "recursion batch item {item} decoded wrong"
        );
    }
}
