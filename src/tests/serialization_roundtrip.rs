//! Round-trips a [`Query`] and a [`Response`] through their wire encoding
//! ([`WriterTo::write_to`] / `read_from`) in the middle of an otherwise normal
//! Server↔Client exchange, then checks the payload still decodes exactly. This
//! exercises the transmitted-message (de)serialization for both collapses.

use std::marker::PhantomData;

use poulpy_cpu_avx::FFT64Avx;

use crate::{
    client::Client,
    config::{Collapse, Config, INSPIRE_INT_32B, INSPIRE_REC_32B},
    database::DatabaseLayout,
    payload::{Payload, U256P65535, U256P65536},
    serialization::{query_component_sizes, response_component_sizes},
    server::{Query, Server},
};

type BE = FFT64Avx;

fn fmt_kib(bytes: usize) -> String {
    format!("{:.1} KiB", bytes as f64 / 1024.0)
}

/// Builds a real query+response at `config`/`layout` for `item`, then prints the
/// serialized wire sizes (totals + per-component breakdown). Reports actual
/// transmitted bytes after base2k=63 repacking + seed compression.
fn report_sizes<P: Payload<[u8; 32]>>(
    label: &str,
    config: Config<[u8; 32], P>,
    layout: DatabaseLayout<P>,
    item: usize,
) {
    let payload = [7u8; 32];
    let mut server = Server::<BE, P>::new(config, layout);
    server.update_shard(item, &[payload]);
    server.offline();

    let mut client = Client::<BE, P>::new(config, layout);
    let (query, state) = client.query(item);
    let response = server.respond(&query);
    // Sanity: the exchange still decodes (sizes are meaningless otherwise).
    assert_eq!(client.decode(&response, &state), payload, "{label} decode");

    let module = server.params().module();
    let mut qbuf = Vec::new();
    query.write_to(module, &mut qbuf).unwrap();
    let mut rbuf = Vec::new();
    response.write_to(module, &mut rbuf).unwrap();

    eprintln!("\n=== {label} (n={}) ===", config.n());
    eprintln!("QUERY    total = {} ({})", qbuf.len(), fmt_kib(qbuf.len()));
    for (name, sz) in query_component_sizes(&query, module) {
        eprintln!("    {name:<18} {sz:>10} B  ({})", fmt_kib(sz));
    }
    eprintln!("RESPONSE total = {} ({})", rbuf.len(), fmt_kib(rbuf.len()));
    for (name, sz) in response_component_sizes(&response, module) {
        eprintln!("    {name:<18} {sz:>10} B  ({})", fmt_kib(sz));
    }
}

/// Measures query sizes at the table-2 database widths (query build is
/// DB-content-independent, so no server/offline needed). Splits out the reusable
/// packing keys, since the paper amortizes them to setup. Run with `--nocapture`.
#[test]
fn measure_query_sizes_vs_table() {
    // (db, cols, rows, paper query KB)
    let interp = [
        ("1GB", 8192usize, 65536usize, 140),
        ("1GB", 16384, 32768, 196),
        ("1GB", 32768, 16384, 308),
        ("8GB", 65536, 65536, 532),
        ("8GB", 131072, 32768, 980),
        ("32GB", 131072, 131072, 980),
        ("32GB", 262144, 65536, 1876),
    ];
    eprintln!("\n=== InsPIRe query vs Table 2 ===");
    eprintln!(
        "{:>4} {:>8} | {:>9} {:>9} {:>9} | {:>7}",
        "db", "cols", "mine", "ex-keys", "keys", "paper"
    );
    for (db, cols, rows, paper) in interp {
        let layout = DatabaseLayout::<U256P65535>::new(rows, cols);
        let mut client = Client::<BE, U256P65535>::new(INSPIRE_INT_32B, layout);
        let (query, _) = client.query(0);
        let module = client.params().module();
        let mut buf = Vec::new();
        query.write_to(module, &mut buf).unwrap();
        let comp = query_component_sizes(&query, module);
        let keys: usize = comp
            .iter()
            .filter(|(n, _)| n.starts_with("key"))
            .map(|(_, s)| *s)
            .sum();
        eprintln!(
            "{db:>4} {cols:>8} | {:>9} {:>9} {:>9} | {:>6}K",
            fmt_kib(buf.len()),
            fmt_kib(buf.len() - keys),
            fmt_kib(keys),
            paper
        );
    }

    // InsPIRe2: production config has gamma0 = 32; compare to table gamma0=32 rows.
    let Collapse::Recursion { gamma0, .. } = INSPIRE_REC_32B.collapse() else {
        unreachable!()
    };
    let rec = [
        ("1GB", 16384usize, 113),
        ("1GB", 32768, 215),
        ("8GB", 131072, 855),
        ("32GB", 65536, 477),
        ("32GB", 131072, 874),
        ("32GB", 262144, 1709),
    ];
    eprintln!("\n=== InsPIRe2 query vs Table 2 (gamma0={gamma0}) ===");
    eprintln!(
        "{:>4} {:>8} | {:>9} {:>9} {:>9} | {:>7}",
        "db", "cols", "mine", "ex-keys", "keys", "paper"
    );
    for (db, cols, paper) in rec {
        let rows = cols; // square-ish; only `cols` drives the one-hot width
        let layout = DatabaseLayout::<U256P65536>::new(rows * gamma0, cols);
        let mut client = Client::<BE, U256P65536>::new(INSPIRE_REC_32B, layout);
        let (query, _) = client.query(0);
        let module = client.params().module();
        let mut buf = Vec::new();
        query.write_to(module, &mut buf).unwrap();
        let comp = query_component_sizes(&query, module);
        let keys: usize = comp
            .iter()
            .filter(|(n, _)| n.starts_with("key"))
            .map(|(_, s)| *s)
            .sum();
        eprintln!(
            "{db:>4} {cols:>8} | {:>9} {:>9} {:>9} | {:>6}K",
            fmt_kib(buf.len()),
            fmt_kib(buf.len() - keys),
            fmt_kib(keys),
            paper
        );
    }
}

/// Reports serialized wire sizes at the production configs (both `n = 2048`).
/// Run with `--nocapture` to see the numbers.
#[test]
fn measure_serialized_sizes() {
    let n = INSPIRE_INT_32B.n();
    report_sizes(
        "InsPIRe (interpolation)",
        INSPIRE_INT_32B,
        DatabaseLayout::<U256P65535>::new(2 * n, n),
        300_000,
    );

    let Collapse::Recursion { gamma0, .. } = INSPIRE_REC_32B.collapse() else {
        unreachable!()
    };
    let layout = DatabaseLayout::<U256P65536>::new(70 * gamma0, 70);
    let item = layout.num_payloads(gamma0) - 1;
    report_sizes("InsPIRe2 (recursion)", INSPIRE_REC_32B, layout, item);
}

/// Serializes `query`, deserializes it against `params`, and asserts the byte
/// stream round-trips (re-serializing the decoded value yields identical bytes).
fn roundtrip_query<P: crate::payload::Payload<[u8; 32]>>(
    query: &Query<BE>,
    params: &crate::parameters::Parameters<BE, [u8; 32], P>,
    label: &str,
) -> Query<BE> {
    let module = params.module();
    let mut bytes = Vec::new();
    query.write_to(module, &mut bytes).expect("query write_to");
    eprintln!("[{label}] query serialized: {} bytes", bytes.len());
    let decoded = Query::read_from(&mut &bytes[..], params).expect("query read_from");
    let mut bytes2 = Vec::new();
    decoded.write_to(module, &mut bytes2).expect("query re-write_to");
    assert_eq!(bytes, bytes2, "query serialization is not stable");
    decoded
}

fn roundtrip_response<P: crate::payload::Payload<[u8; 32]>>(
    response: &crate::client::Response<BE>,
    params: &crate::parameters::Parameters<BE, [u8; 32], P>,
    label: &str,
) -> crate::client::Response<BE> {
    let module = params.module();
    let mut bytes = Vec::new();
    response.write_to(module, &mut bytes).expect("response write_to");
    eprintln!("[{label}] response serialized: {} bytes", bytes.len());
    let decoded =
        crate::client::Response::read_from(&mut &bytes[..], params).expect("response read_from");
    let mut bytes2 = Vec::new();
    decoded.write_to(module, &mut bytes2).expect("response re-write_to");
    assert_eq!(bytes, bytes2, "response serialization is not stable");
    decoded
}

/// Interpolation collapse, full `n = 2048` FHE: run with `--release`.
#[test]
fn serialization_roundtrip_interpolation() {
    let config = Config::<[u8; 32], U256P65535> {
        n: 2048,
        base2k: 18,
        k: 54,
        collapse: Collapse::Interpolation,
        _phantom: PhantomData,
    };
    let n = config.n();
    let layout = DatabaseLayout::<U256P65535>::new(8 * n, n);
    let item = 1_800_000usize;

    let mut server = Server::<BE, U256P65535>::new(config, layout);
    let capacity = layout.num_payloads(n);
    let mut payloads = vec![[0u8; 32]; capacity];
    for (idx, payload) in payloads.iter_mut().enumerate() {
        for (byte_idx, b) in payload.iter_mut().enumerate() {
            *b = (idx as u8)
                .wrapping_mul(29)
                .wrapping_add((byte_idx as u8).wrapping_mul(13))
                .wrapping_add(5);
        }
    }
    let payload = payloads[item];
    server.update_shard(0, &payloads);
    server.offline();

    let mut client = Client::<BE, U256P65535>::new(config, layout);
    let (query, state) = client.query(item);

    // Query travels client → server over the wire.
    let query = roundtrip_query(&query, server.params(), "interpolation");
    let response = server.respond(&query);
    // Response travels server → client over the wire.
    let response = roundtrip_response(&response, client.params(), "interpolation");

    let got = client.decode(&response, &state);
    assert_eq!(got, payload, "interpolation serialized round-trip mismatch");
}

/// Recursion (InsPIRe²) collapse on a tiny `n = 64` instance (fast in debug).
#[test]
fn serialization_roundtrip_recursion() {
    let config = Config::<[u8; 32], U256P65536> {
        n: 64,
        base2k: 18,
        k: 54,
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
    let item = layout.num_payloads(gamma0) - 1;

    let mut payload = [0u8; 32];
    for (i, b) in payload.iter_mut().enumerate() {
        *b = (i as u8).wrapping_mul(19).wrapping_add(11);
    }

    let mut server = Server::<BE, U256P65536>::new(config, layout);
    server.update_shard(item, &[payload]);
    server.offline();

    let mut client = Client::<BE, U256P65536>::new(config, layout);
    let (query, state) = client.query(item);

    let query = roundtrip_query(&query, server.params(), "recursion");
    let response = server.respond(&query);
    let response = roundtrip_response(&response, client.params(), "recursion");

    let got = client.decode(&response, &state);
    assert_eq!(got, payload, "recursion serialized round-trip mismatch");
}
