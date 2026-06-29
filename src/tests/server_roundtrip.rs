use std::marker::PhantomData;

use poulpy_cpu_avx::FFT64Avx;

use crate::{
    client::{Client, Response},
    config::{Collapse, Config, DefaultPirParameters32B, DEFAULT_BASE2K, DEFAULT_K, DEFAULT_N},
    database::DatabaseLayout,
    payload::{U256P65535, U256P65536},
    server::Server,
};

type BE = FFT64Avx;
type Layout = DatabaseLayout<U256P65535>;

/// Full Server↔Client round-trip on a tiny `2 × 1` block grid (interpolation
/// degree 2, so the Horner reduction actually selects between two matrices) over
/// the `FFT64Avx` backend. Retrieves a full-range 256-bit payload from the
/// second matrix and checks it decodes exactly.
// Full n=2048 FHE end-to-end: run the test suite with `--release`.
#[test]
fn server_client_roundtrip_interpolation_generic_u256_chunked() {
    let config = Config::<[u8; 32], U256P65535> {
        n: DEFAULT_N,
        base2k: DEFAULT_BASE2K,
        k: DEFAULT_K,
        collapse: Collapse::Interpolation,
        _phantom: PhantomData,
    };
    let n = config.n();
    let layout = Layout::new(8 * n, n);
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
    let response: Response<BE> = server.respond(&query);
    let got = client.decode(&response, &state);
    assert_eq!(
        got, payload,
        "generic interpolation round-trip mismatch for item {item}"
    );
}

#[test]
fn server_client_roundtrip_full_u256() {
    let config = DefaultPirParameters32B::InspireInt1GiB
        .interpolation()
        .expect("InspireInt1GiB must resolve to interpolation params")
        .config;
    let n = config.n();
    let layout = Layout::new(2 * n, n);

    // Index 300_000 lands in block-row (matrix) 1, exercising the second dim.
    let item: usize = 300_000;
    let mut payload = [0u8; 32];
    for (i, b) in payload.iter_mut().enumerate() {
        *b = (i as u8).wrapping_mul(7).wrapping_add(3);
    }

    let mut server = Server::<BE, U256P65535>::new(config, layout);
    server.update_shard(item, &[payload]);
    server.offline();

    let server_layout = server.layout();
    let address = server_layout.address(item, n, n);
    assert_eq!(
        address.matrix, 1,
        "item should resolve to the second matrix"
    );

    // Client builds the query from the same config/layout, without server state.
    let mut client = Client::<BE, U256P65535>::new(config, layout);
    let (query, state) = client.query(item);

    let response: Response<BE> = server.respond(&query);
    let got = client.decode(&response, &state);
    assert_eq!(
        got, payload,
        "server/client round-trip mismatch for item {item}"
    );
}

#[test]
fn server_client_roundtrip_recursion_generic_u256_chunked() {
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
    let address = state.address();
    assert_eq!(address.column, 69);
    assert_eq!(address.matrix, 69);
    assert_eq!(address.row_offset, 16);

    let response: Response<BE> = server.respond(&query);
    let got = client.decode(&response, &state);
    assert_eq!(
        got, payload,
        "generic Recursion round-trip mismatch for item {item}"
    );

    let expected_record = server.database().record(address.column, address.matrix);
    let noise = client.noise(&response, &state, &expected_record);
    assert!(
        noise.max_log2() < -20.0,
        "InsPIRe2 payload decoded correctly but noise estimate is too large: max={}, std={}",
        noise.max_log2(),
        noise.std_log2()
    );
}
