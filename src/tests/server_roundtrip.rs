use poulpy_core::GGSWEncryptSk;
use poulpy_cpu_avx::FFT64Avx;
use poulpy_hal::{api::ScratchOwnedAlloc, layouts::ScratchOwned};

use crate::{
    client::{Client, Response},
    database::{DatabaseInfos, DatabaseLayout},
    interpolation::Interpolation,
    parameters::Parameters,
    payload::{Payload, U256P65535},
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
fn server_client_roundtrip_full_u256() {
    let params = Parameters::<BE>::default();
    let n = params.n();
    let layout = Layout::new(n, /* block_rows */ 2, /* block_cols */ 1);

    // Index 300_000 lands in block-row (matrix) 1, exercising the second dim.
    let item: usize = 300_000;
    let mut payload = [0u8; 32];
    for (i, b) in payload.iter_mut().enumerate() {
        *b = (i as u8).wrapping_mul(7).wrapping_add(3);
    }

    let mut server = Server::<BE, U256P65535>::new(layout);
    server.update(item, payload);
    server.offline();

    let address = server.layout().address(item);
    assert_eq!(address.matrix, 1, "item should resolve to the second matrix");
    let server_seed = server.server_seed();

    // Client builds the query: common material + the interpolation GGSW root.
    let mut client = Client::<BE>::default();
    let (common, mut ctx, sk) = client.begin_query(&address, &server_seed);
    let interpolation = Interpolation::new(server.layout(), client.params());
    let mut qscratch = ScratchOwned::<BE>::alloc(
        client
            .params()
            .module()
            .ggsw_encrypt_sk_tmp_bytes(&client.params().ggsw_layout()),
    );
    let query = interpolation.build_query(
        client.params().module(),
        common,
        &mut ctx,
        &address,
        &mut qscratch,
    );
    drop(ctx);

    let (response, _timings): (Response<BE>, _) = server.respond(&query);
    let recovered = client.decrypt(&response, &sk);

    let digits: Vec<i16> = (0..address.digits)
        .map(|k| recovered[address.row_offset + k] as i16)
        .collect();
    let mut got = [0u8; 32];
    U256P65535::decode(&mut got, &digits);
    assert_eq!(got, payload, "server/client round-trip mismatch for item {item}");
}
