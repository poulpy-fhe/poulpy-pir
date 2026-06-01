//! End-to-end toy PIR over the [`Server`] / [`Client`] API.
//!
//! 1. The server builds a [`Database`](poulpy_pir::database::Database) from a
//!    [`DatabaseLayout`], fills it with 256-bit payloads, and runs the OFFLINE
//!    interpolation + packing pre-processing.
//! 2. The client takes the server's public [`ServerSeed`](poulpy_pir::client::ServerSeed)
//!    and the layout, resolves a payload index to an address, and builds a query
//!    (one-hot bodies + packing keys + the interpolation GGSW root).
//! 3. The server answers; the client decrypts and decodes the payload.

use std::time::Instant;

use poulpy_core::GGSWEncryptSk;
use poulpy_cpu_avx::FFT64Avx;
use poulpy_hal::{api::ScratchOwnedAlloc, layouts::ScratchOwned};
use poulpy_pir::{
    client::{Client, Response},
    database::{DatabaseInfos, DatabaseLayout},
    interpolation::Interpolation,
    parameters::Parameters,
    payload::{Payload, U256P65535},
    server::Server,
};

/// Backend used by this driver.
type BE = FFT64Avx;
/// Database layout for full 256-bit payloads (`P::EXPONENT = 17` base-65535 digits).
type Layout = DatabaseLayout<U256P65535>;

/// Block grid: `BLOCK_ROWS` interpolation matrices, each `K_BLOCKS · n` columns.
const BLOCK_ROWS: usize = 4;
const K_BLOCKS: usize = 32;

/// Payload (256-bit item) index to retrieve.
const ITEM_INDEX: usize = 5_000_000;

fn format_bytes(bytes: f64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    format!("{value:.3} {}", UNITS[unit])
}

fn main() {
    let params = Parameters::<BE>::default();
    let n = params.n();
    let layout = Layout::new(n, BLOCK_ROWS, K_BLOCKS);
    let address = layout.address(ITEM_INDEX);

    println!(
        "database                    : {} matrices of {} x {} coeffs (block_cols {})",
        layout.block_rows(),
        n,
        layout.cols(),
        layout.block_cols()
    );
    println!(
        "payload capacity            : {} x 32 B = {}",
        layout.num_payloads(),
        format_bytes(layout.total_payload_bytes() as f64)
    );
    println!(
        "payload digits (p = {})  : {}",
        layout.p(),
        layout.payload_digits()
    );
    println!("interpolation degree (t)    : {}", layout.interpolation_t());
    println!(
        "target payload {ITEM_INDEX}     : matrix {}, block_col {}, col_in_block {}, row_offset {}",
        address.matrix, address.block_col, address.col_in_block, address.row_offset
    );

    // ---- SERVER: build, fill, and pre-process the database. ----
    let mut server = Server::<BE, U256P65535>::new(layout);

    // Fill with random payloads — the server keeps the plaintext DB, so we verify
    // the query result against `server.get` rather than recomputing values.
    let t = Instant::now();
    let capacity = layout.num_payloads();
    let chunk = 1usize << 16;
    let mut buf = vec![[0u8; 32]; chunk];
    let mut start = 0;
    while start < capacity {
        let len = chunk.min(capacity - start);
        getrandom::fill(buf[..len].as_flattened_mut()).expect("OS entropy");
        server.update_shard(start, &buf[..len]);
        start += len;
    }
    println!("database fill                : {:?}", t.elapsed());

    // SETUP (DB-independent, reused across DB updates): materialize the masks.
    let t = Instant::now();
    server.generate_query_mask();
    println!("SETUP:");
    println!(
        "  {:<26} : {:?}",
        format!("query mask materialize (x{})", layout.block_cols()),
        t.elapsed()
    );

    let off = server.offline();
    println!("OFFLINE:");
    println!("  {:<26} : {:?}", "interpolation", off.interpolation);
    println!("  {:<26} : {:?}", "prepare U", off.prepare_u);
    println!("  {:<26} : {:?}", "U*A mask multiply", off.ua_mask);
    println!("  {:<26} : {:?}", "packing mask prep", off.mask_prep);
    println!("  {:<26} : {:?}", "pack_precompute", off.pack_precompute);
    println!("  {:<26} : {:?}", "total", off.total());

    // ---- CLIENT: build the query from the server's public seed + layout. ----
    let mut client = Client::<BE>::default();
    let server_seed = server.server_seed();
    let t = Instant::now();
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
    println!("query generation             : {:?}", t.elapsed());

    // ---- SERVER: answer the query (online). ----
    let (response, on): (Response<BE>, _) = server.respond(&query);
    println!("ONLINE:");
    println!("  {:<26} : {:?}", "pack_keys_precompute", on.pack_keys_precompute);
    println!("  {:<26} : {:?}", "U*b data multiply", on.ub_body);
    println!("  {:<26} : {:?}", "pack", on.pack);
    println!("  {:<26} : {:?}", "prepare_root (ggsw)", on.prepare_root);
    println!("  {:<26} : {:?}", "horner reduce", on.reduce);
    println!("  {:<26} : {:?}", "total", on.total());

    // ---- CLIENT: decrypt and decode the payload. ----
    let recovered = client.decrypt(&response, &sk);
    let digits: Vec<i16> = (0..address.digits)
        .map(|k| recovered[address.row_offset + k] as i16)
        .collect();
    let mut got = [0u8; 32];
    U256P65535::decode(&mut got, &digits);
    let want = server.get(ITEM_INDEX); // ground truth from the server's plaintext DB
    println!(
        "retrieved payload {ITEM_INDEX}      : {}",
        if got == want { "OK" } else { "MISMATCH" }
    );
}
