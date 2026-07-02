//! Unified toy-PIR driver: one example, two second-dimension *collapses*, selected
//! at run time, each from its [default config](poulpy_pir::config).
//!
//! Both constructions answer through the same [`Server`] type (and the shared
//! [`Response`]); they differ in the [default config](poulpy_pir::config) —
//! cryptosystem `Collapse`, database layout, and payload type — bundled by the
//! unified [`Config`](poulpy_pir::config::Config). Pick one on the command line:
//!
//! ```text
//! cargo run --release --example pir -- interpolation   # InsPIRe  (U256P65535)
//! cargo run --release --example pir -- recursion        # InsPIRe² (U256P65536)
//! ```

use std::time::Instant;

use poulpy_cpu_avx::FFT64Avx;
use poulpy_pir::{
    client::{Client, Response},
    config::{Collapse, Config, DefaultPirConfig32B, DefaultPirParameters32B, DefaultScheme},
    database::DatabaseLayout,
    payload::Payload,
    server::Server,
};

/// Backend used by this driver.
type BE = FFT64Avx;
const DEFAULT: DefaultPirParameters32B = DefaultPirParameters32B::canonical(DefaultScheme::Recursion { gamma0: 32 }, 32);

fn main() {
    const ITEM_INDEX: usize = 1_000_000;

    match DEFAULT.resolve() {
        DefaultPirConfig32B::Interpolation(params) => run(params.config, params.layout, ITEM_INDEX),
        DefaultPirConfig32B::Recursion(params) => run(params.config, params.layout, ITEM_INDEX),
    }
}

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

fn run<P>(config: Config<[u8; 32], P>, layout: DatabaseLayout<P>, item_index: usize)
where
    P: Payload<[u8; 32]>,
{
    let n = config.n();
    let column_height = config.column_height();
    let collapse = config.collapse();
    let address = layout.address_for(item_index, column_height);

    println!("collapse                    : {:?}", collapse);
    println!("ring degree n               : {}\n", n);
    print_layout_summary(config, layout, item_index, address);

    // ---- SETUP: client and server instantiate from the shared config/layout. ----
    let timer = Instant::now();
    let mut client = Client::<BE, P>::new(config, layout);
    let mut server = Server::<BE, P>::new(config, layout);
    let setup = timer.elapsed();
    println!("SETUP                        : {:?}", setup);

    // ---- SERVER: fill with random 256-bit payloads. ----
    let t = Instant::now();
    let capacity = layout.num_payloads(column_height);
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

    let t = Instant::now();
    server.generate_query_mask();
    println!("SETUP (query mask)           : {:?}", t.elapsed());

    // ---- SERVER: query-independent preprocessing. ----
    let off = server.offline();
    println!("OFFLINE total                : {:?}", off.total());
    for phase in off.phases() {
        println!("  {:<30}: {:?}", phase.name(), phase.duration());
    }

    // ---- CLIENT: build the query for the payload index. ----
    let t = Instant::now();
    let (query, state) = client.query(item_index);
    println!("QUERY                        : {:?}", t.elapsed());

    // ---- SERVER: answer. ----
    let (response, online): (Response<BE>, _) = server.respond_timed(&query);
    println!("ONLINE total                 : {:?}", online.total());
    for phase in online.phases() {
        println!("  {:<30}: {:?}", phase.name(), phase.duration());
    }

    // ---- CLIENT: decrypt + decode the payload. ----
    let t = Instant::now();
    let got = client.decode(&response, &state);
    println!("DECRYPT                      : {:?}", t.elapsed());
    let selected = state.address();
    let expected_record = server.database().record(selected.column, selected.matrix);
    let noise = client.noise(&response, &state, &expected_record);
    println!("NOISE log2(max)              : {:.3}", noise.max_log2());
    println!("NOISE log2(std)              : {:.3}", noise.std_log2());

    let want = server.get(item_index); // ground truth from the server's plaintext DB
    println!(" got : {:?} \n want: {:?}", got, want);
    println!(
        "\nretrieved payload {item_index}      : {}",
        if got == want { "OK" } else { "MISMATCH" }
    );
    assert_eq!(got, want, "{collapse:?} failed to recover the payload");
}

fn print_layout_summary<P>(
    config: Config<[u8; 32], P>,
    layout: DatabaseLayout<P>,
    item_index: usize,
    address: poulpy_pir::database::Address,
) where
    P: Payload<[u8; 32]>,
{
    let n = config.n();
    let column_height = config.column_height();
    let num_payloads = layout.num_payloads(column_height);
    match config.collapse() {
        Collapse::Interpolation => {
            println!(
                "database                    : {} matrices of {} x {} coeffs (block_cols {})",
                layout.block_rows(n),
                n,
                layout.cols(),
                layout.block_cols(n)
            );
            println!(
                "payload capacity            : {} x 32 B = {}",
                num_payloads,
                format_bytes(layout.total_payload_bytes(column_height) as f64)
            );
            println!(
                "interpolation degree (t)    : {}",
                layout.interpolation_t(n)
            );
            println!(
                "target payload {item_index}     : matrix {}, block_col {}, col_in_block {}, row_offset {}",
                address.matrix,
                address.block_col(n),
                address.col_in_block(n),
                address.row_offset
            );
        }
        Collapse::Recursion {
            gamma0,
            gamma1,
            gamma2,
        } => {
            let t_batches = layout.grid_rows_for(gamma0);
            let cols = layout.cols();
            println!(
                "database                    : {} payloads = {} ({} batches x {} cols, γ0={})",
                num_payloads,
                format_bytes((num_payloads * 32) as f64),
                t_batches,
                cols,
                gamma0
            );
            println!(
                "record size γ0              : {} base-{} digits = {} payloads/record",
                gamma0,
                P::BASIS,
                gamma0 / P::EXPONENT
            );
            println!(
                "packing γ0 / γ1 / γ2        : {} / {} / {}",
                gamma0, gamma1, gamma2
            );
            println!("decompose digits τ          : {} (q̃ = 2^{})", 2, 32);
            println!(
                "target payload {item_index}     : batch {}, column {}, row_offset {}",
                address.matrix, address.column, address.row_offset
            );
        }
    }
}
