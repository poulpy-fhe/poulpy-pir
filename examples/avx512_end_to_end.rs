//! End-to-end PIR round-trip driven on the **AVX-512 FHE backend**
//! (`poulpy_cpu_avx512::FFT64Avx512`), with the full offline/online phase
//! breakdown (same detail as the `pir` example).
//!
//! This is a *run-site* binary: it is the only place that names a concrete
//! backend. The library stays fully backend-agnostic — the generic `Server` /
//! `Client` are instantiated here with `BE = FFT64Avx512`.
//!
//! Build/run on a host with AVX-512F (e.g. an AWS `c7i` instance):
//!
//! ```text
//! RUSTFLAGS="-C target-feature=+avx512f" \
//!   cargo run --release --features avx512-fhe --example avx512_end_to_end -- InsPIRe-1GiB-c8192
//! ```
//!
//! Args: `<name> [item_index] [batch]`. `<name>` is a [`DefaultPirParameters32B`]
//! name (default `InsPIRe-1GiB-c8192`); run with an unknown name to print the full
//! list, or e.g. `InsPIRe-32GiB-c262144` / `InsPIRe2-g64-32GiB-c131072` for the
//! large-DB run. `item_index` (default 1_000_000) is clamped to the DB's payload
//! capacity. `batch` (default 1) additionally answers that many queries at once
//! via `respond_batch` and reports batched throughput. `poulpy-cpu-avx512`
//! enforces AVX-512F at compile time, so this only *builds* on an AVX-512F host.

use std::time::Instant;

use poulpy_cpu_avx512::FFT64Avx512;
use poulpy_pir::{
    client::{Client, Response},
    config::{Collapse, Config, DefaultPirConfig32B, DefaultPirParameters32B},
    database::DatabaseLayout,
    payload::Payload,
    server::Server,
};

/// The concrete backend under test. This alias — in a binary, not the library —
/// is the one and only place the AVX-512 backend is named.
type BE = FFT64Avx512;

fn main() {
    // Friendly guard: FFT64Avx512 requires AVX-512F and will otherwise fault.
    if !std::arch::is_x86_feature_detected!("avx512f") {
        eprintln!(
            "error: this CPU does not report AVX-512F; FFT64Avx512 cannot run here.\n\
             Run this on an AVX-512F host (e.g. AWS c7i)."
        );
        std::process::exit(1);
    }

    let args: Vec<String> = std::env::args().collect();
    let name = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "InsPIRe-1GiB-c8192".to_string());
    let item_index: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1_000_000);
    // Optional batch size: answer this many queries at once via `respond_batch`.
    let batch: usize = args
        .get(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1)
        .max(1);

    let variant = DefaultPirParameters32B::from_name(&name).unwrap_or_else(|| {
        eprintln!("unknown parameter '{name}'. Available parameters:");
        for v in DefaultPirParameters32B::all() {
            eprintln!("  {}", v.name());
        }
        std::process::exit(1);
    });

    println!(
        "== AVX-512 (FFT64Avx512) end-to-end: {} ({} GiB DB) ==",
        variant.name(),
        variant.db_size_gib()
    );

    // Worker-thread budget. Mirrors `parallel::num_threads` (pub(crate), so not
    // callable here): `PIR_THREADS` overrides the detected logical-CPU count.
    // Each parallel loop then clamps this to its own work ceiling (e.g. #panels);
    // the dominant recursion loop uses the full budget (cap = usize::MAX).
    let logical_cpus = std::thread::available_parallelism()
        .map(|x| x.get())
        .unwrap_or(1);
    let pir_threads = std::env::var("PIR_THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&t| t >= 1);
    let worker_threads = pir_threads.unwrap_or(logical_cpus);
    println!(
        "worker threads               : {worker_threads}{}  (logical CPUs: {logical_cpus})\n",
        if pir_threads.is_some() {
            " [PIR_THREADS]"
        } else {
            ""
        }
    );

    match variant.resolve() {
        DefaultPirConfig32B::Interpolation(params) => {
            run(params.config, params.layout, item_index, batch)
        }
        DefaultPirConfig32B::Recursion(params) => {
            run(params.config, params.layout, item_index, batch)
        }
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

fn run<P>(config: Config<[u8; 32], P>, layout: DatabaseLayout<P>, requested_item: usize, batch: usize)
where
    P: Payload<[u8; 32]>,
{
    let n = config.n();
    let column_height = config.column_height();
    let collapse = config.collapse();

    // Clamp the requested index to the DB's payload capacity so any DB size works.
    let capacity = layout.num_payloads(column_height);
    let item_index = requested_item.min(capacity.saturating_sub(1));
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

    // ---- SERVER: query-independent preprocessing (per-phase). ----
    let off = server.offline();
    println!("OFFLINE total                : {:?}", off.total());
    for phase in off.phases() {
        println!("  {:<30}: {:?}", phase.name(), phase.duration());
    }

    // ---- CLIENT: build the query for the payload index. ----
    let t = Instant::now();
    let (query, state) = client.query(item_index);
    println!("QUERY                        : {:?}", t.elapsed());

    // ---- SERVER: answer (timed, per-phase). ----
    let (response, online): (Response<BE>, _) = server.respond_timed(&query);
    println!("ONLINE total                 : {:?}", online.total());
    for phase in online.phases() {
        println!("  {:<30}: {:?}", phase.name(), phase.duration());
    }

    // ---- WIRE SIZES: serialize the transmitted messages to measure bytes. ----
    // Both are repacked to base2k=63 by `write_to` (the real on-wire encoding).
    let module = server.params().module();
    let mut qbuf = Vec::new();
    query.write_to(module, &mut qbuf).expect("serialize query");
    let mut rbuf = Vec::new();
    response
        .write_to(module, &mut rbuf)
        .expect("serialize response");
    println!(
        "QUERY size                   : {} B ({})",
        qbuf.len(),
        format_bytes(qbuf.len() as f64)
    );
    println!(
        "RESPONSE size                : {} B ({})",
        rbuf.len(),
        format_bytes(rbuf.len() as f64)
    );

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

    // ---- BATCH: answer `batch` queries at once via `respond_batch`. For
    // interpolation each DB panel is read once for the whole batch (one i16×f64
    // GEMM), so the online cost amortizes; recursion falls back to sequential.
    if batch > 1 {
        println!("\n---- batch of {batch} queries ----");
        // Spread the items across the DB so they land in different panels.
        let stride = (capacity / batch).max(1);
        let items: Vec<usize> = (0..batch)
            .map(|k| (item_index + k * stride) % capacity)
            .collect();

        let t = Instant::now();
        let mut queries = Vec::with_capacity(batch);
        let mut states = Vec::with_capacity(batch);
        for &item in &items {
            let (q, st) = client.query(item);
            queries.push(q);
            states.push(st);
        }
        println!("QUERY (build {batch})            : {:?}", t.elapsed());

        let t = Instant::now();
        let responses = server.respond_batch(&queries);
        let batch_dt = t.elapsed();

        // Decode + verify every response against the plaintext DB.
        let mut ok = 0usize;
        for ((resp, st), &item) in responses.iter().zip(&states).zip(&items) {
            if client.decode(resp, st) == server.get(item) {
                ok += 1;
            }
        }

        println!("ONLINE batch total           : {batch_dt:?}");
        println!(
            "  per query (avg)            : {:?}",
            batch_dt / batch as u32
        );
        println!(
            "  throughput                 : {:.1} queries/s",
            batch as f64 / batch_dt.as_secs_f64()
        );
        println!("BATCH RESULT                 : {ok}/{batch} decoded OK");
        assert_eq!(ok, batch, "{collapse:?} batch decode mismatch");
    }
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
