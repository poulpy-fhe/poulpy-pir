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
//!
//! On a multi-socket (NUMA) host, pick the DB placement for the serving mode:
//! the default build is tuned for batched throughput; add
//! `--features numa-db-interleave` when optimizing single-query latency
//! (interleaves the DB pages across nodes — see the README's NUMA section).
//!
//! PIR_ONLINE_THREADS=1 PIR_OFFLINE_THREADS=64 RUSTFLAGS="-C target-feature=+avx512f,+avx512dq" cargo run --release --features "avx512-fhe, cblas-gemm, numa-db-interleave" --example pir -- recursion
//! PIR_ONLINE_THREADS=1 PIR_OFFLINE_THREADS=64 RUSTFLAGS="-C target-feature=+avx512f,+avx512dq" cargo run --release --features "avx512-fhe, cblas-gemm" --example pir -- recursion

use std::time::Instant;

use poulpy_cpu_avx512::FFT64Avx512;
use poulpy_pir::{
    client::Client,
    config::{Collapse, Config, DefaultPirConfig32B, DefaultPirParameters32B, DefaultScheme},
    database::DatabaseLayout,
    payload::Payload,
    server::Server,
};

/// Backend used by this driver.
type BE = FFT64Avx512;
const DEFAULT: DefaultPirParameters32B = DefaultPirParameters32B::canonical(DefaultScheme::Recursion { gamma0: 32 }, 1);
/// Number of queries answered together per ONLINE batch (`respond_batch_timed`).
const BATCH: usize = 1;
/// Number of times the ONLINE batch is repeated; the online timings are averaged
/// over the repeats for a stable measurement. Use with `PIR_ONLINE_THREADS=1` for
/// single-thread online-phase experiments.
const QUERIES: usize = 10;

fn main() {
    const ITEM_INDEX: usize = 1_000_000;

    // cblas-gemm: the pthread OpenBLAS spawns its worker pool in its ELF
    // constructor — before main — sized from the constructor-time environment.
    // An unsized pool (127 threads here) spin-waits through the whole run and
    // measurably slows the non-BLAS phases (~2× on the single-threaded online
    // path); the runtime `openblas_set_num_threads(1)` pin in `CblasDgemm`
    // only stops dispatch, not the spawn. The env var must therefore be set
    // before the library loads: re-exec once with it set if absent.
    #[cfg(feature = "cblas-gemm")]
    if std::env::var_os("OPENBLAS_NUM_THREADS").is_none() {
        use std::os::unix::process::CommandExt;
        let exe = std::env::current_exe().expect("current_exe for OpenBLAS re-exec");
        let err = std::process::Command::new(exe)
            .args(std::env::args_os().skip(1))
            .env("OPENBLAS_NUM_THREADS", "1")
            .exec();
        panic!("re-exec with OPENBLAS_NUM_THREADS=1 failed: {err}");
    }

    match DEFAULT.resolve() {
        DefaultPirConfig32B::Interpolation(params) => run(params.config, params.layout, ITEM_INDEX),
        DefaultPirConfig32B::Recursion(params) => {
            run(
                params.config,
                DatabaseLayout::new(32768, 1<<18),
                ITEM_INDEX,
            );
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
    #[allow(unused_mut)]
    let mut server = Server::<BE, P>::new(config, layout);
    // With `cblas-gemm`, the offline mask product runs on the system BLAS dgemm
    // instead of private-gemm-x86 (~1.5-1.7× on Granite Rapids; see Cargo.toml).
    #[cfg(feature = "cblas-gemm")]
    {
        server = server.with_gemm(poulpy_pir::server::CblasDgemm);
        println!("gemm backend                 : CblasDgemm (system BLAS)");
    }
    let setup = timer.elapsed();
    println!("SETUP                        : {:?}", setup);

    // ---- SERVER: fill with pseudorandom 256-bit payloads. Content is derived
    // from the payload index (splitmix64), so runs are reproducible.
    //
    // We use one large chunk buffer per pass: the workers parallel-fill disjoint
    // sub-slices of it, then a single `update_shard` scatters the whole chunk
    // into the coefficient matrices (itself parallel over the physical matrices
    // the chunk touches — see `Database::encode_shard`). A large chunk keeps
    // each scatter spanning many matrices (full parallelism) and holds the
    // number of `update_shard` calls — and thus the thread-pool spawn overhead —
    // to a few hundred rather than tens of thousands.
    let t = Instant::now();
    let capacity = layout.num_payloads(column_height);
    let workers = std::thread::available_parallelism().map_or(1, |x| x.get());
    let chunk = (1usize << 22).min(capacity.max(1)); // 4 Mi payloads (~128 MiB)
    let mut buf = vec![[0u8; 32]; chunk];
    let mut start = 0;
    while start < capacity {
        let len = chunk.min(capacity - start);
        let sub = len.div_ceil(workers);
        std::thread::scope(|scope| {
            for (w, part) in buf[..len].chunks_mut(sub).enumerate() {
                let first = start + w * sub;
                scope.spawn(move || fill_payloads(part, first));
            }
        });
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

    // ---- CLIENT: build `BATCH` queries (BATCH = 1 is the single-query case).
    // Items are spread across the DB so they land in different panels.
    let stride = (capacity / BATCH).max(1);
    let items: Vec<usize> = (0..BATCH)
        .map(|k| (item_index + k * stride) % capacity)
        .collect();

    let t = Instant::now();
    let mut queries = Vec::with_capacity(BATCH);
    let mut states = Vec::with_capacity(BATCH);
    for &item in &items {
        let (q, st) = client.query(item);
        queries.push(q);
        states.push(st);
    }
    println!("QUERY (build {BATCH})            : {:?}", t.elapsed());

    // ---- SERVER: answer the `BATCH` at once via `respond_batch_timed`, repeated
    // `QUERIES` times; the ONLINE wall-clock and per-phase work are averaged over
    // the repeats for a stable measurement (with `PIR_ONLINE_THREADS=1`, the
    // single-thread online phase). The phase breakdown is *summed work* across
    // the batch, so it exceeds the wall-clock; throughput uses the wall-clock.
    let mut total_wall = std::time::Duration::ZERO;
    let mut total_work = std::time::Duration::ZERO;
    let mut phase_names: Vec<String> = Vec::new();
    let mut phase_sums: Vec<std::time::Duration> = Vec::new();
    let mut responses = Vec::new();
    for rep in 0..QUERIES {
        let started = Instant::now();
        let (resps, online) = server.respond_batch_timed(&queries);
        total_wall += started.elapsed();
        total_work += online.total();
        for (i, phase) in online.phases().iter().enumerate() {
            if rep == 0 {
                phase_names.push(phase.name().to_string());
                phase_sums.push(std::time::Duration::ZERO);
            }
            phase_sums[i] += phase.duration();
        }
        responses = resps; // keep the last run's responses for verification
    }

    let n = QUERIES as u32;
    let avg_wall = total_wall / n;
    println!("ONLINE avg wall ({BATCH} q × {QUERIES})    : {avg_wall:?}");
    println!("ONLINE avg work (sum of phases): {:?}", total_work / n);
    for (name, sum) in phase_names.iter().zip(&phase_sums) {
        println!("  {:<30}: {:?}", name, *sum / n);
    }
    if BATCH > 1 {
        println!("  per query (wall-clock)     : {:?}", avg_wall / BATCH as u32);
        println!(
            "  throughput                 : {:.1} queries/s",
            BATCH as f64 / avg_wall.as_secs_f64()
        );
    }

    // ---- WIRE SIZES: serialize the first query/response (representative). Both
    // are repacked to base2k=63 by `write_to` (the real on-wire encoding). ----
    let module = server.params().module();
    let mut qbuf = Vec::new();
    queries[0]
        .write_to(module, &mut qbuf)
        .expect("serialize query");
    let mut rbuf = Vec::new();
    responses[0]
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

    // ---- CLIENT: decrypt + verify every response against the plaintext DB. ----
    let mut ok = 0usize;
    for ((resp, st), &item) in responses.iter().zip(&states).zip(&items) {
        if client.decode(resp, st) == server.get(item) {
            ok += 1;
        }
    }

    // Noise estimate on the first response.
    let selected = states[0].address();
    let expected_record = server.database().record(selected.column, selected.matrix);
    let noise = client.noise(&responses[0], &states[0], &expected_record);
    println!("NOISE log2(max)              : {:.3}", noise.max_log2());
    println!("NOISE log2(std)              : {:.3}", noise.std_log2());

    println!("RESULT                       : {ok}/{BATCH} decoded OK");
    if let Some(peak) = peak_rss_bytes() {
        println!("PEAK MEMORY (VmHWM)          : {}", format_bytes(peak as f64));
    }
    assert_eq!(ok, BATCH, "{collapse:?} decode mismatch");
}

/// Peak resident set size (high-water mark) of this process, in bytes, read from
/// `VmHWM` in `/proc/self/status`. Returns `None` off Linux or if unreadable.
fn peak_rss_bytes() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            // Format: "VmHWM:\t   12345 kB"
            let kib: u64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kib * 1024);
        }
    }
    None
}

/// Fill `out` with payloads whose content depends only on the payload index
/// (splitmix64 over `index * 4 + word`), so the DB is identical across runs
/// and thread counts.
fn fill_payloads(out: &mut [[u8; 32]], first_index: usize) {
    for (i, payload) in out.iter_mut().enumerate() {
        let index = (first_index + i) as u64;
        for word in 0..4u64 {
            let mut x = (index * 4 + word).wrapping_add(0x9e3779b97f4a7c15);
            x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
            x ^= x >> 31;
            payload[word as usize * 8..][..8].copy_from_slice(&x.to_le_bytes());
        }
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
