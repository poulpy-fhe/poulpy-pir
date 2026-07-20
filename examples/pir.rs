//! End-to-end PIR driver over the 32-byte default parameterizations.
//!
//! Runs the full round trip — setup, database fill, offline preprocessing,
//! query, answer, decrypt — verifies every recovered payload against ground
//! truth, and prints per-phase timings, wire sizes, and a noise estimate.
//!
//! ```text
//! cargo run --release --features avx512-fhe --example pir -- <preset> [batch]
//! ```
//!
//! `<preset>` is a [`DefaultPirParameters32B`] name such as
//! `InsPIRe2-g32-1GiB-c32768`; run without arguments to list them all.
//! `[batch]` is the number of queries answered together per online batch
//! (default 1).
//!
//! On a multi-socket (NUMA) host, pick the DB placement for the serving mode:
//! add `--features numa-db-interleave` when `batch` is 1 (single-query
//! latency); leave it off for batched serving. See `examples/README.md` for
//! exact command lines and the top-level README's NUMA section for the
//! rationale.

use std::time::Instant;

use poulpy_cpu_avx512::FFT64Avx512;
use poulpy_pir::{
    client::Client,
    config::{Collapse, Config, DefaultPirConfig32B, DefaultPirParameters32B},
    database::DatabaseLayout,
    payload::Payload,
    server::Server,
};

/// Backend used by this driver.
type BE = FFT64Avx512;
/// Payload index retrieved (and verified) by the first query of the batch.
const ITEM_INDEX: usize = 1_000_000;
/// Number of times the ONLINE batch is repeated; the online timings are averaged
/// over the repeats for a stable measurement. Use with `PIR_ONLINE_THREADS=1` for
/// single-thread online-phase experiments.
const REPEATS: usize = 10;

fn main() {
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

    let mut cli = std::env::args().skip(1);
    let Some(name) = cli.next() else { usage() };
    let Some(preset) = DefaultPirParameters32B::from_name(&name) else {
        eprintln!("unknown preset {name:?}\n");
        usage();
    };
    let batch: usize = match cli.next() {
        None => 1,
        Some(s) => match s.parse() {
            Ok(b) if b >= 1 => b,
            _ => {
                eprintln!("batch must be a positive integer, got {s:?}\n");
                usage();
            }
        },
    };

    println!("preset                       : {}", preset.name());
    match preset.resolve() {
        DefaultPirConfig32B::Interpolation(p) => run(p.config, p.layout, ITEM_INDEX, batch),
        DefaultPirConfig32B::Recursion(p) => run(p.config, p.layout, ITEM_INDEX, batch),
    }
}

fn usage() -> ! {
    eprintln!("usage: pir <preset> [batch]\n");
    eprintln!("  <preset>  one of the DefaultPirParameters32B names below");
    eprintln!("  [batch]   queries answered together per online batch (default 1)\n");
    eprintln!("available presets:");
    for preset in DefaultPirParameters32B::ALL {
        eprintln!("  {}", preset.name());
    }
    std::process::exit(2);
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

fn run<P>(config: Config<[u8; 32], P>, layout: DatabaseLayout<P>, item_index: usize, batch: usize)
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

    // ---- CLIENT: build `batch` queries (batch = 1 is the single-query case).
    // Items are spread across the DB so they land in different panels.
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

    // ---- SERVER: answer the batch at once via `respond_batch_timed`, repeated
    // `REPEATS` times; the ONLINE wall-clock and per-phase work are averaged over
    // the repeats for a stable measurement (with `PIR_ONLINE_THREADS=1`, the
    // single-thread online phase). The phase breakdown is *summed work* across
    // the batch, so it exceeds the wall-clock; throughput uses the wall-clock.
    let mut total_wall = std::time::Duration::ZERO;
    let mut total_work = std::time::Duration::ZERO;
    let mut phase_names: Vec<String> = Vec::new();
    let mut phase_sums: Vec<std::time::Duration> = Vec::new();
    let mut responses = Vec::new();
    for rep in 0..REPEATS {
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

    let reps = REPEATS as u32;
    let avg_wall = total_wall / reps;
    println!("ONLINE avg wall ({batch} q × {REPEATS})    : {avg_wall:?}");
    println!("ONLINE avg work (sum of phases): {:?}", total_work / reps);
    for (name, sum) in phase_names.iter().zip(&phase_sums) {
        println!("  {:<30}: {:?}", name, *sum / reps);
    }
    if batch > 1 {
        println!("  per query (wall-clock)     : {:?}", avg_wall / batch as u32);
        println!(
            "  throughput                 : {:.1} queries/s",
            batch as f64 / avg_wall.as_secs_f64()
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

    println!("RESULT                       : {ok}/{batch} decoded OK");
    if let Some(peak) = peak_rss_bytes() {
        println!("PEAK MEMORY (VmHWM)          : {}", format_bytes(peak as f64));
    }
    assert_eq!(ok, batch, "{collapse:?} decode mismatch");
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
