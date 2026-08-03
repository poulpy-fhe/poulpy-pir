//! Keyword PIR driver: retrieve a 64-byte record — the full ETH address plus
//! its full 256-bit token balance — with a single PIR query per lookup.
//!
//! The keyword layer (`poulpy_pir::keyword`) contributes exactly one thing
//! here: a minimal perfect hash function (MPHF, ~2.1 bits/key) mapping each of
//! the 16 M addresses to a record index `i ∈ [0, 16 M)`. The module is generic
//! over `[u8; N]` keys and knows nothing about payloads; this application
//! instantiates it at `N = 20` and stores each record as **one**
//! [`U512P65536`] payload:
//!
//! ```text
//!   payload[0..32]   address, zero-padded to 32 B
//!   payload[32..64]  token balance, little-endian u256
//! ```
//!
//! With one payload per record the MPHF index is the payload index — no
//! placement math — and `Client::decode` returns the whole `[u8; 64]` block
//! from one response.
//!
//! The address half doubles as the in-set check: an address the MPHF never saw
//! still resolves to *some* valid record, but that record's address bytes
//! cannot match the queried address, so the client reports "not in set"
//! instead of trusting an unrelated balance. (Storing the full address rather
//! than a hash tag makes the check exact.)
//!
//! ```text
//! cargo run --release --example key_word_pir [-- <preset> [batch]]
//! ```
//!
//! - `<preset>` — an InsPIRe² [`DefaultPirParameters32B`] name with `γ0 ≥ 32`
//!   (default `InsPIRe2-g32-1GiB-c32768`). The preset supplies the database
//!   geometry and collapse; the payload type is this example's `U512P65536`,
//!   so the byte capacity is identical, grouped as 64 B records.
//! - `[batch]` — address lookups (one query each) answered together per
//!   online batch (default 1).
//!
//! Thread counts default to the logical-CPU count; override per phase with
//! `PIR_THREADS` (base), `PIR_SETUP_THREADS`, `PIR_OFFLINE_THREADS`,
//! `PIR_ONLINE_THREADS` — see `examples/README.md`.
//!
//! The backend is `FFT64Avx` (AVX2/FMA) by default; build with
//! `--features avx512-fhe` (and `RUSTFLAGS="-C target-feature=+avx512f"`) to
//! run on `FFT64Avx512` instead.

use std::time::Instant;

use poulpy_pir::{
    client::Client,
    config::{Config, DefaultPirParameters32B, DefaultScheme},
    database::DatabaseLayout,
    keyword::KeywordIndex,
    payload::{Payload, U512P65536},
    server::Server,
};

/// Backend used by this driver: `FFT64Avx512` when built with `avx512-fhe`,
/// `FFT64Avx` otherwise.
#[cfg(feature = "avx512-fhe")]
type BE = poulpy_cpu_avx512::FFT64Avx512;
#[cfg(not(feature = "avx512-fhe"))]
type BE = poulpy_cpu_avx::FFT64Avx;

/// The payload codec: one 64-byte block = (32 B address, 32 B balance).
type P = U512P65536;
/// The application key: a 20-byte ETH address.
type Address = [u8; 20];

/// Size of the key set the MPHF is built over.
const NUM_ADDRESSES: usize = 16_000_000;
/// Which of the 16 M addresses the first lookup of the batch retrieves.
const QUERIED_ADDRESS: usize = 1_000_000;
/// Preset used when none is given on the command line.
const DEFAULT_PRESET: &str = "InsPIRe2-g32-1GiB-c32768";

fn main() {
    // See examples/pir.rs: the pthread OpenBLAS sizes its worker pool before
    // main from the environment, so the env var must be set via re-exec.
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
    let name = cli.next().unwrap_or_else(|| DEFAULT_PRESET.to_string());
    let Some(preset) = DefaultPirParameters32B::from_name(&name) else {
        eprintln!("unknown preset {name:?}\n");
        usage();
    };
    // `P::EXPONENT` = 32 digits: a 64 B record needs a recursion column of at
    // least γ0 = 32 (interpolation would need a base-65535 codec for 64 B,
    // which the library does not provide).
    match preset.scheme {
        DefaultScheme::Recursion { gamma0 } if gamma0 >= P::EXPONENT => {}
        _ => {
            eprintln!("preset {name:?} cannot hold a 64 B record per column\n");
            usage();
        }
    }
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
    println!("addresses                    : {NUM_ADDRESSES} (64 B records, 1 query each)");
    let config = Config::<P>::with_collapse(preset.collapse());
    let layout = DatabaseLayout::<P>::new(preset.rows(), preset.cols());
    run(config, layout, batch);
}

fn usage() -> ! {
    eprintln!("usage: key_word_pir [preset] [batch]\n");
    eprintln!(
        "  [preset]  an InsPIRe² preset with γ0 ≥ 32 from the list below (default {DEFAULT_PRESET})"
    );
    eprintln!("  [batch]   address lookups answered together per online batch (default 1)\n");
    eprintln!("available presets:");
    for preset in DefaultPirParameters32B::ALL {
        if matches!(preset.scheme, DefaultScheme::Recursion { gamma0 } if gamma0 >= P::EXPONENT) {
            eprintln!("  {}", preset.name());
        }
    }
    std::process::exit(2);
}

fn run(config: Config<P>, layout: DatabaseLayout<P>, batch: usize) {
    let column_height = config.column_height();
    let record_capacity = layout.num_payloads(column_height);
    println!("collapse                     : {:?}", config.collapse());
    println!("ring degree n                : {}", config.n());
    println!("record capacity              : {record_capacity} x 64 B\n");
    assert!(
        NUM_ADDRESSES <= record_capacity,
        "{NUM_ADDRESSES} records exceed the database capacity of {record_capacity}"
    );

    // ---- SETUP: client and server instantiate from the shared config/layout. ----
    let timer = Instant::now();
    let mut client = Client::<BE, P>::new(config, layout);
    #[allow(unused_mut)]
    let mut server = Server::<BE, P>::new(config, layout);
    #[cfg(feature = "cblas-gemm")]
    {
        server = server.with_gemm(poulpy_pir::server::CblasDgemm);
        println!("gemm backend                 : CblasDgemm (system BLAS)");
    }
    println!("SETUP                        : {:?}", timer.elapsed());

    // ---- SERVER: the key universe — 16 M deterministic pseudo-random ETH
    // addresses (splitmix64, so runs are reproducible). In a real deployment
    // this is the set of accounts holding the token.
    let t = Instant::now();
    let keys = addresses(NUM_ADDRESSES);
    println!("address generation           : {:?}", t.elapsed());

    // ---- SERVER: derive the MPHF over the whole key set. This is the batch
    // job that fixes every account's record index.
    let t = Instant::now();
    let index = KeywordIndex::build(&keys).expect("MPHF construction");
    println!("MPHF build                   : {:?}", t.elapsed());

    // ---- WIRE: the MPHF parameters are all a client ever downloads — never
    // the key set. Round-trip through the real serialization to model that.
    let mut blob = Vec::new();
    index.write_to(&mut blob).expect("serialize MPHF");
    let client_index =
        KeywordIndex::<20>::read_from(&mut blob.as_slice()).expect("deserialize MPHF");
    println!(
        "MPHF parameters              : {} B ({:.3} bits/key)",
        blob.len(),
        blob.len() as f64 * 8.0 / NUM_ADDRESSES as f64
    );

    // ---- SERVER: populate the database. The MPHF is minimal, so its indices
    // are exactly [0, 16 M) and each index is the record's payload index —
    // no placement math, `DB[index(key)] = record`.
    let t = Instant::now();
    let mut payloads = vec![[0u8; 64]; NUM_ADDRESSES];
    for key in &keys {
        payloads[index.index(key)] = record_of(key);
    }
    server.update_shard(0, &payloads);
    drop(payloads);
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

    // ---- CLIENT: `batch` lookups, spread across the key set so they land in
    // different panels. Each address resolves locally through the MPHF to its
    // record index and costs exactly one query.
    let stride = (NUM_ADDRESSES / batch).max(1);
    let targets: Vec<Address> = (0..batch)
        .map(|k| keys[(QUERIED_ADDRESS + k * stride) % NUM_ADDRESSES])
        .collect();

    let t = Instant::now();
    let mut queries = Vec::with_capacity(batch);
    let mut states = Vec::with_capacity(batch);
    for target in &targets {
        let (q, st) = client.query(client_index.index(target));
        queries.push(q);
        states.push(st);
    }
    println!("QUERY build ({batch})              : {:?}", t.elapsed());
    println!("queried address              : 0x{}", hex(&targets[0]));
    println!(
        "MPHF record index            : {}",
        client_index.index(&targets[0])
    );

    // ---- SERVER: answer the whole batch at once. ----
    let started = Instant::now();
    let (responses, online) = server.respond_batch_timed(&queries);
    let wall = started.elapsed();
    println!("ONLINE wall ({batch} queries)      : {wall:?}");
    println!("ONLINE work (sum of phases)  : {:?}", online.total());
    for phase in online.phases() {
        println!("  {:<30}: {:?}", phase.name(), phase.duration());
    }
    if batch > 1 {
        println!("  per lookup (wall-clock)     : {:?}", wall / batch as u32);
        println!(
            "  throughput                  : {:.1} lookups/s",
            batch as f64 / wall.as_secs_f64()
        );
    }

    // ---- WIRE SIZES: the real on-wire encodings (repacked to base2k=63). One
    // query and one response per lookup. ----
    let module = server.params().module();
    let mut qbuf = Vec::new();
    queries[0]
        .write_to(module, &mut qbuf)
        .expect("serialize query");
    let mut rbuf = Vec::new();
    responses[0]
        .write_to(module, &mut rbuf)
        .expect("serialize response");
    println!("QUERY size                   : {} B", qbuf.len());
    println!("RESPONSE size                : {} B", rbuf.len());

    // ---- CLIENT: decode each response into its 64 B record. The address half
    // must equal the queried address exactly — that is the in-set check — and
    // the balance half must match ground truth.
    let mut ok = 0;
    for (target, (response, state)) in targets.iter().zip(responses.iter().zip(&states)) {
        let record = client.decode(response, state);
        if record == record_of(target) {
            ok += 1;
        }
    }
    let first = record_of(&targets[0]);
    println!(
        "balance                      : {} (LE storage, verified)",
        u256_hex(first[32..].try_into().unwrap())
    );

    // ---- CLIENT: an address the MPHF never saw. It still resolves to a valid
    // record index and the PIR round trip proceeds identically — but the
    // retrieved record's address bytes belong to some other account, so the
    // mismatch exposes it and the lookup reports "not in set" instead of a
    // stranger's balance.
    let stranger = address_at((NUM_ADDRESSES + 42) as u64);
    let (q, st) = client.query(client_index.index(&stranger));
    let response = server.respond(&q);
    let record = client.decode(&response, &st);
    assert_ne!(record[..20], stranger, "out-of-set address must not verify");
    println!(
        "out-of-set address 0x{}..    : rejected (address mismatch)",
        hex(&stranger[..4])
    );

    if let Some(peak) = peak_rss_bytes() {
        println!(
            "PEAK MEMORY (VmHWM)          : {:.3} GiB",
            peak as f64 / (1u64 << 30) as f64
        );
    }
    println!("RESULT                       : {ok}/{batch} lookups OK");
    assert_eq!(ok, batch, "lookup verification mismatch");
}

/// The 64-byte record of an address: the address zero-padded to 32 bytes,
/// followed by its little-endian `u256` token balance.
fn record_of(key: &Address) -> [u8; 64] {
    let mut record = [0u8; 64];
    record[..20].copy_from_slice(key);
    record[32..].copy_from_slice(&balance_of(key));
    record
}

/// The `i`-th deterministic pseudo-random address (splitmix64 chain, as in the
/// keyword module's tests). The first 8 bytes are injective in `i`, so the
/// generated set is duplicate-free by construction.
fn address_at(i: u64) -> Address {
    let mut key = [0u8; 20];
    let mut z = i.wrapping_add(0x9e3779b97f4a7c15);
    for chunk in key.chunks_mut(8) {
        z = z.wrapping_add(0x9e3779b97f4a7c15);
        let x = splitmix64(z);
        chunk.copy_from_slice(&x.to_le_bytes()[..chunk.len()]);
    }
    key
}

/// The first `count` addresses of [`address_at`], generated in parallel.
fn addresses(count: usize) -> Vec<Address> {
    let mut keys = vec![[0u8; 20]; count];
    let workers = std::thread::available_parallelism().map_or(1, |x| x.get());
    let chunk = count.div_ceil(workers).max(1);
    std::thread::scope(|scope| {
        for (w, part) in keys.chunks_mut(chunk).enumerate() {
            let first = w * chunk;
            scope.spawn(move || {
                for (i, key) in part.iter_mut().enumerate() {
                    *key = address_at((first + i) as u64);
                }
            });
        }
    });
    keys
}

/// The token balance of an address: a deterministic pseudo-random full `u256`,
/// **little-endian** (byte 0 is least significant), derived from the address
/// itself so ground truth is recomputable at verification time.
fn balance_of(key: &Address) -> [u8; 32] {
    let mut z = u64::from_le_bytes(key[..8].try_into().unwrap());
    let mut value = [0u8; 32];
    for word in 0..4 {
        z = z.wrapping_add(0x9e3779b97f4a7c15);
        value[word * 8..][..8].copy_from_slice(&splitmix64(z).to_le_bytes());
    }
    value
}

fn splitmix64(z: u64) -> u64 {
    let mut x = z;
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Displays a little-endian `u256` as conventional big-endian `0x…` hex.
fn u256_hex(le: &[u8; 32]) -> String {
    let be: Vec<u8> = le.iter().rev().copied().collect();
    format!("0x{}", hex(&be))
}

/// Peak resident set size (high-water mark) of this process, in bytes, read
/// from `VmHWM` in `/proc/self/status`. Returns `None` off Linux or if unreadable.
fn peak_rss_bytes() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            let kib: u64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kib * 1024);
        }
    }
    None
}
