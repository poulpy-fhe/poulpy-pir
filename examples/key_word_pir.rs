//! Keyword PIR driver: retrieve a 64-byte record — the full ETH address plus
//! its full 256-bit token balance — with a **single** PIR query per lookup.
//!
//! The keyword layer (`poulpy_pir::keyword`) contributes exactly one thing
//! here: a minimal perfect hash function (MPHF, ~2.1 bits/key) mapping each of
//! the 16 M addresses to a record number `r ∈ [0, 16 M)`. Everything else —
//! the record layout and its verification — is this application's own scheme.
//!
//! ## One query, 64 bytes
//!
//! A query encrypts only the *column* selector; the row offset within the
//! selected column never leaves the client. The response therefore carries the
//! entire logical column — `γ0 × 16` bits (64 B at `γ0 = 32`) for InsPIRe²,
//! `n` digits for InsPIRe — and `Client::decrypt_digits` exposes all of it.
//! So a 64-byte record costs one query as long as both of its 32-byte payload
//! slots live in the *same* column. Payload indices are column-minor
//! (`column = index % cols`), so the co-located pair is `i` and `i + cols`,
//! not `i` and `i + 1`:
//!
//! ```text
//!   slot base(r)         address, zero-padded to 32 B   (row offset 0)
//!   slot base(r) + cols  token balance, little-endian u256 (row offset 16)
//! ```
//!
//! The address slot doubles as the in-set check: an address the MPHF never saw
//! still resolves to *some* valid record, but that record's address slot
//! cannot match the queried address, so the client reports "not in set"
//! instead of trusting an unrelated balance. (Storing the full address rather
//! than a hash tag makes the check exact.)
//!
//! ```text
//! cargo run --release --example key_word_pir [-- <preset> [batch]]
//! ```
//!
//! - `<preset>` — a [`DefaultPirParameters32B`] name (default
//!   `InsPIRe2-g32-1GiB-c32768`); any 1 GiB-or-larger preset whose column
//!   holds a whole record (`γ0 ≥ 32` for InsPIRe²; any InsPIRe preset).
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
    config::{Config, DefaultPirConfig32B, DefaultPirParameters32B},
    database::DatabaseLayout,
    keyword::{EthAddress, KeywordIndex},
    payload::Payload,
    server::Server,
};

/// Backend used by this driver: `FFT64Avx512` when built with `avx512-fhe`,
/// `FFT64Avx` otherwise.
#[cfg(feature = "avx512-fhe")]
type BE = poulpy_cpu_avx512::FFT64Avx512;
#[cfg(not(feature = "avx512-fhe"))]
type BE = poulpy_cpu_avx::FFT64Avx;

/// Payload slots per record: one for the address, one for the balance.
const SLOTS_PER_RECORD: usize = 2;
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
    match preset.resolve() {
        DefaultPirConfig32B::Interpolation(p) => run(p.config, p.layout, batch),
        DefaultPirConfig32B::Recursion(p) => run(p.config, p.layout, batch),
    }
}

fn usage() -> ! {
    eprintln!("usage: key_word_pir [preset] [batch]\n");
    eprintln!("  [preset]  one of the DefaultPirParameters32B names below (default {DEFAULT_PRESET})");
    eprintln!("  [batch]   address lookups answered together per online batch (default 1)\n");
    eprintln!("available presets:");
    for preset in DefaultPirParameters32B::ALL {
        eprintln!("  {}", preset.name());
    }
    std::process::exit(2);
}

/// Maps record number `r` to the payload index of its *first* slot; slot `k`
/// of the record is `record_base(..) + k * cols`, which `address_for` places
/// in the same `(matrix, column)` at row offset `k · P::EXPONENT` past the
/// first slot — the co-location that makes one query retrieve the whole
/// record.
fn record_base(r: usize, cols: usize, payloads_per_column: usize) -> usize {
    let records_per_column = payloads_per_column / SLOTS_PER_RECORD;
    let per_band = cols * records_per_column;
    let band = r / per_band;
    let sub = (r % per_band) / cols;
    let col = r % cols;
    band * payloads_per_column * cols + (SLOTS_PER_RECORD * sub) * cols + col
}

fn run<P>(config: Config<[u8; 32], P>, layout: DatabaseLayout<P>, batch: usize)
where
    P: Payload<[u8; 32]>,
{
    let column_height = config.column_height();
    let cols = layout.cols();
    let payloads_per_column = layout.payloads_per_column(column_height);
    assert!(
        payloads_per_column >= SLOTS_PER_RECORD,
        "one column holds {payloads_per_column} payload slot(s), a 64 B record needs \
         {SLOTS_PER_RECORD} (pick γ0 ≥ 32 for InsPIRe² presets)"
    );
    let records_per_column = payloads_per_column / SLOTS_PER_RECORD;
    let record_capacity = layout.num_records(column_height) * records_per_column;
    println!("collapse                     : {:?}", config.collapse());
    println!("ring degree n                : {}", config.n());
    println!(
        "record capacity              : {record_capacity} x 64 B ({} payload slots)\n",
        layout.num_payloads(column_height)
    );
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
    // job that fixes every account's record number.
    let t = Instant::now();
    let index = KeywordIndex::build(&keys).expect("MPHF construction");
    println!("MPHF build                   : {:?}", t.elapsed());

    // ---- WIRE: the MPHF parameters are all a client ever downloads — never
    // the key set. Round-trip through the real serialization to model that.
    let mut blob = Vec::new();
    index.write_to(&mut blob).expect("serialize MPHF");
    let client_index = KeywordIndex::read_from(&mut blob.as_slice()).expect("deserialize MPHF");
    println!(
        "MPHF parameters              : {} B ({:.3} bits/key)",
        blob.len(),
        blob.len() as f64 * 8.0 / NUM_ADDRESSES as f64
    );

    // ---- SERVER: populate the database. Record `r = MPHF(address)` occupies
    // the same-column slot pair: the address itself, then its little-endian
    // u256 balance one payload row below (`+ cols`). The vector spans whole
    // bands so the trailing partially-used band is still addressable.
    let t = Instant::now();
    let bands_used = NUM_ADDRESSES.div_ceil(cols * records_per_column);
    let mut payloads = vec![[0u8; 32]; bands_used * payloads_per_column * cols];
    for key in &keys {
        let base = record_base(index.index(key), cols, payloads_per_column);
        payloads[base] = address_slot(key);
        payloads[base + cols] = balance_of(key);
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
    // record's column and costs exactly one query.
    let stride = (NUM_ADDRESSES / batch).max(1);
    let targets: Vec<EthAddress> = (0..batch)
        .map(|k| keys[(QUERIED_ADDRESS + k * stride) % NUM_ADDRESSES])
        .collect();

    let t = Instant::now();
    let mut queries = Vec::with_capacity(batch);
    let mut states = Vec::with_capacity(batch);
    for target in &targets {
        let base = record_base(client_index.index(target), cols, payloads_per_column);
        let (q, st) = client.query(base);
        queries.push(q);
        states.push(st);
    }
    println!("QUERY build ({batch})              : {:?}", t.elapsed());
    println!("queried address              : 0x{}", hex(&targets[0]));
    println!("MPHF record index            : {}", client_index.index(&targets[0]));

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
    queries[0].write_to(module, &mut qbuf).expect("serialize query");
    let mut rbuf = Vec::new();
    responses[0].write_to(module, &mut rbuf).expect("serialize response");
    println!("QUERY size                   : {} B", qbuf.len());
    println!("RESPONSE size                : {} B", rbuf.len());

    // ---- CLIENT: decrypt each response once and slice both record halves out
    // of the recovered column digits. The address half must equal the queried
    // address exactly — that is the in-set check — and the balance half must
    // match ground truth.
    let mut ok = 0;
    for (target, (response, state)) in targets.iter().zip(responses.iter().zip(&states)) {
        let [address, balance] = decode_record::<P>(&mut client, response, state);
        if address == address_slot(target) && balance == balance_of(target) {
            ok += 1;
        }
    }
    let first = balance_of(&targets[0]);
    println!("balance                      : {} (LE storage, verified)", u256_hex(&first));

    // ---- CLIENT: an address the MPHF never saw. It still resolves to a valid
    // record and the PIR round trip proceeds identically — but the retrieved
    // address slot belongs to some other account, so the mismatch exposes it
    // and the lookup reports "not in set" instead of a stranger's balance.
    let stranger = address_at((NUM_ADDRESSES + 42) as u64);
    let base = record_base(client_index.index(&stranger), cols, payloads_per_column);
    let (q, st) = client.query(base);
    let response = server.respond(&q);
    let [address, _balance] = decode_record::<P>(&mut client, &response, &st);
    assert_ne!(address, address_slot(&stranger), "out-of-set address must not verify");
    println!("out-of-set address 0x{}..    : rejected (address slot mismatch)", hex(&stranger[..4]));

    if let Some(peak) = peak_rss_bytes() {
        println!("PEAK MEMORY (VmHWM)          : {:.3} GiB", peak as f64 / (1u64 << 30) as f64);
    }
    println!("RESULT                       : {ok}/{batch} lookups OK");
    assert_eq!(ok, batch, "lookup verification mismatch");
}

/// Decrypts one response and extracts the record's two 32-byte halves from the
/// recovered column digits: the query's own slot at the state's row offset,
/// then the slot one payload row below it (`+ P::EXPONENT` digits).
fn decode_record<P>(
    client: &mut Client<BE, P>,
    response: &poulpy_pir::client::Response<BE>,
    state: &poulpy_pir::client::QueryState<BE>,
) -> [[u8; 32]; SLOTS_PER_RECORD]
where
    P: Payload<[u8; 32]>,
{
    let digits = client.decrypt_digits(response, state);
    let base = state.address().row_offset;
    let mut out = [[0u8; 32]; SLOTS_PER_RECORD];
    for (k, half) in out.iter_mut().enumerate() {
        let start = base + k * P::EXPONENT;
        let slice: Vec<i16> = digits[start..start + P::EXPONENT]
            .iter()
            .map(|&v| v as i16)
            .collect();
        P::decode(half, &slice);
    }
    out
}

/// The address's 32-byte payload slot: the 20 address bytes, zero-padded.
fn address_slot(key: &EthAddress) -> [u8; 32] {
    let mut slot = [0u8; 32];
    slot[..20].copy_from_slice(key);
    slot
}

/// The `i`-th deterministic pseudo-random address (splitmix64 chain, as in the
/// keyword module's tests). The first 8 bytes are injective in `i`, so the
/// generated set is duplicate-free by construction.
fn address_at(i: u64) -> EthAddress {
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
fn addresses(count: usize) -> Vec<EthAddress> {
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
fn balance_of(key: &EthAddress) -> [u8; 32] {
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
