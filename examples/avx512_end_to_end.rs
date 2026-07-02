//! End-to-end PIR round-trip driven on the **AVX-512 FHE backend**
//! (`poulpy_cpu_avx512::FFT64Avx512`).
//!
//! This is a *run-site* binary: it is the only place that names a concrete
//! backend. The library stays fully backend-agnostic — the generic `Server` /
//! `Client` are instantiated here with `BE = FFT64Avx512`.
//!
//! Build/run on a host with AVX-512F (e.g. an AWS `c7i` instance):
//!
//! ```text
//! RUSTFLAGS="-C target-feature=+avx512f" \
//!   cargo run --release --features avx512-fhe --example avx512_end_to_end -- InspireInt1GiB
//! ```
//!
//! The argument names a [`DefaultPirParameters32B`] variant (default
//! `InspireInt1GiB`); pass any of the 24 variants, e.g. `InspireInt32GiB` or
//! `InspireRecGamma64_32GiB`, for the large-DB end-to-end run. `poulpy-cpu-avx512`
//! enforces AVX-512F at compile time, so this example only *builds* on an
//! AVX-512F host; the flag above is required.

use std::time::Instant;

use poulpy_cpu_avx512::FFT64Avx512;
use poulpy_pir::{
    client::Client,
    config::{DefaultPirInterpolationParams32B, DefaultPirParameters32B, DefaultPirRecursionParams32B},
    payload::{U256P65535, U256P65536},
    server::Server,
};

/// The concrete backend under test. This alias — in a binary, not the library —
/// is the one and only place the AVX-512 backend is named.
type BE = FFT64Avx512;

fn deterministic_payload() -> [u8; 32] {
    let mut payload = [0u8; 32];
    for (i, b) in payload.iter_mut().enumerate() {
        *b = (i as u8).wrapping_mul(7).wrapping_add(3);
    }
    payload
}

fn main() {
    // Friendly guard: FFT64Avx512 requires AVX-512F and will otherwise fault.
    if !std::arch::is_x86_feature_detected!("avx512f") {
        eprintln!(
            "error: this CPU does not report AVX-512F; FFT64Avx512 cannot run here.\n\
             Run this on an AVX-512F host (e.g. AWS c7i)."
        );
        std::process::exit(1);
    }

    let name = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "InspireInt1GiB".to_string());

    let variant = DefaultPirParameters32B::ALL
        .into_iter()
        .find(|v| format!("{v:?}") == name)
        .unwrap_or_else(|| {
            eprintln!("unknown parameter variant '{name}'. Available variants:");
            for v in DefaultPirParameters32B::ALL {
                eprintln!("  {v:?}");
            }
            std::process::exit(1);
        });

    println!(
        "== AVX-512 (FFT64Avx512) end-to-end: {variant:?} ({} GiB DB) ==",
        variant.db_size_gib()
    );

    if let Some(params) = variant.interpolation() {
        run_interpolation(params);
    } else if let Some(params) = variant.recursion() {
        run_recursion(params);
    } else {
        unreachable!("every DefaultPirParameters32B variant resolves to one construction");
    }
}

/// InsPIRe (interpolation collapse), payload type `U256P65535`.
fn run_interpolation(params: DefaultPirInterpolationParams32B) {
    let config = params.config;
    let layout = params.layout;
    let n = config.n();

    // Pick a valid retrieval index (interpolation payload granularity is `n`).
    let capacity = layout.num_payloads(n);
    let item = 300_000usize.min(capacity.saturating_sub(1));
    let payload = deterministic_payload();

    let mut server = Server::<BE, U256P65535>::new(config, layout);
    server.update_shard(item, &[payload]);

    let t = Instant::now();
    let _ = server.offline();
    let offline_dt = t.elapsed();

    let mut client = Client::<BE, U256P65535>::new(config, layout);
    let (query, state) = client.query(item);

    let t = Instant::now();
    let response = server.respond(&query);
    let respond_dt = t.elapsed();

    let got = client.decode(&response, &state);
    assert_eq!(got, payload, "interpolation round-trip mismatch at item {item}");

    report("InsPIRe (interpolation)", item, offline_dt, respond_dt);
}

/// InsPIRe² (recursion collapse), payload type `U256P65536`.
fn run_recursion(params: DefaultPirRecursionParams32B) {
    let config = params.config;
    let layout = params.layout;

    // Recursion payload granularity is `gamma0`; last index is always valid.
    let item = layout.num_payloads(params.gamma0).saturating_sub(1);
    let payload = deterministic_payload();

    let mut server = Server::<BE, U256P65536>::new(config, layout);
    server.update_shard(item, &[payload]);

    let t = Instant::now();
    let _ = server.offline();
    let offline_dt = t.elapsed();

    let mut client = Client::<BE, U256P65536>::new(config, layout);
    let (query, state) = client.query(item);

    let t = Instant::now();
    let response = server.respond(&query);
    let respond_dt = t.elapsed();

    let got = client.decode(&response, &state);
    assert_eq!(got, payload, "recursion round-trip mismatch at item {item}");

    report(
        &format!(
            "InsPIRe² (recursion γ0={}, γ1={}, γ2={})",
            params.gamma0, params.gamma1, params.gamma2
        ),
        item,
        offline_dt,
        respond_dt,
    );
}

fn report(construction: &str, item: usize, offline_dt: std::time::Duration, respond_dt: std::time::Duration) {
    println!("  construction : {construction}");
    println!("  item         : {item}");
    println!("  offline      : {:.3?}", offline_dt);
    println!("  respond      : {:.3?}", respond_dt);
    println!("  RESULT       : OK (decoded payload matches)");
}
