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
    config::{Collapse, Config, INSPIRE_INT_32B, INSPIRE_REC_32B},
    database::DatabaseLayout,
    payload::{Payload, U256P65535, U256P65536},
    server::Server,
};

/// Backend used by this driver.
type BE = FFT64Avx;

const DB_ROWS: usize = 16384;
const DB_COLS: usize = 32768;

fn main() {
    const ITEM_INDEX: usize = 5_000_000;

    if true {
        run(
            INSPIRE_INT_32B,
            DatabaseLayout::<U256P65535>::new(DB_ROWS, DB_COLS),
            ITEM_INDEX,
        );
    } else {
        run(
            INSPIRE_REC_32B,
            DatabaseLayout::<U256P65536>::new(DB_ROWS, DB_COLS),
            ITEM_INDEX,
        );
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

    // DEBUG: decrypt the first step U·(A,b) BEFORE packing and check each panel
    // against the plaintext interpolated-U column it must equal. (0, 0) = correct.
    let first_steps = server.debug_interpolation_first_step(&query, &selected);
    if !first_steps.is_empty() {
        let checks = client.debug_decrypt_first_step(&first_steps, &state);
        for (panel, (max_err, mismatches)) in checks.iter().enumerate() {
            println!(
                "  first_step[{panel}] U·(A,b) max_err / mismatches : {} / {}",
                max_err, mismatches
            );
        }
        // Same U-column reference, but checked against the PACKED panel values:
        // splits the value bug between pack (wrong here) and reduce (correct here).
        let expected_cols: Vec<Vec<i64>> = first_steps.iter().map(|(_, e)| e.clone()).collect();
        let packed_checks = client.debug_decrypt_packed_values(&response, &expected_cols, &state);
        for (panel, (max_err, mismatches)) in packed_checks.iter().enumerate() {
            println!(
                "  packed[{panel}] value max_err / mismatches    : {} / {}",
                max_err, mismatches
            );
        }

        // PLAINTEXT Horner of the (correct) U columns at root X^exp, compared to the
        // decrypted `selected`. Match ⇒ the ciphertext Horner is correct (bug is in
        // root/decode); mismatch ⇒ the GGSW×GLWE Horner itself is wrong.
        let p = client.params().p();
        let t_deg = layout.interpolation_t(n);
        let exp = selected.matrix * (2 * n / t_deg);
        let center = |x: i64| -> i64 {
            let mut r = x.rem_euclid(p);
            if r > p / 2 {
                r -= p;
            }
            r
        };
        let mut recon = vec![0i64; n];
        for (j, col) in expected_cols.iter().enumerate() {
            let e = (exp * j) % (2 * n);
            for i in 0..n {
                let pos = (i + e) % (2 * n);
                if pos < n {
                    recon[pos] = center(recon[pos] + col[i]);
                } else {
                    recon[pos - n] = center(recon[pos - n] - col[i]);
                }
            }
        }
        let decoded_selected = client.decrypt_digits(&response, &state);
        let horner_mismatch = recon
            .iter()
            .zip(&decoded_selected)
            .filter(|(a, b)| a != b)
            .count();
        println!(
            "  plaintext-horner vs decrypt(selected) (exp={exp}) mismatches : {} / {}",
            horner_mismatch, n
        );
        println!(
            "  recon[off..off+8]   : {:?}",
            &recon[selected.row_offset..(selected.row_offset + 8).min(n)]
        );
        println!(
            "  selected[off..off+8]: {:?}",
            &decoded_selected[selected.row_offset..(selected.row_offset + 8).min(n)]
        );

        // Which interpolation POINT actually reconstructs each raw DB matrix?
        // Reveals the panel/point ordering (e.g. bit-reversal) mismatch.
        let kb = layout.block_cols(n);
        let nb = layout.block_rows(n);
        let block_col = selected.block_col(n);
        let col_in_block = selected.col_in_block(n);
        let raw_cols: Vec<Vec<i64>> = (0..t_deg)
            .map(|m| {
                if m < nb {
                    (0..n)
                        .map(|r| {
                            server.database().matrices()[m * kb + block_col].row(r)[col_in_block]
                                as i64
                        })
                        .collect()
                } else {
                    vec![0i64; n]
                }
            })
            .collect();
        for m in 0..t_deg {
            let e0 = m * (2 * n / t_deg);
            let mut rec = vec![0i64; n];
            for (j, col) in expected_cols.iter().enumerate() {
                let e = (e0 * j) % (2 * n);
                for i in 0..n {
                    let pos = (i + e) % (2 * n);
                    if pos < n {
                        rec[pos] = center(rec[pos] + col[i]);
                    } else {
                        rec[pos - n] = center(rec[pos - n] - col[i]);
                    }
                }
            }
            let (best_m, best_hits) = (0..t_deg)
                .map(|mm| {
                    (
                        mm,
                        rec.iter()
                            .zip(&raw_cols[mm])
                            .filter(|(a, b)| center(**a - **b) == 0)
                            .count(),
                    )
                })
                .max_by_key(|x| x.1)
                .unwrap();
            println!(
                "  point m={m} (e0={e0}) best-matches raw matrix {best_m} : {best_hits}/{n}"
            );
        }
    }

    // DEBUG: per-panel self-noise of the packed GLWEs (pre-reduce). Healthy ~ -20;
    // ~ -1 means the panel is already garbage before the Horner reduce.
    let packed_noise = client.debug_packed_noise(&response, &state);
    let n_panels = packed_noise.len().saturating_sub(1);
    for (idx, (max_log2, std_log2)) in packed_noise.iter().enumerate() {
        let label = if idx < n_panels {
            format!("packed[{idx}]")
        } else {
            "selected   ".to_string()
        };
        println!(
            "  {label} self-noise log2(max/std) : {:.3} / {:.3}",
            max_log2, std_log2
        );
    }

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
