//! Full InsPIRe² double-PIR correctness and the public-API round-trips.

use crate::config::{Collapse, Config, DEFAULT_BASE2K, DEFAULT_K};
use crate::packing::{
    Packing, PackingKeysGenerate, PackingMaskAggregation,
    recursion::{decompose_digits, partial_pack_batch, qtilde_glwe_layout},
};
use poulpy_core::{
    EncryptionLayout, GLWECompressedEncryptSk, GLWEDecrypt, GLWEExpandLWEMatrix, GLWENormalize,
    LWEMatrixDecrypt,
    layouts::{
        Base2K, Degree, GLWEAutomorphismKeyLayout, GLWEDecompress, GLWELayout,
        GLWESecretPreparedFactory, LWEInfos, LWEMatrixLayout, LWESecret, ModuleCoreAlloc,
        ModuleCoreCompressedAlloc, Rank, SecretConversion, TorusPrecision,
    },
};
use poulpy_cpu_avx::FFT64Avx;

use crate::database::CoeffMatrix;
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, ScratchOwned, VecZnx, ZnxView, ZnxViewMut, ZnxZero},
    source::Source,
};

use super::glwe_secret_wrap_lwe;

type BE = FFT64Avx;

/// Full InsPIRe² double-PIR correctness (without the 2nd-level packing
/// optimization): level 1 selects column `i0` of each batch `D_b` → `resp0`;
/// Decompose → `D1`; level 2 selects batch `i1`, recovering ALL of `resp0[i1]`'s
/// digits (mask `τn` + body `τγ0`); recompose into a base2k=16 RLWE and final
/// decrypt with `s̃` → the record `D_{i1}[:, i0] ∈ Z_p^{γ0}`.
#[allow(clippy::needless_range_loop)]
fn run_end_to_end(t: usize, cols: usize, gamma0: usize, i0: usize, i1: usize) {
    let n = 64usize;
    let module = Module::<BE>::new(n as u64);
    let base2k = 18usize;
    let k_ct = 2 * base2k;
    let k_pt = 16usize;
    let qtilde_bits = 32usize;
    let baby_size = 8usize;
    let stride = (n / 2) / gamma0;
    let tau = decompose_digits(qtilde_bits);
    let mask_digits = n * tau; // τn
    let body_digits = gamma0 * tau; // τγ0

    let src_infos = EncryptionLayout::new_from_default_sigma(GLWELayout {
        n: Degree(n as u32),
        base2k: Base2K(base2k as u32),
        k: TorusPrecision(k_ct as u32),
        rank: Rank(1),
    })
    .unwrap();
    let query1_infos = LWEMatrixLayout {
        rows: cols,
        n: Degree(n as u32),
        base2k: src_infos.base2k(),
        k: src_infos.max_k(),
    };
    let res1_infos = LWEMatrixLayout {
        rows: gamma0,
        n: Degree(n as u32),
        base2k: src_infos.base2k(),
        k: src_infos.max_k(),
    };
    let query2_infos = LWEMatrixLayout {
        rows: t,
        n: Degree(n as u32),
        base2k: src_infos.base2k(),
        k: src_infos.max_k(),
    };
    // Digit selects are chunked to ≤ n outputs (matmul requires rows_out ≤ ring n).
    let res_chunk_infos = LWEMatrixLayout {
        rows: n,
        n: Degree(n as u32),
        base2k: src_infos.base2k(),
        k: src_infos.max_k(),
    };
    let key_infos = EncryptionLayout::new_from_default_sigma(GLWEAutomorphismKeyLayout {
        n: Degree(n as u32),
        base2k: Base2K(base2k as u32),
        k: TorusPrecision((k_ct + base2k) as u32),
        rank: Rank(1),
        dnum: k_ct.div_ceil(base2k).into(),
        dsize: 1usize.into(),
    })
    .unwrap();
    let qtilde_infos = qtilde_glwe_layout(Degree(n as u32), qtilde_bits);
    let size = res1_infos.size();

    let mut scratch = ScratchOwned::<BE>::alloc(
        module
            .glwe_compressed_encrypt_sk_tmp_bytes(&src_infos)
            .max(module.glwe_decrypt_tmp_bytes(&qtilde_infos))
            .max(module.glwe_expand_lwe_matrix_tmp_bytes(&query1_infos, &src_infos))
            .max(module.glwe_expand_lwe_matrix_tmp_bytes(&query2_infos, &src_infos))
            .max(module.lwe_matrix_decrypt_tmp_bytes(&res_chunk_infos))
            .max(module.pack_partial_mask_preprocessing_tmp_bytes(gamma0, size))
            .max(module.pack_keys_generate_tmp_bytes(&key_infos))
            .max(module.pack_keys_precompute_tmp_bytes(&key_infos, &key_infos, baby_size))
            .max(module.glwe_normalize_tmp_bytes())
            .max(module.pack_precompute_tmp_bytes(
                crate::packing::PackingPrecomputeInfos::new(n - 1, size, base2k, baby_size),
                &module.vec_znx_alloc(n, size),
                &key_infos,
            )),
    );

    let mut source_xs = Source::new([31u8; 32]);
    let mut source_xe = Source::new([32u8; 32]);
    let mut sk_lwe = module.lwe_secret_alloc(Degree(n as u32));
    sk_lwe.fill_ternary_prob(0.5, &mut source_xs);
    let sk_src = module.glwe_secret_from_lwe_secret(&sk_lwe);
    let sk_base = glwe_secret_wrap_lwe(&module, &sk_lwe);
    let mut sk_src_prep = module.glwe_secret_prepared_alloc_from_infos(&sk_src);
    module.glwe_secret_prepare(&mut sk_src_prep, &sk_src);
    let mut sk_dst_prep = module.glwe_secret_prepared_alloc_from_infos(&sk_base);
    module.glwe_secret_prepare(&mut sk_dst_prep, &sk_base);

    let key_partial = module.pack_partial_key_generate(
        &key_infos,
        &sk_lwe,
        [9u8; 32],
        stride,
        &mut source_xe,
        &mut scratch.borrow(),
    );
    let key_precomputations =
        module.pack_partial_keys_precompute(&key_partial, stride, baby_size, &mut scratch.borrow());

    // Level-1 query: one-hot e_{i0} over `cols`.
    let q1 = expand_one_hot(
        &module,
        &src_infos,
        &query1_infos,
        &sk_src_prep,
        i0,
        k_pt,
        &mut source_xe,
        &mut scratch,
    );

    // Per batch: D_b (γ0 × cols), select i0 → resp0 input. Track answer D_{i1}[:,i0].
    let mut precomputes = Vec::with_capacity(t);
    let mut bodies: Vec<VecZnx<<BE as Backend>::OwnedBuf>> = Vec::with_capacity(t);
    let mut answer = vec![0i64; gamma0];
    for b in 0..t {
        let db: Vec<Vec<i64>> = (0..gamma0)
            .map(|j| {
                (0..cols)
                    .map(|k| ((b * 11 + j * 5 + k * 3) % 29) as i64)
                    .collect()
            })
            .collect();
        if b == i1 {
            for j in 0..gamma0 {
                answer[j] = db[j][i0];
            }
        }
        let mut u = CoeffMatrix::zeros(gamma0, cols);
        for j in 0..gamma0 {
            for k in 0..cols {
                u.row_mut(j)[k] = db[j][k] as i16;
            }
        }
        let mut res = module.lwe_matrix_alloc_from_infos(&res1_infos);
        crate::test_oracle::lwe_matrix_mul(&mut res, &u, &q1);
        let mut aggregate = module.vec_znx_alloc(gamma0, size);
        module.packing_partial_mask_preprocessing(
            &mut aggregate,
            base2k,
            gamma0,
            res.mask(),
            &mut scratch.borrow(),
        );
        let mut body = module.vec_znx_alloc(1, size);
        body.zero();
        for limb in 0..size {
            body.at_mut(0, limb)[..gamma0].copy_from_slice(res.body().at(0, limb));
        }
        let mut precompute =
            module.pack_partial_precompute_alloc(gamma0 - 1, size, base2k, baby_size, stride);
        module.pack_partial_precompute(
            &mut precompute,
            &aggregate,
            &key_partial,
            &mut scratch.borrow(),
        );
        precomputes.push(precompute);
        bodies.push(body);
    }
    let inputs: Vec<_> = precomputes.iter().zip(bodies.iter()).collect();
    let resp0 = partial_pack_batch(
        &module,
        &src_infos,
        qtilde_bits,
        &inputs,
        &key_precomputations,
        &mut scratch.borrow(),
    );

    // Decompose resp0 → per-digit values across batches (D1 columns = batches).
    let mut mask_data: Vec<Vec<i16>> = vec![vec![0i16; t]; mask_digits];
    let mut body_data: Vec<Vec<i16>> = vec![vec![0i16; t]; body_digits];
    for (k, glwe) in resp0.iter().enumerate() {
        let data = glwe.data();
        for c in 0..n {
            for l in 0..tau {
                mask_data[c * tau + l][k] = data.at(1, l)[c] as i16;
            }
        }
        for c in 0..gamma0 {
            for l in 0..tau {
                body_data[c * tau + l][k] = data.at(0, l)[c] as i16;
            }
        }
    }

    // Level-2 query: one-hot e_{i1} over `t`; chunked select of batch i1's digits.
    let q2 = expand_one_hot(
        &module,
        &src_infos,
        &query2_infos,
        &sk_src_prep,
        i1,
        k_pt,
        &mut source_xe,
        &mut scratch,
    );
    let src_k = src_infos.max_k();
    let mask_digit_vals = select_digits_chunked(
        &module,
        t,
        &mask_data,
        &q2,
        &sk_lwe,
        base2k,
        k_pt,
        src_k,
        &mut scratch,
    );
    let body_digit_vals = select_digits_chunked(
        &module,
        t,
        &body_data,
        &q2,
        &sk_lwe,
        base2k,
        k_pt,
        src_k,
        &mut scratch,
    );

    // Recompose resp0[i1] from its digits and final-decrypt with s̃.
    let mut recomposed = module.glwe_alloc_from_infos(&qtilde_infos);
    recomposed.data_mut().zero();
    for c in 0..n {
        for l in 0..tau {
            recomposed.data_mut().at_mut(1, l)[c] = mask_digit_vals[c * tau + l];
        }
    }
    for c in 0..gamma0 {
        for l in 0..tau {
            recomposed.data_mut().at_mut(0, l)[c] = body_digit_vals[c * tau + l];
        }
    }

    let mut out_pt = module.glwe_plaintext_alloc_from_infos(&qtilde_infos);
    module.glwe_decrypt(
        &recomposed,
        &mut out_pt,
        &sk_dst_prep,
        &mut scratch.borrow(),
    );
    let mut got = vec![0i64; n];
    out_pt.decode_vec_i64(&mut got, TorusPrecision(k_pt as u32));

    assert_eq!(
        &got[..gamma0],
        answer.as_slice(),
        "InsPIRe² double PIR did not recover D_{{i1}}[:, i0]"
    );
}

/// Encrypts a one-hot `e_idx` GLWE and expands it to a query LWE matrix.
fn expand_one_hot(
    module: &Module<BE>,
    src_infos: &EncryptionLayout<GLWELayout>,
    query_infos: &LWEMatrixLayout,
    sk_src_prep: &poulpy_core::layouts::GLWESecretPrepared<<BE as Backend>::OwnedBuf, BE>,
    idx: usize,
    k_pt: usize,
    source_xe: &mut Source,
    scratch: &mut ScratchOwned<BE>,
) -> poulpy_core::layouts::LWEMatrix<<BE as Backend>::OwnedBuf> {
    let n = module.n();
    let mut sel = vec![0i64; n];
    sel[idx] = 1;
    let mut pt = module.glwe_plaintext_alloc_from_infos(src_infos);
    pt.encode_vec_i64(&sel, TorusPrecision(k_pt as u32));
    let mut src = module.glwe_compressed_alloc_from_infos(src_infos);
    module.glwe_compressed_encrypt_sk(
        &mut src,
        &pt,
        sk_src_prep,
        [60u8; 32],
        src_infos,
        source_xe,
        &mut scratch.borrow(),
    );
    let mut glwe = module.glwe_alloc_from_infos(src_infos);
    module.decompress_glwe(&mut glwe, &src);
    let mut query = module.lwe_matrix_alloc_from_infos(query_infos);
    module.glwe_expand_lwe_matrix(&mut query, &glwe, &mut scratch.borrow());
    query
}

/// `D · one-hot-query`, decrypted with the LWE secret → the `rows` plaintext values.
fn select_and_decrypt(
    module: &Module<BE>,
    res_infos: &LWEMatrixLayout,
    db: &CoeffMatrix,
    query: &poulpy_core::layouts::LWEMatrix<<BE as Backend>::OwnedBuf>,
    sk_lwe: &LWESecret<<BE as Backend>::OwnedBuf>,
    base2k: usize,
    k_pt: usize,
    scratch: &mut ScratchOwned<BE>,
) -> Vec<i64> {
    let n = module.n();
    let rows = res_infos.rows;
    let mut res = module.lwe_matrix_alloc_from_infos(res_infos);
    crate::test_oracle::lwe_matrix_mul(&mut res, db, query);
    let layout = GLWELayout {
        n: Degree(n as u32),
        base2k: Base2K(base2k as u32),
        k: res_infos.k,
        rank: Rank(1),
    };
    let mut pt = module.glwe_plaintext_alloc_from_infos(&layout);
    module.lwe_matrix_decrypt(&res, &mut pt, sk_lwe, &mut scratch.borrow());
    let mut out = vec![0i64; n.max(rows)];
    pt.data().decode_vec_i64(base2k, 0, k_pt, &mut out);
    out.truncate(rows);
    out
}

/// Selects all `all_digits` (length = total digits) for the queried batch, in
/// chunks of ≤ n outputs (the matmul caps `rows_out` at the ring degree). This
/// mirrors the paper's γ-batched second level without yet packing the result.
#[allow(clippy::needless_range_loop)]
fn select_digits_chunked(
    module: &Module<BE>,
    t: usize,
    all_digits: &[Vec<i16>],
    query: &poulpy_core::layouts::LWEMatrix<<BE as Backend>::OwnedBuf>,
    sk_lwe: &LWESecret<<BE as Backend>::OwnedBuf>,
    base2k: usize,
    k_pt: usize,
    src_k: TorusPrecision,
    scratch: &mut ScratchOwned<BE>,
) -> Vec<i64> {
    let n = module.n();
    let total = all_digits.len();
    let mut out = Vec::with_capacity(total);
    let mut start = 0;
    while start < total {
        let chunk = (total - start).min(n);
        let chunk_infos = LWEMatrixLayout {
            rows: chunk,
            n: Degree(n as u32),
            base2k: Base2K(base2k as u32),
            k: src_k,
        };
        let mut db = CoeffMatrix::zeros(chunk, t);
        for j in 0..chunk {
            for b in 0..t {
                db.row_mut(j)[b] = all_digits[start + j][b];
            }
        }
        let vals = select_and_decrypt(
            module,
            &chunk_infos,
            &db,
            query,
            sk_lwe,
            base2k,
            k_pt,
            scratch,
        );
        out.extend_from_slice(&vals);
        start += chunk;
    }
    out
}

#[test]
fn recursion_end_to_end_double_pir() {
    run_end_to_end(4, 64, 8, 5, 2);
}

#[test]
fn recursion_api_roundtrip() {
    use crate::client::Client;
    use crate::database::DatabaseLayout;
    use crate::payload::Payload;
    use crate::server::Server;
    use std::marker::PhantomData;
    struct RawRecordPayload;
    impl Payload<[u8; 32]> for RawRecordPayload {
        const BASIS: u32 = 65536;
        const EXPONENT: usize = 1;
        fn encode(_digits: &mut [i16], _a: [u8; 32]) {}
        fn decode(_a: &mut [u8; 32], _digits: &[i16]) {}
    }
    let (t, cols, gamma0) = (4usize, 64usize, 8usize);
    let num_records = t * cols;
    let config = || Config::<[u8; 32], RawRecordPayload> {
        n: 64,
        base2k: DEFAULT_BASE2K,
        k: DEFAULT_K,
        collapse: Collapse::Recursion {
            gamma0,
            gamma1: 32,
            gamma2: 8,
        },
        _phantom: PhantomData,
    };
    let layout = DatabaseLayout::<RawRecordPayload>::new(t * gamma0, cols);
    let mut client = Client::<FFT64Avx, RawRecordPayload>::new(config(), layout);
    let mut server = Server::<FFT64Avx, RawRecordPayload>::new(config(), layout);
    let records: Vec<Vec<i64>> = (0..num_records)
        .map(|r| (0..gamma0).map(|j| ((r * 3 + j * 7) % 41) as i64).collect())
        .collect();
    server.encode(&records);
    server.offline();

    let (i0, i1) = (5usize, 2usize);
    let payload_index = i1 * gamma0 * cols + i0;
    let (query, state) = client.query(payload_index);
    let resp = server.respond(&query);
    let got = client.decrypt_digits(&resp, &state);

    assert_eq!(
        got,
        server.database().record(i0, i1),
        "Recursion API did not recover record (i0,i1)"
    );
}

#[test]
fn recursion_api_roundtrip_larger() {
    use crate::client::Client;
    use crate::database::DatabaseLayout;
    use crate::payload::Payload;
    use crate::server::Server;
    use std::marker::PhantomData;
    struct RawRecordPayload;
    impl Payload<[u8; 32]> for RawRecordPayload {
        const BASIS: u32 = 65536;
        const EXPONENT: usize = 1;
        fn encode(_digits: &mut [i16], _a: [u8; 32]) {}
        fn decode(_a: &mut [u8; 32], _digits: &[i16]) {}
    }
    let (t, cols, gamma0, k_pt) = (8usize, 128usize, 8usize, 16usize);
    let num_records = t * cols;
    let p = 1i64 << k_pt;
    let config = || Config::<[u8; 32], RawRecordPayload> {
        n: 128,
        base2k: DEFAULT_BASE2K,
        k: DEFAULT_K,
        collapse: Collapse::Recursion {
            gamma0,
            gamma1: 32,
            gamma2: 8,
        },
        _phantom: PhantomData,
    };
    let layout = DatabaseLayout::<RawRecordPayload>::new(t * gamma0, cols);
    let mut client = Client::<FFT64Avx, RawRecordPayload>::new(config(), layout);
    let mut server = Server::<FFT64Avx, RawRecordPayload>::new(config(), layout);
    // Full-range Z_p records (stress the decompose digits' i16 range).
    let records: Vec<Vec<i64>> = (0..num_records)
        .map(|r| {
            (0..gamma0)
                .map(|j| (r as i64 * 131 + j as i64 * 97717).rem_euclid(p))
                .collect()
        })
        .collect();
    server.encode(&records);
    server.offline();

    for &(i0, i1) in &[(0usize, 0usize), (5, 3), (127, 7), (63, 4)] {
        let payload_index = i1 * gamma0 * cols + i0;
        let (query, state) = client.query(payload_index);
        let resp = server.respond(&query);
        let got: Vec<i64> = client
            .decrypt_digits(&resp, &state)
            .iter()
            .map(|v| v.rem_euclid(p))
            .collect();
        // The PIR recovers each record digit mod p. The database stores digits
        // centered into i16, so `record()` returns the signed representative
        // (e.g. -1174 for 64362 when p = 2^16); reduce it to the same [0, p)
        // canonical form as `got` before comparing.
        let want: Vec<i64> = server
            .database()
            .record(i0, i1)
            .iter()
            .map(|v| v.rem_euclid(p))
            .collect();
        assert_eq!(got, want, "record ({i0}, {i1}) mismatch");
    }
}

#[test]
fn recursion_api_roundtrip_chunked_dimensions() {
    use crate::client::Client;
    use crate::database::DatabaseLayout;
    use crate::payload::Payload;
    use crate::server::Server;
    use std::marker::PhantomData;
    struct RawRecordPayload;
    impl Payload<[u8; 32]> for RawRecordPayload {
        const BASIS: u32 = 65536;
        const EXPONENT: usize = 1;
        fn encode(_digits: &mut [i16], _a: [u8; 32]) {}
        fn decode(_a: &mut [u8; 32], _digits: &[i16]) {}
    }

    let (n, t, cols, gamma0) = (64usize, 128usize, 128usize, 8usize);
    let config = || Config::<[u8; 32], RawRecordPayload> {
        n,
        base2k: DEFAULT_BASE2K,
        k: DEFAULT_K,
        collapse: Collapse::Recursion {
            gamma0,
            gamma1: 32,
            gamma2: gamma0,
        },
        _phantom: PhantomData,
    };
    let layout = DatabaseLayout::<RawRecordPayload>::new(t * gamma0, cols);
    let mut client = Client::<FFT64Avx, RawRecordPayload>::new(config(), layout);
    let mut server = Server::<FFT64Avx, RawRecordPayload>::new(config(), layout);
    let records: Vec<Vec<i64>> = (0..t * cols)
        .map(|r| {
            (0..gamma0)
                .map(|j| ((r * 5 + j * 13) % 101) as i64)
                .collect()
        })
        .collect();
    server.encode(&records);
    server.offline();

    for &(i0, i1) in &[(0usize, 0usize), (65, 3), (127, 127)] {
        let payload_index = i1 * gamma0 * cols + i0;
        let (query, state) = client.query(payload_index);
        let resp = server.respond(&query);
        let got = client.decrypt_digits(&resp, &state);
        assert_eq!(
            got,
            server.database().record(i0, i1),
            "chunked record ({i0}, {i1}) mismatch"
        );
    }
}
