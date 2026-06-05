//! First-level (phase 1) one-hot select + partial pack, single and batched.

use crate::packing::{
    Packing, PackingKeysGenerate, PackingMaskAggregation,
    recursion::{partial_pack_batch, qtilde_glwe_layout},
};
use poulpy_core::{
    EncryptionLayout, GLWECompressedEncryptSk, GLWEDecrypt, GLWEExpandLWEMatrix, GLWENormalize,
    layouts::{
        Base2K, Degree, GLWEAutomorphismKeyLayout, GLWEDecompress, GLWELayout,
        GLWESecretPreparedFactory, LWEInfos, LWEMatrixLayout, ModuleCoreAlloc,
        ModuleCoreCompressedAlloc, Rank, SecretConversion, TorusPrecision,
    },
};
use poulpy_cpu_avx::FFT64Avx;
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, ScratchOwned, VecZnx, ZnxView, ZnxViewMut, ZnxZero},
    source::Source,
};

use super::glwe_secret_wrap_lwe;

type BE = FFT64Avx;

/// Phase 1 (first level): one-hot LWE PIR select `D·e_{i0}` via `lwe_matrix_mul`,
/// then partial-pack the γ₀ selected LWEs. Recovers `D[:, i0]` — the γ₀ entries
/// of the selected column. This is the per-batch building block of `Respond`'s
/// first level (`resp0 = PartialPackBatch(t, γ0, D·[A0,b0], K_{g_{γ0}})`).
#[allow(clippy::needless_range_loop)]
fn run_first_level(cols: usize, gamma: usize, i0: usize) {
    let n = 64usize;
    let module = Module::<BE>::new(n as u64);
    let base2k = 18usize;
    let k_ct = 2 * base2k;
    let k_pt = 16usize;
    let baby_size = 8usize;
    let stride = (n / 2) / gamma;

    let src_infos = EncryptionLayout::new_from_default_sigma(GLWELayout {
        n: Degree(n as u32),
        base2k: Base2K(base2k as u32),
        k: TorusPrecision(k_ct as u32),
        rank: Rank(1),
    })
    .unwrap();
    // Query LWE matrix has one row per DB column (input dimension `cols`); the
    // selected result has `gamma` rows (the γ₀ batch).
    let query_infos = LWEMatrixLayout {
        rows: cols,
        n: Degree(n as u32),
        base2k: src_infos.base2k(),
        k: src_infos.max_k(),
    };
    let res_infos = LWEMatrixLayout {
        rows: gamma,
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
    let size = res_infos.size();

    let mut scratch = ScratchOwned::<BE>::alloc(
        module
            .glwe_compressed_encrypt_sk_tmp_bytes(&src_infos)
            .max(module.glwe_decrypt_tmp_bytes(&src_infos))
            .max(module.glwe_expand_lwe_matrix_tmp_bytes(&query_infos, &src_infos))
            .max(module.pack_partial_mask_preprocessing_tmp_bytes(gamma, size))
            .max(module.pack_keys_generate_tmp_bytes(&key_infos))
            .max(module.pack_keys_precompute_tmp_bytes(&key_infos, &key_infos, baby_size))
            .max(module.pack_precompute_tmp_bytes(
                crate::packing::PackingPrecomputeInfos::new(n - 1, size, base2k, baby_size),
                &module.vec_znx_alloc(n, size),
                &key_infos,
            )),
    );

    let mut source_xs = Source::new([3u8; 32]);
    let mut source_xe = Source::new([4u8; 32]);
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

    // Plaintext DB column block D ∈ Z_p^{gamma × cols}; select column i0.
    let db: Vec<Vec<i64>> = (0..gamma)
        .map(|j| (0..cols).map(|k| ((j * 7 + k * 3) % 41) as i64).collect())
        .collect();
    let expected: Vec<i64> = (0..gamma).map(|j| db[j][i0]).collect();

    let mut u = module.coeff_matrix_alloc::<i16>(
        cols,
        gamma,
        Base2K(base2k as u32),
        TorusPrecision(base2k as u32),
    );
    for j in 0..gamma {
        for k in 0..cols {
            u.data_mut().at_mut(j, 0)[k] = db[j][k];
        }
    }

    // One-hot query: GLWE encrypting e_{i0} (length `cols`), expanded to `cols` LWEs.
    let mut sel = vec![0i64; n];
    sel[i0] = 1;
    let mut pt = module.glwe_plaintext_alloc_from_infos(&src_infos);
    pt.encode_vec_i64(&sel, TorusPrecision(k_pt as u32));
    let mut src = module.glwe_compressed_alloc_from_infos(&src_infos);
    module.glwe_compressed_encrypt_sk(
        &mut src,
        &pt,
        &sk_src_prep,
        [50u8; 32],
        &src_infos,
        &mut source_xe,
        &mut scratch.borrow(),
    );
    let mut src_glwe = module.glwe_alloc_from_infos(&src_infos);
    module.decompress_glwe(&mut src_glwe, &src);
    let mut query = module.lwe_matrix_alloc_from_infos(&query_infos);
    module.glwe_expand_lwe_matrix(&mut query, &src_glwe, &mut scratch.borrow());

    // First-level select: res = D · query  (= D[:, i0], γ₀ LWEs).
    let mut res = module.lwe_matrix_alloc_from_infos(&res_infos);
    crate::test_oracle::lwe_matrix_mul(&mut res, &u, &query);

    // Partial-pack the selected γ₀ LWEs (same flow as `partial_pack`).
    let mut aggregate = module.vec_znx_alloc(gamma, size);
    module.packing_partial_mask_preprocessing(
        &mut aggregate,
        base2k,
        gamma,
        res.mask(),
        &mut scratch.borrow(),
    );
    let mut body = module.vec_znx_alloc(1, size);
    body.zero();
    for limb in 0..size {
        body.at_mut(0, limb)[..gamma].copy_from_slice(res.body().at(0, limb));
    }
    let mut precompute =
        module.pack_partial_precompute_alloc(gamma - 1, size, base2k, baby_size, stride);
    module.pack_partial_precompute(
        &mut precompute,
        &aggregate,
        &key_partial,
        &mut scratch.borrow(),
    );

    let mut packed = module.glwe_alloc_from_infos(&src_infos);
    module.pack(
        &mut packed,
        &body,
        &precompute,
        &key_precomputations,
        1,
        &mut scratch.borrow(),
    );

    let mut decoded_pt = module.glwe_plaintext_alloc_from_infos(&src_infos);
    module.glwe_decrypt(
        &packed,
        &mut decoded_pt,
        &sk_dst_prep,
        &mut scratch.borrow(),
    );
    let mut decoded = vec![0i64; n];
    decoded_pt.decode_vec_i64(&mut decoded, TorusPrecision(k_pt as u32));

    assert_eq!(
        &decoded[..gamma],
        expected.as_slice(),
        "first-level select+pack did not recover D[:, i0]"
    );
}

#[test]
fn first_level_select_then_partial_pack() {
    run_first_level(64, 8, 5);
    run_first_level(64, 16, 30);
}

/// Phase 1 (full first level): `t` γ₀-batches, each selecting column `i0` of its
/// own `D_b ∈ Z_p^{γ0 × cols}`, then `partial_pack_batch` (mod-switch to q̃ at
/// base2k=16) → `resp0` (the t RLWEs whose balanced limbs are `D1`). Checks every
/// batch decrypts to `D_b[:, i0]`, i.e. the full `resp0 = PartialPackBatch(t, γ0,
/// D·[A0,b0], K_{g_{γ0}})` data flow.
#[allow(clippy::needless_range_loop)]
fn run_first_level_batched(t: usize, cols: usize, gamma: usize, i0: usize) {
    let n = 64usize;
    let module = Module::<BE>::new(n as u64);
    let base2k = 18usize;
    let k_ct = 2 * base2k;
    let k_pt = 16usize;
    let qtilde_bits = 32usize;
    let baby_size = 8usize;
    let stride = (n / 2) / gamma;

    let src_infos = EncryptionLayout::new_from_default_sigma(GLWELayout {
        n: Degree(n as u32),
        base2k: Base2K(base2k as u32),
        k: TorusPrecision(k_ct as u32),
        rank: Rank(1),
    })
    .unwrap();
    let query_infos = LWEMatrixLayout {
        rows: cols,
        n: Degree(n as u32),
        base2k: src_infos.base2k(),
        k: src_infos.max_k(),
    };
    let res_infos = LWEMatrixLayout {
        rows: gamma,
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
    let size = res_infos.size();

    let mut scratch = ScratchOwned::<BE>::alloc(
        module
            .glwe_compressed_encrypt_sk_tmp_bytes(&src_infos)
            .max(module.glwe_decrypt_tmp_bytes(&qtilde_infos))
            .max(module.glwe_expand_lwe_matrix_tmp_bytes(&query_infos, &src_infos))
            .max(module.pack_partial_mask_preprocessing_tmp_bytes(gamma, size))
            .max(module.pack_keys_generate_tmp_bytes(&key_infos))
            .max(module.pack_keys_precompute_tmp_bytes(&key_infos, &key_infos, baby_size))
            .max(module.glwe_normalize_tmp_bytes())
            .max(module.pack_precompute_tmp_bytes(
                crate::packing::PackingPrecomputeInfos::new(n - 1, size, base2k, baby_size),
                &module.vec_znx_alloc(n, size),
                &key_infos,
            )),
    );

    let mut source_xs = Source::new([13u8; 32]);
    let mut source_xe = Source::new([14u8; 32]);
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

    // One shared one-hot query selecting column i0 (same query across batches).
    let mut sel = vec![0i64; n];
    sel[i0] = 1;
    let mut pt = module.glwe_plaintext_alloc_from_infos(&src_infos);
    pt.encode_vec_i64(&sel, TorusPrecision(k_pt as u32));
    let mut src = module.glwe_compressed_alloc_from_infos(&src_infos);
    module.glwe_compressed_encrypt_sk(
        &mut src,
        &pt,
        &sk_src_prep,
        [50u8; 32],
        &src_infos,
        &mut source_xe,
        &mut scratch.borrow(),
    );
    let mut src_glwe = module.glwe_alloc_from_infos(&src_infos);
    module.decompress_glwe(&mut src_glwe, &src);
    let mut query = module.lwe_matrix_alloc_from_infos(&query_infos);
    module.glwe_expand_lwe_matrix(&mut query, &src_glwe, &mut scratch.borrow());

    // Per-batch: D_b, select i0 → γ₀ LWEs → (precompute, body).
    let mut precomputes = Vec::with_capacity(t);
    let mut bodies: Vec<VecZnx<<BE as Backend>::OwnedBuf>> = Vec::with_capacity(t);
    let mut expected: Vec<Vec<i64>> = Vec::with_capacity(t);

    for b in 0..t {
        let db: Vec<Vec<i64>> = (0..gamma)
            .map(|j| {
                (0..cols)
                    .map(|k| ((b * 5 + j * 7 + k * 3) % 37) as i64)
                    .collect()
            })
            .collect();
        expected.push((0..gamma).map(|j| db[j][i0]).collect());

        let mut u = module.coeff_matrix_alloc::<i16>(
            cols,
            gamma,
            Base2K(base2k as u32),
            TorusPrecision(base2k as u32),
        );
        for j in 0..gamma {
            for k in 0..cols {
                u.data_mut().at_mut(j, 0)[k] = db[j][k];
            }
        }
        let mut res = module.lwe_matrix_alloc_from_infos(&res_infos);
        crate::test_oracle::lwe_matrix_mul(&mut res, &u, &query);

        let mut aggregate = module.vec_znx_alloc(gamma, size);
        module.packing_partial_mask_preprocessing(
            &mut aggregate,
            base2k,
            gamma,
            res.mask(),
            &mut scratch.borrow(),
        );
        let mut body = module.vec_znx_alloc(1, size);
        body.zero();
        for limb in 0..size {
            body.at_mut(0, limb)[..gamma].copy_from_slice(res.body().at(0, limb));
        }
        let mut precompute =
            module.pack_partial_precompute_alloc(gamma - 1, size, base2k, baby_size, stride);
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
    assert_eq!(resp0.len(), t);

    for (b, glwe) in resp0.iter().enumerate() {
        let mut decoded_pt = module.glwe_plaintext_alloc_from_infos(&qtilde_infos);
        module.glwe_decrypt(glwe, &mut decoded_pt, &sk_dst_prep, &mut scratch.borrow());
        let mut decoded = vec![0i64; n];
        decoded_pt.decode_vec_i64(&mut decoded, TorusPrecision(k_pt as u32));
        assert_eq!(
            &decoded[..gamma],
            expected[b].as_slice(),
            "batch {b}: first-level resp0 did not recover D_b[:, i0]"
        );
    }
}

#[test]
fn first_level_batched_recovers_columns() {
    run_first_level_batched(3, 64, 8, 5);
}
