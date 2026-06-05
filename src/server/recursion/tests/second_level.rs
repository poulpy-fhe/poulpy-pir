//! Second-level (phase 2 recursion core) body-digit select on the decomposed DB.

use crate::packing::{
    Packing, PackingKeysGenerate, PackingMaskAggregation,
    recursion::{decompose_digits, partial_pack_batch, qtilde_glwe_layout},
};
use poulpy_core::{
    EncryptionLayout, GLWECompressedEncryptSk, GLWEDecrypt, GLWEExpandLWEMatrix, GLWENormalize,
    LWEMatrixDecrypt,
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

type BE = FFT64Avx;

/// Phase 2 (recursion core): the decomposed first-level output `D1` is used as a
/// PLAINTEXT database for the second PIR level. This builds `resp0` (t RLWEs),
/// reshapes the body-digit portion (`τγ0` i16 digits per batch) into a
/// `CoeffMatrix<i16>` `D1_body`, then runs a second-level one-hot select of batch
/// `i1` and checks the decrypted `τγ0` LWEs recover `resp0[i1]`'s body digits —
/// i.e. PIR-select on the decomposed digit DB round-trips (the heart of Decompose
/// → 2nd layer). The full second level additionally packs (γ1/γ2) + Extract.
fn run_second_level_body_select(t: usize, gamma0: usize, i1: usize) {
    let n = 64usize;
    let module = Module::<BE>::new(n as u64);
    let base2k = 18usize;
    let k_ct = 2 * base2k;
    let k_pt = 16usize;
    let qtilde_bits = 32usize;
    let baby_size = 8usize;
    let stride = (n / 2) / gamma0;
    let tau = decompose_digits(qtilde_bits);
    let n_digits = gamma0 * tau; // body digits kept per batch (τγ0)

    let src_infos = EncryptionLayout::new_from_default_sigma(GLWELayout {
        n: Degree(n as u32),
        base2k: Base2K(base2k as u32),
        k: TorusPrecision(k_ct as u32),
        rank: Rank(1),
    })
    .unwrap();
    let matrix_infos = LWEMatrixLayout {
        rows: gamma0,
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
    let size = matrix_infos.size();
    // Second-level query/result layouts: input dim = t batches, output = τγ0 digits.
    let query2_infos = LWEMatrixLayout {
        rows: t,
        n: Degree(n as u32),
        base2k: src_infos.base2k(),
        k: src_infos.max_k(),
    };
    let res2_infos = LWEMatrixLayout {
        rows: n_digits,
        n: Degree(n as u32),
        base2k: src_infos.base2k(),
        k: src_infos.max_k(),
    };

    let mut scratch = ScratchOwned::<BE>::alloc(
        module
            .glwe_compressed_encrypt_sk_tmp_bytes(&src_infos)
            .max(module.glwe_decrypt_tmp_bytes(&qtilde_infos))
            .max(module.glwe_expand_lwe_matrix_tmp_bytes(&matrix_infos, &src_infos))
            .max(module.glwe_expand_lwe_matrix_tmp_bytes(&query2_infos, &src_infos))
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

    let mut source_xs = Source::new([21u8; 32]);
    let mut source_xe = Source::new([22u8; 32]);
    let mut sk_lwe = module.lwe_secret_alloc(Degree(n as u32));
    sk_lwe.fill_ternary_prob(0.5, &mut source_xs);
    let sk_src = module.glwe_secret_from_lwe_secret(&sk_lwe);
    let mut sk_src_prep = module.glwe_secret_prepared_alloc_from_infos(&sk_src);
    module.glwe_secret_prepare(&mut sk_src_prep, &sk_src);

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

    // --- First level: build resp0 (t RLWEs) via partial_pack_batch. ---
    let mut precomputes = Vec::with_capacity(t);
    let mut bodies: Vec<VecZnx<<BE as Backend>::OwnedBuf>> = Vec::with_capacity(t);
    for b in 0..t {
        let data: Vec<i64> = (0..n).map(|i| ((i + 3 * b) as i64 % 7) - 3).collect();
        let mut pt = module.glwe_plaintext_alloc_from_infos(&src_infos);
        pt.encode_vec_i64(&data, TorusPrecision(k_pt as u32));
        let mut src = module.glwe_compressed_alloc_from_infos(&src_infos);
        module.glwe_compressed_encrypt_sk(
            &mut src,
            &pt,
            &sk_src_prep,
            [40u8 + b as u8; 32],
            &src_infos,
            &mut source_xe,
            &mut scratch.borrow(),
        );
        let mut src_glwe = module.glwe_alloc_from_infos(&src_infos);
        module.decompress_glwe(&mut src_glwe, &src);
        let mut lwe_matrix = module.lwe_matrix_alloc_from_infos(&matrix_infos);
        module.glwe_expand_lwe_matrix(&mut lwe_matrix, &src_glwe, &mut scratch.borrow());
        let mut aggregate = module.vec_znx_alloc(gamma0, size);
        module.packing_partial_mask_preprocessing(
            &mut aggregate,
            base2k,
            gamma0,
            lwe_matrix.mask(),
            &mut scratch.borrow(),
        );
        let mut body = module.vec_znx_alloc(1, size);
        body.zero();
        for limb in 0..size {
            body.at_mut(0, limb)[..gamma0].copy_from_slice(lwe_matrix.body().at(0, limb));
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

    // --- D1_body: reshape resp0 body digits into a plaintext CoeffMatrix<i16>. ---
    // Digit index j = c*τ + l for body coeff c∈[γ0], limb l∈[τ]; columns = batches.
    let mut d1_body = module.coeff_matrix_alloc::<i16>(
        t,
        n_digits,
        Base2K(base2k as u32),
        TorusPrecision(base2k as u32),
    );
    for (k, glwe) in resp0.iter().enumerate() {
        let data = glwe.data();
        for c in 0..gamma0 {
            for l in 0..tau {
                let j = c * tau + l;
                d1_body.data_mut().at_mut(j, 0)[k] = data.at(0, l)[c]; // body column 0
            }
        }
    }
    // Reference: the actual body digits of resp0[i1].
    let want: Vec<i64> = {
        let data = resp0[i1].data();
        let mut v = vec![0i64; n_digits];
        for c in 0..gamma0 {
            for l in 0..tau {
                v[c * tau + l] = data.at(0, l)[c];
            }
        }
        v
    };

    // --- Second level: one-hot select of batch i1 over the digit DB. ---
    let mut sel = vec![0i64; n];
    sel[i1] = 1;
    let mut pt2 = module.glwe_plaintext_alloc_from_infos(&src_infos);
    pt2.encode_vec_i64(&sel, TorusPrecision(k_pt as u32));
    let mut src2 = module.glwe_compressed_alloc_from_infos(&src_infos);
    module.glwe_compressed_encrypt_sk(
        &mut src2,
        &pt2,
        &sk_src_prep,
        [70u8; 32],
        &src_infos,
        &mut source_xe,
        &mut scratch.borrow(),
    );
    let mut src2_glwe = module.glwe_alloc_from_infos(&src_infos);
    module.decompress_glwe(&mut src2_glwe, &src2);
    let mut query2 = module.lwe_matrix_alloc_from_infos(&query2_infos);
    module.glwe_expand_lwe_matrix(&mut query2, &src2_glwe, &mut scratch.borrow());

    let mut res2 = module.lwe_matrix_alloc_from_infos(&res2_infos);
    crate::test_oracle::lwe_matrix_mul(&mut res2, &d1_body, &query2);

    // Decrypt the τγ0 selected LWEs (LWE matrix) → recovered digits.
    let mut res2_pt = module.glwe_plaintext_alloc_from_infos(&src_infos);
    module.lwe_matrix_decrypt(&res2, &mut res2_pt, &sk_lwe, &mut scratch.borrow());
    let mut got = vec![0i64; n];
    res2_pt.data().decode_vec_i64(base2k, 0, k_pt, &mut got);

    assert_eq!(
        &got[..n_digits],
        want.as_slice(),
        "second-level select did not recover resp0[i1]'s body digits"
    );
}

#[test]
fn second_level_body_select_recovers_digits() {
    run_second_level_body_select(4, 8, 2);
}
