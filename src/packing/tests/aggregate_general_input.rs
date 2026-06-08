//! Regression test: `packing_mask_preprocessing` produces a correct MLWE-like
//! aggregate `(â_agg, b̄_agg)` for a GENERAL (non-expansion) LWE matrix, i.e. one
//! produced by `U * expand(query)` with a non-diagonal `U`.
//!
//! The aggregate is "decrypted" directly as the MLWE ciphertext it represents
//! (`m̂ = b̄_agg + Σ_col â_agg[col] · ŝ[col]`, the same procedure used by
//! `aggregate_decrypts_expanded_glwe_ciphertext`) and compared against the LWE
//! matrix's own decryption under the raw LWE secret.
//!
//! The aggregate is masked by the inverse share `ŝ[col] = τ_{g^{-col}}(s̄)`
//! (the implementation-wide convention; see the `galois_element_inv` in
//! `packing_mask_preprocessing`). For symmetric expansion masks the positive and
//! inverse shares coincide, so this test — which uses a permutation `U` — is what
//! pins the share direction for general input. It previously exposed a sign bug
//! where the aggregate and the COLLAPSE disagreed on the share direction.
use crate::packing::PackingMaskAggregation;
use poulpy_core::{
    EncryptionLayout, GLWECompressedEncryptSk, GLWEExpandLWEMatrix, LWEMatrixDecrypt,
    layouts::{
        Base2K, Degree, GLWEDecompress, GLWELayout, GLWESecretPreparedFactory, LWEInfos,
        LWEMatrixLayout, ModuleCoreAlloc, ModuleCoreCompressedAlloc, Rank, SecretConversion,
        TorusPrecision,
    },
};
use poulpy_cpu_avx::FFT64Avx;

use crate::database::CoeffMatrix;
use poulpy_hal::{
    api::{
        ScalarZnxAutomorphismBackend, ScratchOwnedAlloc, ScratchOwnedBorrow, SvpApplyDft,
        SvpPPolAlloc, SvpPrepare, VecZnxBigAddAssign, VecZnxBigAlloc, VecZnxBigFromSmallBackend,
        VecZnxBigNormalize, VecZnxDftAddAssign, VecZnxDftAlloc, VecZnxIdftApplyTmpA,
    },
    layouts::{
        GaloisElement, Module, ScalarZnxToBackendMut, ScalarZnxToBackendRef, ScratchOwned,
        SvpPPolToBackendMut, SvpPPolToBackendRef, VecZnxBigToBackendMut, VecZnxBigToBackendRef,
        VecZnxDftToBackendMut, VecZnxDftToBackendRef, VecZnxToBackendMut, VecZnxToBackendRef,
    },
    source::Source,
};

const N: usize = 64;

fn run() {
    type BE = FFT64Avx;
    let n = N;
    let module = Module::<BE>::new(n as u64);
    let base2k = Base2K(18);
    let k_ct = TorusPrecision(36);
    let k_pt = 16usize;

    let src_infos = EncryptionLayout::new_from_default_sigma(GLWELayout {
        n: Degree(n as u32),
        base2k,
        k: k_ct,
        rank: Rank(1),
    })
    .unwrap();
    let matrix_infos = LWEMatrixLayout {
        rows: n,
        n: Degree(n as u32),
        base2k: src_infos.base2k(),
        k: src_infos.max_k(),
    };
    let mut scratch = ScratchOwned::<BE>::alloc(
        module
            .glwe_compressed_encrypt_sk_tmp_bytes(&src_infos)
            .max(module.lwe_matrix_decrypt_tmp_bytes(&matrix_infos))
            .max(module.packing_mask_preprocessing_tmp_bytes(matrix_infos.size()))
            .max(1 << 20),
    );

    let mut source_xs = Source::new([11u8; 32]);
    let mut source_xe = Source::new([12u8; 32]);

    let mut sk_lwe = module.lwe_secret_alloc(Degree(n as u32));
    sk_lwe.fill_ternary_prob(0.5, &mut source_xs);
    let sk_src = module.glwe_secret_from_lwe_secret(&sk_lwe);
    let mut sk_src_prep = module.glwe_secret_prepared_alloc_from_infos(&sk_src);
    module.glwe_secret_prepare(&mut sk_src_prep, &sk_src);

    // Encrypt the query (compressed) encoding `data`.
    let data: Vec<i64> = (0..n).map(|i| (i as i64 % 7) - 3).collect();
    let mut pt = module.glwe_plaintext_alloc_from_infos(&src_infos);
    pt.encode_vec_i64(&data, TorusPrecision(k_pt as u32));
    let mut query = module.glwe_compressed_alloc_from_infos(&src_infos);
    module.glwe_compressed_encrypt_sk(
        &mut query,
        &pt,
        &sk_src_prep,
        [17u8; 32],
        &src_infos,
        &mut source_xe,
        &mut scratch.borrow(),
    );

    // Non-diagonal U: cyclic-shift permutation (row i selects query row (i+1)).
    let mut u = CoeffMatrix::zeros(n, n);
    for i in 0..n {
        u.row_mut(i)[(i + 1) % n] = 1;
    }

    // Expand the compressed query to an LWE matrix, then apply U via the test
    // oracle (the homomorphic `U · query` product, removed from the library).
    let mut query_glwe = module.glwe_alloc_from_infos(&src_infos);
    module.decompress_glwe(&mut query_glwe, &query);
    let mut query_expanded = module.lwe_matrix_alloc_from_infos(&matrix_infos);
    module.glwe_expand_lwe_matrix(&mut query_expanded, &query_glwe, &mut scratch.borrow());
    let mut product = module.lwe_matrix_alloc_from_infos(&matrix_infos);
    crate::test_oracle::lwe_matrix_mul_mask(&mut product, &u, &query_expanded);
    crate::test_oracle::lwe_matrix_mul_body(&mut product, &u, &query_expanded);

    // Reference: decrypt the LWE matrix directly under the raw LWE secret.
    let mut ref_pt = module.glwe_plaintext_alloc_from_infos(&src_infos);
    module.lwe_matrix_decrypt(&product, &mut ref_pt, &sk_lwe, &mut scratch.borrow());
    let mut reference = vec![0i64; n];
    ref_pt.decode_vec_i64(&mut reference, TorusPrecision(k_pt as u32));

    // Build the aggregate.
    let size = matrix_infos.size();
    let mut aggregate = module.vec_znx_alloc(n, size);
    module.packing_mask_preprocessing(
        &mut aggregate,
        base2k.as_usize(),
        product.mask(),
        &mut scratch.borrow(),
    );

    // Decrypt the aggregate as the MLWE ciphertext it represents:
    //   m̂ = b̄_agg + Σ_col â_agg[col] · ŝ[col].
    // Try BOTH secret-share signs to disambiguate whether the aggregate is
    // correct (and the share-sign convention is the bug) or the aggregate is
    // wrong. `use_inv = true` mirrors the existing expansion decrypt test.
    let n_half = n >> 1;
    let h_list: Vec<i64> = (0..n_half)
        .map(|i| module.galois_element(i as i64))
        .collect();

    let decrypt_with_sign = |use_inv: bool, scratch: &mut ScratchOwned<BE>| -> Vec<i64> {
        let mut acc_big = module.vec_znx_big_alloc(1, size);
        {
            let mut acc_big_mut = acc_big.to_backend_mut();
            let body_ref = VecZnxToBackendRef::<BE>::to_backend_ref(product.body());
            module.vec_znx_big_from_small_backend(&mut acc_big_mut, 0, &body_ref, 0);
        }

        let mut svp = module.svp_ppol_alloc(1);
        let mut acc_dft = module.vec_znx_dft_alloc(1, size);
        let mut tmp_dft = module.vec_znx_dft_alloc(1, size);
        let aggregate_ref = VecZnxToBackendRef::<BE>::to_backend_ref(&aggregate);

        for col in 0..n {
            let p = if col < n_half {
                h_list[col]
            } else {
                -h_list[col - n_half]
            };
            let secret_auto = if use_inv {
                module.galois_element_inv(p)
            } else {
                p
            };
            let mut auto_secret = module.scalar_znx_alloc(1);
            {
                let secret_ref = ScalarZnxToBackendRef::<BE>::to_backend_ref(sk_lwe.data());
                let mut auto_secret_mut =
                    ScalarZnxToBackendMut::<BE>::to_backend_mut(&mut auto_secret);
                module.scalar_znx_automorphism_backend(
                    secret_auto,
                    &mut auto_secret_mut,
                    0,
                    &secret_ref,
                    0,
                );
            }
            {
                let mut svp_mut = svp.to_backend_mut();
                let auto_secret_ref = ScalarZnxToBackendRef::<BE>::to_backend_ref(&auto_secret);
                module.svp_prepare(&mut svp_mut, 0, &auto_secret_ref, 0);
            }
            {
                let mut tmp_dft_mut = tmp_dft.to_backend_mut();
                let svp_ref = svp.to_backend_ref();
                module.svp_apply_dft(&mut tmp_dft_mut, 0, &svp_ref, 0, &aggregate_ref, col);
            }
            {
                let mut acc_dft_mut = acc_dft.to_backend_mut();
                let tmp_dft_ref = tmp_dft.to_backend_ref();
                module.vec_znx_dft_add_assign(&mut acc_dft_mut, 0, &tmp_dft_ref, 0);
            }
        }

        let mut product_big = module.vec_znx_big_alloc(1, size);
        {
            let mut product_big_mut = product_big.to_backend_mut();
            let mut acc_dft_mut = acc_dft.to_backend_mut();
            module.vec_znx_idft_apply_tmpa(&mut product_big_mut, 0, &mut acc_dft_mut, 0);
        }
        {
            let product_big_ref = product_big.to_backend_ref();
            let mut acc_big_mut = acc_big.to_backend_mut();
            module.vec_znx_big_add_assign(&mut acc_big_mut, 0, &product_big_ref, 0);
        }

        let mut decrypted_pt = module.vec_znx_alloc(1, size);
        {
            let acc_big_ref = acc_big.to_backend_ref();
            let mut decrypted_pt_mut = VecZnxToBackendMut::<BE>::to_backend_mut(&mut decrypted_pt);
            module.vec_znx_big_normalize(
                &mut decrypted_pt_mut,
                base2k.as_usize(),
                0,
                0,
                &acc_big_ref,
                base2k.as_usize(),
                0,
                &mut scratch.borrow(),
            );
        }

        let mut decoded = vec![0i64; n];
        decrypted_pt.decode_vec_i64(base2k.as_usize(), 0, k_pt, &mut decoded);
        decoded
    };

    let decoded_inv = decrypt_with_sign(true, &mut scratch);
    let decoded_pos = decrypt_with_sign(false, &mut scratch);
    let exact = |d: &[i64]| {
        d.iter()
            .zip(reference.iter())
            .filter(|(g, w)| g == w)
            .count()
    };
    println!(
        "aggregate decrypt (general U): inv-share {}/{n}, pos-share {}/{n}",
        exact(&decoded_inv),
        exact(&decoded_pos)
    );

    // The aggregate is a CORRECT MLWE for general input: it decrypts to U*data
    // under the positive secret share ŝ[col] = τ_{g^{+col}}(s̄), matching the
    // paper's Algorithm 1 TRANSFORM convention (â[j] = τ_g^{+j}(ã)) that the
    // whole pipeline now agrees on. The inverse share only matches for
    // symmetric expansion masks.
    assert_eq!(
        decoded_pos, reference,
        "aggregate of U * expand(query) must decrypt to U * data under the positive share τ_{{g^{{+col}}}}"
    );
    assert_ne!(
        decoded_inv, reference,
        "the inverse share τ_{{g^{{-col}}}} only matches for symmetric (expansion) masks"
    );
}

#[test]
fn aggregate_decrypts_general_lwe_matrix() {
    run();
}
