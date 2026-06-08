//! Packs `U * query` where `U` is applied to the expanded GLWE (an LWE matrix),
//! then runs the usual packing pipeline (mask preprocessing, precompute, pack,
//! decrypt) and checks the result against the LWE-matrix's own decryption.
//!
//! `U` is a database-style coefficient matrix multiplied into the expanded
//! query via [`poulpy_core::LWEMatrixMul`]. The reference is obtained by
//! decrypting the resulting LWE matrix directly under the raw LWE secret, so the
//! test asserts that packing reproduces exactly what the LWE matrix encrypts.
use crate::packing::{
    Packing, PackingKeysGenerate, PackingMaskAggregation, PackingPrecomputeInfos,
};
use poulpy_core::{
    EncryptionLayout, GLWECompressedEncryptSk, GLWEDecrypt, GLWEExpandLWEMatrix, LWEMatrixDecrypt,
    layouts::{
        Base2K, Degree, GLWEAutomorphismKeyLayout, GLWEDecompress, GLWELayout, GLWEPlaintext,
        GLWESecret, GLWESecretPreparedFactory, LWEInfos, LWEMatrixLayout, LWESecret,
        ModuleCoreAlloc, ModuleCoreCompressedAlloc, Rank, SecretConversion, TorusPrecision,
    },
};
use poulpy_cpu_avx::FFT64Avx;

use crate::database::CoeffMatrix;
use poulpy_hal::{
    api::{ScalarZnxAutomorphismBackend, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, ScalarZnxToBackendMut, ScalarZnxToBackendRef, ScratchOwned},
    source::Source,
};

const N: usize = 64;

fn run() {
    type BE = FFT64Avx;
    let n = N;
    let module = Module::<BE>::new(n as u64);
    let base2k = 18usize;
    let k_ct = 2 * base2k;
    let k_pt = 16usize;
    let dsize = 1usize;
    let dnum = k_ct.div_ceil(base2k * dsize);
    let k_ksk = k_ct + base2k * dsize;
    let baby_size = 8usize;
    let query_seed = [17u8; 32];
    let packing_key_seed = [21u8; 32];

    let src_infos = EncryptionLayout::new_from_default_sigma(GLWELayout {
        n: Degree(n as u32),
        base2k: Base2K(base2k as u32),
        k: TorusPrecision(k_ct as u32),
        rank: Rank(1),
    })
    .unwrap();
    let matrix_infos = LWEMatrixLayout {
        rows: n,
        n: Degree(n as u32),
        base2k: src_infos.base2k(),
        k: src_infos.max_k(),
    };
    let key_infos = EncryptionLayout::new_from_default_sigma(GLWEAutomorphismKeyLayout {
        n: Degree(n as u32),
        base2k: Base2K(base2k as u32),
        k: TorusPrecision(k_ksk as u32),
        rank: Rank(1),
        dnum: dnum.into(),
        dsize: dsize.into(),
    })
    .unwrap();
    let precompute_metadata =
        PackingPrecomputeInfos::new(n - 1, matrix_infos.size(), base2k, baby_size);
    let mut aggregate = module.vec_znx_alloc(n, matrix_infos.size());

    let mut scratch = ScratchOwned::<BE>::alloc(
        module
            .glwe_compressed_encrypt_sk_tmp_bytes(&src_infos)
            .max(module.glwe_decrypt_tmp_bytes(&src_infos))
            .max(module.lwe_matrix_decrypt_tmp_bytes(&matrix_infos))
            .max(module.packing_mask_preprocessing_tmp_bytes(matrix_infos.size()))
            .max(module.pack_keys_generate_tmp_bytes(&key_infos))
            .max(module.pack_keys_precompute_tmp_bytes(&key_infos, &key_infos, baby_size))
            .max(module.pack_precompute_tmp_bytes(precompute_metadata, &aggregate, &key_infos)),
    );

    let mut source_xs = Source::new([11u8; 32]);
    let mut source_xe = Source::new([12u8; 32]);

    let mut sk_lwe = module.lwe_secret_alloc(Degree(n as u32));
    sk_lwe.fill_ternary_prob(0.5, &mut source_xs);
    let sk_src = module.glwe_secret_from_lwe_secret(&sk_lwe);
    let sk_pack = glwe_secret_wrap_lwe(&module, &sk_lwe);

    let mut sk_src_prep = module.glwe_secret_prepared_alloc_from_infos(&sk_src);
    module.glwe_secret_prepare(&mut sk_src_prep, &sk_src);
    let mut sk_pack_prep = module.glwe_secret_prepared_alloc_from_infos(&sk_pack);
    module.glwe_secret_prepare(&mut sk_pack_prep, &sk_pack);

    let (key_g, key_h) = module.pack_keys_generate(
        &key_infos,
        &sk_lwe,
        packing_key_seed,
        &mut source_xe,
        &mut scratch.borrow(),
    );
    let key_precomputations =
        module.pack_keys_precompute(&key_g, &key_h, baby_size, &mut scratch.borrow());

    // Encrypt the query (compressed) encoding `data`.
    let data: Vec<i64> = (0..n).map(|i| (i as i64 % 7) - 3).collect();
    let mut pt = module.glwe_plaintext_alloc_from_infos(&src_infos);
    pt.encode_vec_i64(&data, TorusPrecision(k_pt as u32));
    let mut query = module.glwe_compressed_alloc_from_infos(&src_infos);
    module.glwe_compressed_encrypt_sk(
        &mut query,
        &pt,
        &sk_src_prep,
        query_seed,
        &src_infos,
        &mut source_xe,
        &mut scratch.borrow(),
    );

    // Database multiplier U: a cyclic-shift permutation (row i selects query row
    // (i + 1) mod n). This is a non-diagonal U applied to the expanded query.
    let mut u = CoeffMatrix::zeros(n, n);
    for i in 0..n {
        u.row_mut(i)[(i + 1) % n] = 1;
    }

    // product = U * expand(query): mask (precompute side) then body (hot side).
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

    // Usual packing pipeline on the U-multiplied LWE matrix.
    module.packing_mask_preprocessing(
        &mut aggregate,
        base2k,
        product.mask(),
        &mut scratch.borrow(),
    );
    let mut precompute = module.pack_precompute_alloc(
        precompute_metadata.steps(),
        precompute_metadata.size(),
        precompute_metadata.base2k(),
        precompute_metadata.baby_size(),
    );
    module.pack_precompute(&mut precompute, &aggregate, &key_g, &mut scratch.borrow());

    let mut packed = module.glwe_alloc_from_infos(&src_infos);
    module.pack(
        &mut packed,
        product.body(),
        &precompute,
        &key_precomputations,
        1,
        &mut scratch.borrow(),
    );

    let mut decoded_pt: GLWEPlaintext<_> = module.glwe_plaintext_alloc_from_infos(&src_infos);
    module.glwe_decrypt(
        &packed,
        &mut decoded_pt,
        &sk_pack_prep,
        &mut scratch.borrow(),
    );
    let mut decoded = vec![0i64; n];
    decoded_pt.decode_vec_i64(&mut decoded, TorusPrecision(k_pt as u32));

    let exact = decoded
        .iter()
        .zip(reference.iter())
        .filter(|(g, w)| g == w)
        .count();
    println!("pack(U * expand(query)): {exact}/{n} exact slots");
    println!("decoded   = {:?}", &decoded[..8]);
    println!("reference = {:?}", &reference[..8]);

    assert_eq!(
        decoded, reference,
        "packing U * expand(query) must decrypt to U * data"
    );
}

fn glwe_secret_wrap_lwe(
    module: &Module<FFT64Avx>,
    sk_lwe: &LWESecret<<FFT64Avx as Backend>::OwnedBuf>,
) -> GLWESecret<<FFT64Avx as Backend>::OwnedBuf> {
    let mut sk_glwe = module.glwe_secret_alloc(Rank(1));
    sk_glwe.fill_zero();
    {
        let src_ref = ScalarZnxToBackendRef::<FFT64Avx>::to_backend_ref(sk_lwe.data());
        let mut dst_mut = ScalarZnxToBackendMut::<FFT64Avx>::to_backend_mut(sk_glwe.data_mut());
        module.scalar_znx_automorphism_backend(1, &mut dst_mut, 0, &src_ref, 0);
    }
    sk_glwe
}

// Regression test for the secret-share SIGN bug that broke packing for
// non-diagonal `U`. Per the InspiRING paper (Algorithm 1), `pack` is general and
// consumes arbitrary LWE ciphertexts. The bug was a share-direction mismatch:
// the mask aggregate (`packing_mask_preprocessing`) and the COLLAPSE disagreed on
// whether component `col` is masked by `τ_{g^{+col}}(s̄)` or `τ_{g^{-col}}(s̄)`.
// Symmetric expansion masks hid it (both signs coincide); a non-diagonal `U`
// exposed it. Fixed by aligning the aggregate to the COLLAPSE's inverse-share
// convention (the `galois_element_inv` in `packing_mask_preprocessing`). This
// test packs `U * expand(query)` for a cyclic-shift `U` and checks it decrypts
// to `U * data`.
#[test]
fn pack_u_times_expanded_query_decrypts() {
    run();
}
