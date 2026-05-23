//! Verifies that the BSGS DFT-hot online collapse path
//! ([`crate::packing::Packing`]) decrypts to the original
//! plaintext.
//!
//! The body accumulator stays in the DFT domain through the whole schedule;
//! per-step normalizations are deferred to a single IDFT + normalize at the
//! end. So we verify decryption equality rather than ciphertext byte equality.
use super::aggregate::AggregateLWE;
use crate::packing::{Packing, PackingPrecomputeInfos};
use poulpy_core::{
    EncryptionLayout, GLWEAutomorphismKeyCompressedEncryptSk, GLWECompressedEncryptSk, GLWEDecrypt,
    GLWEExpandLWEMatrix, GLWENoise,
    layouts::{
        Base2K, Degree, GGLWEPreparedFactory, GLWEAutomorphismKeyCompressed,
        GLWEAutomorphismKeyLayout, GLWEDecompress, GLWELayout, GLWESecret,
        GLWESecretPreparedFactory, LWEInfos, LWEMatrixLayout, LWESecret, ModuleCoreAlloc,
        ModuleCoreCompressedAlloc, Rank, SecretConversion, TorusPrecision,
    },
};
use poulpy_cpu_avx::FFT64Avx;
use poulpy_hal::{
    api::{ScalarZnxAutomorphismBackend, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{
        Backend, GaloisElement, Module, ScalarZnxToBackendMut, ScalarZnxToBackendRef, ScratchOwned,
    },
    source::Source,
};

fn run() {
    type BE = FFT64Avx;
    let n = 64usize;
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
            .max(module.glwe_noise_tmp_bytes(&src_infos))
            .max(module.glwe_expand_lwe_matrix_tmp_bytes(&matrix_infos, &src_infos))
            .max(module.aggregate_lwe_tmp_bytes(matrix_infos.size()))
            .max(module.glwe_automorphism_key_compressed_encrypt_sk_tmp_bytes(&key_infos))
            .max(module.gglwe_prepare_tmp_bytes(&key_infos))
            .max(module.pack_keys_precompute_tmp_bytes(&key_infos, &key_infos, baby_size))
            .max(module.pack_precompute_tmp_bytes(precompute_metadata, &aggregate, &key_infos)),
    );

    let mut source_xs = Source::new([11u8; 32]);
    let mut source_xe = Source::new([12u8; 32]);

    let mut sk_lwe = module.lwe_secret_alloc(Degree(n as u32));
    sk_lwe.fill_ternary_prob(0.5, &mut source_xs);
    let sk_src = module.glwe_secret_from_lwe_secret(&sk_lwe);
    let sk_base = glwe_secret_wrap_lwe(&module, &sk_lwe);

    let mut sk_src_prep = module.glwe_secret_prepared_alloc_from_infos(&sk_src);
    module.glwe_secret_prepare(&mut sk_src_prep, &sk_src);
    let mut sk_dst_prep = module.glwe_secret_prepared_alloc_from_infos(&sk_base);
    module.glwe_secret_prepare(&mut sk_dst_prep, &sk_base);

    let (key_g, key_h) = encrypt_packing_keys(
        &module,
        &key_infos,
        &sk_base,
        packing_key_seed,
        &mut source_xe,
        &mut scratch,
    );
    let key_precomputations =
        module.pack_keys_precompute(&key_g, &key_h, baby_size, &mut scratch.borrow());

    let data: Vec<i64> = (0..n).map(|i| (i as i64 % 7) - 3).collect();
    let mut pt = module.glwe_plaintext_alloc_from_infos(&src_infos);
    pt.encode_vec_i64(&data, TorusPrecision(k_pt as u32));

    // Encrypts GLWE compressed (i.e. query)
    let mut src = module.glwe_compressed_alloc_from_infos(&src_infos);
    module.glwe_compressed_encrypt_sk(
        &mut src,
        &pt,
        &sk_src_prep,
        query_seed,
        &src_infos,
        &mut source_xe,
        &mut scratch.borrow(),
    );

    // Maps it to the LWEMatrix (but we skip the U * A part, doesn't matter for this test)
    let mut src_glwe = module.glwe_alloc_from_infos(&src_infos);
    module.decompress_glwe(&mut src_glwe, &src);
    let mut lwe_matrix = module.lwe_matrix_alloc_from_infos(&matrix_infos);
    module.glwe_expand_lwe_matrix(&mut lwe_matrix, &src_glwe, &mut scratch.borrow());
    module.aggregate_lwe(
        &mut aggregate,
        base2k,
        lwe_matrix.mask(),
        &mut scratch.borrow(),
    );

    // Precomputes packing shared masks
    let mut precompute = module.pack_precompute_alloc(
        precompute_metadata.steps(),
        precompute_metadata.size(),
        precompute_metadata.base2k(),
        precompute_metadata.baby_size(),
    );
    let precompute_bytes =
        module.pack_precompute_tmp_bytes(precompute_metadata, &aggregate, &key_g);
    assert!(BE::len_bytes(&scratch.data) >= precompute_bytes);
    module.pack_precompute(&mut precompute, &aggregate, &key_g, &mut scratch.borrow());

    // Uses precomputations to pack
    let mut bsgs_res = module.glwe_alloc_from_infos(&src_infos);
    module.pack(
        &mut bsgs_res,
        lwe_matrix.body(),
        &precompute,
        &key_precomputations,
        1,
        &mut scratch.borrow(),
    );

    let mut bsgs_decoded_pt = module.glwe_plaintext_alloc_from_infos(&src_infos);
    let bsgs_noise_log2 = module
        .glwe_noise(&bsgs_res, &pt, &sk_dst_prep, &mut scratch.borrow())
        .std()
        .log2();
    println!("BSGS DFT-hot collapse noise log2(std) = {bsgs_noise_log2:.3}");
    module.glwe_decrypt(
        &bsgs_res,
        &mut bsgs_decoded_pt,
        &sk_dst_prep,
        &mut scratch.borrow(),
    );
    let mut bsgs_decoded = vec![0; n];
    bsgs_decoded_pt.decode_vec_i64(&mut bsgs_decoded, TorusPrecision(k_pt as u32));

    assert_eq!(bsgs_decoded, data);
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

fn encrypt_packing_keys(
    module: &Module<FFT64Avx>,
    key_infos: &EncryptionLayout<GLWEAutomorphismKeyLayout>,
    sk_base: &GLWESecret<<FFT64Avx as Backend>::OwnedBuf>,
    key_seed: [u8; 32],
    source_xe: &mut Source,
    scratch: &mut ScratchOwned<FFT64Avx>,
) -> (
    GLWEAutomorphismKeyCompressed<<FFT64Avx as Backend>::OwnedBuf>,
    GLWEAutomorphismKeyCompressed<<FFT64Avx as Backend>::OwnedBuf>,
) {
    let mut key_g = module.glwe_automorphism_key_compressed_alloc_from_infos(key_infos);
    module.glwe_automorphism_key_compressed_encrypt_sk(
        &mut key_g,
        module.galois_element_inv(module.galois_element(1)),
        sk_base,
        key_seed,
        key_infos,
        source_xe,
        &mut scratch.borrow(),
    );
    let mut key_h = module.glwe_automorphism_key_compressed_alloc_from_infos(key_infos);
    module.glwe_automorphism_key_compressed_encrypt_sk(
        &mut key_h,
        -1,
        sk_base,
        key_seed,
        key_infos,
        source_xe,
        &mut scratch.borrow(),
    );
    (key_g, key_h)
}

#[test]
fn bsgs_dft_hot_collapse_decrypts() {
    run();
}
