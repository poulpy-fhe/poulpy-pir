use std::time::Instant;

use crate::common::{scalar_mut, scalar_ref, split_output_key};
use poulpy_core::{
    EncryptionLayout, GLWEDecrypt, GLWEEncryptSk, GLWEExpandLWEMatrix, GLWEKeyswitch,
    GLWESwitchingKeyEncryptSk,
    layouts::{
        Base2K, Degree, GGLWEPreparedFactory, GLWELayout, GLWEPlaintext, GLWESecret,
        GLWESecretPreparedFactory, GLWESwitchingKey, GLWESwitchingKeyLayout, LWEInfos,
        LWEMatrixLayout, ModuleCoreAlloc, Rank, SecretConversion, TorusPrecision,
        prepared::GLWESecretPrepared,
    },
};
use poulpy_cpu_avx::FFT64Avx;
use poulpy_hal::{
    api::{
        ModuleNew, ScalarZnxAutomorphismBackend, ScratchOwnedAlloc, ScratchOwnedBorrow,
        VecZnxAddAssignBackend, VecZnxAutomorphismBackend, VecZnxBigAddSmallAssign,
        VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxCopyBackend,
        VecZnxDftApply, VecZnxDftBytesOf, VecZnxIdftApply, VecZnxIdftApplyTmpBytes,
        VmpApplyDftToDft, VmpApplyDftToDftTmpBytes,
    },
    layouts::{
        Backend, GaloisElement, HostDataMut, HostDataRef, Module, ScalarZnx, ScalarZnxToBackendMut,
        ScalarZnxToBackendRef, ScratchOwned, VecZnx, VecZnxToBackendMut, VecZnxToBackendRef,
    },
    source::Source,
};
use poulpy_pir::circuit::{
    AggregateLWE, precompute_sequential_keyswitch_collapse_aggregate_mask,
    precompute_sequential_keyswitch_collapse_aggregate_mask_tmp_bytes,
    sequential_collapse_mask_precompute_alloc,
    sequential_keyswitch_collapse_aggregate_mask_precomputed,
    sequential_keyswitch_collapse_aggregate_mask_precomputed_tmp_bytes,
    sequential_keyswitch_collapse_aggregate_mask_tmp_bytes,
};

fn run<BE>()
where
    BE: Backend,
    BE::OwnedBuf: HostDataMut + HostDataRef,
    Module<BE>: AggregateLWE<BE>
        + GGLWEPreparedFactory<BE>
        + GLWEDecrypt<BE>
        + GLWEEncryptSk<BE>
        + GLWEExpandLWEMatrix<BE>
        + GLWEKeyswitch<BE>
        + GLWESwitchingKeyEncryptSk<BE>
        + GLWESecretPreparedFactory<BE>
        + GaloisElement
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>
        + ModuleNew<BE>
        + ScalarZnxAutomorphismBackend<BE>
        + SecretConversion<BE>
        + VecZnxAddAssignBackend<BE>
        + VecZnxAutomorphismBackend<BE>
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxBigBytesOf
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxCopyBackend<BE>
        + VecZnxDftApply<BE>
        + VecZnxDftBytesOf
        + VecZnxIdftApply<BE>
        + VecZnxIdftApplyTmpBytes
        + VmpApplyDftToDft<BE>
        + VmpApplyDftToDftTmpBytes,
    ScalarZnx<BE::OwnedBuf>: ScalarZnxToBackendMut<BE> + ScalarZnxToBackendRef<BE>,
    VecZnx<BE::OwnedBuf>: VecZnxToBackendMut<BE> + VecZnxToBackendRef<BE>,
{
    let n = 1024usize;
    let module = Module::<BE>::new(n as u64);
    let base2k = 18usize;
    let k_ct = 2 * base2k;
    let k_pt = 16usize;
    let dsize = 1usize;
    let dnum = k_ct.div_ceil(base2k * dsize);
    let k_ksk = k_ct + base2k * dsize;

    let src_infos = EncryptionLayout::new_from_default_sigma(GLWELayout {
        n: Degree(n as u32),
        base2k: Base2K(base2k as u32),
        k: TorusPrecision(k_ct as u32),
        rank: Rank(1),
    })
    .unwrap();
    let dst_infos = EncryptionLayout::new_from_default_sigma(GLWELayout {
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
    let key_infos = EncryptionLayout::new_from_default_sigma(GLWESwitchingKeyLayout {
        n: Degree(n as u32),
        base2k: Base2K(base2k as u32),
        k: TorusPrecision(k_ksk as u32),
        rank_in: Rank(1),
        rank_out: Rank(1),
        dnum: dnum.into(),
        dsize: dsize.into(),
    })
    .unwrap();

    let mut scratch = ScratchOwned::<BE>::alloc(
        module
            .glwe_encrypt_sk_tmp_bytes(&src_infos)
            .max(module.glwe_decrypt_tmp_bytes(&dst_infos))
            .max(module.glwe_expand_lwe_matrix_tmp_bytes(&matrix_infos, &src_infos))
            .max(module.aggregate_lwe_tmp_bytes(matrix_infos.size()))
            .max(module.glwe_switching_key_encrypt_sk_tmp_bytes(&key_infos))
            .max(module.gglwe_prepare_tmp_bytes(&key_infos))
            .max(sequential_keyswitch_collapse_aggregate_mask_tmp_bytes(
                &module,
                &dst_infos,
                &module.vec_znx_alloc(n, matrix_infos.size()),
                &key_infos,
                &key_infos,
            ))
            .max(1 << 26),
    );

    let mut source_xs = Source::new([11u8; 32]);
    let mut source_xe = Source::new([12u8; 32]);
    let mut source_xa = Source::new([13u8; 32]);

    let mut sk_src: GLWESecret<BE::OwnedBuf> = module.glwe_secret_alloc_from_infos(&src_infos);
    sk_src.fill_ternary_prob(0.5, &mut source_xs);
    let sk_lwe = module.lwe_secret_from_glwe_secret(&sk_src);

    let automorphic_lwe_share = |p: i64| {
        let mut share: GLWESecret<BE::OwnedBuf> = module.glwe_secret_alloc(Rank(1));
        share.fill_zero();
        {
            let src_ref = scalar_ref::<BE>(sk_lwe.data());
            let mut share_mut = scalar_mut::<BE>(share.data_mut());
            module.scalar_znx_automorphism_backend(
                module.galois_element_inv(p),
                &mut share_mut,
                0,
                &src_ref,
                0,
            );
        }
        share
    };

    let sk_base = automorphic_lwe_share(1);
    let sk_g = automorphic_lwe_share(module.galois_element(1));
    let sk_h = automorphic_lwe_share(-1);

    let mut sk_src_prep: GLWESecretPrepared<_, BE> =
        module.glwe_secret_prepared_alloc_from_infos(&sk_src);
    module.glwe_secret_prepare(&mut sk_src_prep, &sk_src);
    let mut sk_dst_prep: GLWESecretPrepared<_, BE> =
        module.glwe_secret_prepared_alloc_from_infos(&sk_base);
    module.glwe_secret_prepare(&mut sk_dst_prep, &sk_base);

    let mut key_g: GLWESwitchingKey<BE::OwnedBuf> =
        module.glwe_switching_key_alloc_from_infos(&key_infos);
    module.glwe_switching_key_encrypt_sk(
        &mut key_g,
        &sk_g,
        &sk_base,
        &key_infos,
        &mut source_xe,
        &mut source_xa,
        &mut scratch.borrow(),
    );
    let key_g_body = split_output_key(&module, &key_g, 0, &mut scratch);
    let key_g_mask = split_output_key(&module, &key_g, 1, &mut scratch);

    let mut key_h: GLWESwitchingKey<BE::OwnedBuf> =
        module.glwe_switching_key_alloc_from_infos(&key_infos);
    module.glwe_switching_key_encrypt_sk(
        &mut key_h,
        &sk_h,
        &sk_base,
        &key_infos,
        &mut source_xe,
        &mut source_xa,
        &mut scratch.borrow(),
    );
    let key_h_body = split_output_key(&module, &key_h, 0, &mut scratch);
    let key_h_mask = split_output_key(&module, &key_h, 1, &mut scratch);

    let data: Vec<i64> = (0..n).map(|i| (i as i64 % 7) - 3).collect();
    let mut pt: GLWEPlaintext<BE::OwnedBuf> = module.glwe_plaintext_alloc_from_infos(&src_infos);
    pt.encode_vec_i64(&data, TorusPrecision(k_pt as u32));

    let mut src = module.glwe_alloc_from_infos(&src_infos);
    module.glwe_encrypt_sk(
        &mut src,
        &pt,
        &sk_src_prep,
        &src_infos,
        &mut source_xe,
        &mut source_xa,
        &mut scratch.borrow(),
    );

    let mut lwe_matrix = module.lwe_matrix_alloc_from_infos(&matrix_infos);
    module.glwe_expand_lwe_matrix(&mut lwe_matrix, &src, &mut scratch.borrow());

    let mut aggregate = module.vec_znx_alloc(n, matrix_infos.size());
    module.aggregate_lwe(
        &mut aggregate,
        base2k,
        lwe_matrix.mask(),
        &mut scratch.borrow(),
    );

    let mut precompute = sequential_collapse_mask_precompute_alloc::<BE>(
        &module,
        n - 1,
        matrix_infos.size(),
        base2k,
        base2k,
        Rank(1),
    );
    let precompute_bytes =
        precompute_sequential_keyswitch_collapse_aggregate_mask_tmp_bytes::<BE, _, _, _>(
            &module,
            &aggregate,
            &key_g_mask,
            &key_h_mask,
            key_infos.size(),
            key_infos.size(),
        );
    assert!(BE::len_bytes(&scratch.data) >= precompute_bytes);

    let now = Instant::now();
    precompute_sequential_keyswitch_collapse_aggregate_mask(
        &module,
        &mut precompute,
        &aggregate,
        &key_g_mask,
        &key_h_mask,
        key_infos.size(),
        key_infos.size(),
        &mut scratch.borrow(),
    );
    println!(
        "precompute_sequential_keyswitch_collapse_aggregate_mask: {:?}",
        now.elapsed()
    );

    let online_bytes = sequential_keyswitch_collapse_aggregate_mask_precomputed_tmp_bytes(
        &module,
        &dst_infos,
        lwe_matrix.body(),
        &precompute,
        &key_g_body,
        &key_h_body,
        key_infos.size(),
        key_infos.size(),
    );
    assert!(BE::len_bytes(&scratch.data) >= online_bytes);

    let mut res = module.glwe_alloc_from_infos(&dst_infos);
    let now = Instant::now();
    sequential_keyswitch_collapse_aggregate_mask_precomputed(
        &module,
        &mut res,
        lwe_matrix.body(),
        &precompute,
        &key_g_body,
        &key_h_body,
        key_infos.size(),
        key_infos.size(),
        &mut scratch.borrow(),
    );
    println!(
        "sequential_keyswitch_collapse_aggregate_mask_precomputed: {:?}",
        now.elapsed()
    );

    let mut decoded_pt = module.glwe_plaintext_alloc_from_infos(&dst_infos);
    module.glwe_decrypt(&res, &mut decoded_pt, &sk_dst_prep, &mut scratch.borrow());
    let mut decoded = vec![0; n];
    decoded_pt.decode_vec_i64(&mut decoded, TorusPrecision(k_pt as u32));

    assert_eq!(decoded, data);
}

#[test]
fn precomputed_sequential_collapse_decrypts() {
    run::<FFT64Avx>();
}
