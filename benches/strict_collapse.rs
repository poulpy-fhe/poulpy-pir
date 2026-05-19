use poulpy_core::{
    EncryptionLayout, GLWEAutomorphismKeyEncryptSk, GLWECopy, GLWEEncryptSk, GLWEExpandLWEMatrix,
    GLWEKeyswitch, GLWEPacking, GLWESwitchingKeyEncryptSk, GLWEZero,
    layouts::{
        Base2K, Degree, GGLWEAtViewMut, GGLWEAtViewRef, GGLWEInfos, GGLWEPreparedFactory, GLWE,
        GLWEAutomorphismKey, GLWEAutomorphismKeyLayout, GLWEAutomorphismKeyPreparedFactory,
        GLWELayout, GLWESecret, GLWESecretPreparedFactory, GLWESwitchingKey,
        GLWESwitchingKeyLayout, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, LWEMatrixLayout,
        ModuleCoreAlloc, Rank, SecretConversion, TorusPrecision,
        prepared::{
            GGLWEPrepared, GLWEAutomorphismKeyPrepared, GLWESecretPrepared,
            GLWESwitchingKeyPreparedFactory,
        },
    },
};
use poulpy_cpu_avx::FFT64Avx;
use poulpy_hal::{
    api::{
        ModuleNew, ScalarZnxAutomorphismBackend, ScratchOwnedAlloc, ScratchOwnedBorrow,
        VecZnxAddAssignBackend, VecZnxAutomorphismBackend, VecZnxBigAddSmallAssign,
        VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxCopyBackend,
        VecZnxDftAddAssign, VecZnxDftApply, VecZnxDftBytesOf, VecZnxIdftApply,
        VecZnxIdftApplyTmpBytes, VmpApplyDftToDft, VmpApplyDftToDftTmpBytes,
    },
    layouts::{
        Backend, GaloisElement, HostDataMut, HostDataRef, Module, ScalarZnx, ScalarZnxToBackendMut,
        ScalarZnxToBackendRef, ScratchOwned, VecZnx, VecZnxToBackendMut, VecZnxToBackendRef,
        ZnxView,
    },
    source::Source,
};
use poulpy_pir::circuit::{
    AggregateLWE, precompute_sequential_keyswitch_collapse_aggregate_mask,
    precompute_sequential_keyswitch_collapse_aggregate_mask_dft_accumulated,
    precompute_sequential_keyswitch_collapse_aggregate_mask_dft_accumulated_tmp_bytes,
    precompute_sequential_keyswitch_collapse_aggregate_mask_tmp_bytes,
    sequential_collapse_mask_precompute_alloc, sequential_keyswitch_collapse_aggregate_mask,
    sequential_keyswitch_collapse_aggregate_mask_precomputed,
    sequential_keyswitch_collapse_aggregate_mask_precomputed_tmp_bytes,
    sequential_keyswitch_collapse_aggregate_mask_tmp_bytes,
};
use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

const ITERATIONS: usize = 5;

fn main() {
    run::<FFT64Avx>(18);
}

fn run<BE>(base2k: usize)
where
    BE: Backend,
    BE::OwnedBuf: HostDataMut + HostDataRef,
    Module<BE>: AggregateLWE<BE>
        + GGLWEPreparedFactory<BE>
        + GLWEAutomorphismKeyEncryptSk<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + GLWECopy<BE>
        + GLWEEncryptSk<BE>
        + GLWEExpandLWEMatrix<BE>
        + GLWEKeyswitch<BE>
        + GLWEPacking<BE>
        + GLWESwitchingKeyEncryptSk<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEZero<BE>
        + GLWESwitchingKeyPreparedFactory<BE>
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
        + VecZnxDftAddAssign<BE>
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
    let k_ct: usize = 36;
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
    let auto_key_infos = EncryptionLayout::new_from_default_sigma(GLWEAutomorphismKeyLayout {
        n: Degree(n as u32),
        base2k: Base2K(base2k as u32),
        k: TorusPrecision(k_ksk as u32),
        rank: Rank(1),
        dnum: dnum.into(),
        dsize: dsize.into(),
    })
    .unwrap();

    let mut scratch = ScratchOwned::<BE>::alloc(
        module
            .glwe_encrypt_sk_tmp_bytes(&src_infos)
            .max(module.glwe_expand_lwe_matrix_tmp_bytes(&matrix_infos, &src_infos))
            .max(module.aggregate_lwe_tmp_bytes(matrix_infos.size()))
            .max(module.glwe_switching_key_encrypt_sk_tmp_bytes(&key_infos))
            .max(module.glwe_switching_key_prepare_tmp_bytes(&key_infos))
            .max(module.gglwe_prepare_tmp_bytes(&key_infos))
            .max(module.glwe_automorphism_key_encrypt_sk_tmp_bytes(&auto_key_infos))
            .max(module.glwe_automorphism_key_prepare_tmp_bytes(&auto_key_infos))
            .max(module.glwe_pack_tmp_bytes(&dst_infos, &auto_key_infos))
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
            let src_ref = ScalarZnxToBackendRef::<BE>::to_backend_ref(sk_lwe.data());
            let mut share_mut = ScalarZnxToBackendMut::<BE>::to_backend_mut(share.data_mut());
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

    let gal_els = module.glwe_pack_galois_elements();
    let mut auto_keys: HashMap<i64, GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE>> = HashMap::new();
    let mut auto_key: GLWEAutomorphismKey<BE::OwnedBuf> =
        module.glwe_automorphism_key_alloc_from_infos(&auto_key_infos);
    for gal_el in gal_els {
        module.glwe_automorphism_key_encrypt_sk(
            &mut auto_key,
            gal_el,
            &sk_src,
            &auto_key_infos,
            &mut source_xe,
            &mut source_xa,
            &mut scratch.borrow(),
        );
        let mut auto_key_prepared =
            module.glwe_automorphism_key_prepared_alloc_from_infos(&auto_key);
        module.glwe_automorphism_key_prepare(
            &mut auto_key_prepared,
            &auto_key,
            &mut scratch.borrow(),
        );
        auto_keys.insert(gal_el, auto_key_prepared);
    }

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
    let mut key_g_prepared = module.glwe_switching_key_prepared_alloc_from_infos(&key_g);
    module.glwe_switching_key_prepare(&mut key_g_prepared, &key_g, &mut scratch.borrow());
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
    let mut key_h_prepared = module.glwe_switching_key_prepared_alloc_from_infos(&key_h);
    module.glwe_switching_key_prepare(&mut key_h_prepared, &key_h, &mut scratch.borrow());
    let key_h_body = split_output_key(&module, &key_h, 0, &mut scratch);
    let key_h_mask = split_output_key(&module, &key_h, 1, &mut scratch);

    let data: Vec<i64> = (0..n).map(|i| (i as i64 % 7) - 3).collect();
    let mut pt = module.glwe_plaintext_alloc_from_infos(&src_infos);
    pt.encode_vec_i64(&data, TorusPrecision(16));

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

    let packing_cts_template: Vec<GLWE<BE::OwnedBuf>> = (0..n)
        .map(|_| {
            let mut ct = module.glwe_alloc_from_infos(&dst_infos);
            module.glwe_copy(&mut ct, &src);
            ct
        })
        .collect();
    let mut packing_cts_work: Vec<GLWE<BE::OwnedBuf>> = (0..n)
        .map(|_| module.glwe_alloc_from_infos(&dst_infos))
        .collect();
    let mut packing_res = module.glwe_alloc_from_infos(&dst_infos);

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
    let mut precompute_dft = sequential_collapse_mask_precompute_alloc::<BE>(
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
    let precompute_dft_bytes =
        precompute_sequential_keyswitch_collapse_aggregate_mask_dft_accumulated_tmp_bytes::<
            BE,
            _,
            _,
            _,
        >(
            &module,
            &aggregate,
            &key_g_mask,
            &key_h_mask,
            key_infos.size(),
            key_infos.size(),
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
    assert!(
        BE::len_bytes(&scratch.data)
            >= precompute_bytes.max(precompute_dft_bytes).max(online_bytes)
    );

    let mut baseline = module.glwe_alloc_from_infos(&dst_infos);
    let mut optimized = module.glwe_alloc_from_infos(&dst_infos);

    sequential_keyswitch_collapse_aggregate_mask(
        &module,
        &mut baseline,
        lwe_matrix.body(),
        &aggregate,
        &key_g_prepared,
        &key_h_prepared,
        key_infos.size(),
        key_infos.size(),
        &mut scratch.borrow(),
    );
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
    sequential_keyswitch_collapse_aggregate_mask_precomputed(
        &module,
        &mut optimized,
        lwe_matrix.body(),
        &precompute,
        &key_g_body,
        &key_h_body,
        key_infos.size(),
        key_infos.size(),
        &mut scratch.borrow(),
    );
    assert_eq!(baseline.data().raw(), optimized.data().raw());

    let baseline_avg = time_average(ITERATIONS, || {
        sequential_keyswitch_collapse_aggregate_mask(
            &module,
            &mut baseline,
            lwe_matrix.body(),
            &aggregate,
            &key_g_prepared,
            &key_h_prepared,
            key_infos.size(),
            key_infos.size(),
            &mut scratch.borrow(),
        );
    });

    let precompute_avg = time_average(ITERATIONS, || {
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
    });

    let precompute_dft_avg = time_average(ITERATIONS, || {
        precompute_sequential_keyswitch_collapse_aggregate_mask_dft_accumulated(
            &module,
            &mut precompute_dft,
            &aggregate,
            &key_g_mask,
            &key_h_mask,
            key_infos.size(),
            key_infos.size(),
            &mut scratch.borrow(),
        );
    });

    let online_avg = time_average(ITERATIONS, || {
        sequential_keyswitch_collapse_aggregate_mask_precomputed(
            &module,
            &mut optimized,
            lwe_matrix.body(),
            &precompute,
            &key_g_body,
            &key_h_body,
            key_infos.size(),
            key_infos.size(),
            &mut scratch.borrow(),
        );
    });

    let mut packing_total = Duration::ZERO;
    for _ in 0..ITERATIONS {
        for (dst, src) in packing_cts_work.iter_mut().zip(packing_cts_template.iter()) {
            module.glwe_copy(dst, src);
        }
        let mut cts_map: HashMap<usize, &mut GLWE<BE::OwnedBuf>> = HashMap::new();
        for (slot, ct) in packing_cts_work.iter_mut().enumerate() {
            cts_map.insert(slot, ct);
        }
        let start = Instant::now();
        module.glwe_pack(
            &mut packing_res,
            cts_map,
            0,
            &auto_keys,
            auto_key_infos.size(),
            &mut scratch.borrow(),
        );
        packing_total += start.elapsed();
    }
    let packing_avg = packing_total / ITERATIONS as u32;

    println!("strict collapse benchmark");
    println!("  n: {n}");
    println!("  iterations: {ITERATIONS}");
    println!(
        "  baseline online full 1x2 collapse:      {:>10.3} ms",
        millis(baseline_avg)
    );
    println!(
        "  offline fixed-mask precompute:          {:>10.3} ms",
        millis(precompute_avg)
    );
    println!(
        "  offline DFT-accumulated precompute:     {:>10.3} ms",
        millis(precompute_dft_avg)
    );
    println!(
        "  online precomputed body-only collapse:  {:>10.3} ms",
        millis(online_avg)
    );
    println!(
        "  core GLWE pack of n GLWEs (log n keys): {:>10.3} ms",
        millis(packing_avg)
    );
}

fn time_average(iterations: usize, mut f: impl FnMut()) -> Duration {
    let start = Instant::now();
    for _ in 0..iterations {
        f();
    }
    start.elapsed() / iterations as u32
}

fn millis(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn split_output_key<BE, K>(
    module: &Module<BE>,
    key: &K,
    output_col: usize,
    scratch: &mut ScratchOwned<BE>,
) -> GGLWEPrepared<BE::OwnedBuf, BE>
where
    BE: Backend,
    Module<BE>:
        GGLWEPreparedFactory<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf> + VecZnxCopyBackend<BE>,
    K: GGLWEAtViewRef<BE> + GGLWEInfos,
{
    let mut split = module.gglwe_alloc(
        key.base2k(),
        key.max_k(),
        key.rank_in(),
        Rank(0),
        key.dnum(),
        key.dsize(),
    );

    for row in 0..key.dnum().as_usize() {
        for col in 0..key.rank_in().as_usize() {
            let src = GGLWEAtViewRef::<BE>::at_view(key, row, col);
            let mut dst = GGLWEAtViewMut::<BE>::at_view_mut(&mut split, row, col);
            let src_ref = src.to_backend_ref();
            let mut dst_mut = dst.to_backend_mut();
            module.vec_znx_copy_backend(dst_mut.data_mut(), 0, src_ref.data(), output_col);
        }
    }

    let mut prepared = module.gglwe_prepared_alloc_from_infos(&split);
    module.gglwe_prepare(&mut prepared, &split, &mut scratch.borrow());
    prepared
}
