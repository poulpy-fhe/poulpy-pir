use crate::{
    encoding::ModPEncoder,
    interpolation::{HornerCoeffs, HornerEvaluation, MonomialInterpolation},
};
use poulpy_core::{
    EncryptionLayout, GGSWEncryptSk, GLWEAdd, GLWECopy, GLWEDecrypt, GLWEEncryptSk,
    GLWEExternalProduct, GLWENoise, ScratchArenaTakeCore,
    layouts::{
        Base2K, Degree, Dnum, Dsize, GGSWLayout, GGSWPreparedFactory, GLWELayout, GLWEPlaintext,
        GLWESecret, GLWESecretPreparedFactory, ModuleCoreAlloc, Rank, TorusPrecision,
    },
};
use poulpy_cpu_avx::FFT64Avx;
use poulpy_hal::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{
        Backend, HostBackend, HostDataMut, HostDataRef, Module, ScalarZnx, ScratchArena,
        ScratchOwned, VecZnx, VecZnxToBackendMut, VecZnxToBackendRef, ZnxView, ZnxViewMut,
    },
    source::Source,
};

fn root_monomial<BE>(module: &Module<BE>, exponent: usize) -> ScalarZnx<BE::OwnedBuf>
where
    BE: Backend,
    Module<BE>: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
    BE::OwnedBuf: HostDataMut,
{
    let n = module.n();
    let mut root = module.scalar_znx_alloc(1);
    if exponent < n {
        root.at_mut(0, 0)[exponent] = 1;
    } else {
        root.at_mut(0, 0)[exponent - n] = -1;
    }
    root
}

#[allow(clippy::needless_range_loop)]
fn run<BE>()
where
    BE: Backend + HostBackend,
    BE::OwnedBuf: HostDataMut + HostDataRef,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: GGSWEncryptSk<BE>
        + GGSWPreparedFactory<BE>
        + GLWEAdd<BE>
        + GLWECopy<BE>
        + GLWEDecrypt<BE>
        + GLWEEncryptSk<BE>
        + GLWEExternalProduct<BE>
        + GLWENoise<BE>
        + GLWESecretPreparedFactory<BE>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>
        + ModuleNew<BE>
        + MonomialInterpolation<BE>
        + HornerEvaluation<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    ScalarZnx<BE::OwnedBuf>: poulpy_hal::layouts::ScalarZnxToBackendRef<BE>,
    VecZnx<BE::OwnedBuf>: VecZnxToBackendMut<BE> + VecZnxToBackendRef<BE>,
{
    let n: usize = 32;
    let t: usize = 2 * n;
    let p: i64 = 65535;
    let base2k: usize = 18;
    let glwe_torus_bits: usize = 54;
    let ggsw_torus_bits: usize = 54;
    let rank = Rank(1);
    let encoder = ModPEncoder::new(p, glwe_torus_bits);
    let module = Module::<BE>::new(n as u64);

    let glwe_infos = EncryptionLayout::new_from_default_sigma(GLWELayout {
        n: Degree(n as u32),
        base2k: Base2K(base2k as u32),
        k: TorusPrecision(glwe_torus_bits as u32),
        rank,
    })
    .unwrap();
    let ggsw_infos = EncryptionLayout::new_from_default_sigma(GGSWLayout {
        n: Degree(n as u32),
        base2k: Base2K(base2k as u32),
        k: TorusPrecision(ggsw_torus_bits as u32),
        dnum: Dnum(3),
        dsize: Dsize(1),
        rank,
    })
    .unwrap();

    let mut eval_polys: Vec<VecZnx<BE::OwnedBuf>> =
        (0..t).map(|_| module.vec_znx_alloc(1, 1)).collect();
    let mut expected: Vec<Vec<i64>> = vec![vec![0i64; n]; t];

    for (point, (values, eval_poly)) in expected.iter_mut().zip(eval_polys.iter_mut()).enumerate() {
        for coeff in 0..n {
            let value = ((point as i64 * 8191 + coeff as i64 * 1297 + (point * coeff) as i64 * 59)
                % p)
                - (p / 2);
            values[coeff] = encoder.normalize(value);
        }
        eval_poly.at_mut(0, 0).copy_from_slice(values);
    }

    let scratch_bytes = module
        .monomial_interpolate_tmp_bytes(1)
        .max(module.glwe_encrypt_sk_tmp_bytes(&glwe_infos))
        .max(module.glwe_decrypt_tmp_bytes(&glwe_infos))
        .max(module.glwe_noise_tmp_bytes(&glwe_infos))
        .max(module.ggsw_encrypt_sk_tmp_bytes(&ggsw_infos))
        .max(module.ggsw_prepare_tmp_bytes(&ggsw_infos))
        .max(module.horner_evaluate_tmp_bytes(&glwe_infos, &ggsw_infos));
    let mut scratch = ScratchOwned::<BE>::alloc(scratch_bytes);

    module.monomial_interpolate(&mut eval_polys, 0, &mut scratch.borrow());

    let inv_t = encoder.inv(t as i64);
    let mut coeff_values: Vec<Vec<i64>> = vec![vec![0i64; n]; t];
    for (coeff_poly, values) in eval_polys.iter().zip(coeff_values.iter_mut()) {
        for (value, &raw) in values.iter_mut().zip(coeff_poly.at(0, 0)) {
            *value = encoder.mul(encoder.normalize(raw), inv_t);
        }
    }

    let mut source_xs = Source::new([11u8; 32]);
    let mut source_xe = Source::new([12u8; 32]);
    let mut source_xa = Source::new([13u8; 32]);

    let mut sk: GLWESecret<BE::OwnedBuf> = module.glwe_secret_alloc(rank);
    sk.fill_ternary_prob(0.5, &mut source_xs);
    let mut sk_prepared = module.glwe_secret_prepared_alloc(rank);
    module.glwe_secret_prepare(&mut sk_prepared, &sk);

    let mut coeff_cts = Vec::with_capacity(t);
    let mut plaintext = module.glwe_plaintext_alloc_from_infos(&glwe_infos);
    for values in &coeff_values {
        encoder.encode_vec_i64(&mut plaintext.data, base2k, 0, values);

        let mut ct = module.glwe_alloc_from_infos(&glwe_infos);
        module.glwe_encrypt_sk(
            &mut ct,
            &plaintext,
            &sk_prepared,
            &glwe_infos,
            &mut source_xe,
            &mut source_xa,
            &mut scratch.borrow(),
        );
        coeff_cts.push(ct);
    }

    for point in 0..t {
        let root_pt = root_monomial(&module, point);
        let mut root_ct = module.ggsw_alloc_from_infos(&ggsw_infos);
        module.ggsw_encrypt_sk(
            &mut root_ct,
            &root_pt,
            &sk_prepared,
            &ggsw_infos,
            &mut source_xe,
            &mut source_xa,
            &mut scratch.borrow(),
        );

        let mut root_prepared = module.ggsw_prepared_alloc_from_infos(&root_ct);
        module.ggsw_prepare(&mut root_prepared, &root_ct, &mut scratch.borrow());

        let helper = HornerCoeffs(&coeff_cts);

        let mut res = module.glwe_alloc_from_infos(&glwe_infos);
        module.horner_evaluate(&mut res, &helper, &root_prepared, &mut scratch.borrow());

        let mut want_pt: GLWEPlaintext<BE::OwnedBuf> =
            module.glwe_plaintext_alloc_from_infos(&glwe_infos);
        encoder.encode_vec_i64(&mut want_pt.data, base2k, 0, &expected[point]);
        let noise_log2 = module
            .glwe_noise(&res, &want_pt, &sk_prepared, &mut scratch.borrow())
            .std()
            .log2();
        println!("encrypted Horner w^{point} noise log2(std) = {noise_log2:.3}");

        let mut got_pt: GLWEPlaintext<BE::OwnedBuf> =
            module.glwe_plaintext_alloc_from_infos(&glwe_infos);
        module.glwe_decrypt(&res, &mut got_pt, &sk_prepared, &mut scratch.borrow());

        let mut decoded = vec![0i64; n];
        encoder.decode_vec_i64(&got_pt.data, base2k, 0, &mut decoded);
        assert_eq!(decoded, expected[point], "encrypted Horner at w^{point}");
    }
}

#[test]
fn encrypted_horner_rgsw_selects_root_point() {
    run::<FFT64Avx>();
}
