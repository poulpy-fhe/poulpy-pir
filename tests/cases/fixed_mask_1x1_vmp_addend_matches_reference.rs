use poulpy_core::layouts::{
    Base2K, Dnum, Dsize, GGLWEPreparedFactory, GGLWEPreparedVmpPMatRef, LWEInfos, ModuleCoreAlloc,
    Rank, TorusPrecision,
};
use poulpy_cpu_ref::FFT64Ref;
use poulpy_hal::{
    api::{
        ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxBigAddSmallAssign, VecZnxBigAlloc,
        VecZnxBigNormalize, VecZnxDftAddAssign, VecZnxDftAlloc, VecZnxDftApply, VecZnxDftCopy,
        VecZnxDftZero, VecZnxIdftApply, VmpApplyDft, VmpApplyDftToDft, VmpPrepare,
    },
    layouts::{
        Backend, FillUniform, HostDataMut, HostDataRef, Module, ScratchOwned, VecZnx,
        VecZnxBigToBackendMut, VecZnxBigToBackendRef, VecZnxDftToBackendRef, VecZnxToBackendMut,
        VecZnxToBackendRef, ZnxView,
    },
    source::Source,
};
use poulpy_pir::circuit::fixed_mask_1x1_vmp_body_addend;

fn run<BE>()
where
    BE: Backend,
    BE::OwnedBuf: HostDataMut + HostDataRef,
    Module<BE>: GGLWEPreparedFactory<BE>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>
        + ModuleNew<BE>
        + VecZnxBigAlloc<BE>
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxDftAddAssign<BE>
        + VecZnxDftAlloc<BE>
        + VecZnxDftApply<BE>
        + VecZnxDftCopy<BE>
        + VecZnxDftZero<BE>
        + VecZnxIdftApply<BE>
        + VmpApplyDft<BE>
        + VmpApplyDftToDft<BE>
        + VmpPrepare<BE>,
{
    let n = 1024usize;
    let module = Module::<BE>::new(n as u64);
    let base2k = Base2K(18);
    let k = TorusPrecision(36);
    let dnum = Dnum(2);
    let dsize = Dsize(1);
    let size = k.0.div_ceil(base2k.0) as usize;

    let mut source = Source::new([27u8; 32]);
    let mut mask = module.vec_znx_alloc(1, size);
    mask.fill_uniform(base2k.as_usize(), &mut source);

    let mut key = module.gglwe_alloc(base2k, k, Rank(1), Rank(0), dnum, dsize);
    key.fill_uniform(base2k.as_usize(), &mut source);
    let mut prepared = module.gglwe_prepared_alloc_from_infos(&key);

    let scratch_bytes = module.gglwe_prepare_tmp_bytes(&key).max(1 << 20);
    let mut scratch = ScratchOwned::<BE>::alloc(scratch_bytes);
    module.gglwe_prepare(&mut prepared, &key, &mut scratch.borrow());

    let mut actual = module.vec_znx_alloc(1, size);
    fixed_mask_1x1_vmp_body_addend(
        &module,
        &mut actual,
        base2k.as_usize(),
        &mask,
        0,
        &prepared,
        prepared.size(),
        &mut scratch.borrow(),
    );

    let mut ref_dft = module.vec_znx_dft_alloc(1, prepared.size());
    module.vmp_apply_dft(
        &mut ref_dft,
        &<VecZnx<BE::OwnedBuf> as VecZnxToBackendRef<BE>>::to_backend_ref(&mask),
        &prepared.vmp_pmat_backend_ref(),
        &mut scratch.borrow(),
    );

    let mut ref_big = module.vec_znx_big_alloc(1, prepared.size());
    module.vec_znx_idft_apply(
        &mut ref_big.to_backend_mut(),
        0,
        &ref_dft.to_backend_ref(),
        0,
        &mut scratch.borrow(),
    );

    let mut expected = module.vec_znx_alloc(1, size);
    module.vec_znx_big_normalize(
        &mut <VecZnx<BE::OwnedBuf> as VecZnxToBackendMut<BE>>::to_backend_mut(&mut expected),
        base2k.as_usize(),
        0,
        0,
        &ref_big.to_backend_ref(),
        prepared.base2k().as_usize(),
        0,
        &mut scratch.borrow(),
    );

    assert_eq!(actual.raw(), expected.raw());
}

#[test]
fn fixed_mask_1x1_vmp_addend_matches_reference() {
    run::<FFT64Ref>();
}
