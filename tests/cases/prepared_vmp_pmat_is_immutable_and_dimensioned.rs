use poulpy_core::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWEInfos, GGLWEPreparedFactory, GGLWEPreparedVmpPMatRef,
    LWEInfos, ModuleCoreAlloc, Rank, TorusPrecision,
};
use poulpy_cpu_ref::FFT64Ref;
use poulpy_hal::{
    api::ModuleNew,
    layouts::{Backend, Module},
};

fn run<BE>()
where
    BE: Backend,
    Module<BE>: GGLWEPreparedFactory<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf> + ModuleNew<BE>,
{
    let n = 1024usize;
    let module = Module::<BE>::new(n as u64);
    let base2k = Base2K(18);
    let k = TorusPrecision(36);
    let dnum = Dnum(2);
    let dsize = Dsize(1);

    let prepared = module.gglwe_prepared_alloc(base2k, k, Rank(1), Rank(0), dnum, dsize);
    let vmp = prepared.vmp_pmat_backend_ref();

    assert_eq!(prepared.n(), Degree(n as u32));
    assert_eq!(prepared.base2k(), base2k);
    assert_eq!(prepared.dnum(), dnum);
    assert_eq!(prepared.dsize(), dsize);
    assert_eq!(vmp.rows(), dnum.as_usize());
    assert_eq!(vmp.cols_in(), 1);
    assert_eq!(vmp.cols_out(), 1);
    assert_eq!(vmp.size(), k.0.div_ceil(base2k.0) as usize);
}

#[test]
fn prepared_vmp_pmat_is_immutable_and_dimensioned() {
    run::<FFT64Ref>();
}
