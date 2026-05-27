use crate::packing::PackingMaskAggregation;
use poulpy_core::layouts::{Base2K, ModuleCoreAlloc, TorusPrecision};
use poulpy_cpu_ref::FFT64Ref;
use poulpy_hal::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{
        Backend, HostDataMut, HostDataRef, Module, ScratchOwned, VecZnx, VecZnxToBackendMut,
        ZnxView, ZnxViewMut,
    },
};

fn run<BE>()
where
    BE: Backend,
    BE::OwnedBuf: HostDataMut + HostDataRef,
    Module<BE>:
        PackingMaskAggregation<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf> + ModuleNew<BE>,
    VecZnx<BE::OwnedBuf>: VecZnxToBackendMut<BE>,
{
    let n = 1024usize;
    let module = Module::<BE>::new(n as u64);
    let base2k = Base2K(18);
    let k = TorusPrecision(36);
    let mut lwe_matrix = module.lwe_matrix_alloc(n, n.into(), base2k, k);
    let size = lwe_matrix.mask().size();
    let mut lhs = module.vec_znx_alloc(n, size);
    let mut rhs = module.vec_znx_alloc(n, size);
    let mut changed = module.vec_znx_alloc(n, size);
    let mut scratch = ScratchOwned::<BE>::alloc(module.packing_mask_aggregate_tmp_bytes(size));

    for row in 0..n {
        lwe_matrix.body_mut().at_mut(0, 0)[row] = 100 + row as i64;
        for col in 0..n {
            for limb in 0..size {
                lwe_matrix.mask_mut().at_mut(col, limb)[row] =
                    512 * ((limb as i64 + 1) * 100 + (row as i64 + 1) * 10 + col as i64 + 1);
            }
        }
    }

    module.packing_mask_aggregate(
        &mut lhs,
        base2k.as_usize(),
        lwe_matrix.mask(),
        &mut scratch.borrow(),
    );
    for row in 0..n {
        lwe_matrix.body_mut().at_mut(0, 0)[row] += 1_000;
    }
    module.packing_mask_aggregate(
        &mut rhs,
        base2k.as_usize(),
        lwe_matrix.mask(),
        &mut scratch.borrow(),
    );

    let changed_limb = size - 1;
    lwe_matrix.mask_mut().at_mut(2, changed_limb)[1] += 4096;
    module.packing_mask_aggregate(
        &mut changed,
        base2k.as_usize(),
        lwe_matrix.mask(),
        &mut scratch.borrow(),
    );

    assert_eq!(lhs.n(), n);
    assert_eq!(lhs.cols(), n);
    assert_eq!(lhs.size(), size);
    assert_eq!(lhs.raw(), rhs.raw());
    assert_ne!(lhs.raw(), changed.raw());
}

#[test]
fn packing_mask_aggregate_uses_lwe_matrix_mask_rows() {
    run::<FFT64Ref>();
}
