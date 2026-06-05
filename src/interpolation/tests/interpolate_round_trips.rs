use poulpy_core::layouts::ModuleCoreAlloc;
use poulpy_cpu_avx::FFT64Avx;
use poulpy_hal::{
    api::{
        ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAddAssignBackend,
        VecZnxRotateBackend, VecZnxSubBackend,
    },
    layouts::{
        Backend, HostDataMut, HostDataRef, Module, ScratchOwned, VecZnx, VecZnxToBackendMut,
        VecZnxToBackendRef, ZnxView, ZnxViewMut,
    },
};

use crate::interpolation::MonomialInterpolation;

#[allow(clippy::needless_range_loop)]
fn run<BE>()
where
    BE: Backend,
    BE::OwnedBuf: HostDataMut + HostDataRef,
    Module<BE>: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>
        + ModuleNew<BE>
        + VecZnxAddAssignBackend<BE>
        + VecZnxRotateBackend<BE>
        + VecZnxSubBackend<BE>
        + MonomialInterpolation<BE>,
    VecZnx<BE::OwnedBuf>: VecZnxToBackendMut<BE> + VecZnxToBackendRef<BE>,
{
    let n: usize = 64;
    let t: usize = 8;
    let size: usize = 1;

    let module = Module::<BE>::new(n as u64);

    let mut c_polys: Vec<VecZnx<BE::OwnedBuf>> =
        (0..t).map(|_| module.vec_znx_alloc(1, size)).collect();
    let mut y_polys: Vec<VecZnx<BE::OwnedBuf>> =
        (0..t).map(|_| module.vec_znx_alloc(1, size)).collect();

    let mut c_values: Vec<Vec<i64>> = vec![vec![0i64; n]; t];
    for (j, c_j) in c_values.iter_mut().enumerate() {
        for (i, slot) in c_j.iter_mut().enumerate() {
            let v = (((j * 31 + i * 17) % 19) as i64) - 9;
            *slot = v;
        }
    }
    for (j, c_j) in c_values.iter().enumerate() {
        let coeffs = c_polys[j].at_mut(0, 0);
        coeffs.copy_from_slice(c_j);
    }

    let step: i64 = (2 * n / t) as i64;
    let mut tmp_eval: VecZnx<BE::OwnedBuf> = module.vec_znx_alloc(1, size);
    for k_idx in 0..t {
        for slot in y_polys[k_idx].at_mut(0, 0).iter_mut() {
            *slot = 0;
        }
        for j in 0..t {
            let exponent: i64 = (j as i64) * (k_idx as i64) * step;
            {
                let c_ref = c_polys[j].to_backend_ref();
                let mut tmp_mut = tmp_eval.to_backend_mut();
                module.vec_znx_rotate_backend(exponent, &mut tmp_mut, 0, &c_ref, 0);
            }
            let tmp_ref = tmp_eval.to_backend_ref();
            let mut y_k_mut = y_polys[k_idx].to_backend_mut();
            module.vec_znx_add_assign_backend(&mut y_k_mut, 0, &tmp_ref, 0);
        }
    }

    let mut scratch = ScratchOwned::<BE>::alloc(module.monomial_interpolate_tmp_bytes(size));
    module.monomial_interpolate(&mut y_polys, 0, &mut scratch.borrow());

    let t_i64 = t as i64;
    for j in 0..t {
        let got = y_polys[j].at(0, 0);
        let want = &c_values[j];
        for i in 0..n {
            assert_eq!(
                got[i],
                t_i64 * want[i],
                "coefficient mismatch at point j={j}, coeff i={i} (got {}, want t·c = {})",
                got[i],
                t_i64 * want[i],
            );
        }
    }
}

#[test]
fn interpolate_round_trips_naive_dft() {
    run::<FFT64Avx>();
}
