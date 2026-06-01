use poulpy_core::layouts::ModuleCoreAlloc;
use poulpy_cpu_avx::FFT64Avx;
use poulpy_hal::{
    api::{
        ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAddAssignBackend,
        VecZnxRotateBackend,
    },
    layouts::{
        Backend, HostDataMut, HostDataRef, Module, ScratchOwned, VecZnx, VecZnxToBackendMut,
        VecZnxToBackendRef, ZnxView, ZnxViewMut,
    },
};

use crate::interpolation::MonomialInterpolation;

fn select_ref<T>(value: &T) -> &T {
    value
}

fn select_mut<T>(value: &mut T) -> &mut T {
    value
}

/// Each `y_k` carries `cols` independent channels. Interpolating them one column
/// at a time over the *same* slice must recover each channel exactly, proving the
/// column-local bit-reversal does not corrupt neighbouring columns. This is the
/// property `Interpolation::prepare` relies on when it interpolates every
/// mask column of a shared `LWEMatrix`.
fn run<BE>()
where
    BE: Backend,
    BE::OwnedBuf: HostDataMut + HostDataRef,
    Module<BE>: ModuleNew<BE>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>
        + MonomialInterpolation<BE>
        + VecZnxAddAssignBackend<BE>
        + VecZnxRotateBackend<BE>,
    VecZnx<BE::OwnedBuf>: VecZnxToBackendMut<BE> + VecZnxToBackendRef<BE>,
{
    let n: usize = 64;
    let t: usize = 8;
    let cols: usize = 3;
    let size: usize = 1;

    let module = Module::<BE>::new(n as u64);

    let mut c_polys: Vec<VecZnx<BE::OwnedBuf>> =
        (0..t).map(|_| module.vec_znx_alloc(cols, size)).collect();
    let mut y_polys: Vec<VecZnx<BE::OwnedBuf>> =
        (0..t).map(|_| module.vec_znx_alloc(cols, size)).collect();

    // c_values[j][col] holds the n target coefficients of point j on channel col.
    let mut c_values: Vec<Vec<Vec<i64>>> = vec![vec![vec![0i64; n]; cols]; t];
    for j in 0..t {
        for col in 0..cols {
            for i in 0..n {
                let v = (((j * 31 + col * 53 + i * 17) % 23) as i64) - 11;
                c_values[j][col][i] = v;
            }
            c_polys[j].at_mut(col, 0).copy_from_slice(&c_values[j][col]);
        }
    }

    // Build the evaluation form per channel: y_k[col] = sum_j c_j[col] * X^(j*k*2n/t).
    let step: i64 = (2 * n / t) as i64;
    let mut tmp_eval: VecZnx<BE::OwnedBuf> = module.vec_znx_alloc(1, size);
    for k_idx in 0..t {
        for col in 0..cols {
            for slot in y_polys[k_idx].at_mut(col, 0).iter_mut() {
                *slot = 0;
            }
        }
        for col in 0..cols {
            for j in 0..t {
                let exponent: i64 = (j as i64) * (k_idx as i64) * step;
                {
                    let c_ref = c_polys[j].to_backend_ref();
                    let mut tmp_mut = tmp_eval.to_backend_mut();
                    module.vec_znx_rotate_backend(exponent, &mut tmp_mut, 0, &c_ref, col);
                }
                let tmp_ref = tmp_eval.to_backend_ref();
                let mut y_k_mut = y_polys[k_idx].to_backend_mut();
                module.vec_znx_add_assign_backend(&mut y_k_mut, col, &tmp_ref, 0);
            }
        }
    }

    let mut scratch = ScratchOwned::<BE>::alloc(module.monomial_interpolate_tmp_bytes(size));
    // Interpolate one column at a time over the shared slice.
    for col in 0..cols {
        module.monomial_interpolate(&mut y_polys, col, &mut scratch.borrow());
    }

    let t_i64 = t as i64;
    for j in 0..t {
        for col in 0..cols {
            let got = y_polys[j].at(col, 0);
            let want = &c_values[j][col];
            for i in 0..n {
                assert_eq!(
                    got[i],
                    t_i64 * want[i],
                    "mismatch at point j={j}, col={col}, coeff i={i} (got {}, want t·c = {})",
                    got[i],
                    t_i64 * want[i],
                );
            }
        }
    }
}

#[test]
fn interpolate_columns_independent_naive_dft() {
    run::<FFT64Avx>();
}
