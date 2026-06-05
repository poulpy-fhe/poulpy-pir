use crate::{encoding::ModPEncoder, interpolation::MonomialInterpolation};
use poulpy_core::layouts::ModuleCoreAlloc;
use poulpy_cpu_avx::FFT64Avx;
use poulpy_hal::{
    api::{
        ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAddAssignBackend,
        VecZnxCopyBackend, VecZnxRotateAssignBackend, VecZnxRotateBackend, VecZnxSubBackend,
    },
    layouts::{
        Backend, HostDataMut, HostDataRef, Module, ScratchArena, ScratchOwned, VecZnx,
        VecZnxToBackendMut, VecZnxToBackendRef,
    },
};

fn horner_eval_at_monomial_root<BE>(
    module: &Module<BE>,
    coeffs: &[VecZnx<BE::OwnedBuf>],
    root_exp: i64,
    scratch: &mut ScratchArena<'_, BE>,
) -> VecZnx<BE::OwnedBuf>
where
    BE: Backend,
    Module<BE>: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>
        + VecZnxAddAssignBackend<BE>
        + VecZnxRotateAssignBackend<BE>
        + VecZnxCopyBackend<BE>,
    VecZnx<BE::OwnedBuf>: VecZnxToBackendMut<BE> + VecZnxToBackendRef<BE>,
{
    assert!(!coeffs.is_empty());
    let size = coeffs[0].size();
    let mut acc = module.vec_znx_alloc(1, size);

    let coeffs_last = coeffs.last().unwrap();
    let coeffs_last_ref = coeffs_last.to_backend_ref();

    module.vec_znx_copy_backend(&mut acc.to_backend_mut(), 0, &coeffs_last_ref, 0);

    for coeff in coeffs[..coeffs.len() - 1].iter().rev() {
        module.vec_znx_rotate_assign_backend(root_exp, &mut acc.to_backend_mut(), 0, scratch);
        module.vec_znx_add_assign_backend(&mut acc.to_backend_mut(), 0, &coeff.to_backend_ref(), 0);
    }

    acc
}

#[allow(clippy::needless_range_loop)]
fn run<BE>()
where
    BE: Backend,
    BE::OwnedBuf: HostDataMut + HostDataRef,
    Module<BE>: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>
        + ModuleNew<BE>
        + VecZnxAddAssignBackend<BE>
        + VecZnxRotateAssignBackend<BE>
        + VecZnxSubBackend<BE>
        + VecZnxCopyBackend<BE>
        + VecZnxRotateBackend<BE>
        + MonomialInterpolation<BE>,
    VecZnx<BE::OwnedBuf>: VecZnxToBackendMut<BE> + VecZnxToBackendRef<BE>,
{
    let n: usize = 32;
    let t: usize = 2 * n;
    let base2k: usize = 18;
    let glwe_torus_bits: usize = 36;
    let p: i64 = 65537;
    let size: usize = glwe_torus_bits.div_ceil(base2k);
    let encoder = ModPEncoder::new(p, glwe_torus_bits);
    let module = Module::<BE>::new(n as u64);

    let mut eval_polys: Vec<VecZnx<BE::OwnedBuf>> =
        (0..t).map(|_| module.vec_znx_alloc(1, size)).collect();
    let mut expected: Vec<Vec<i64>> = vec![vec![0i64; n]; t];

    for (point, eval_poly) in eval_polys.iter_mut().enumerate() {
        let mut values = vec![0i64; n];
        for coeff in 0..n {
            let value = ((point as i64 * 4099 + coeff as i64 * 917 + (point * coeff) as i64 * 37)
                % p)
                - (p / 2);
            values[coeff] = encoder.normalize(value);
        }
        encoder.encode_vec_i64(eval_poly, base2k, 0, &values);
        expected[point] = values;
    }

    let mut scratch = ScratchOwned::<BE>::alloc(module.monomial_interpolate_tmp_bytes(size));
    module.monomial_interpolate(&mut eval_polys, 0, &mut scratch.borrow());

    // `interpolate` returns t*c. Normalize over Z_p, then re-embed
    // with the encoder's 1/p torus scaling before Horner evaluation.
    let inv_t = encoder.inv(t as i64);
    let mut decoded_coeffs = vec![0i64; n];
    for coeff_poly in &mut eval_polys {
        encoder.decode_vec_i64(coeff_poly, base2k, 0, &mut decoded_coeffs);
        for coeff in decoded_coeffs.iter_mut() {
            *coeff = encoder.mul(*coeff, inv_t);
        }
        encoder.encode_vec_i64(coeff_poly, base2k, 0, &decoded_coeffs);
    }

    for point in 0..t {
        let got =
            horner_eval_at_monomial_root(&module, &eval_polys, point as i64, &mut scratch.borrow());
        let mut decoded = vec![0i64; n];
        encoder.decode_vec_i64(&got, base2k, 0, &mut decoded);

        assert_eq!(
            decoded, expected[point],
            "Horner evaluation at w^{point} did not select point {point}",
        );
    }
}

#[test]
fn interpolate_horner_selects_root_point() {
    run::<FFT64Avx>();
}
