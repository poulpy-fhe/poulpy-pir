use crate::encoding::ModPEncoder;
use poulpy_core::layouts::ModuleCoreAlloc;
use poulpy_cpu_ref::FFT64Ref;
use poulpy_hal::{
    api::{ModuleNew, VecZnxAddAssignBackend, VecZnxRotateBackend},
    layouts::{
        Backend, HostDataMut, HostDataRef, Module, VecZnx, VecZnxToBackendMut, VecZnxToBackendRef,
    },
};

fn negacyclic_rotate_mod_p(src: &[i64], shift: i64, encoder: ModPEncoder) -> Vec<i64> {
    let n = src.len() as i64;
    let period = 2 * n;
    let shift = shift.rem_euclid(period);
    let mut dst = vec![0i64; src.len()];

    for (idx, &value) in src.iter().enumerate() {
        let rotated = (idx as i64 + shift).rem_euclid(period);
        if rotated < n {
            dst[rotated as usize] = encoder.normalize(value);
        } else {
            dst[(rotated - n) as usize] = encoder.sub(0, value);
        }
    }

    dst
}

fn run_forward_backward<BE>()
where
    BE: Backend,
    BE::OwnedBuf: HostDataMut + HostDataRef,
    Module<BE>: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf> + ModuleNew<BE>,
{
    let n = 32usize;
    let p = 65537i64;
    let module = Module::<BE>::new(n as u64);

    let mut values = vec![0i64; n];
    for (idx, value) in values.iter_mut().enumerate() {
        *value = ((idx as i64 * 12345) - 70000).rem_euclid(p);
    }
    values[0] = 0;
    values[1] = 1;
    values[2] = -1;
    values[3] = p / 2;
    values[4] = p / 2 + 1;

    for (base2k, torus_bits) in [(18usize, 36usize), (18, 54), (27, 54)] {
        let encoder = ModPEncoder::new(p, torus_bits);
        let mut encoded = module.vec_znx_alloc(1, torus_bits.div_ceil(base2k));

        encoder.encode_vec_i64(&mut encoded, base2k, 0, &values);

        let mut decoded = vec![0i64; n];
        encoder.decode_vec_i64(&encoded, base2k, 0, &mut decoded);

        let expected: Vec<i64> = values
            .iter()
            .map(|&value| encoder.normalize(value))
            .collect();
        assert_eq!(
            decoded, expected,
            "base2k {base2k}, torus_bits {torus_bits}"
        );
    }
}

fn run_rotate_add<BE>()
where
    BE: Backend,
    BE::OwnedBuf: HostDataMut + HostDataRef,
    Module<BE>: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>
        + ModuleNew<BE>
        + VecZnxAddAssignBackend<BE>
        + VecZnxRotateBackend<BE>,
    VecZnx<BE::OwnedBuf>: VecZnxToBackendMut<BE> + VecZnxToBackendRef<BE>,
{
    let n = 32usize;
    let base2k = 18usize;
    let torus_bits = 36usize;
    let p = 65537i64;
    let encoder = ModPEncoder::new(p, torus_bits);
    let module = Module::<BE>::new(n as u64);
    let size = torus_bits.div_ceil(base2k);

    let mut lhs_values = vec![0i64; n];
    let mut rhs_values = vec![0i64; n];
    for idx in 0..n {
        lhs_values[idx] = ((idx as i64 * 223 + 17) % p) - (p / 2);
        rhs_values[idx] = ((idx as i64 * 397 + 991) % p) - (p / 2);
    }

    for shift in [-35i64, -7, -1, 0, 1, 9, 31, 34] {
        let mut lhs = module.vec_znx_alloc(1, size);
        let mut rhs = module.vec_znx_alloc(1, size);
        let mut got = module.vec_znx_alloc(1, size);

        encoder.encode_vec_i64(&mut lhs, base2k, 0, &lhs_values);
        encoder.encode_vec_i64(&mut rhs, base2k, 0, &rhs_values);

        module.vec_znx_rotate_backend(
            shift,
            &mut got.to_backend_mut(),
            0,
            &lhs.to_backend_ref(),
            0,
        );
        module.vec_znx_add_assign_backend(&mut got.to_backend_mut(), 0, &rhs.to_backend_ref(), 0);

        let mut decoded = vec![0i64; n];
        encoder.decode_vec_i64(&got, base2k, 0, &mut decoded);

        let rotated = negacyclic_rotate_mod_p(&lhs_values, shift, encoder);
        let mut expected = vec![0i64; n];
        for idx in 0..n {
            expected[idx] = encoder.add(rotated[idx], rhs_values[idx]);
        }

        assert_eq!(decoded, expected, "shift {shift}");
    }
}

#[test]
fn mod_p_encoder_round_trips() {
    run_forward_backward::<FFT64Ref>();
}

#[test]
fn mod_p_encoder_rotate_add_matches_mod_p() {
    run_rotate_add::<FFT64Ref>();
}
