use poulpy_core::{
    GLWEAdd, GLWECopy, GLWEExternalProduct, ScratchArenaTakeCore,
    layouts::{
        GGSWInfos, GGSWPreparedToBackendRef, GLWE, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef,
    },
};
use poulpy_hal::{
    api::{
        ModuleN, ScratchArenaTakeBasic, VecZnxAddAssignBackend, VecZnxRotateBackend,
        VecZnxSubBackend,
    },
    layouts::{
        Backend, Module, ScratchArena, VecZnx, VecZnxToBackendMut, VecZnxToBackendRef, ZnxInfos,
    },
};

use crate::interpolation::HornerHelper;

/// Blanket impl: every backend's `Module` gets the default monomial
/// interpolation. The trait is local to this crate, so this is allowed by the
/// orphan rules and keeps the interpolation layer fully backend-agnostic (no
/// concrete backend is named anywhere in the library).
impl<BE: Backend> MonomialInterpolationDefault<BE> for Module<BE> {}

pub trait MonomialInterpolationDefault<BE: Backend>
where
    Self: Sized,
{
    fn monomial_interpolate_tmp_bytes_default(&self, size: usize) -> usize
    where
        Self: ModuleN,
    {
        VecZnx::<Vec<u8>>::bytes_of(self.n(), 1, size)
    }
    fn monomial_interpolate_default<Y>(
        &self,
        y: &mut [Y],
        col: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        Self: VecZnxAddAssignBackend<BE> + VecZnxRotateBackend<BE> + VecZnxSubBackend<BE> + ModuleN,
        Y: VecZnxToBackendMut<BE> + VecZnxToBackendRef<BE> + ZnxInfos,
    {
        idft(self, y, col, scratch);
    }
}

fn idft<BE, M, E>(module: &M, elems: &mut [E], col: usize, scratch: &mut ScratchArena<'_, BE>)
where
    BE: Backend,
    M: VecZnxAddAssignBackend<BE> + VecZnxRotateBackend<BE> + VecZnxSubBackend<BE> + ModuleN,
    E: VecZnxToBackendMut<BE> + VecZnxToBackendRef<BE> + ZnxInfos,
{
    let t: usize = elems.len();
    assert!(
        t.is_power_of_two(),
        "interpolate: number of points must be a power of two (got {t})"
    );
    if t < 2 {
        return;
    }

    let n: usize = module.n();
    assert!(
        n.is_power_of_two(),
        "interpolate: ring degree must be a power of two"
    );
    let size: usize = elems[0].size();

    #[cfg(debug_assertions)]
    {
        for slot in elems.iter() {
            debug_assert_eq!(slot.n(), n, "interpolate: y_k degree mismatch");
            debug_assert_eq!(slot.size(), size, "interpolate: y_k size mismatch");
            debug_assert!(col < slot.cols(), "interpolate: column index out of range");
        }
    }

    let arena = scratch.borrow();
    let (mut tmp, _) = arena.take_vec_znx_scratch(n, 1, size);

    let log_t: usize = t.trailing_zeros() as usize;

    // Bit-reversal permutation restricted to column `col` of the selected
    // channel, so that interpolating a different column (e.g. another mask
    // column) of the same `elems` slice stays independent. A whole-`VecZnx` swap
    // would shuffle every column and corrupt already-interpolated ones, so the
    // exchange is done column-by-column through `tmp` (a rotation by `0` is a
    // plain copy, reusing the only backend op this routine already requires).
    for i in 0..t {
        let j = bit_reverse(i, log_t);
        if i < j {
            let (lo_half, hi_half) = elems.split_at_mut(j);
            let lo: &mut E = &mut lo_half[i];
            let hi: &mut E = &mut hi_half[0];

            // tmp[0] <- hi[col]
            module.vec_znx_rotate_backend(
                0,
                &mut tmp.to_backend_mut(),
                0,
                &hi.to_backend_ref(),
                col,
            );
            // hi[col] <- lo[col]
            {
                let lo_ref = lo.to_backend_ref();
                module.vec_znx_rotate_backend(0, &mut hi.to_backend_mut(), col, &lo_ref, col);
            }
            // lo[col] <- tmp[0]
            module.vec_znx_rotate_backend(
                0,
                &mut lo.to_backend_mut(),
                col,
                &tmp.to_backend_ref(),
                0,
            );
        }
    }

    let two_n: i64 = (2 * n) as i64;
    let mut m: usize = 2;
    while m <= t {
        let half: usize = m >> 1;
        let step: i64 = two_n / m as i64;

        let mut start: usize = 0;
        while start < t {
            for j in 0..half {
                let lo_idx: usize = start + j;
                let hi_idx: usize = lo_idx + half;

                let exponent: i64 = -((j as i64) * step);

                let (lo_half, hi_half) = elems.split_at_mut(hi_idx);
                let lo: &mut E = &mut lo_half[lo_idx];
                let hi: &mut E = &mut hi_half[0];

                module.vec_znx_rotate_backend(
                    exponent,
                    &mut tmp.to_backend_mut(),
                    0,
                    &hi.to_backend_ref(),
                    col,
                );

                {
                    let lo_ref = lo.to_backend_ref();
                    module.vec_znx_sub_backend(
                        &mut hi.to_backend_mut(),
                        col,
                        &lo_ref,
                        col,
                        &tmp.to_backend_ref(),
                        0,
                    );
                }

                module.vec_znx_add_assign_backend(
                    &mut lo.to_backend_mut(),
                    col,
                    &tmp.to_backend_ref(),
                    0,
                );
            }
            start += m;
        }
        m <<= 1;
    }
}

#[inline]
fn bit_reverse(mut x: usize, bits: usize) -> usize {
    let mut r: usize = 0;
    for _ in 0..bits {
        r = (r << 1) | (x & 1);
        x >>= 1;
    }
    r
}

/// Blanket impl: every backend's `Module` whose ops satisfy the Horner
/// supertraits gets the default Horner evaluation. Local trait ⇒ orphan-rule
/// safe, and no concrete backend is named.
impl<BE: Backend> HornerEvaluationDefault<BE> for Module<BE> where
    Module<BE>: GLWEExternalProduct<BE> + GLWECopy<BE> + GLWEAdd<BE>
{
}

pub trait HornerEvaluationDefault<BE: Backend>
where
    Self: GLWEExternalProduct<BE> + GLWECopy<BE> + GLWEAdd<BE>,
{
    fn horner_evaluate_tmp_bytes_default<G, H>(&self, glwe_infos: &G, ggsw_infos: &H) -> usize
    where
        G: GLWEInfos,
        H: GGSWInfos,
    {
        self.glwe_external_product_tmp_bytes(glwe_infos, glwe_infos, ggsw_infos)
            + GLWE::<Vec<u8>>::bytes_of_from_infos(glwe_infos)
    }

    fn horner_evaluate_default<R, H, C, G>(
        &self,
        res: &mut R,
        coeffs: &H,
        power: &G,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        H: HornerHelper<BE, C>,
        C: GLWEToBackendRef<BE>,
        G: GGSWPreparedToBackendRef<BE> + GGSWInfos,
    {
        let t = coeffs.nb_coeffs();
        assert!(t > 0, "horner: no coefficients");
        let last = coeffs.get_coeff(t - 1);
        self.glwe_copy(res, last);
        if t == 1 {
            return;
        }

        let (mut product, mut scratch_1) = scratch.borrow().take_glwe_scratch(res);

        let size = power.size();
        for idx in (0..t - 1).rev() {
            self.glwe_external_product(&mut product, res, power, size, &mut scratch_1.borrow());
            self.glwe_add_into(res, &product, coeffs.get_coeff(idx));
        }
    }
}
