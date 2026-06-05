//! Tests for the recursive (InsPIRe²) server collapse, split by phase:
//! [`partial_pack`] (phase 0), [`first_level`], [`second_level`], and the
//! [`end_to_end`] double-PIR + API round-trips.

mod end_to_end;
mod first_level;
mod partial_pack;
mod second_level;

use poulpy_core::layouts::{GLWESecret, LWESecret, ModuleCoreAlloc, Rank};
use poulpy_cpu_avx::FFT64Avx;
use poulpy_hal::{
    api::ScalarZnxAutomorphismBackend,
    layouts::{Backend, Module, ScalarZnxToBackendMut, ScalarZnxToBackendRef},
};

type BE = FFT64Avx;

pub(super) fn glwe_secret_wrap_lwe(
    module: &Module<BE>,
    sk_lwe: &LWESecret<<BE as Backend>::OwnedBuf>,
) -> GLWESecret<<BE as Backend>::OwnedBuf> {
    let mut sk_glwe = module.glwe_secret_alloc(Rank(1));
    sk_glwe.fill_zero();
    {
        let src_ref = ScalarZnxToBackendRef::<BE>::to_backend_ref(sk_lwe.data());
        let mut dst_mut = ScalarZnxToBackendMut::<BE>::to_backend_mut(sk_glwe.data_mut());
        module.scalar_znx_automorphism_backend(1, &mut dst_mut, 0, &src_ref, 0);
    }
    sk_glwe
}
