use poulpy_core::layouts::{
    GGLWEInfos, GGLWEToBackendRef, GLWEInfos, GLWEToBackendMut, GetGaloisElement,
    LWECompressedToBackendRef, LWEInfos,
};
use poulpy_hal::layouts::{Backend, Module};

pub trait InspirePackLWEToGLWE<BE: Backend> {
    fn lwe_pack_to_glwe_tmp_bytes(&self) -> usize;
    fn lwe_pack_to_glwe<R, L, G, H>(&self, _dst: &mut R, _src: &[L], _key_g: &G, _key_h: &H)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        L: LWECompressedToBackendRef<BE> + LWEInfos,
        G: GGLWEToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GGLWEToBackendRef<BE> + GetGaloisElement + GGLWEInfos;
}

impl<BE: Backend> InspirePackLWEToGLWE<BE> for Module<BE> {
    fn lwe_pack_to_glwe_tmp_bytes(&self) -> usize {
        0
    }

    fn lwe_pack_to_glwe<R, L, G, H>(&self, _dst: &mut R, _src: &[L], _key_g: &G, _key_h: &H)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        L: LWECompressedToBackendRef<BE> + LWEInfos,
        G: GGLWEToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GGLWEToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
    {
    }
}
