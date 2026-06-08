//! Delegate blanket impl: wire the user-facing [`Packing`] trait on
//! `Module<BE>` to the OEP `*Impl<BE>` surface.

#![allow(clippy::too_many_arguments)]

use poulpy_core::layouts::{
    GGLWECompressedSeed, GGLWEInfos, GLWEInfos, GLWEToBackendMut, GetGaloisElement,
    compressed::GGLWECompressedToBackendRef,
};
use poulpy_hal::layouts::{
    Backend, Module, ScratchArena, VecZnxToBackendMut, VecZnxToBackendRef, ZnxInfos,
};

use crate::packing::{
    Packing, PackingKeys, PackingMaskAggregation,
    oep::{PackingImpl, PackingMaskAggregationImpl},
    packing_precomputations::{PackingPrecomputations, PackingPrecomputeInfos},
};

impl<BE> PackingMaskAggregation<BE> for Module<BE>
where
    BE: Backend + PackingMaskAggregationImpl<BE>,
{
    fn packing_mask_preprocessing_tmp_bytes(&self, size: usize) -> usize {
        BE::packing_mask_preprocessing_tmp_bytes_impl(self, size)
    }

    fn packing_mask_preprocessing<R, A>(
        &self,
        dst: &mut R,
        base2k: usize,
        a: &A,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: VecZnxToBackendMut<BE> + ZnxInfos,
        A: VecZnxToBackendRef<BE> + ZnxInfos,
    {
        BE::packing_mask_preprocessing_impl(self, dst, base2k, a, scratch);
    }

    fn pack_partial_mask_preprocessing_tmp_bytes(&self, gamma: usize, size: usize) -> usize {
        BE::pack_partial_mask_preprocessing_tmp_bytes_impl(self, gamma, size)
    }

    fn packing_partial_mask_preprocessing<R, A>(
        &self,
        dst: &mut R,
        base2k: usize,
        gamma: usize,
        a: &A,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: VecZnxToBackendMut<BE> + ZnxInfos,
        A: VecZnxToBackendRef<BE> + ZnxInfos,
    {
        BE::pack_partial_mask_preprocessing_impl(self, dst, base2k, gamma, a, scratch);
    }
}

impl<BE> Packing<BE> for Module<BE>
where
    BE: Backend + PackingImpl<BE>,
{
    fn pack_keys_precompute_tmp_bytes<KG, KH>(
        &self,
        key_g: &KG,
        key_h: &KH,
        baby_size: usize,
    ) -> usize
    where
        KG: GGLWEInfos,
        KH: GGLWEInfos,
    {
        BE::pack_keys_precompute_tmp_bytes_impl(self, key_g, key_h, baby_size)
    }

    fn pack_keys_precompute<KG, KH>(
        &self,
        key_g: &KG,
        key_h: &KH,
        baby_size: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> PackingKeys<BE>
    where
        KG: GGLWECompressedSeed + GGLWECompressedToBackendRef<BE> + GGLWEInfos + GetGaloisElement,
        KH: GGLWECompressedSeed + GGLWECompressedToBackendRef<BE> + GGLWEInfos + GetGaloisElement,
    {
        BE::pack_keys_precompute_impl(self, key_g, key_h, baby_size, scratch)
    }

    fn pack_partial_keys_precompute<KG>(
        &self,
        key_g: &KG,
        stride: usize,
        baby_size: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> PackingKeys<BE>
    where
        KG: GGLWECompressedSeed + GGLWECompressedToBackendRef<BE> + GGLWEInfos + GetGaloisElement,
    {
        BE::pack_partial_keys_precompute_impl(self, key_g, stride, baby_size, scratch)
    }

    fn pack_precompute_alloc(
        &self,
        steps: usize,
        size: usize,
        base2k: usize,
        baby_size: usize,
    ) -> PackingPrecomputations<BE> {
        BE::pack_precompute_alloc_impl(self, steps, size, base2k, baby_size)
    }

    fn pack_partial_precompute_alloc(
        &self,
        steps: usize,
        size: usize,
        base2k: usize,
        baby_size: usize,
        stride: usize,
    ) -> PackingPrecomputations<BE> {
        BE::pack_partial_precompute_alloc_impl(self, steps, size, base2k, baby_size, stride)
    }

    fn pack_precompute_tmp_bytes<A, KMask>(
        &self,
        metadata: PackingPrecomputeInfos,
        aggregate_mask: &A,
        key_mask_source: &KMask,
    ) -> usize
    where
        A: ZnxInfos,
        KMask: GGLWEInfos,
    {
        BE::pack_precompute_tmp_bytes_impl(self, metadata, aggregate_mask, key_mask_source)
    }

    fn pack_precompute<A, KMask>(
        &self,
        precomputations: &mut PackingPrecomputations<BE>,
        aggregate_mask: &A,
        key_mask_source: &KMask,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        A: VecZnxToBackendRef<BE> + ZnxInfos,
        KMask: GGLWECompressedSeed + GGLWEInfos,
    {
        BE::pack_precompute_impl(
            self,
            precomputations,
            aggregate_mask,
            key_mask_source,
            scratch,
        );
    }

    fn pack_partial_precompute<A, KMask>(
        &self,
        precomputations: &mut PackingPrecomputations<BE>,
        aggregate_mask: &A,
        key_mask_source: &KMask,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        A: VecZnxToBackendRef<BE> + ZnxInfos,
        KMask: GGLWECompressedSeed + GGLWEInfos,
    {
        BE::pack_partial_precompute_impl(
            self,
            precomputations,
            aggregate_mask,
            key_mask_source,
            scratch,
        );
    }

    /// Forwards the public module API to the backend OEP hook.
    ///
    /// Keeping the delegate this small makes the split explicit: API shape
    /// lives in `api.rs`, backend specialization in `oep.rs`, and algorithmic
    /// work in `default.rs`/`bsgs_pack.rs`.
    fn pack<R, B>(
        &self,
        res: &mut R,
        body: &B,
        precomputations: &PackingPrecomputations<BE>,
        key_precomputations: &PackingKeys<BE>,
        chunk_size: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        B: VecZnxToBackendRef<BE> + ZnxInfos,
    {
        BE::pack_impl(
            self,
            res,
            body,
            precomputations,
            key_precomputations,
            chunk_size,
            scratch,
        );
    }
}
