//! Reference implementation of the BSGS DFT-hot packing `*Default` trait on
//! `Module<BE>`. The online method delegates to the free function in
//! [`crate::packing::bsgs_pack`]; bounds mirror those of the underlying function.

#![allow(clippy::too_many_arguments)]

use poulpy_core::{
    GLWEMaskFillDefault, ScratchArenaTakeCore,
    layouts::{
        GGLWECompressedSeed, GGLWEInfos, GGLWEPreparedFactory, GLWEInfos, GLWEToBackendMut,
        GetGaloisElement,
        compressed::GGLWECompressedToBackendRef,
        prepared::{GGLWEPrepared, GGLWEPreparedVmpPMatRef},
    },
};
use poulpy_hal::{
    api::{
        ScratchArenaTakeBasic, VecZnxAddAssignBackend, VecZnxAlloc, VecZnxAutomorphismBackend,
        VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxCopyBackend,
        VecZnxDftAddAssign, VecZnxDftAlloc, VecZnxDftApply, VecZnxDftAutomorphism,
        VecZnxDftAutomorphismPlan, VecZnxDftBytesOf, VecZnxDftZero, VecZnxIdftApply,
        VecZnxIdftApplyTmpBytes, VecZnxRotateAssignBackend, VecZnxRotateAssignTmpBytes,
        VecZnxRshAssignBackend, VecZnxRshTmpBytes, VecZnxTransposeBackend, VmpApplyDftToDft,
        VmpApplyDftToDftTmpBytes,
    },
    layouts::{
        Backend, GaloisElement, Module, ScratchArena, VecZnx, VecZnxToBackendMut,
        VecZnxToBackendRef, ZnxInfos,
    },
};

use crate::packing::{
    PackingKeyPrecomputations, PackingKeyPrecomputationsHelper,
    aggregate::{packing_mask_aggregate_default, packing_mask_aggregate_tmp_bytes_default},
    bsgs_pack,
    collapse_precompute::{
        PackingPrecomputations, PackingPrecomputeInfos, pack_precompute_alloc_default,
        precompute_sequential_keyswitch_collapse_aggregate_mask,
        precompute_sequential_keyswitch_collapse_aggregate_mask_tmp_bytes,
        sequential_collapse_bsgs_dft_build, sequential_collapse_bsgs_dft_build_tmp_bytes,
    },
    key_precompute::{
        mask_key_layout, pack_keys_precompute_default, packing_keys_precompute_tmp_bytes,
        packing_mask_keys_precompute, packing_mask_keys_precompute_tmp_bytes,
    },
};

#[doc(hidden)]
pub trait PackingMaskAggregationDefault<BE: Backend> {
    /// Default scratch estimate for packing-mask aggregation.
    fn packing_mask_aggregate_tmp_bytes_default(&self, size: usize) -> usize;

    /// Default packing-mask aggregation implementation.
    fn packing_mask_aggregate_default<R, A>(
        &self,
        dst: &mut R,
        base2k: usize,
        a: &A,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: VecZnxToBackendMut<BE> + ZnxInfos,
        A: VecZnxToBackendRef<BE> + ZnxInfos;
}

#[doc(hidden)]
#[allow(private_bounds)]
pub trait PackingDefault<BE: Backend> {
    /// Default scratch estimate for client-key-side precomputation.
    fn pack_keys_precompute_tmp_bytes_default<KG, KH>(
        &self,
        key_g: &KG,
        key_h: &KH,
        baby_size: usize,
    ) -> usize
    where
        KG: GGLWEInfos,
        KH: GGLWEInfos;

    /// Default client-key-side precompute implementation.
    fn pack_keys_precompute_default<KG, KH>(
        &self,
        key_g: &KG,
        key_h: &KH,
        baby_size: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> PackingKeyPrecomputations<GGLWEPrepared<BE::OwnedBuf, BE>>
    where
        KG: GGLWECompressedSeed + GGLWECompressedToBackendRef<BE> + GGLWEInfos + GetGaloisElement,
        KH: GGLWECompressedSeed + GGLWECompressedToBackendRef<BE> + GGLWEInfos + GetGaloisElement;

    /// Default fixed mask-side allocation implementation.
    fn pack_precompute_alloc_default(
        &self,
        steps: usize,
        size: usize,
        base2k: usize,
        baby_size: usize,
    ) -> PackingPrecomputations<BE>;

    /// Default scratch estimate for fixed mask-side precompute.
    fn pack_precompute_tmp_bytes_default<A, KMask>(
        &self,
        metadata: PackingPrecomputeInfos,
        aggregate_mask: &A,
        key_mask_source: &KMask,
    ) -> usize
    where
        A: ZnxInfos,
        KMask: GGLWEInfos;

    /// Default fixed mask-side precompute implementation.
    fn pack_precompute_default<A, KMask>(
        &self,
        precomputations: &mut PackingPrecomputations<BE>,
        aggregate_mask: &A,
        key_mask_source: &KMask,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        A: VecZnxToBackendRef<BE> + ZnxInfos,
        KMask: GGLWECompressedSeed + GGLWEInfos;

    /// Default online implementation used by backends that do not provide a
    /// specialized `PackingImpl`. It delegates to `bsgs_pack` so the OEP layer
    /// can stay thin.
    fn pack_default<R, B, P, K>(
        &self,
        res: &mut R,
        body: &B,
        precomputations: &PackingPrecomputations<BE>,
        key_precomputations: &P,
        chunk_size: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        B: VecZnxToBackendRef<BE> + ZnxInfos,
        P: PackingKeyPrecomputationsHelper<BE, K>,
        K: GGLWEPreparedVmpPMatRef<BE> + GGLWEInfos;
}

impl<BE: Backend> PackingMaskAggregationDefault<BE> for Module<BE>
where
    Self: VecZnxCopyBackend<BE>
        + VecZnxTransposeBackend<BE>
        + VecZnxAutomorphismBackend<BE>
        + VecZnxAddAssignBackend<BE>
        + VecZnxRotateAssignBackend<BE>
        + VecZnxRotateAssignTmpBytes
        + VecZnxRshAssignBackend<BE>
        + VecZnxRshTmpBytes
        + GaloisElement,
{
    fn packing_mask_aggregate_tmp_bytes_default(&self, size: usize) -> usize {
        packing_mask_aggregate_tmp_bytes_default(self, size)
    }

    fn packing_mask_aggregate_default<R, A>(
        &self,
        dst: &mut R,
        base2k: usize,
        a: &A,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: VecZnxToBackendMut<BE> + ZnxInfos,
        A: VecZnxToBackendRef<BE> + ZnxInfos,
    {
        packing_mask_aggregate_default(self, dst, base2k, a, scratch);
    }
}

impl<BE: Backend> PackingDefault<BE> for Module<BE>
where
    Module<BE>: GaloisElement
        + VecZnxAddAssignBackend<BE>
        + VecZnxAlloc<BE>
        + VecZnxAutomorphismBackend<BE>
        + VecZnxBigBytesOf
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxCopyBackend<BE>
        + VecZnxDftAddAssign<BE>
        + VecZnxDftAlloc<BE>
        + VecZnxDftApply<BE>
        + VecZnxDftAutomorphism<BE>
        + VecZnxDftBytesOf
        + VecZnxDftZero<BE>
        + VecZnxIdftApply<BE>
        + VecZnxIdftApplyTmpBytes
        + VmpApplyDftToDft<BE>,
    Module<BE>: GGLWEPreparedFactory<BE> + GLWEMaskFillDefault<BE>,
    Module<BE>: VmpApplyDftToDftTmpBytes,
    <Module<BE> as VecZnxDftAutomorphismPlan<BE>>::Plan: 'static,
    VecZnx<BE::OwnedBuf>: VecZnxToBackendRef<BE>,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeBasic<'a, BE> + ScratchArenaTakeCore<'a, BE>,
{
    fn pack_keys_precompute_tmp_bytes_default<KG, KH>(
        &self,
        key_g: &KG,
        key_h: &KH,
        baby_size: usize,
    ) -> usize
    where
        KG: GGLWEInfos,
        KH: GGLWEInfos,
    {
        packing_keys_precompute_tmp_bytes(self, key_g, key_h, baby_size)
    }

    fn pack_keys_precompute_default<KG, KH>(
        &self,
        key_g: &KG,
        key_h: &KH,
        baby_size: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> PackingKeyPrecomputations<GGLWEPrepared<BE::OwnedBuf, BE>>
    where
        KG: GGLWECompressedSeed + GGLWECompressedToBackendRef<BE> + GGLWEInfos + GetGaloisElement,
        KH: GGLWECompressedSeed + GGLWECompressedToBackendRef<BE> + GGLWEInfos + GetGaloisElement,
    {
        pack_keys_precompute_default(self, key_g, key_h, baby_size, scratch)
    }

    fn pack_precompute_alloc_default(
        &self,
        steps: usize,
        size: usize,
        base2k: usize,
        baby_size: usize,
    ) -> PackingPrecomputations<BE> {
        pack_precompute_alloc_default(self, steps, size, base2k, baby_size)
    }

    fn pack_precompute_tmp_bytes_default<A, KMask>(
        &self,
        metadata: PackingPrecomputeInfos,
        aggregate_mask: &A,
        key_mask_source: &KMask,
    ) -> usize
    where
        A: ZnxInfos,
        KMask: GGLWEInfos,
    {
        let key_mask_bytes = packing_mask_keys_precompute_tmp_bytes(self, key_mask_source);
        let key_mask_layout = mask_key_layout(key_mask_source);
        let mask_bytes = precompute_sequential_keyswitch_collapse_aggregate_mask_tmp_bytes(
            self,
            aggregate_mask,
            &key_mask_layout,
            &key_mask_layout,
        );
        let bsgs_bytes = sequential_collapse_bsgs_dft_build_tmp_bytes(self, metadata);
        key_mask_bytes.max(mask_bytes).max(bsgs_bytes)
    }

    fn pack_precompute_default<A, KMask>(
        &self,
        precomputations: &mut PackingPrecomputations<BE>,
        aggregate_mask: &A,
        key_mask_source: &KMask,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        A: VecZnxToBackendRef<BE> + ZnxInfos,
        KMask: GGLWECompressedSeed + GGLWEInfos,
    {
        let (key_g_mask, key_h_mask) =
            packing_mask_keys_precompute(self, key_mask_source, &mut scratch.borrow());
        precompute_sequential_keyswitch_collapse_aggregate_mask(
            self,
            precomputations,
            aggregate_mask,
            &key_g_mask,
            &key_h_mask,
            scratch,
        );
        sequential_collapse_bsgs_dft_build(self, precomputations, scratch);
    }

    fn pack_default<R, B, P, K>(
        &self,
        res: &mut R,
        body: &B,
        precomputations: &PackingPrecomputations<BE>,
        key_precomputations: &P,
        chunk_size: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        B: VecZnxToBackendRef<BE> + ZnxInfos,
        P: PackingKeyPrecomputationsHelper<BE, K>,
        K: GGLWEPreparedVmpPMatRef<BE> + GGLWEInfos,
    {
        bsgs_pack::pack_default(
            self,
            res,
            body,
            precomputations,
            key_precomputations,
            chunk_size,
            scratch,
        )
    }
}
