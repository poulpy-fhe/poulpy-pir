//! Reference implementation of the BSGS DFT-hot packing `*Default` trait on
//! `Module<BE>`. The online method delegates to the free function in
//! [`crate::packing::bsgs_pack`]; bounds mirror those of the underlying function.

#![allow(clippy::too_many_arguments)]

use poulpy_core::{
    GLWEMaskFillDefault,
    layouts::{
        GGLWECompressedSeed, GGLWEInfos, GGLWEPreparedFactory, GLWEInfos, GLWEToBackendMut,
        GetGaloisElement, ModuleCoreAlloc, compressed::GGLWECompressedToBackendRef,
    },
};
use poulpy_hal::{
    api::{
        VecZnxAddAssignBackend, VecZnxAlloc, VecZnxAutomorphismBackend,
        VecZnxAutomorphismRotateBackend, VecZnxBigBytesOf, VecZnxBigNormalize,
        VecZnxBigNormalizeTmpBytes, VecZnxCopyBackend, VecZnxDftAddAssign, VecZnxDftAlloc,
        VecZnxDftApply, VecZnxDftAutomorphism, VecZnxDftAutomorphismPlan, VecZnxDftBytesOf,
        VecZnxDftZero, VecZnxIdftApply, VecZnxIdftApplyTmpBytes, VecZnxNormalize,
        VecZnxNormalizeTmpBytes, VecZnxRshAssignBackend, VecZnxRshTmpBytes, VecZnxTransposeBackend,
        VecZnxZeroBackend, VmpApplyDftToDft, VmpApplyDftToDftTmpBytes,
    },
    layouts::{
        Backend, GaloisElement, Module, ScratchArena, VecZnx, VecZnxToBackendMut,
        VecZnxToBackendRef, ZnxInfos,
    },
};

use crate::packing::{
    PackingKeys, packing,
    packing_keys::{
        mask_key_layout, pack_keys_precompute_default, pack_partial_keys_precompute,
        packing_keys_precompute_tmp_bytes, packing_mask_key_precompute_partial,
        packing_mask_keys_precompute, packing_mask_keys_precompute_tmp_bytes,
    },
    packing_mask_preprocessing::{
        packing_mask_preprocessing_default, packing_mask_preprocessing_partial_default,
        packing_mask_preprocessing_partial_threaded,
        packing_mask_preprocessing_partial_tmp_bytes_default, packing_mask_preprocessing_threaded,
        packing_mask_preprocessing_tmp_bytes_default,
    },
    packing_precomputations::{
        PackingPrecomputations, PackingPrecomputeInfos, arithmetic_precompute_metadata,
        normalize_precompute_aggregate, normalize_precompute_coefficients,
        pack_precompute_alloc_default, pack_precompute_alloc_partial, precompute_collapse_mask,
        precompute_sequential_keyswitch_collapse_aggregate_mask_tmp_bytes_for_size,
        sequential_collapse_bsgs_dft_build, sequential_collapse_bsgs_dft_build_tmp_bytes,
    },
};

#[doc(hidden)]
pub trait PackingMaskAggregationDefault<BE: Backend> {
    /// Default scratch estimate for packing-mask aggregation.
    fn packing_mask_preprocessing_tmp_bytes_default(&self, size: usize) -> usize;

    /// Default packing-mask aggregation implementation.
    fn packing_mask_preprocessing_default<R, A>(
        &self,
        dst: &mut R,
        base2k: usize,
        a: &A,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: VecZnxToBackendMut<BE> + ZnxInfos,
        A: VecZnxToBackendRef<BE> + ZnxInfos;

    /// Default threaded packing-mask aggregation implementation.
    fn packing_mask_preprocessing_threaded_default<R, A>(
        &self,
        dst: &mut R,
        base2k: usize,
        a: &A,
        intra_threads: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: VecZnxToBackendMut<BE> + ZnxInfos,
        A: VecZnxToBackendRef<BE> + ZnxInfos;

    /// Default scratch estimate for partial packing-mask aggregation.
    fn packing_mask_preprocessing_partial_tmp_bytes_default(
        &self,
        gamma: usize,
        size: usize,
    ) -> usize;

    /// Default partial packing-mask aggregation implementation.
    fn packing_mask_preprocessing_partial_default<R, A>(
        &self,
        dst: &mut R,
        base2k: usize,
        gamma: usize,
        a: &A,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: VecZnxToBackendMut<BE> + ZnxInfos,
        A: VecZnxToBackendRef<BE> + ZnxInfos;

    /// Default threaded partial packing-mask aggregation implementation.
    fn packing_mask_preprocessing_partial_threaded_default<R, A>(
        &self,
        dst: &mut R,
        base2k: usize,
        gamma: usize,
        a: &A,
        intra_threads: usize,
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
    ) -> PackingKeys<BE>
    where
        KG: GGLWECompressedSeed + GGLWECompressedToBackendRef<BE> + GGLWEInfos + GetGaloisElement,
        KH: GGLWECompressedSeed + GGLWECompressedToBackendRef<BE> + GGLWEInfos + GetGaloisElement;

    /// Default partial client-key-side precompute implementation.
    fn pack_partial_keys_precompute_default<KG>(
        &self,
        key_g: &KG,
        stride: usize,
        baby_size: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> PackingKeys<BE>
    where
        KG: GGLWECompressedSeed + GGLWECompressedToBackendRef<BE> + GGLWEInfos + GetGaloisElement;

    /// Default fixed mask-side allocation implementation.
    fn pack_precompute_alloc_default(
        &self,
        steps: usize,
        size: usize,
        base2k: usize,
        baby_size: usize,
    ) -> PackingPrecomputations<BE>;

    /// Default partial fixed mask-side allocation implementation.
    fn pack_precompute_alloc_partial_default(
        &self,
        steps: usize,
        size: usize,
        base2k: usize,
        baby_size: usize,
        stride: usize,
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

    /// Default partial fixed mask-side precompute implementation.
    fn pack_precompute_partial_default<A, KMask>(
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
    fn pack_default<R, B>(
        &self,
        res: &mut R,
        body: &B,
        precomputations: &PackingPrecomputations<BE>,
        key_precomputations: &PackingKeys<BE>,
        chunk_size: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        B: VecZnxToBackendRef<BE> + ZnxInfos;
}

impl<BE: Backend<OwnedBuf = Vec<u8>>> PackingMaskAggregationDefault<BE> for Module<BE>
where
    Self: VecZnxCopyBackend<BE>
        + VecZnxTransposeBackend<BE>
        + VecZnxAutomorphismRotateBackend<BE>
        + VecZnxAddAssignBackend<BE>
        + VecZnxNormalize<BE>
        + VecZnxNormalizeTmpBytes
        + VecZnxRshAssignBackend<BE>
        + VecZnxRshTmpBytes
        + VecZnxZeroBackend<BE>
        + GaloisElement
        + ModuleCoreAlloc<OwnedBuf = Vec<u8>>,
    VecZnx<Vec<u8>>: VecZnxToBackendMut<BE> + VecZnxToBackendRef<BE>,
{
    fn packing_mask_preprocessing_tmp_bytes_default(&self, size: usize) -> usize {
        packing_mask_preprocessing_tmp_bytes_default(self, size)
    }

    fn packing_mask_preprocessing_default<R, A>(
        &self,
        dst: &mut R,
        base2k: usize,
        a: &A,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: VecZnxToBackendMut<BE> + ZnxInfos,
        A: VecZnxToBackendRef<BE> + ZnxInfos,
    {
        packing_mask_preprocessing_default(self, dst, base2k, a, scratch);
    }

    fn packing_mask_preprocessing_threaded_default<R, A>(
        &self,
        dst: &mut R,
        base2k: usize,
        a: &A,
        intra_threads: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: VecZnxToBackendMut<BE> + ZnxInfos,
        A: VecZnxToBackendRef<BE> + ZnxInfos,
    {
        packing_mask_preprocessing_threaded(self, dst, base2k, a, intra_threads, scratch);
    }

    fn packing_mask_preprocessing_partial_tmp_bytes_default(
        &self,
        gamma: usize,
        size: usize,
    ) -> usize {
        packing_mask_preprocessing_partial_tmp_bytes_default(self, gamma, size)
    }

    fn packing_mask_preprocessing_partial_default<R, A>(
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
        packing_mask_preprocessing_partial_default(self, dst, base2k, gamma, a, scratch);
    }

    fn packing_mask_preprocessing_partial_threaded_default<R, A>(
        &self,
        dst: &mut R,
        base2k: usize,
        gamma: usize,
        a: &A,
        intra_threads: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: VecZnxToBackendMut<BE> + ZnxInfos,
        A: VecZnxToBackendRef<BE> + ZnxInfos,
    {
        packing_mask_preprocessing_partial_threaded(
            self,
            dst,
            base2k,
            gamma,
            a,
            intra_threads,
            scratch,
        );
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
        + VecZnxNormalize<BE>
        + VecZnxNormalizeTmpBytes
        + VmpApplyDftToDft<BE>,
    Module<BE>: GGLWEPreparedFactory<BE>
        + GLWEMaskFillDefault<BE>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
    Module<BE>: VmpApplyDftToDftTmpBytes,
    <Module<BE> as VecZnxDftAutomorphismPlan<BE>>::Plan: 'static + Send + Sync,
    VecZnx<BE::OwnedBuf>: VecZnxToBackendRef<BE>,
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
    ) -> PackingKeys<BE>
    where
        KG: GGLWECompressedSeed + GGLWECompressedToBackendRef<BE> + GGLWEInfos + GetGaloisElement,
        KH: GGLWECompressedSeed + GGLWECompressedToBackendRef<BE> + GGLWEInfos + GetGaloisElement,
    {
        pack_keys_precompute_default(self, key_g, key_h, baby_size, scratch)
    }

    fn pack_partial_keys_precompute_default<KG>(
        &self,
        key_g: &KG,
        stride: usize,
        baby_size: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> PackingKeys<BE>
    where
        KG: GGLWECompressedSeed + GGLWECompressedToBackendRef<BE> + GGLWEInfos + GetGaloisElement,
    {
        pack_partial_keys_precompute(self, key_g, stride, baby_size, scratch)
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

    fn pack_precompute_alloc_partial_default(
        &self,
        steps: usize,
        size: usize,
        base2k: usize,
        baby_size: usize,
        stride: usize,
    ) -> PackingPrecomputations<BE> {
        pack_precompute_alloc_partial(self, steps, size, base2k, baby_size, stride)
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
        assert_eq!(
            metadata.size(),
            aggregate_mask.size(),
            "packing precompute metadata and aggregate mask sizes differ"
        );
        let key_mask_bytes = packing_mask_keys_precompute_tmp_bytes(self, key_mask_source);
        let key_mask_layout = mask_key_layout(key_mask_source);
        let arithmetic_metadata = arithmetic_precompute_metadata(metadata);
        let mask_bytes = precompute_sequential_keyswitch_collapse_aggregate_mask_tmp_bytes_for_size(
            self,
            arithmetic_metadata.size(),
            metadata.size(),
            &key_mask_layout,
            &key_mask_layout,
        );
        let bsgs_bytes = sequential_collapse_bsgs_dft_build_tmp_bytes(self, metadata);
        key_mask_bytes
            .max(mask_bytes)
            .max(bsgs_bytes)
            .max(self.vec_znx_normalize_tmp_bytes())
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
        assert_eq!(
            precomputations.size(),
            aggregate_mask.size(),
            "packing precompute storage and aggregate mask sizes differ"
        );
        let target_metadata = precomputations.metadata();
        let arithmetic_metadata = arithmetic_precompute_metadata(target_metadata);

        let (key_g_mask, key_h_mask) =
            packing_mask_keys_precompute(self, key_mask_source, &mut scratch.borrow());

        let mut arithmetic_aggregate =
            self.vec_znx_alloc(aggregate_mask.cols(), arithmetic_metadata.size());
        normalize_precompute_aggregate(
            self,
            &mut arithmetic_aggregate,
            arithmetic_metadata.base2k(),
            aggregate_mask,
            target_metadata.base2k(),
            scratch,
        );

        let mut arithmetic_precomputations = pack_precompute_alloc_default(
            self,
            arithmetic_metadata.steps(),
            arithmetic_metadata.size(),
            arithmetic_metadata.base2k(),
            arithmetic_metadata.baby_size(),
        );
        precompute_collapse_mask(
            self,
            &mut arithmetic_precomputations,
            &arithmetic_aggregate,
            target_metadata.base2k(),
            target_metadata.size(),
            &key_g_mask,
            Some(&key_h_mask),
            scratch,
        );
        normalize_precompute_coefficients(
            self,
            precomputations,
            &arithmetic_precomputations,
            scratch,
        );
        sequential_collapse_bsgs_dft_build(self, precomputations, scratch);
    }

    fn pack_precompute_partial_default<A, KMask>(
        &self,
        precomputations: &mut PackingPrecomputations<BE>,
        aggregate_mask: &A,
        key_mask_source: &KMask,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        A: VecZnxToBackendRef<BE> + ZnxInfos,
        KMask: GGLWECompressedSeed + GGLWEInfos,
    {
        pack_precompute_partial(
            self,
            precomputations,
            aggregate_mask,
            key_mask_source,
            scratch,
        );
    }

    fn pack_default<R, B>(
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
        packing::pack_default(
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

/// Fills a **partial** (Algorithm 2) fixed mask-side precompute: a single
/// `key_g` collapse half over the first `γ` aggregate columns, no `key_h`.
/// `precomputations` must be allocated via
/// [`pack_precompute_alloc_partial`](crate::packing::packing_precomputations::pack_precompute_alloc_partial)
/// with `steps = γ - 1`. The `key_mask_source` provides the (seed-derived)
/// `key_g` mask; its `key_h` mask is not used.
#[allow(private_bounds)]
pub(crate) fn pack_precompute_partial<BE, A, KMask>(
    module: &Module<BE>,
    precomputations: &mut PackingPrecomputations<BE>,
    aggregate_mask: &A,
    key_mask_source: &KMask,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
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
        + VecZnxNormalize<BE>
        + VecZnxNormalizeTmpBytes
        + VmpApplyDftToDft<BE>
        + VmpApplyDftToDftTmpBytes
        + GGLWEPreparedFactory<BE>
        + GLWEMaskFillDefault<BE>,
    <Module<BE> as VecZnxDftAutomorphismPlan<BE>>::Plan: 'static + Send + Sync,
    VecZnx<BE::OwnedBuf>: VecZnxToBackendRef<BE>,
    A: VecZnxToBackendRef<BE> + ZnxInfos,
    KMask: GGLWECompressedSeed + GGLWEInfos,
{
    assert_eq!(
        precomputations.size(),
        aggregate_mask.size(),
        "packing precompute storage and aggregate mask sizes differ"
    );
    assert!(
        precomputations.partial(),
        "precompute storage is not partial"
    );

    let target_metadata = precomputations.metadata();
    let arithmetic_metadata = arithmetic_precompute_metadata(target_metadata);
    let stride = target_metadata.stride();

    // Partial packing uses only the order-γ generator key K_{g_γ}; its mask
    // projection is rotated by galois_element(stride). No key_h.
    let key_g_mask =
        packing_mask_key_precompute_partial(module, key_mask_source, stride, &mut scratch.borrow());

    let mut arithmetic_aggregate =
        module.vec_znx_alloc(aggregate_mask.cols(), arithmetic_metadata.size());
    normalize_precompute_aggregate(
        module,
        &mut arithmetic_aggregate,
        arithmetic_metadata.base2k(),
        aggregate_mask,
        target_metadata.base2k(),
        scratch,
    );

    let mut arithmetic_precomputations = pack_precompute_alloc_partial(
        module,
        arithmetic_metadata.steps(),
        arithmetic_metadata.size(),
        arithmetic_metadata.base2k(),
        arithmetic_metadata.baby_size(),
        stride,
    );
    precompute_collapse_mask(
        module,
        &mut arithmetic_precomputations,
        &arithmetic_aggregate,
        target_metadata.base2k(),
        target_metadata.size(),
        &key_g_mask,
        None,
        scratch,
    );
    normalize_precompute_coefficients(
        module,
        precomputations,
        &arithmetic_precomputations,
        scratch,
    );
    sequential_collapse_bsgs_dft_build(module, precomputations, scratch);
}
