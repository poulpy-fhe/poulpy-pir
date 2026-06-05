//! OEP triple for the BSGS DFT-hot collapse (`*Impl` backend trait, `*Default`
//! reference-impl trait, plus a blanket `*Impl for BE where Module<BE>: *Default`).
//!
//! Structure mirrors `poulpy-core/src/oep/keyswitching.rs`: a backend that
//! implements the `*Default` trait on its `Module<BE>` automatically satisfies
//! the corresponding `*Impl<BE>` trait, which the delegate in
//! delegate uses to wire the user-facing [`crate::packing::Packing`] trait.

#![allow(clippy::too_many_arguments)]

use poulpy_core::layouts::{
    GGLWECompressedSeed, GGLWEInfos, GLWEInfos, GLWEToBackendMut, GetGaloisElement,
    compressed::GGLWECompressedToBackendRef,
};
use poulpy_hal::layouts::{
    Backend, Module, ScratchArena, VecZnxToBackendMut, VecZnxToBackendRef, ZnxInfos,
};

use crate::packing::{
    PackingKeys,
    default::{PackingDefault, PackingMaskAggregationDefault},
    packing_precomputations::{PackingPrecomputations, PackingPrecomputeInfos},
};

/// # Safety
/// Implementations must preserve the mask aggregation semantics documented on
/// [`crate::packing::PackingMaskAggregation`] and must keep all backend reads/writes
/// within the provided layouts.
#[allow(private_bounds)]
pub unsafe trait PackingMaskAggregationImpl<BE: Backend>: Backend {
    /// Backend hook for packing-mask aggregation scratch estimation.
    fn packing_mask_preprocessing_tmp_bytes_impl(module: &Module<BE>, size: usize) -> usize;

    /// Backend hook for packing-mask aggregation.
    fn packing_mask_preprocessing_impl<R, A>(
        module: &Module<BE>,
        dst: &mut R,
        base2k: usize,
        a: &A,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: VecZnxToBackendMut<BE> + ZnxInfos,
        A: VecZnxToBackendRef<BE> + ZnxInfos;

    /// Backend hook for partial packing-mask aggregation scratch estimation.
    fn pack_partial_mask_preprocessing_tmp_bytes_impl(
        module: &Module<BE>,
        gamma: usize,
        size: usize,
    ) -> usize;

    /// Backend hook for partial packing-mask aggregation.
    fn pack_partial_mask_preprocessing_impl<R, A>(
        module: &Module<BE>,
        dst: &mut R,
        base2k: usize,
        gamma: usize,
        a: &A,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: VecZnxToBackendMut<BE> + ZnxInfos,
        A: VecZnxToBackendRef<BE> + ZnxInfos;
}

#[allow(private_bounds)]
unsafe impl<BE: Backend> PackingMaskAggregationImpl<BE> for BE
where
    Module<BE>: PackingMaskAggregationDefault<BE>,
{
    fn packing_mask_preprocessing_tmp_bytes_impl(module: &Module<BE>, size: usize) -> usize {
        module.packing_mask_preprocessing_tmp_bytes_default(size)
    }

    fn packing_mask_preprocessing_impl<R, A>(
        module: &Module<BE>,
        dst: &mut R,
        base2k: usize,
        a: &A,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: VecZnxToBackendMut<BE> + ZnxInfos,
        A: VecZnxToBackendRef<BE> + ZnxInfos,
    {
        module.packing_mask_preprocessing_default(dst, base2k, a, scratch);
    }

    fn pack_partial_mask_preprocessing_tmp_bytes_impl(
        module: &Module<BE>,
        gamma: usize,
        size: usize,
    ) -> usize {
        module.packing_mask_preprocessing_partial_tmp_bytes_default(gamma, size)
    }

    fn pack_partial_mask_preprocessing_impl<R, A>(
        module: &Module<BE>,
        dst: &mut R,
        base2k: usize,
        gamma: usize,
        a: &A,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: VecZnxToBackendMut<BE> + ZnxInfos,
        A: VecZnxToBackendRef<BE> + ZnxInfos,
    {
        module.packing_mask_preprocessing_partial_default(dst, base2k, gamma, a, scratch);
    }
}

/// # Safety
/// Implementations must satisfy the documented collapse semantics, honor
/// layout metadata and prepared-key interpretation, and keep all reads/writes
/// within the described backend buffers.
///
/// Backends implement this trait when they want to override the default packing
/// path. The blanket impl below wires any backend whose `Module<BE>` implements
/// [`PackingDefault`] to the default code in `default.rs`.
#[allow(private_bounds)]
pub unsafe trait PackingImpl<BE: Backend>: Backend {
    /// Backend hook for client-key-side precompute scratch estimation.
    fn pack_keys_precompute_tmp_bytes_impl<KG, KH>(
        module: &Module<BE>,
        key_g: &KG,
        key_h: &KH,
        baby_size: usize,
    ) -> usize
    where
        KG: GGLWEInfos,
        KH: GGLWEInfos;

    /// Backend hook for client-key-side precompute.
    fn pack_keys_precompute_impl<KG, KH>(
        module: &Module<BE>,
        key_g: &KG,
        key_h: &KH,
        baby_size: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> PackingKeys<BE>
    where
        KG: GGLWECompressedSeed + GGLWECompressedToBackendRef<BE> + GGLWEInfos + GetGaloisElement,
        KH: GGLWECompressedSeed + GGLWECompressedToBackendRef<BE> + GGLWEInfos + GetGaloisElement;

    /// Backend hook for partial client-key-side precompute.
    fn pack_partial_keys_precompute_impl<KG>(
        module: &Module<BE>,
        key_g: &KG,
        stride: usize,
        baby_size: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> PackingKeys<BE>
    where
        KG: GGLWECompressedSeed + GGLWECompressedToBackendRef<BE> + GGLWEInfos + GetGaloisElement;

    /// Backend hook for fixed mask-side allocation.
    fn pack_precompute_alloc_impl(
        module: &Module<BE>,
        steps: usize,
        size: usize,
        base2k: usize,
        baby_size: usize,
    ) -> PackingPrecomputations<BE>;

    /// Backend hook for partial fixed mask-side allocation.
    fn pack_partial_precompute_alloc_impl(
        module: &Module<BE>,
        steps: usize,
        size: usize,
        base2k: usize,
        baby_size: usize,
        stride: usize,
    ) -> PackingPrecomputations<BE>;

    /// Backend hook for fixed mask-side precompute scratch estimation.
    fn pack_precompute_tmp_bytes_impl<A, KMask>(
        module: &Module<BE>,
        metadata: PackingPrecomputeInfos,
        aggregate_mask: &A,
        key_mask_source: &KMask,
    ) -> usize
    where
        A: ZnxInfos,
        KMask: GGLWEInfos;

    /// Backend hook for fixed mask-side precompute.
    fn pack_precompute_impl<A, KMask>(
        module: &Module<BE>,
        precomputations: &mut PackingPrecomputations<BE>,
        aggregate_mask: &A,
        key_mask_source: &KMask,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        A: VecZnxToBackendRef<BE> + ZnxInfos,
        KMask: GGLWECompressedSeed + GGLWEInfos;

    /// Backend hook for partial fixed mask-side precompute.
    fn pack_partial_precompute_impl<A, KMask>(
        module: &Module<BE>,
        precomputations: &mut PackingPrecomputations<BE>,
        aggregate_mask: &A,
        key_mask_source: &KMask,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        A: VecZnxToBackendRef<BE> + ZnxInfos,
        KMask: GGLWECompressedSeed + GGLWEInfos;

    /// Backend hook for the online packing pass.
    fn pack_impl<R, B>(
        module: &Module<BE>,
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

#[allow(private_bounds)]
unsafe impl<BE: Backend> PackingImpl<BE> for BE
where
    Module<BE>: PackingDefault<BE>,
{
    fn pack_keys_precompute_tmp_bytes_impl<KG, KH>(
        module: &Module<BE>,
        key_g: &KG,
        key_h: &KH,
        baby_size: usize,
    ) -> usize
    where
        KG: GGLWEInfos,
        KH: GGLWEInfos,
    {
        module.pack_keys_precompute_tmp_bytes_default(key_g, key_h, baby_size)
    }

    fn pack_keys_precompute_impl<KG, KH>(
        module: &Module<BE>,
        key_g: &KG,
        key_h: &KH,
        baby_size: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> PackingKeys<BE>
    where
        KG: GGLWECompressedSeed + GGLWECompressedToBackendRef<BE> + GGLWEInfos + GetGaloisElement,
        KH: GGLWECompressedSeed + GGLWECompressedToBackendRef<BE> + GGLWEInfos + GetGaloisElement,
    {
        module.pack_keys_precompute_default(key_g, key_h, baby_size, scratch)
    }

    fn pack_partial_keys_precompute_impl<KG>(
        module: &Module<BE>,
        key_g: &KG,
        stride: usize,
        baby_size: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> PackingKeys<BE>
    where
        KG: GGLWECompressedSeed + GGLWECompressedToBackendRef<BE> + GGLWEInfos + GetGaloisElement,
    {
        module.pack_partial_keys_precompute_default(key_g, stride, baby_size, scratch)
    }

    fn pack_precompute_alloc_impl(
        module: &Module<BE>,
        steps: usize,
        size: usize,
        base2k: usize,
        baby_size: usize,
    ) -> PackingPrecomputations<BE> {
        module.pack_precompute_alloc_default(steps, size, base2k, baby_size)
    }

    fn pack_partial_precompute_alloc_impl(
        module: &Module<BE>,
        steps: usize,
        size: usize,
        base2k: usize,
        baby_size: usize,
        stride: usize,
    ) -> PackingPrecomputations<BE> {
        module.pack_precompute_alloc_partial_default(steps, size, base2k, baby_size, stride)
    }

    fn pack_precompute_tmp_bytes_impl<A, KMask>(
        module: &Module<BE>,
        metadata: PackingPrecomputeInfos,
        aggregate_mask: &A,
        key_mask_source: &KMask,
    ) -> usize
    where
        A: ZnxInfos,
        KMask: GGLWEInfos,
    {
        module.pack_precompute_tmp_bytes_default(metadata, aggregate_mask, key_mask_source)
    }

    fn pack_precompute_impl<A, KMask>(
        module: &Module<BE>,
        precomputations: &mut PackingPrecomputations<BE>,
        aggregate_mask: &A,
        key_mask_source: &KMask,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        A: VecZnxToBackendRef<BE> + ZnxInfos,
        KMask: GGLWECompressedSeed + GGLWEInfos,
    {
        module.pack_precompute_default(precomputations, aggregate_mask, key_mask_source, scratch);
    }

    fn pack_partial_precompute_impl<A, KMask>(
        module: &Module<BE>,
        precomputations: &mut PackingPrecomputations<BE>,
        aggregate_mask: &A,
        key_mask_source: &KMask,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        A: VecZnxToBackendRef<BE> + ZnxInfos,
        KMask: GGLWECompressedSeed + GGLWEInfos,
    {
        module.pack_precompute_partial_default(
            precomputations,
            aggregate_mask,
            key_mask_source,
            scratch,
        );
    }

    fn pack_impl<R, B>(
        module: &Module<BE>,
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
        module.pack_default(
            res,
            body,
            precomputations,
            key_precomputations,
            chunk_size,
            scratch,
        );
    }
}
