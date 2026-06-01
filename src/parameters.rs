//! Single source of truth for the toy-PIR parameters and the derived ciphertext
//! / matrix layouts used by the `pack_then_horner` example.
//!
//! Three base2k regimes are used:
//! - `base2k` (18, 3 limbs): the pack / repacking-key regime (non-linear ops).
//! - `matmul_base2k` (16, 4 limbs): the query / `U·(A,b)` GEMM input regime
//!   (i16 kernel at `ap = 1`).
//! - `mask_base2k` (32, 2 limbs): the expanded query mask `A` (cheaper `X^-i`
//!   expansion than 4x16, and a same-base `U·A` matmul accumulator that does not
//!   overflow i64 across the block-column sum).

use poulpy_core::{
    EncryptionLayout,
    layouts::{
        Base2K, Degree, Dnum, Dsize, GGSWLayout, GLWEAutomorphismKeyLayout, GLWELayout,
        LWEMatrixLayout, Rank, TorusPrecision,
    },
};
use poulpy_hal::{
    api::ModuleNew,
    layouts::{Backend, Module},
};

use crate::{encoding::ModPEncoder, packing::PackingPrecomputeInfos};

/// All toy-PIR parameters plus the backend [`Module`] they instantiate. Pass it
/// by reference everywhere; it is the single source of truth for dimensions and
/// layouts.
pub struct Parameters<BE: Backend> {
    module: Module<BE>,
}

impl<BE: Backend> Default for Parameters<BE>
where
    Module<BE>: ModuleNew<BE>,
{
    fn default() -> Self {
        Self {
            module: Module::<BE>::new(2048),
        }
    }
}

impl<BE: Backend> Parameters<BE> {
    /// The backend module (ring degree `n`), instantiated once.
    pub fn module(&self) -> &Module<BE> {
        &self.module
    }

    pub const fn n(&self) -> usize {
        2048
    }

    pub const fn p(&self) -> i64 {
        65535
    }

    /// Pack / repacking-key regime base2k (3x18).
    pub const fn base2k(&self) -> usize {
        18
    }

    /// Linear-matmul input regime base2k (4x16).
    pub const fn matmul_base2k(&self) -> usize {
        16
    }

    /// Query-mask regime base2k. Using 16 (matching the matmul regime) keeps the
    /// `U·A` inner-product element at i16 so the i64 accumulator stays clear of
    /// overflow even for many block-columns; 32 (i32 element) overflows at ~54.
    pub const fn mask_base2k(&self) -> usize {
        32
    }

    /// Torus precision shared by every regime.
    pub const fn k(&self) -> usize {
        54
    }

    /// Key-switch torus precision.
    pub const fn ks_k(&self) -> usize {
        54
    }

    pub const fn dnum(&self) -> usize {
        3
    }

    pub const fn dsize(&self) -> usize {
        1
    }

    pub const fn baby_size(&self) -> usize {
        8
    }

    /// Number of base2k limbs needed for `k` at the given base2k.
    pub const fn size_at(&self, base2k: usize) -> usize {
        self.k().div_ceil(base2k)
    }

    pub fn encoder(&self) -> ModPEncoder {
        ModPEncoder::new(self.p(), self.k())
    }

    fn glwe_layout(&self, base2k: usize) -> EncryptionLayout<GLWELayout> {
        EncryptionLayout::new_from_default_sigma(GLWELayout {
            n: Degree(self.n() as u32),
            base2k: Base2K(base2k as u32),
            k: TorusPrecision(self.k() as u32),
            rank: Rank(1),
        })
        .unwrap()
    }

    /// Query ciphertext layout: 4x16 matmul regime.
    pub fn glwe_query(&self) -> EncryptionLayout<GLWELayout> {
        self.glwe_layout(self.matmul_base2k())
    }

    /// Packed / result ciphertext layout: 3x18 pack regime.
    pub fn glwe_pack(&self) -> EncryptionLayout<GLWELayout> {
        self.glwe_layout(self.base2k())
    }

    /// Coarse query-mask `A` layout: 2x32 regime.
    pub fn glwe_mask(&self) -> EncryptionLayout<GLWELayout> {
        self.glwe_layout(self.mask_base2k())
    }

    pub fn key_layout(&self) -> EncryptionLayout<GLWEAutomorphismKeyLayout> {
        EncryptionLayout::new_from_default_sigma(GLWEAutomorphismKeyLayout {
            n: Degree(self.n() as u32),
            base2k: Base2K(self.base2k() as u32),
            k: TorusPrecision(self.ks_k() as u32),
            rank: Rank(1),
            dnum: Dnum(self.dnum() as u32),
            dsize: Dsize(self.dsize() as u32),
        })
        .unwrap()
    }

    pub fn ggsw_layout(&self) -> EncryptionLayout<GGSWLayout> {
        EncryptionLayout::new_from_default_sigma(GGSWLayout {
            n: Degree(self.n() as u32),
            base2k: Base2K(self.base2k() as u32),
            k: TorusPrecision(self.ks_k() as u32),
            rank: Rank(1),
            dnum: Dnum(self.dnum() as u32),
            dsize: Dsize(self.dsize() as u32),
        })
        .unwrap()
    }

    /// The single canonical `n x n` `LWEMatrix` layout, in the 3x18 pack regime.
    ///
    /// Every `LWEMatrix` that crosses an API boundary uses this layout (the
    /// query mask `A`, the `U·(A,b)` products, the selected coefficient). The
    /// faster mask (2x32) and body (4x16) matmul regimes are internal,
    /// value-preserving intermediate representations of the operations that need
    /// them, never surfaced here.
    pub fn lwe_matrix_infos(&self) -> LWEMatrixLayout {
        LWEMatrixLayout {
            rows: self.n(),
            n: Degree(self.n() as u32),
            base2k: Base2K(self.base2k() as u32),
            k: TorusPrecision((self.size_at(self.base2k()) * self.base2k()) as u32),
        }
    }

    pub fn packing_precompute_infos(&self) -> PackingPrecomputeInfos {
        PackingPrecomputeInfos::new(
            self.n() - 1,
            self.size_at(self.base2k()),
            self.base2k(),
            self.baby_size(),
        )
    }
}
