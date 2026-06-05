use poulpy_core::{
    EncryptionLayout,
    layouts::{
        Base2K, Degree, Dnum, Dsize, GGSWLayout, GLWEAutomorphismKeyLayout, GLWELayout,
        LWEMatrixLayout, Rank, TorusPrecision,
    },
};
use poulpy_hal::layouts::{Backend, Module};

use crate::{
    config::{Collapse, Config},
    encoding::ModPEncoder,
    packing::PackingPrecomputeInfos,
    payload::Payload,
};

/// All toy-PIR parameters plus the backend [`Module`] they instantiate. Pass it
/// by reference everywhere; it is the single source of truth for dimensions and
/// layouts. The shared cryptosystem lives directly on the struct; the
/// second-dimension reduction is the [`Collapse`] enum.
pub struct Parameters<BE: Backend, B, P>
where
    P: Payload<B>,
{
    pub(crate) params: Config<B, P>,
    pub(crate) module: Module<BE>,
}

impl<BE: Backend, B, P> Parameters<BE, B, P>
where
    P: Payload<B>,
{
    /// The backend module (ring degree `n`), instantiated once.
    pub fn module(&self) -> &Module<BE> {
        &self.module
    }

    /// The second-dimension collapse method.
    pub fn collapse(&self) -> Collapse {
        self.params.collapse
    }

    /// Logical column/record height for the selected collapse.
    ///
    /// Interpolation returns one full ring column (`n`); InsPIRe² returns the
    /// first-level record-packing parameter (`γ0`).
    pub fn column_height(&self) -> usize {
        self.params.column_height()
    }

    pub fn n(&self) -> usize {
        self.params.n
    }

    pub fn p(&self) -> i64 {
        P::BASIS as i64
    }

    pub fn digits(&self) -> usize {
        P::EXPONENT
    }

    pub fn encode(&self, digits: &mut [i16], value: B) {
        P::encode(digits, value)
    }

    pub fn decode(&self, value: &mut B, digits: &[i16]) {
        P::decode(value, digits)
    }

    /// Pack / repacking-key regime base2k (3x18).
    pub const fn base2k(&self) -> usize {
        self.params.base2k
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
        self.params.k
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
            k: TorusPrecision(self.k() as u32),
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
            k: TorusPrecision(self.k() as u32),
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
