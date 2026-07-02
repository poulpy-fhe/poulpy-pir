use std::mem::size_of;

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
    database::DatabaseLayout,
    encoding::ModPEncoder,
    packing::PackingPrecomputeInfos,
    payload::Payload,
};

/// Densest signed-`i64` limb packing used by query/response serialization.
const TRANSMIT_BASE2K: usize = 63;
const U8_BYTES: usize = 1;
const U32_BYTES: usize = 4;
const U64_BYTES: usize = 8;
const SEED_BYTES: usize = 32;
const I64_BYTES: usize = size_of::<i64>();

/// Actual serialized byte size of a PIR query, split by transmitted component.
///
/// Coefficients are counted as serialized today: one `i64` per transmitted limb
/// after base2k=63 repacking, plus the framing bytes written by the serializers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct QuerySize {
    /// Collapse tag byte.
    pub tag: usize,
    /// `Vec<_>` length prefixes (`u64` each).
    pub length_prefixes: usize,
    /// Interpolation one-hot blocks.
    pub one_hot: usize,
    /// InsPIRe² first one-hot source (`src0`).
    pub src0_one_hot: usize,
    /// InsPIRe² second one-hot source (`src1`).
    pub src1_one_hot: usize,
    /// Interpolation key `g`.
    pub key_g: usize,
    /// Interpolation key `h`.
    pub key_h: usize,
    /// Interpolation root GGSW.
    pub ggsw: usize,
    /// InsPIRe² `gamma0` key.
    pub key_gamma0: usize,
    /// InsPIRe² `gamma1` key.
    pub key_gamma1: usize,
    /// InsPIRe² `gamma2` key.
    pub key_gamma2: usize,
}

impl QuerySize {
    /// Total serialized query size in bytes.
    pub fn total_size(&self) -> usize {
        self.tag
            + self.length_prefixes
            + self.one_hot
            + self.src0_one_hot
            + self.src1_one_hot
            + self.key_g
            + self.key_h
            + self.ggsw
            + self.key_gamma0
            + self.key_gamma1
            + self.key_gamma2
    }
}

/// Actual serialized byte size of a PIR response, split by transmitted
/// component.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResponseSize {
    /// Collapse tag byte.
    pub tag: usize,
    /// `Vec<_>` length prefixes (`u64` each).
    pub length_prefixes: usize,
    /// Interpolation selected GLWE.
    pub selected: usize,
    /// InsPIRe² first response vector.
    pub resp1: usize,
    /// InsPIRe² second response vector.
    pub resp2: usize,
}

impl ResponseSize {
    /// Total serialized response size in bytes.
    pub fn total_size(&self) -> usize {
        self.tag + self.length_prefixes + self.selected + self.resp1 + self.resp2
    }
}

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

    // The cryptosystem scalars below are pure functions of the [`Config`] (no
    // backend), so they live on `Config` and are forwarded here for the 50+
    // call sites that already hold a `Parameters`.

    /// Pack / repacking-key regime base2k (3x18).
    pub const fn base2k(&self) -> usize {
        self.params.base2k()
    }

    /// Linear-matmul input regime base2k (4x16).
    pub const fn matmul_base2k(&self) -> usize {
        self.params.matmul_base2k()
    }

    /// Query-mask regime base2k. Using 16 (matching the matmul regime) keeps the
    /// `U·A` inner-product element at i16 so the i64 accumulator stays clear of
    /// overflow even for many block-columns; 32 (i32 element) overflows at ~54.
    pub const fn mask_base2k(&self) -> usize {
        self.params.mask_base2k()
    }

    /// Torus precision shared by every regime.
    pub const fn k(&self) -> usize {
        self.params.k()
    }

    pub const fn dnum(&self) -> usize {
        self.params.dnum()
    }

    pub const fn dsize(&self) -> usize {
        self.params.dsize()
    }

    pub const fn baby_size(&self) -> usize {
        self.params.baby_size()
    }

    /// Number of base2k limbs needed for `k` at the given base2k.
    pub const fn size_at(&self, base2k: usize) -> usize {
        self.params.size_at(base2k)
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

    /// Actual serialized query size for `layout`, counted in bytes. Backend-free;
    /// forwards to [`Config::query_size`].
    pub fn query_size(&self, layout: DatabaseLayout<P>) -> QuerySize
    where
        P: Payload<[u8; 32]>,
    {
        self.params.query_size(layout)
    }

    /// Actual serialized response size for `layout`, counted in bytes.
    /// Backend-free; forwards to [`Config::response_size`].
    pub fn response_size(&self, layout: DatabaseLayout<P>) -> ResponseSize
    where
        P: Payload<[u8; 32]>,
    {
        self.params.response_size(layout)
    }
}

/// Backend-free cryptosystem scalars and serialized-size computation. These are
/// pure functions of the [`Config`] — no [`Module`] is needed — so they live
/// here and [`Parameters`] forwards to them. This lets callers that only have a
/// `Config` (e.g. [`crate::config::DefaultPirParameters32B`]) size a query /
/// response without instantiating a backend.
impl<B, P> Config<B, P>
where
    P: Payload<B>,
{
    /// Pack / repacking-key regime base2k (3x18).
    pub const fn base2k(&self) -> usize {
        self.base2k
    }

    /// Linear-matmul input regime base2k (4x16).
    pub const fn matmul_base2k(&self) -> usize {
        16
    }

    /// Query-mask regime base2k (2x32).
    pub const fn mask_base2k(&self) -> usize {
        32
    }

    /// Torus precision shared by every regime.
    pub const fn k(&self) -> usize {
        self.k
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

    /// Actual serialized query size for `layout`, counted in bytes.
    pub fn query_size(&self, layout: DatabaseLayout<P>) -> QuerySize
    where
        P: Payload<[u8; 32]>,
    {
        match self.collapse() {
            Collapse::Interpolation => {
                let one_hot_blocks = layout.column_blocks(self.n());
                QuerySize {
                    tag: U8_BYTES,
                    length_prefixes: U64_BYTES,
                    one_hot: one_hot_blocks * glwe_compressed_wire_bytes(self.n(), self.k()),
                    key_g: self.auto_key_wire_bytes(),
                    key_h: self.auto_key_wire_bytes(),
                    ggsw: self.ggsw_wire_bytes(),
                    ..QuerySize::default()
                }
            }
            Collapse::Recursion {
                gamma0,
                gamma1: _,
                gamma2: _,
            } => {
                let t = layout.grid_rows_for(gamma0);
                QuerySize {
                    tag: U8_BYTES,
                    length_prefixes: 2 * U64_BYTES,
                    src0_one_hot: layout.column_blocks(self.n())
                        * glwe_compressed_wire_bytes(self.n(), self.k()),
                    src1_one_hot: t.div_ceil(self.n())
                        * glwe_compressed_wire_bytes(self.n(), self.k()),
                    key_gamma0: self.compressed_key_wire_bytes(),
                    key_gamma1: self.compressed_key_wire_bytes(),
                    key_gamma2: self.compressed_key_wire_bytes(),
                    ..QuerySize::default()
                }
            }
        }
    }

    /// Actual serialized response size, counted in bytes. The size depends only
    /// on the cryptosystem scalars and collapse (not the DB shape), so `_layout`
    /// is unused; it is kept for signature symmetry with [`Self::query_size`].
    pub fn response_size(&self, _layout: DatabaseLayout<P>) -> ResponseSize
    where
        P: Payload<[u8; 32]>,
    {
        match self.collapse() {
            Collapse::Interpolation => ResponseSize {
                tag: U8_BYTES,
                selected: glwe_wire_bytes(self.n(), self.k()),
                ..ResponseSize::default()
            },
            Collapse::Recursion {
                gamma0,
                gamma1,
                gamma2,
            } => {
                let qtilde_bits = 2 * self.matmul_base2k();
                let tau = qtilde_bits.div_ceil(self.matmul_base2k());
                ResponseSize {
                    tag: U8_BYTES,
                    length_prefixes: 2 * U64_BYTES,
                    resp1: (self.n() * tau).div_ceil(gamma1)
                        * glwe_wire_bytes(self.n(), qtilde_bits),
                    resp2: (gamma0 * tau).div_ceil(gamma2) * glwe_wire_bytes(self.n(), qtilde_bits),
                    ..ResponseSize::default()
                }
            }
        }
    }

    fn auto_key_wire_bytes(&self) -> usize {
        let entries = self.dnum();
        U64_BYTES
            + U64_BYTES
            + entries * SEED_BYTES
            + entries * rank0_glwe_wire_bytes(self.n(), self.k())
    }

    fn compressed_key_wire_bytes(&self) -> usize {
        U64_BYTES + self.auto_key_wire_bytes()
    }

    fn ggsw_wire_bytes(&self) -> usize {
        let entries = self.dnum() * 2;
        U64_BYTES + entries * SEED_BYTES + entries * rank0_glwe_wire_bytes(self.n(), self.k())
    }
}

fn transmitted_limbs(k: usize) -> usize {
    k.div_ceil(TRANSMIT_BASE2K)
}

fn vec_znx_wire_bytes(n: usize, cols: usize, size: usize) -> usize {
    5 * U64_BYTES + n * cols * size * I64_BYTES
}

fn glwe_wire_bytes(n: usize, k: usize) -> usize {
    U32_BYTES + vec_znx_wire_bytes(n, 2, transmitted_limbs(k))
}

fn glwe_compressed_wire_bytes(n: usize, k: usize) -> usize {
    U32_BYTES + U32_BYTES + SEED_BYTES + vec_znx_wire_bytes(n, 1, transmitted_limbs(k))
}

fn rank0_glwe_wire_bytes(n: usize, k: usize) -> usize {
    U32_BYTES + vec_znx_wire_bytes(n, 1, transmitted_limbs(k))
}
