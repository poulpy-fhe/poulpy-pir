use poulpy_core::{
    EncryptionInfos, GGSWCompressedEncryptSk, GLWECompressedEncryptSk,
    layouts::{
        GGSWCompressed, GGSWInfos, GLWECompressed, GLWEInfos, ModuleCoreAlloc,
        ModuleCoreCompressedAlloc, prepared::GLWESecretPreparedToBackendRef,
    },
};
use poulpy_hal::{
    layouts::{
        Backend, HostDataMut, Module, ScalarZnx, ScalarZnxToBackendRef, ScratchArena, ZnxViewMut,
    },
    source::Source,
};

use crate::database::DatabaseLayout;
use crate::encoding::ModPEncoder;

/// Tagged Common Reference Seed. A thin newtype around `[u8; 32]` so call
/// sites can pass an explicit `CRS(seed)` rather than a bare byte array.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CRS(pub [u8; 32]);

impl CRS {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Encrypted PIR query for a single retrieval target.
///
/// The retrieval target is the triple `(target_matrix, target_row_block,
/// target_column)`. The row-block index is only used client-side after
/// decryption and is **not** carried in the query; the query encrypts the
/// first-dim column selector and the second-dim matrix selector.
///
/// # In-memory layout
///
/// * [`crs`](Self::crs) — single 32-byte Common Reference Seed. Every public
///   mask inside `blocks` and `selector` is derived from it via
///   [`Source::new_seed`]; the server regenerates the same masks from `crs`.
/// * [`blocks`](Self::blocks) — `k_blocks` [`GLWECompressed`] one-hot
///   first-dim selectors. Block `b_target = target_column / n` encrypts
///   `Δ · X^{target_column mod n}`; every other block encrypts the zero
///   polynomial.
/// * [`selector`](Self::selector) — a [`GGSWCompressed`] encrypting the IDFT
///   root of unity `X^{(2n · target_matrix) / interpolation_t}` (signed
///   monomial in `Z[X]/(X^n+1)`).
///
/// `Query` is **host-owned** (the inner buffers are plain `Vec<u8>`), like
/// every other compressed ciphertext in `poulpy-core`; it carries no
/// `Backend` type parameter. Pass it to backends whose
/// `Backend::OwnedBuf = Vec<u8>` (the host backends — `FFT64Ref`, `NTT120Ref`,
/// etc.); other backends would need a backend-specific conversion.
///
/// # Wire format vs. in-memory representation
///
/// The compact "paper-style" form `(crs, dim_0: VecZnx[k_blocks columns],
/// dim_1: MatZnx)` is only meaningful for serialization: every seed inside
/// `blocks` / `selector` is CRS-derived and need not be transmitted, and
/// only the bodies / matrix data carry information. `poulpy-core` currently
/// hides those bodies behind the encryption path (no public field accessor
/// / no public constructor from external bytes), so the in-memory `Query`
/// keeps the full envelopes intact; a custom `WriterTo` / `ReaderFrom` on
/// `Query` is the right place to add the packed wire format later.
pub struct Query {
    crs: [u8; 32],
    blocks: Vec<GLWECompressed<Vec<u8>>>,
    selector: GGSWCompressed<Vec<u8>>,
}

impl Query {
    /// Encrypts a fresh query for retrieving `(target_matrix, target_column)`.
    ///
    /// `glwe_infos` / `ggsw_infos` must carry the noise parameter (`σ_χ`),
    /// typically obtained from
    /// `EncryptionLayout::new_from_default_sigma(GLWELayout { … })`.
    #[allow(clippy::too_many_arguments)]
    pub fn new<BE, G, H, SK>(
        module: &Module<BE>,
        crs: [u8; 32],
        layout: &DatabaseLayout,
        glwe_infos: &G,
        ggsw_infos: &H,
        encoder: &ModPEncoder,
        sk_prepared: &SK,
        target_matrix: usize,
        target_column: usize,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Self
    where
        BE: Backend<OwnedBuf = Vec<u8>>,
        Vec<u8>: HostDataMut,
        Module<BE>: ModuleCoreAlloc<OwnedBuf = Vec<u8>>
            + ModuleCoreCompressedAlloc
            + GLWECompressedEncryptSk<BE>
            + GGSWCompressedEncryptSk<BE>,
        G: GLWEInfos + EncryptionInfos,
        H: GGSWInfos + EncryptionInfos,
        SK: GLWESecretPreparedToBackendRef<BE>,
        ScalarZnx<Vec<u8>>: ScalarZnxToBackendRef<BE>,
    {
        let n = module.n();
        let k_blocks = layout.k_blocks;
        assert_eq!(
            layout.n, n,
            "DatabaseLayout n ({}) does not match module degree ({})",
            layout.n, n,
        );
        assert!(
            target_matrix < layout.nb_matrices.max(1),
            "target_matrix {target_matrix} \u{2265} nb_matrices {}",
            layout.nb_matrices.max(1),
        );
        assert!(
            target_column < layout.cols,
            "target_column {target_column} \u{2265} cols {}",
            layout.cols,
        );
        assert_eq!(
            glwe_infos.n().as_usize(),
            n,
            "glwe_infos.n must match module degree",
        );
        assert_eq!(
            ggsw_infos.n().as_usize(),
            n,
            "ggsw_infos.n must match module degree",
        );

        // CRS sub-seeds: k_blocks for the GLWE block masks, then one for the GGSW.
        let mut source = Source::new(crs);
        let block_seeds: Vec<[u8; 32]> = (0..k_blocks).map(|_| source.new_seed()).collect();
        let ggsw_seed = source.new_seed();

        // ---- First-dim blocks --------------------------------------------------
        let b_target = target_column / n;
        let c_within = target_column % n;
        let glwe_base2k = glwe_infos.base2k().as_usize();
        let mut pt = module.glwe_plaintext_alloc_from_infos(glwe_infos);
        let mut blocks: Vec<GLWECompressed<Vec<u8>>> = Vec::with_capacity(k_blocks);
        for b in 0..k_blocks {
            if b == b_target {
                encoder.encode_one_hot_into(&mut pt.data, glwe_base2k, 0, c_within);
            } else {
                encoder.encode_zero_into(&mut pt.data, 0);
            }
            let mut block = module.glwe_compressed_alloc_from_infos(glwe_infos);
            module.glwe_compressed_encrypt_sk(
                &mut block,
                &pt,
                sk_prepared,
                block_seeds[b],
                glwe_infos,
                source_xe,
                scratch,
            );
            blocks.push(block);
        }

        // ---- Second-dim selector -----------------------------------------------
        // IDFT primitive root `ω = X^{2n/t}`. Selecting matrix `m` encrypts
        // `ω^m = X^{(2n·m)/t}` (signed monomial when ≥ n).
        let exponent = (2 * n * target_matrix) / layout.interpolation_t.max(1);
        let root_pt = root_monomial(module, exponent);
        let mut selector = module.ggsw_compressed_alloc_from_infos(ggsw_infos);
        module.ggsw_compressed_encrypt_sk(
            &mut selector,
            &root_pt,
            sk_prepared,
            ggsw_seed,
            ggsw_infos,
            source_xe,
            scratch,
        );

        Self {
            crs,
            blocks,
            selector,
        }
    }

    /// Scratch bytes required by [`Query::new`].
    pub fn new_tmp_bytes<BE, G, H>(module: &Module<BE>, glwe_infos: &G, ggsw_infos: &H) -> usize
    where
        BE: Backend,
        Module<BE>: GLWECompressedEncryptSk<BE> + GGSWCompressedEncryptSk<BE>,
        G: GLWEInfos,
        H: GGSWInfos,
    {
        module
            .glwe_compressed_encrypt_sk_tmp_bytes(glwe_infos)
            .max(module.ggsw_compressed_encrypt_sk_tmp_bytes(ggsw_infos))
    }

    pub fn crs(&self) -> [u8; 32] {
        self.crs
    }

    pub fn blocks(&self) -> &[GLWECompressed<Vec<u8>>] {
        &self.blocks
    }

    pub fn selector(&self) -> &GGSWCompressed<Vec<u8>> {
        &self.selector
    }

    pub fn k_blocks(&self) -> usize {
        self.blocks.len()
    }
}

/// Builds the monomial plaintext for an IDFT root `ω^m = X^e` in
/// `Z[X]/(X^n+1)`.
///
/// The monomial group of the ring has order `2n` (`X^{2n} = 1`), so the second
/// dimension — which addresses up to `2n` matrix-axis ciphertexts — is reduced
/// **modulo `2n`**. The reduced exponent maps to `+X^e` for `e < n` and to
/// `-X^{e-n}` for `n ≤ e < 2n`.
fn root_monomial<BE: Backend>(module: &Module<BE>, exponent: usize) -> ScalarZnx<BE::OwnedBuf>
where
    Module<BE>: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
    BE::OwnedBuf: HostDataMut,
{
    let n = module.n();
    let exponent = exponent % (2 * n);
    let mut root = module.scalar_znx_alloc(1);
    if exponent < n {
        root.at_mut(0, 0)[exponent] = 1;
    } else {
        root.at_mut(0, 0)[exponent - n] = -1;
    }
    root
}
