//! PIR client: builds the common first-dimension query material (one-hot
//! selector bodies + packing keys) and decrypts the server's [`Response`]. The
//! reduction-specific part of the query (e.g. the interpolation GGSW root) is
//! built by the chosen reduction unit from the ephemeral [`QueryContext`] this
//! returns — see [`crate::interpolation::Interpolation`].
//!
//! Seeds. The server hands the client one public root seed, wrapped in
//! [`ServerSeed`], from which three domain-separated sub-seeds are derived:
//! - [`ServerSeed::mask`]: the query mask `A`. Wrapped in [`MaskSeeds`], each
//!   block-column derives its own seed (`mask[0..28] ‖ i`) so the per-block masks
//!   are independent. The server re-derives the identical seeds to materialize
//!   `A`, so the `a·s` terms cancel.
//! - [`ServerSeed::keys`]: the packing keys' public `a` part.
//! - [`ServerSeed::root_a`]: the (full, self-contained) GGSW root's public `a`.
//!
//! The client's own secret (the LWE secret and all encryption error) is sampled
//! locally from OS entropy on each [`Client::begin_query`] — never an input.
//!
//! Generic over the backend `BE`, but assumes a **host** backend
//! (`BE::OwnedBuf = Vec<u8>`) so the query/answer buffers are plain `Vec<u8>`.

use poulpy_core::{
    GLWECompressedEncryptSk, GLWEDecrypt,
    layouts::{
        BackendGLWESecretPrepared, Degree, GLWE, GLWEAutomorphismKeyCompressed, GLWECompressed,
        GLWESecret, GLWESecretPreparedFactory, LWESecret, ModuleCoreAlloc,
        ModuleCoreCompressedAlloc, Rank, SecretConversion,
    },
};
use poulpy_hal::{
    api::{
        ModuleN, ModuleNew, ScalarZnxAutomorphismBackend, ScratchOwnedAlloc, ScratchOwnedBorrow,
    },
    layouts::{
        Backend, HostDataMut, HostDataRef, Module, ScalarZnxToBackendMut, ScalarZnxToBackendRef,
        ScratchOwned,
    },
    source::Source,
};

use crate::{database::PayloadAddress, packing::PackingKeysGenerate, parameters::Parameters};

/// The single public root seed the server hands the client. All public
/// randomness a query needs — the mask `A`, the packing keys' `a`, and the GGSW
/// root's `a` — is derived from it as domain-separated sub-seeds (independent
/// ChaCha streams). The server holds the same root and re-derives [`mask`](Self::mask)
/// and [`keys`](Self::keys) to reconstruct its half of the protocol.
#[derive(Copy, Clone, Debug)]
pub struct ServerSeed {
    root: [u8; 32],
}

impl ServerSeed {
    pub fn new(root: [u8; 32]) -> Self {
        Self { root }
    }

    /// The `n`-th sub-seed: the `(n+1)`-th draw of a PRNG seeded by the root.
    /// Each is a full 32-byte ChaCha output, so distinct sub-seeds share no
    /// prefix (unlike a stamped index) and [`MaskSeeds`] can safely re-stamp the
    /// low bytes of [`mask`](Self::mask) without colliding with the others.
    fn sub_seed(&self, n: usize) -> [u8; 32] {
        let mut source = Source::new(self.root);
        let mut seed = [0u8; 32];
        for _ in 0..=n {
            seed = source.new_seed();
        }
        seed
    }

    /// Sub-seed for the query mask `A` (feed to [`MaskSeeds`]).
    pub fn mask(&self) -> [u8; 32] {
        self.sub_seed(0)
    }

    /// Sub-seed for the packing keys' public `a`.
    pub fn keys(&self) -> [u8; 32] {
        self.sub_seed(1)
    }

    /// Sub-seed for the GGSW root's public `a`.
    pub fn root_a(&self) -> [u8; 32] {
        self.sub_seed(2)
    }
}

/// Per-block-column mask seeds, addressable in O(1). The seed for block `i` is
/// the public root with its low 4 bytes overwritten by `i` (`root[0..28] ‖ i`),
/// so any block's seed is computed directly — no sequential PRNG walk, no
/// materialized table — which scales to thousands (up to `2^32`) of blocks.
/// Distinct indices yield distinct ChaCha keys, so the per-block masks are
/// independent. The root is [`ServerSeed::mask`]; the client seeds its query
/// bodies and the server its masks from it, so the `a·s` terms cancel.
#[derive(Copy, Clone, Debug)]
pub struct MaskSeeds {
    root: [u8; 32],
}

impl MaskSeeds {
    pub fn new(root: [u8; 32]) -> Self {
        Self { root }
    }

    /// Seed for block-column `block`. Requires `block <= u32::MAX`.
    pub fn seed(&self, block: usize) -> [u8; 32] {
        assert!(
            block <= u32::MAX as usize,
            "block index {block} exceeds 2^32"
        );
        let mut seed = self.root;
        seed[28..].copy_from_slice(&(block as u32).to_le_bytes());
        seed
    }
}

/// The reduction-independent half of the query: the packing keys and the per
/// block-column one-hot selector bodies. A reduction unit wraps this in its own
/// query struct alongside its reduction-specific selector.
pub struct QueryCommon<BE: Backend> {
    pub key_g: GLWEAutomorphismKeyCompressed<BE::OwnedBuf>,
    pub key_h: GLWEAutomorphismKeyCompressed<BE::OwnedBuf>,
    /// One-hot query bodies, one per block-column (compressed; seed-derived `a`).
    pub blocks: Vec<GLWECompressed<BE::OwnedBuf>>,
}

/// Ephemeral secret handles the client hands to a reduction unit so it can build
/// its query selector (e.g. encrypt the GGSW root under the same secret, with the
/// error stream continued from the common material). Holds nothing the server
/// ever sees; drop it as soon as the reduction has built its query.
pub struct QueryContext<BE: Backend> {
    pub sk_pack_prep: BackendGLWESecretPrepared<BE>,
    pub source_xe: Source,
    pub source_xa: Source,
}

/// The client's secret state, returned alongside the query and kept locally to
/// decrypt the [`Response`]. Never transmitted.
pub struct Sk<BE: Backend> {
    sk_lwe: LWESecret<BE::OwnedBuf>,
}

impl<BE: Backend> Sk<BE> {
    pub fn sk_lwe(&self) -> &LWESecret<BE::OwnedBuf> {
        &self.sk_lwe
    }
}

/// The server's answer: one packed GLWE holding the selected coefficient column.
pub struct Response<BE: Backend> {
    pub selected: GLWE<BE::OwnedBuf>,
}

/// PIR client: owns the shared [`Parameters`] (and its [`Module`]) plus its own
/// scratch. Host backends only (`BE::OwnedBuf = Vec<u8>`).
pub struct Client<BE: Backend> {
    params: Parameters<BE>,
    scratch: ScratchOwned<BE>,
}

impl<BE: Backend<OwnedBuf = Vec<u8>>> Default for Client<BE>
where
    Module<BE>:
        ModuleNew<BE> + PackingKeysGenerate<BE> + GLWECompressedEncryptSk<BE> + GLWEDecrypt<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
{
    fn default() -> Self {
        let params = Parameters::default();
        let scratch = ScratchOwned::<BE>::alloc(client_scratch_bytes(&params));
        Self { params, scratch }
    }
}

impl<BE: Backend<OwnedBuf = Vec<u8>>> Client<BE>
where
    Module<BE>: ModuleN
        + ModuleCoreAlloc<OwnedBuf = Vec<u8>>
        + ModuleCoreCompressedAlloc
        + SecretConversion<BE>
        + GLWESecretPreparedFactory<BE>
        + PackingKeysGenerate<BE>
        + GLWECompressedEncryptSk<BE>
        + GLWEDecrypt<BE>
        + ScalarZnxAutomorphismBackend<BE>,
    ScratchOwned<BE>: ScratchOwnedBorrow<BE>,
    for<'b> BE::BufRef<'b>: HostDataRef,
    for<'b> BE::BufMut<'b>: HostDataMut,
{
    /// The shared parameters (handy for callers that build reduction units).
    pub fn params(&self) -> &Parameters<BE> {
        &self.params
    }

    /// Builds the reduction-independent query material for `address`. `server_seed`
    /// is the server's public root (mask `A`, packing keys' `a`, GGSW root's `a`);
    /// the client's secret (LWE secret + all error) is sampled locally from OS
    /// entropy. Returns the public [`QueryCommon`], the ephemeral [`QueryContext`]
    /// (secret handles for the reduction's selector), and the client-only [`Sk`].
    pub fn begin_query(
        &mut self,
        address: &PayloadAddress,
        server_seed: &ServerSeed,
    ) -> (QueryCommon<BE>, QueryContext<BE>, Sk<BE>) {
        let params = &self.params;
        let module = params.module();
        let encoder = params.encoder();
        let glwe_query = params.glwe_query();

        // The client's secret randomness (secret key + all encryption error) is
        // drawn fresh from OS entropy: independent PRNG sub-streams for the secret
        // key and the error. The public `a` parts come from `server_seed`.
        let mut seed_secret = [0u8; 32];
        getrandom::fill(&mut seed_secret).expect("OS entropy");
        let mut secret = Source::new(seed_secret);
        let mut source_xs = Source::new(secret.new_seed());
        let mut source_xe = Source::new(secret.new_seed());
        let source_xa = Source::new(server_seed.root_a());

        let mut sk_lwe = module.lwe_secret_alloc(Degree(params.n() as u32));
        sk_lwe.fill_ternary_prob(0.5, &mut source_xs);
        let sk_query = module.glwe_secret_from_lwe_secret(&sk_lwe);
        let sk_pack = glwe_secret_wrap_lwe(module, &sk_lwe);
        let mut sk_query_prep = module.glwe_secret_prepared_alloc_from_infos(&sk_query);
        module.glwe_secret_prepare(&mut sk_query_prep, &sk_query);
        let mut sk_pack_prep = module.glwe_secret_prepared_alloc_from_infos(&sk_pack);
        module.glwe_secret_prepare(&mut sk_pack_prep, &sk_pack);

        // Packing keys: public `a` from `server_seed.keys()`, error from the secret stream.
        let (key_g, key_h) = module.pack_keys_generate(
            &params.key_layout(),
            &sk_lwe,
            server_seed.keys(),
            &mut source_xe,
            &mut self.scratch.borrow(),
        );

        // One-hot query bodies. Each block-column's `a` is seeded by the same
        // index-addressed seed the server uses to materialize its mask `A`.
        let mask_seeds = MaskSeeds::new(server_seed.mask());
        let mut blocks = Vec::with_capacity(address.block_cols);
        for block_col in 0..address.block_cols {
            let seed = mask_seeds.seed(block_col);
            let mut query_pt = module.glwe_plaintext_alloc_from_infos(&glwe_query);
            if block_col == address.block_col {
                encoder.encode_one_hot_into(
                    &mut query_pt.data,
                    params.matmul_base2k(),
                    0,
                    address.col_in_block,
                );
            } else {
                encoder.encode_zero_into(&mut query_pt.data, 0);
            }
            let mut block = module.glwe_compressed_alloc_from_infos(&glwe_query);
            module.glwe_compressed_encrypt_sk(
                &mut block,
                &query_pt,
                &sk_query_prep,
                seed,
                &glwe_query,
                &mut source_xe,
                &mut self.scratch.borrow(),
            );
            blocks.push(block);
        }

        let common = QueryCommon {
            key_g,
            key_h,
            blocks,
        };
        let ctx = QueryContext {
            sk_pack_prep,
            source_xe,
            source_xa,
        };
        (common, ctx, Sk { sk_lwe })
    }

    /// Decrypts the server's [`Response`] with `sk` and decodes the `n` recovered
    /// coefficients (the queried column). The retrieved payload's `16` base-65535
    /// digits are `out[row_offset .. row_offset + 16]`.
    pub fn decrypt(&mut self, response: &Response<BE>, sk: &Sk<BE>) -> Vec<i64> {
        let params = &self.params;
        let module = params.module();
        let sk_pack = glwe_secret_wrap_lwe(module, &sk.sk_lwe);
        let mut sk_pack_prep = module.glwe_secret_prepared_alloc_from_infos(&sk_pack);
        module.glwe_secret_prepare(&mut sk_pack_prep, &sk_pack);
        let mut pt = module.glwe_plaintext_alloc_from_infos(&params.glwe_pack());
        module.glwe_decrypt(
            &response.selected,
            &mut pt,
            &sk_pack_prep,
            &mut self.scratch.borrow(),
        );
        let mut out = vec![0i64; params.n()];
        params
            .encoder()
            .decode_vec_i64(&pt.data, params.base2k(), 0, &mut out);
        out
    }
}

/// Wrap an LWE secret into a rank-1 GLWE secret (the pack-regime secret).
pub(crate) fn glwe_secret_wrap_lwe<BE: Backend<OwnedBuf = Vec<u8>>>(
    module: &Module<BE>,
    sk_lwe: &LWESecret<BE::OwnedBuf>,
) -> GLWESecret<BE::OwnedBuf>
where
    Module<BE>: ModuleCoreAlloc<OwnedBuf = Vec<u8>> + ScalarZnxAutomorphismBackend<BE>,
    for<'b> BE::BufRef<'b>: HostDataRef,
    for<'b> BE::BufMut<'b>: HostDataMut,
{
    let mut sk_glwe = module.glwe_secret_alloc(Rank(1));
    sk_glwe.fill_zero();
    {
        let src_ref = ScalarZnxToBackendRef::<BE>::to_backend_ref(sk_lwe.data());
        let mut dst_mut = ScalarZnxToBackendMut::<BE>::to_backend_mut(sk_glwe.data_mut());
        module.scalar_znx_automorphism_backend(1, &mut dst_mut, 0, &src_ref, 0);
    }
    sk_glwe
}

/// Scratch large enough for every client operation in [`Client::begin_query`] /
/// [`Client::decrypt`]. (The reduction's selector encryption uses its own scratch.)
fn client_scratch_bytes<BE: Backend>(params: &Parameters<BE>) -> usize
where
    Module<BE>: PackingKeysGenerate<BE> + GLWECompressedEncryptSk<BE> + GLWEDecrypt<BE>,
{
    let module = params.module();
    0usize
        .max(module.pack_keys_generate_tmp_bytes(&params.key_layout()))
        .max(module.glwe_compressed_encrypt_sk_tmp_bytes(&params.glwe_query()))
        .max(module.glwe_decrypt_tmp_bytes(&params.glwe_pack()))
}
