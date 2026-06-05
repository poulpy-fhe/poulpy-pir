use poulpy_core::{
    EncryptionLayout, GGSWEncryptSk, GLWECompressedEncryptSk, GLWEDecrypt, GLWENoise,
    layouts::{
        BackendGLWESecretPrepared, Degree, GLWESecret, GLWESecretPreparedFactory, LWESecret,
        ModuleCoreAlloc, ModuleCoreCompressedAlloc, Rank, SecretConversion, TorusPrecision,
    },
};
use poulpy_hal::{
    api::{
        ModuleN, ModuleNew, ScalarZnxAutomorphismBackend, ScratchOwnedAlloc, ScratchOwnedBorrow,
    },
    layouts::{
        Backend, HostBackend, HostDataMut, HostDataRef, Module, ScalarZnxToBackendMut,
        ScalarZnxToBackendRef, ScratchOwned,
    },
    source::Source,
};

use crate::{
    config::{Collapse, Config, INSPIRE_INT_32B},
    database::{DatabaseLayout, PayloadAddress},
    interpolation::{Interpolation, InterpolationKeys},
    packing::PackingKeysGenerate,
    packing::recursion::qtilde_glwe_layout,
    parameters::Parameters,
    payload::{P65535, Payload},
    server::{Query, RecursionKeys, RecursionQuery, generate_recursion_key},
};

use super::{
    seed::{MaskSeeds, ServerSeed},
    state::{QueryCommon, QueryContext, QueryState, Response, ResponseNoise, Sk},
};

/// PIR client: owns the shared [`Parameters`] (and its [`Module`]) plus its own
/// scratch. Host backends only (`BE::OwnedBuf = Vec<u8>`).
pub struct Client<BE: Backend, P: Payload<[u8; 32]> = P65535<[u8; 32]>> {
    params: Parameters<BE, [u8; 32], P>,
    layout: DatabaseLayout<P>,
    scratch: ScratchOwned<BE>,
}

struct QueryMaterial<BE: Backend> {
    sk_lwe: LWESecret<BE::OwnedBuf>,
    sk_query_prep: BackendGLWESecretPrepared<BE>,
    sk_pack_prep: BackendGLWESecretPrepared<BE>,
    source_xe: Source,
    source_xa: Source,
}

#[derive(Clone, Copy)]
enum OneHotEncoding {
    ModP,
    Native { precision: usize },
}

impl<BE: Backend<OwnedBuf = Vec<u8>>> Default for Client<BE, P65535<[u8; 32]>>
where
    Module<BE>: ModuleNew<BE>
        + PackingKeysGenerate<BE>
        + GLWECompressedEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GGSWEncryptSk<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
{
    fn default() -> Self {
        let config = INSPIRE_INT_32B;
        let params = config.new::<BE>();
        let layout = DatabaseLayout::<P65535<[u8; 32]>>::new(params.n(), params.n());
        let scratch = ScratchOwned::<BE>::alloc(client_scratch_bytes(&params));
        Self {
            params,
            layout,
            scratch,
        }
    }
}

impl<BE: Backend<OwnedBuf = Vec<u8>>, P: Payload<[u8; 32]>> Client<BE, P>
where
    Module<BE>: ModuleN
        + ModuleNew<BE>
        + ModuleCoreAlloc<OwnedBuf = Vec<u8>>
        + ModuleCoreCompressedAlloc
        + SecretConversion<BE>
        + GLWESecretPreparedFactory<BE>
        + PackingKeysGenerate<BE>
        + GLWECompressedEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GGSWEncryptSk<BE>
        + ScalarZnxAutomorphismBackend<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'b> BE::BufRef<'b>: HostDataRef,
    for<'b> BE::BufMut<'b>: HostDataMut,
{
    pub fn new(config: Config<[u8; 32], P>, layout: DatabaseLayout<P>) -> Self {
        let params = config.new::<BE>();
        let scratch = ScratchOwned::<BE>::alloc(client_scratch_bytes(&params));
        Self {
            params,
            layout,
            scratch,
        }
    }

    /// The shared parameters (handy for callers that build reduction units).
    pub fn params(&self) -> &Parameters<BE, [u8; 32], P> {
        &self.params
    }

    /// The shared database layout.
    pub fn layout(&self) -> DatabaseLayout<P> {
        self.layout
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
        let glwe_query = params.glwe_query();
        let mut material = self.fresh_query_material(server_seed.root_a());

        // Packing keys: public `a` from `server_seed.keys()`, error from the secret stream.
        let (key_g, key_h) = module.pack_keys_generate(
            &params.key_layout(),
            &material.sk_lwe,
            server_seed.keys(),
            &mut material.source_xe,
            &mut self.scratch.borrow(),
        );

        let blocks = self.encrypt_one_hot_blocks(
            address.column,
            self.layout.cols(),
            server_seed.mask(),
            &glwe_query,
            &material.sk_query_prep,
            &mut material.source_xe,
            OneHotEncoding::ModP,
        );

        let common = QueryCommon { blocks };
        let ctx = QueryContext {
            sk_pack_prep: material.sk_pack_prep,
            source_xe: material.source_xe,
            source_xa: material.source_xa,
            interpolation_keys: Some(InterpolationKeys::new(key_g, key_h)),
        };
        (common, ctx, Sk::new(material.sk_lwe))
    }

    /// Build a query for `payload_index`, dispatching internally on the selected
    /// collapse.
    pub fn query(&mut self, payload_index: usize) -> (Query<BE>, QueryState<BE>) {
        let address = self
            .layout
            .address_for(payload_index, self.params.column_height());
        let server_seed = ServerSeed::default();
        match self.params.collapse() {
            Collapse::Interpolation => {
                let (common, mut ctx, sk) = self.begin_query(&address, &server_seed);
                let interpolation = Interpolation::new(&self.layout, &self.params);
                let query = Query::Interpolation(interpolation.build_query(
                    self.params.module(),
                    common,
                    &mut ctx,
                    &address,
                    &mut self.scratch,
                ));
                (query, QueryState::new(sk, address))
            }
            Collapse::Recursion {
                gamma0,
                gamma1,
                gamma2,
            } => {
                let mut material = self.fresh_query_material(server_seed.root_a());
                let src_infos = self.params.glwe_pack();
                let precision = self.params.matmul_base2k();
                let t = self.layout.grid_rows_for(gamma0);
                let src0 = self.encrypt_one_hot_blocks(
                    address.column,
                    self.layout.cols(),
                    server_seed.recursion_a0(),
                    &src_infos,
                    &material.sk_query_prep,
                    &mut material.source_xe,
                    OneHotEncoding::Native { precision },
                );
                let src1 = self.encrypt_one_hot_blocks(
                    address.matrix,
                    t,
                    server_seed.recursion_a1(),
                    &src_infos,
                    &material.sk_query_prep,
                    &mut material.source_xe,
                    OneHotEncoding::Native { precision },
                );
                let keys = self.recursion_keys(
                    &material.sk_lwe,
                    &mut material.source_xe,
                    [gamma0, gamma1, gamma2],
                    &server_seed,
                );
                let query = Query::Recursion(RecursionQuery { src0, src1, keys });
                (query, QueryState::new(Sk::new(material.sk_lwe), address))
            }
        }
    }

    /// Decrypts the server's [`Response`] with `sk`, dispatching on the collapse
    /// variant. For [`Response::Interpolation`] the result is the `n` recovered
    /// coefficients (the queried column; a payload's `16` base-65535 digits are
    /// `out[row_offset .. row_offset + 16]`). For [`Response::Recursion`] it is the
    /// `γ0` `Z_p` record.
    pub fn decrypt(&mut self, response: &Response<BE>, sk: &Sk<BE>) -> Vec<i64> {
        let params = &self.params;
        let module = params.module();
        let sk_pack_prep = self.prepare_pack_secret(sk);
        match response {
            Response::Interpolation(response) => {
                let mut pt = module.glwe_plaintext_alloc_from_infos(&params.glwe_pack());
                module.glwe_decrypt(
                    response.selected(),
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
            Response::Recursion(response) => {
                // Digits are base-2^16; q̃ = 2·matmul_base2k. The dst secret is the
                // same wrapped LWE secret used above.
                let k_pt = params.matmul_base2k();
                let Collapse::Recursion {
                    gamma0,
                    gamma1,
                    gamma2,
                } = params.collapse()
                else {
                    panic!("Response::Recursion requires Collapse::Recursion parameters");
                };
                super::recursion::extract_response(
                    module,
                    &sk_pack_prep,
                    gamma0,
                    gamma1,
                    gamma2,
                    k_pt,
                    2 * k_pt,
                    response,
                )
            }
        }
    }

    pub fn decrypt_digits(&mut self, response: &Response<BE>, state: &QueryState<BE>) -> Vec<i64> {
        self.decrypt(response, state.sk())
    }

    /// Computes the final response noise against the full selected plaintext
    /// record, after the same extraction/recomposition used by [`Self::decrypt`].
    pub fn noise(
        &self,
        response: &Response<BE>,
        state: &QueryState<BE>,
        expected_record: &[i64],
    ) -> ResponseNoise
    where
        BE: HostBackend,
        Module<BE>: GLWENoise<BE>,
    {
        let params = &self.params;
        let module = params.module();
        assert_eq!(
            expected_record.len(),
            params.column_height(),
            "noise expects the full selected record"
        );
        let sk_pack_prep = self.prepare_pack_secret(state.sk());

        match response {
            Response::Interpolation(response) => {
                let mut expected_values = vec![0i64; params.n()];
                expected_values.copy_from_slice(expected_record);

                let mut expected = module.glwe_plaintext_alloc_from_infos(&params.glwe_pack());
                params.encoder().encode_vec_i64(
                    &mut expected.data,
                    params.base2k(),
                    0,
                    &expected_values,
                );

                let mut scratch =
                    ScratchOwned::<BE>::alloc(module.glwe_noise_tmp_bytes(response.selected()));
                let stats = module.glwe_noise(
                    response.selected(),
                    &expected,
                    &sk_pack_prep,
                    &mut scratch.borrow(),
                );
                ResponseNoise::new(stats.max(), stats.std())
            }
            Response::Recursion(response) => {
                let k_pt = params.matmul_base2k();
                let Collapse::Recursion {
                    gamma0,
                    gamma1,
                    gamma2,
                } = params.collapse()
                else {
                    panic!("Response::Recursion requires Collapse::Recursion parameters");
                };

                let recomposed = super::recursion::recompose_response(
                    module,
                    &sk_pack_prep,
                    gamma0,
                    gamma1,
                    gamma2,
                    k_pt,
                    2 * k_pt,
                    response,
                );
                let qtilde_infos = qtilde_glwe_layout(Degree(params.n() as u32), 2 * k_pt);
                let mut scratch =
                    ScratchOwned::<BE>::alloc(module.glwe_noise_tmp_bytes(&recomposed));
                let mut expected = module.glwe_plaintext_alloc_from_infos(&qtilde_infos);
                module.glwe_decrypt(
                    &recomposed,
                    &mut expected,
                    &sk_pack_prep,
                    &mut scratch.borrow(),
                );
                // InsPIRe2 only exposes the first gamma0 slots after the second decrypt.
                // Keep the non-output tail neutral so GLWENoise scores the decoded surface.
                for (idx, &value) in expected_record.iter().take(gamma0).enumerate() {
                    expected.encode_coeff_i64(value, TorusPrecision(k_pt as u32), idx);
                }
                let stats =
                    module.glwe_noise(&recomposed, &expected, &sk_pack_prep, &mut scratch.borrow());
                ResponseNoise::new(stats.max(), stats.std())
            }
        }
    }

    pub fn decode(&mut self, response: &Response<BE>, state: &QueryState<BE>) -> [u8; 32] {
        let digits = self.decrypt_digits(response, state);
        let start = state.address().row_offset;
        let end = start + P::EXPONENT;
        assert!(
            end <= digits.len(),
            "response has {} digits but payload slice ends at {end}",
            digits.len()
        );
        let payload_digits: Vec<i16> = digits[start..end].iter().map(|&v| v as i16).collect();
        let mut out = [0u8; 32];
        P::decode(&mut out, &payload_digits);
        out
    }

    fn prepare_pack_secret(&self, sk: &Sk<BE>) -> BackendGLWESecretPrepared<BE> {
        let module = self.params.module();
        let sk_pack = glwe_secret_wrap_lwe(module, sk.sk_lwe());
        let mut sk_pack_prep = module.glwe_secret_prepared_alloc_from_infos(&sk_pack);
        module.glwe_secret_prepare(&mut sk_pack_prep, &sk_pack);
        sk_pack_prep
    }

    fn fresh_query_material(&self, root_a_seed: [u8; 32]) -> QueryMaterial<BE> {
        let params = &self.params;
        let module = params.module();
        let mut seed_secret = [0u8; 32];
        getrandom::fill(&mut seed_secret).expect("OS entropy");
        let mut secret = Source::new(seed_secret);
        let mut source_xs = Source::new(secret.new_seed());
        let source_xe = Source::new(secret.new_seed());
        let source_xa = Source::new(root_a_seed);

        let mut sk_lwe = module.lwe_secret_alloc(Degree(params.n() as u32));
        sk_lwe.fill_ternary_prob(0.5, &mut source_xs);
        let sk_query = module.glwe_secret_from_lwe_secret(&sk_lwe);
        let sk_pack = glwe_secret_wrap_lwe(module, &sk_lwe);
        let mut sk_query_prep = module.glwe_secret_prepared_alloc_from_infos(&sk_query);
        module.glwe_secret_prepare(&mut sk_query_prep, &sk_query);
        let mut sk_pack_prep = module.glwe_secret_prepared_alloc_from_infos(&sk_pack);
        module.glwe_secret_prepare(&mut sk_pack_prep, &sk_pack);

        QueryMaterial {
            sk_lwe,
            sk_query_prep,
            sk_pack_prep,
            source_xe,
            source_xa,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn encrypt_one_hot_blocks(
        &mut self,
        idx: usize,
        len: usize,
        seed_root: [u8; 32],
        infos: &EncryptionLayout<poulpy_core::layouts::GLWELayout>,
        sk_query_prep: &BackendGLWESecretPrepared<BE>,
        source_xe: &mut Source,
        encoding: OneHotEncoding,
    ) -> Vec<poulpy_core::layouts::GLWECompressed<BE::OwnedBuf>> {
        assert!(
            idx < len,
            "query index {idx} out of bounds for length {len}"
        );
        let params = &self.params;
        let module = params.module();
        let n = params.n();
        let blocks = len.div_ceil(n);
        let mask_seeds = MaskSeeds::new(seed_root);
        (0..blocks)
            .map(|block| {
                let start = block * n;
                let width = (len - start).min(n);
                let mut query_pt = module.glwe_plaintext_alloc_from_infos(infos);
                match encoding {
                    OneHotEncoding::ModP => {
                        if idx >= start && idx < start + width {
                            params.encoder().encode_one_hot_into(
                                &mut query_pt.data,
                                params.matmul_base2k(),
                                0,
                                idx - start,
                            );
                        } else {
                            params.encoder().encode_zero_into(&mut query_pt.data, 0);
                        }
                    }
                    OneHotEncoding::Native { precision } => {
                        let mut sel = vec![0i64; n];
                        if idx >= start && idx < start + width {
                            sel[idx - start] = 1;
                        }
                        query_pt.encode_vec_i64(&sel, TorusPrecision(precision as u32));
                    }
                }
                let mut block_ct = module.glwe_compressed_alloc_from_infos(infos);
                module.glwe_compressed_encrypt_sk(
                    &mut block_ct,
                    &query_pt,
                    sk_query_prep,
                    mask_seeds.seed(block),
                    infos,
                    source_xe,
                    &mut self.scratch.borrow(),
                );
                block_ct
            })
            .collect()
    }

    fn recursion_keys(
        &mut self,
        sk_lwe: &LWESecret<BE::OwnedBuf>,
        source_xe: &mut Source,
        gammas: [usize; 3],
        server_seed: &ServerSeed,
    ) -> RecursionKeys<BE> {
        let key_infos = self.params.key_layout();
        let n = self.params.n();
        let module = self.params.module();
        RecursionKeys {
            gamma0: generate_recursion_key(
                module,
                &key_infos,
                sk_lwe,
                n,
                gammas[0],
                server_seed.recursion_key(0),
                source_xe,
                &mut self.scratch.borrow(),
            ),
            gamma1: generate_recursion_key(
                module,
                &key_infos,
                sk_lwe,
                n,
                gammas[1],
                server_seed.recursion_key(1),
                source_xe,
                &mut self.scratch.borrow(),
            ),
            gamma2: generate_recursion_key(
                module,
                &key_infos,
                sk_lwe,
                n,
                gammas[2],
                server_seed.recursion_key(2),
                source_xe,
                &mut self.scratch.borrow(),
            ),
        }
    }
}

/// Wrap an LWE secret into a rank-1 GLWE secret (the pack-regime secret).
fn glwe_secret_wrap_lwe<BE: Backend<OwnedBuf = Vec<u8>>>(
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
fn client_scratch_bytes<BE: Backend, P: Payload<[u8; 32]>>(
    params: &Parameters<BE, [u8; 32], P>,
) -> usize
where
    Module<BE>:
        PackingKeysGenerate<BE> + GLWECompressedEncryptSk<BE> + GLWEDecrypt<BE> + GGSWEncryptSk<BE>,
{
    let module = params.module();
    module
        .pack_keys_generate_tmp_bytes(&params.key_layout())
        .max(module.glwe_compressed_encrypt_sk_tmp_bytes(&params.glwe_query()))
        .max(module.ggsw_encrypt_sk_tmp_bytes(&params.ggsw_layout()))
        .max(module.glwe_decrypt_tmp_bytes(&params.glwe_pack()))
}
