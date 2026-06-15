//! Server construction and the fixed CRS query-mask expansion (SETUP).

use std::time::{Duration, Instant};

use poulpy_core::{
    GLWEDecrypt, GLWEExpandLWEMatrix, GLWENormalize,
    layouts::{
        Degree, LWEInfos, LWEMatrix, LWEMatrixLayout, LWEMatrixToBackendMut, ModuleCoreAlloc,
    },
};
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxDftAutomorphismPlan},
    layouts::{
        Backend, HostDataMut, HostDataRef, Module, ScratchOwned, VecZnx, VecZnxToBackendMut,
        VecZnxToBackendRef,
    },
    source::Source,
};

use crate::{
    client::ServerSeed,
    config::Collapse,
    database::DatabaseLayout,
    packing::{Packing, PackingMaskAggregation, recursion::qtilde_glwe_layout},
    parameters::Parameters,
    payload::Payload,
    server::{
        OfflineTimings, Server, ServerCollapse, ServerPrecomputation,
        api::RecursionServerModule,
        common::{
            QueryMask, default_query_mask_tmp_bytes, fill_default_query_mask, mask_regime_infos,
        },
    },
};

use super::{
    RecursionPrecomputation, RecursionState, assert_params_valid, generate_recursion_key,
    qtilde_bits, src_infos_for,
};

#[allow(private_bounds)]
impl<BE: Backend<OwnedBuf = Vec<u8>>, P: Payload<[u8; 32]>> Server<BE, P>
where
    BE: poulpy_cpu_ref::reference::fft64::reim::ReimArith,
    Module<BE>: RecursionServerModule<BE>,
    <Module<BE> as VecZnxDftAutomorphismPlan<BE>>::Plan: 'static + Send + Sync,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    VecZnx<Vec<u8>>: VecZnxToBackendMut<BE> + VecZnxToBackendRef<BE>,
    LWEMatrix<Vec<u8>>: LWEMatrixToBackendMut<BE>,
    for<'b> BE::BufRef<'b>: HostDataRef,
    for<'b> BE::BufMut<'b>: HostDataMut,
{
    /// Build the shared server with InsPIRe²-specific state.
    pub(crate) fn new_recursion(
        params: Parameters<BE, [u8; 32], P>,
        layout: DatabaseLayout<P>,
    ) -> Self {
        let key_infos = params.key_layout();
        let Collapse::Recursion {
            gamma0,
            gamma1,
            gamma2,
        } = params.collapse()
        else {
            panic!("Server::new_recursion requires Collapse::Recursion parameters");
        };
        assert_eq!(
            layout.rows() % gamma0,
            0,
            "layout rows must be a multiple of γ0"
        );
        let t = layout.grid_rows_for(gamma0);
        assert_params_valid(&params, t, layout.cols());
        let module = params.module();
        let src_infos = src_infos_for(&params);
        let mask_infos = mask_regime_infos(&params);
        let server_seed = ServerSeed::default();

        let scratch_bytes = module
            .glwe_decrypt_tmp_bytes(&qtilde_glwe_layout(
                Degree(params.n() as u32),
                qtilde_bits(&params),
            ))
            .max(default_query_mask_tmp_bytes(
                module,
                &mask_infos,
                &params.glwe_mask(),
            ))
            .max(module.glwe_expand_lwe_matrix_tmp_bytes(
                &LWEMatrixLayout {
                    rows: params.n(),
                    n: Degree(params.n() as u32),
                    base2k: src_infos.base2k(),
                    k: src_infos.max_k(),
                },
                &src_infos,
            ))
            .max(
                module.pack_partial_mask_preprocessing_tmp_bytes(
                    gamma0.max(gamma1),
                    src_infos.size(),
                ),
            )
            .max(module.pack_keys_precompute_tmp_bytes(&key_infos, &key_infos, params.baby_size()))
            .max(module.glwe_normalize_tmp_bytes())
            .max(module.pack_precompute_tmp_bytes(
                crate::packing::PackingPrecomputeInfos::new(
                    params.n() - 1,
                    src_infos.size(),
                    params.base2k(),
                    params.baby_size(),
                ),
                &module.vec_znx_alloc(params.n(), src_infos.size()),
                &key_infos,
            ));
        let mut scratch = ScratchOwned::<BE>::alloc(scratch_bytes);

        let mut sk = module.lwe_secret_alloc(Degree(params.n() as u32));
        sk.fill_ternary_prob(0.5, &mut Source::new([0u8; 32]));
        let mut source_xe = Source::new([0u8; 32]);
        let mut mask_key = |gamma: usize, idx: usize, sc: &mut ScratchOwned<BE>| {
            generate_recursion_key(
                module,
                &key_infos,
                &sk,
                params.n(),
                gamma,
                server_seed.recursion_key(idx),
                &mut source_xe,
                &mut sc.borrow(),
            )
        };
        let key0_mask = mask_key(gamma0, 0, &mut scratch);
        let key1_mask = mask_key(gamma1, 1, &mut scratch);
        let _key2_mask = mask_key(gamma2, 2, &mut scratch);

        let database = layout.instantiate(module, params.base2k(), gamma0);

        Self {
            params,
            layout,
            server_seed,
            database,
            scratch,
            scratch_pool: Vec::new(),
            collapse: ServerCollapse::Recursion(RecursionState {
                src_infos,
                key0_mask,
                key1_mask,
                q0_masks: Vec::new(),
                q1_masks: Vec::new(),
            }),
            precomputation: ServerPrecomputation::Recursion(RecursionPrecomputation::default()),
        }
    }

    /// SETUP: materialize the fixed CRS query masks `A0` and `A1`. These depend
    /// only on public seeds and database shape, so they are reused across offline
    /// runs after database updates.
    pub(crate) fn generate_recursion_query_mask(&mut self) {
        let _ = self.generate_recursion_query_mask_timed();
    }

    fn generate_recursion_query_mask_timed(&mut self) -> (Duration, Duration) {
        let started = Instant::now();
        let q0_masks = self.expand_crs_masks(self.server_seed.recursion_a0(), self.database.cols());
        let expand_a0 = started.elapsed();

        let started = Instant::now();
        let q1_masks = self.expand_crs_masks(self.server_seed.recursion_a1(), self.database.t());
        let expand_a1 = started.elapsed();

        let state = self.recursion_state_mut();
        state.q0_masks = q0_masks;
        state.q1_masks = q1_masks;
        (expand_a0, expand_a1)
    }

    pub(super) fn ensure_recursion_query_mask(&mut self, timings: &mut OfflineTimings) {
        let missing = {
            let state = self.recursion_state();
            state.q0_masks.is_empty() || state.q1_masks.is_empty()
        };
        if missing {
            let (expand_a0, expand_a1) = self.generate_recursion_query_mask_timed();
            timings.record_phase("recursion.expand_a0", expand_a0);
            timings.record_phase("recursion.expand_a1", expand_a1);
        }
    }

    /// Materializes a CRS query mask `A` (from a fixed public `crs_seed`) in the
    /// shared coarse mask regime.
    fn expand_crs_masks(&self, crs_seed: [u8; 32], rows: usize) -> Vec<QueryMask> {
        let module = self.params.module();
        let n = self.params.n();
        let src_infos = &self.recursion_state().src_infos;
        let dst_infos = mask_regime_infos(&self.params);
        let glwe_mask = self.params.glwe_mask();
        (0..rows.div_ceil(n))
            .map(|block| {
                let mut q = module.lwe_matrix_alloc_from_infos(&dst_infos);
                let mut sc = ScratchOwned::<BE>::alloc(default_query_mask_tmp_bytes(
                    module, &dst_infos, &glwe_mask,
                ));
                fill_default_query_mask(
                    module,
                    &mut q,
                    block_seed(crs_seed, block),
                    src_infos,
                    &glwe_mask,
                    &mut sc.borrow(),
                );
                QueryMask::new(q, self.params.k())
            })
            .collect()
    }
}

pub(super) fn block_seed(seed: [u8; 32], block: usize) -> [u8; 32] {
    assert!(block <= u32::MAX as usize, "query block index exceeds 2^32");
    let mut s = seed;
    s[28..].copy_from_slice(&(block as u32).to_le_bytes());
    s
}
