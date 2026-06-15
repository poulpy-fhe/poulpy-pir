//! Server construction and per-block-column query-mask generation (SETUP).

use poulpy_core::{
    GLWEExpandLWEMatrix,
    layouts::{
        GGSWPreparedFactory, GLWE, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos,
        LWEMatrix, LWEMatrixToBackendMut, ModuleCoreAlloc,
    },
};
use poulpy_hal::{
    api::{
        ModuleN, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxNormalizeTmpBytes, VmpPrepare,
        VmpPrepareTmpBytes, VmpZero,
    },
    layouts::{
        Backend, HostDataMut, HostDataRef, Module, ScratchOwned, VecZnx, VecZnxToBackendMut,
        VecZnxToBackendRef,
    },
};

use crate::{
    client::{MaskSeeds, ServerSeed},
    config::Collapse,
    database::DatabaseLayout,
    interpolation::{HornerEvaluation, Interpolation, MonomialInterpolation},
    packing::{Packing, PackingKeysGenerate, PackingMaskAggregation},
    parameters::Parameters,
    payload::Payload,
    server::{
        Server, ServerCollapse, ServerPrecomputation,
        api::InterpolationServerModule,
        common::{
            QueryMask, default_query_mask_tmp_bytes, fill_default_query_mask, mask_regime_infos,
        },
    },
};

use super::{InterpolationPrecomputation, InterpolationState};

#[allow(private_bounds)]
impl<BE: Backend<OwnedBuf = Vec<u8>>, P: Payload<[u8; 32]>> Server<BE, P>
where
    BE: poulpy_cpu_ref::reference::fft64::reim::ReimArith,
    Module<BE>: InterpolationServerModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    VecZnx<Vec<u8>>:
        VecZnxToBackendMut<BE> + VecZnxToBackendRef<BE> + poulpy_hal::layouts::ZnxInfos,
    LWEMatrix<Vec<u8>>: LWEMatrixToBackendMut<BE>,
    GLWE<Vec<u8>>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos,
    for<'b> BE::BufRef<'b>: HostDataRef,
    for<'b> BE::BufMut<'b>: HostDataMut,
{
    /// Build the shared server with interpolation-specific state.
    pub(crate) fn new_interpolation(
        params: Parameters<BE, [u8; 32], P>,
        layout: DatabaseLayout<P>,
    ) -> Self {
        assert!(
            matches!(params.collapse(), Collapse::Interpolation),
            "params must use the interpolation collapse"
        );
        let module = params.module();
        let base2k = params.base2k();
        let n = params.n();
        assert_eq!(
            layout.rows() % n,
            0,
            "interpolation database rows must be a multiple of n"
        );
        assert_eq!(
            layout.cols() % n,
            0,
            "interpolation database cols must be a multiple of n"
        );

        let server_seed = ServerSeed::default();

        let database = layout.instantiate(module, base2k, n);
        let matrix_layout = DatabaseLayout::<P>::new(layout.interpolation_t(n) * n, layout.cols());
        let matrix = matrix_layout.instantiate(module, base2k, n);

        let interpolation = Interpolation::new(&layout, &params);
        let scratch = ScratchOwned::<BE>::alloc(server_scratch_bytes(&params));

        Self {
            params,
            layout,
            server_seed,
            database,
            scratch,
            scratch_pool: Vec::new(),
            collapse: ServerCollapse::Interpolation(InterpolationState {
                interpolation,
                matrix,
            }),
            precomputation: ServerPrecomputation::Interpolation(
                InterpolationPrecomputation::default(),
            ),
        }
    }

    /// SETUP: materialize the per-block-column query masks `A` from the public
    /// `server_seed.mask()`. Reused across DB updates and queries.
    pub(crate) fn generate_interpolation_query_mask(&mut self) {
        let glwe_query = self.params.glwe_query();
        let mask_seeds = MaskSeeds::new(self.server_seed.mask());
        let block_cols = self.layout.block_cols(self.params.n());
        let mut masks = Vec::with_capacity(block_cols);
        for bc in 0..block_cols {
            let mut mask = self
                .params
                .module()
                .lwe_matrix_alloc_from_infos(&mask_regime_infos(&self.params));
            fill_default_query_mask(
                self.params.module(),
                &mut mask,
                mask_seeds.seed(bc),
                &glwe_query,
                &self.params.glwe_mask(),
                &mut self.scratch.borrow(),
            );
            masks.push(QueryMask::new(mask, self.params.k()));
        }
        let ServerPrecomputation::Interpolation(precomputation) = &mut self.precomputation else {
            panic!("interpolation precomputation requested for non-interpolation server");
        };
        precomputation.masks = masks;
    }
}

/// Scratch large enough for every server operation.
pub(super) fn server_scratch_bytes<BE: Backend, P: Payload<[u8; 32]>>(
    params: &Parameters<BE, [u8; 32], P>,
) -> usize
where
    Module<BE>: ModuleN
        + ModuleCoreAlloc<OwnedBuf = Vec<u8>>
        + GLWEExpandLWEMatrix<BE>
        + VecZnxNormalizeTmpBytes
        + MonomialInterpolation<BE>
        + PackingMaskAggregation<BE>
        + Packing<BE>
        + PackingKeysGenerate<BE>
        + GGSWPreparedFactory<BE>
        + HornerEvaluation<BE>
        + VmpPrepare<BE>
        + VmpPrepareTmpBytes
        + VmpZero<BE>,
{
    let module = params.module();
    let glwe_pack = params.glwe_pack();
    let glwe_mask = params.glwe_mask();
    let lwe_infos = params.lwe_matrix_infos();
    let key_infos = params.key_layout();
    let ggsw_infos = params.ggsw_layout();
    let precompute_metadata = params.packing_precompute_infos();
    let aggregate = module.vec_znx_alloc(params.n(), lwe_infos.size());
    module
        .monomial_interpolate_tmp_bytes(1)
        .max(module.pack_keys_generate_tmp_bytes(&key_infos))
        .max(module.pack_keys_precompute_tmp_bytes(&key_infos, &key_infos, params.baby_size()))
        .max(module.ggsw_prepare_tmp_bytes(&ggsw_infos))
        .max(module.horner_evaluate_tmp_bytes(&glwe_pack, &ggsw_infos))
        .max(default_query_mask_tmp_bytes(
            module,
            &mask_regime_infos(params),
            &glwe_mask,
        ))
        .max(module.vec_znx_normalize_tmp_bytes())
        .max(module.packing_mask_preprocessing_tmp_bytes(lwe_infos.size()))
        .max(module.pack_precompute_tmp_bytes(precompute_metadata, &aggregate, &key_infos))
}
