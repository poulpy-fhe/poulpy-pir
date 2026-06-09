//! Per-query ONLINE work: the panel body product `U·b`, pack, and the Horner
//! reduction at the query's GGSW root.

use std::time::{Duration, Instant};

use poulpy_core::layouts::{
    GLWE, GLWECompressed, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, ModuleCoreAlloc,
};
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{
        Backend, HostDataMut, HostDataRef, Module, ScratchOwned, VecZnx, VecZnxToBackendMut,
        VecZnxToBackendRef,
    },
};

use crate::{
    client::Response,
    interpolation::{InterpolationQuery, InterpolationResponse},
    packing::Packing,
    parameters::Parameters,
    payload::Payload,
    server::{
        OnlineTimings, Server, ServerCollapse, ServerPrecomputation,
        api::InterpolationServerModule,
        common::{PreparedF64, full_torus_f64_body_product},
    },
};

#[allow(private_bounds)]
impl<BE: Backend<OwnedBuf = Vec<u8>>, P: Payload<[u8; 32]>> Server<BE, P>
where
    BE: poulpy_cpu_ref::reference::fft64::reim::ReimArith,
    Module<BE>: InterpolationServerModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    VecZnx<Vec<u8>>:
        VecZnxToBackendMut<BE> + VecZnxToBackendRef<BE> + poulpy_hal::layouts::ZnxInfos,
    GLWE<Vec<u8>>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos,
    for<'b> BE::BufRef<'b>: HostDataRef,
    for<'b> BE::BufMut<'b>: HostDataMut,
{
    /// ONLINE: answer `query`. Per panel computes `U·b`, packs it, then evaluates
    /// the interpolated polynomial at the query's GGSW root (Horner). Returns the
    /// [`Response`] and a per-step timing breakdown.
    pub(crate) fn respond_interpolation(
        &mut self,
        query: &InterpolationQuery<BE>,
    ) -> (Response<BE>, OnlineTimings) {
        let ServerPrecomputation::Interpolation(precomputation) = &self.precomputation else {
            panic!("interpolation respond requested for non-interpolation server");
        };
        assert!(
            !precomputation.precomputations.is_empty(),
            "call offline() before respond()"
        );
        let mut timings = OnlineTimings::default();
        let module = self.params.module();
        let glwe_pack = self.params.glwe_pack();
        let ServerCollapse::Interpolation(state) = &self.collapse else {
            panic!("interpolation respond requested for non-interpolation server");
        };
        let panels = state.interpolation.num_panels();

        let t = Instant::now();
        let key_precomputations = module.pack_keys_precompute(
            query.keys.key_g(),
            query.keys.key_h(),
            self.params.baby_size(),
            &mut self.scratch.borrow(),
        );
        timings.add_key_precompute("interpolation.key_precompute", t.elapsed());

        let body_size = self.params.size_at(self.params.base2k());
        let mut body_product = Duration::default();
        let mut pack = Duration::default();
        let mut packed_coeffs = Vec::with_capacity(panels);
        for panel in 0..panels {
            let mut body = module.vec_znx_alloc(1, body_size);
            let t = Instant::now();
            accumulate_body_product(
                &self.params,
                &mut body,
                &precomputation.prepared_u[panel],
                &query.common.blocks,
            );
            body_product += t.elapsed();
            let mut packed = module.glwe_alloc_from_infos(&glwe_pack);
            let t = Instant::now();
            module.pack(
                &mut packed,
                &body,
                &precomputation.precomputations[panel],
                &key_precomputations,
                1,
                &mut self.scratch.borrow(),
            );
            pack += t.elapsed();
            packed_coeffs.push(packed);
        }
        timings.add_body_product("interpolation.body_product", body_product);
        timings.add_pack("interpolation.pack", pack);

        let t = Instant::now();
        let root_prepared =
            state
                .interpolation
                .prepare_root(module, &query.root, &mut self.scratch);
        timings.add_reduce_precompute("interpolation.reduce_precompute", t.elapsed());
        let mut selected = module.glwe_alloc_from_infos(&glwe_pack);
        let t = Instant::now();
        state.interpolation.reduce(
            module,
            &packed_coeffs,
            &root_prepared,
            &mut selected,
            &mut self.scratch,
        );
        timings.add_reduce("interpolation.reduce", t.elapsed());
        (
            Response::Interpolation(InterpolationResponse::new(selected)),
            timings,
        )
    }
}

/// `out_body = sum_bc U[bc] · b[bc]` for one interpolation panel, via the dense
/// `f64` GEMV. Writes a small body `VecZnx` (online only needs the body, not the
/// mask) directly in the pack regime.
fn accumulate_body_product<BE, P: Payload<[u8; 32]>>(
    params: &Parameters<BE, [u8; 32], P>,
    out_body: &mut VecZnx<BE::OwnedBuf>,
    u_panel: &[PreparedF64],
    bodies: &[GLWECompressed<BE::OwnedBuf>],
) where
    BE: Backend<OwnedBuf = Vec<u8>> + poulpy_cpu_ref::reference::fft64::reim::ReimArith,
{
    full_torus_f64_body_product::<BE>(
        out_body,
        params.base2k(),
        u_panel,
        bodies,
        params.matmul_base2k(),
        params.k(),
    );
}
