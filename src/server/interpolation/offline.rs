//! Interpolate the plaintext DB into the matrix DB, then run each
//! query-independent packing precomputation phase (OFFLINE).

use std::time::{Duration, Instant};

use poulpy_core::layouts::{
    Degree, GLWE, GLWEAutomorphismKeyCompressed, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef,
    LWEInfos, LWEMatrix, LWEMatrixToBackendMut, ModuleCoreAlloc,
};
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{
        Backend, HostDataMut, HostDataRef, Module, ScratchOwned, VecZnx, VecZnxToBackendMut,
        VecZnxToBackendRef,
    },
    source::Source,
};

use crate::{
    packing::{Packing, PackingKeysGenerate, PackingMaskAggregation},
    payload::Payload,
    server::{
        OfflineTimings, Server, ServerCollapse, ServerPrecomputation,
        api::InterpolationServerModule,
        common::{PreparedF64, mask_product_to_pack},
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
    LWEMatrix<Vec<u8>>: LWEMatrixToBackendMut<BE>,
    GLWE<Vec<u8>>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos,
    for<'b> BE::BufRef<'b>: HostDataRef,
    for<'b> BE::BufMut<'b>: HostDataMut,
{
    /// OFFLINE: interpolate the plaintext DB into the matrix DB, then run each
    /// query-independent packing precomputation phase in sequence. Re-run after
    /// DB updates.
    pub(crate) fn offline_interpolation(&mut self) -> OfflineTimings {
        let mut timings = OfflineTimings::default();

        if let Some(duration) = self.offline_interpolation_query_mask() {
            timings.record_phase("interpolation.query_mask", duration);
        }
        timings.add_interpolation(
            "interpolation.interpolate",
            self.offline_interpolation_interpolate(),
        );
        timings.add_prepare_u(
            "interpolation.prepare_u",
            self.offline_interpolation_prepare_u(),
        );

        let t = Instant::now();
        let key_mask_src = self.interpolation_pack_key_mask();
        timings.record_phase("interpolation.pack_key_mask", t.elapsed());

        let (ua_mask, mask_prep, pack_precompute) =
            self.offline_interpolation_pack_precomputations(&key_mask_src);
        timings.add_ua_mask("interpolation.ua_mask", ua_mask);
        timings.add_mask_prep("interpolation.mask_prep", mask_prep);
        timings.add_pack_precompute("interpolation.pack_precompute", pack_precompute);
        timings
    }

    fn offline_interpolation_query_mask(&mut self) -> Option<Duration> {
        let masks_empty = match &self.precomputation {
            ServerPrecomputation::Interpolation(precomputation) => precomputation.masks.is_empty(),
            ServerPrecomputation::Recursion(_) => {
                panic!("interpolation offline requested for non-interpolation server")
            }
        };
        if masks_empty {
            let t = Instant::now();
            self.generate_interpolation_query_mask();
            return Some(t.elapsed());
        }
        None
    }

    fn offline_interpolation_interpolate(&mut self) -> Duration {
        let encoder = self.params.encoder();
        let t = Instant::now();
        let ServerCollapse::Interpolation(state) = &mut self.collapse else {
            panic!("interpolation offline requested for non-interpolation server");
        };
        state.interpolation.interpolate_into(
            self.params.module(),
            &self.database,
            &mut state.matrix,
            &encoder,
            &mut self.scratch,
        );
        t.elapsed()
    }

    fn offline_interpolation_prepare_u(&mut self) -> Duration {
        let _module = self.params.module();
        let block_cols = self.layout.block_cols(self.params.n());
        let t = Instant::now();
        let prepared_u = {
            let ServerCollapse::Interpolation(state) = &self.collapse else {
                panic!("interpolation offline requested for non-interpolation server");
            };
            let panels = state.interpolation.num_panels();
            let mut prepared_u: Vec<Vec<PreparedF64>> = Vec::with_capacity(panels);
            for panel in 0..panels {
                let row: Vec<_> = (0..block_cols)
                    .map(|bc| PreparedF64::new(&state.matrix.matrices()[panel * block_cols + bc]))
                    .collect();
                prepared_u.push(row);
            }
            prepared_u
        };
        let ServerPrecomputation::Interpolation(precomputation) = &mut self.precomputation else {
            panic!("interpolation precomputation requested for non-interpolation server");
        };
        precomputation.prepared_u = prepared_u;
        t.elapsed()
    }

    /// A throwaway key whose mask is seeded by `server_seed.keys()`; its body is
    /// ignored by `pack_precompute`, which consumes only the mask seed.
    fn interpolation_pack_key_mask(&mut self) -> GLWEAutomorphismKeyCompressed<BE::OwnedBuf> {
        let module = self.params.module();
        let mut sk = module.lwe_secret_alloc(Degree(self.params.n() as u32));
        sk.fill_ternary_prob(0.5, &mut Source::new([0u8; 32]));
        let (key_g, _key_h) = module.pack_keys_generate(
            &self.params.key_layout(),
            &sk,
            self.server_seed.keys(),
            &mut Source::new([0u8; 32]),
            &mut self.scratch.borrow(),
        );
        key_g
    }

    fn offline_interpolation_pack_precomputations(
        &mut self,
        key_mask_src: &GLWEAutomorphismKeyCompressed<BE::OwnedBuf>,
    ) -> (Duration, Duration, Duration) {
        let module = self.params.module();
        let lwe_infos = self.params.lwe_matrix_infos();
        let precompute_metadata = self.params.packing_precompute_infos();
        let mut aggregate = module.vec_znx_alloc(self.params.n(), lwe_infos.size());
        let panels = {
            let ServerCollapse::Interpolation(state) = &self.collapse else {
                panic!("interpolation offline requested for non-interpolation server");
            };
            state.interpolation.num_panels()
        };

        let mut ua_mask = Duration::default();
        let mut mask_prep = Duration::default();
        let mut pack_precompute = Duration::default();
        let mut precomputations = Vec::with_capacity(panels);
        {
            let ServerPrecomputation::Interpolation(precomputation) = &self.precomputation else {
                panic!("interpolation precomputation requested for non-interpolation server");
            };
            for panel in 0..panels {
                let t = Instant::now();
                let product = mask_product_to_pack(
                    module,
                    &self.params,
                    &lwe_infos,
                    &precomputation.prepared_u[panel],
                    &precomputation.masks,
                );
                ua_mask += t.elapsed();
                let t = Instant::now();
                module.packing_mask_preprocessing(
                    &mut aggregate,
                    self.params.base2k(),
                    product.mask(),
                    &mut self.scratch.borrow(),
                );
                mask_prep += t.elapsed();
                let mut precompute = module.pack_precompute_alloc(
                    precompute_metadata.steps(),
                    precompute_metadata.size(),
                    precompute_metadata.base2k(),
                    precompute_metadata.baby_size(),
                );
                let t = Instant::now();
                module.pack_precompute(
                    &mut precompute,
                    &aggregate,
                    key_mask_src,
                    &mut self.scratch.borrow(),
                );
                pack_precompute += t.elapsed();
                precomputations.push(precompute);
            }
        }
        let ServerPrecomputation::Interpolation(precomputation) = &mut self.precomputation else {
            panic!("interpolation precomputation requested for non-interpolation server");
        };
        precomputation.precomputations = precomputations;
        (ua_mask, mask_prep, pack_precompute)
    }
}
