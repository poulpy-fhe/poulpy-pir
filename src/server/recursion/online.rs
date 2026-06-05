//! The light per-query ONLINE work: level-1 body select `D·b0` packed into
//! `resp0`, decompose into body digits, and the `resp1`/`resp2` responses.

use std::time::{Duration, Instant};

use poulpy_core::layouts::{LWEInfos, LWEMatrix, LWEMatrixToBackendMut};
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxDftAutomorphismPlan},
    layouts::{
        Backend, HostDataMut, HostDataRef, Module, ScratchOwned, VecZnx, VecZnxToBackendMut,
        VecZnxToBackendRef, ZnxView, ZnxZero,
    },
};

use crate::{
    client::{RecursionResponse, Response},
    config::Collapse,
    packing::{Packing, recursion::partial_pack_batch},
    payload::Payload,
    server::{
        OnlineTimings, Server,
        api::RecursionServerModule,
        common::{copy_vec_znx_rows, full_torus_f64_body_product},
    },
};

use super::{CompressedKey, KeyBundle, PackBodyPhaseNames, RecursionQuery, qtilde_bits, tau};

#[allow(private_bounds)]
impl<BE: Backend<OwnedBuf = Vec<u8>>, P: Payload<[u8; 32]>> Server<BE, P>
where
    BE: poulpy_cpu_ref::reference::fft64::reim::ReimArith,
    Module<BE>: RecursionServerModule<BE>,
    <Module<BE> as VecZnxDftAutomorphismPlan<BE>>::Plan: 'static,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    VecZnx<Vec<u8>>: VecZnxToBackendMut<BE> + VecZnxToBackendRef<BE>,
    LWEMatrix<Vec<u8>>: LWEMatrixToBackendMut<BE>,
    for<'b> BE::BufRef<'b>: HostDataRef,
    for<'b> BE::BufMut<'b>: HostDataMut,
{
    /// Answers `query` ONLINE, using the [`offline`](Self::offline_recursion)
    /// preprocessing: level-1 body select `D·b0` packed with the offline mask
    /// precomputes → `resp0` bodies; Decompose → body digits; `resp1` = offline
    /// mask precomputes + online body; `resp2` = the (query-dependent) body digits,
    /// fully online.
    pub(crate) fn respond_recursion(
        &mut self,
        query: &RecursionQuery<BE>,
    ) -> (Response<BE>, OnlineTimings) {
        let (size, tau, t, gamma0, gamma1, gamma2, base2k, qtilde_bits) = {
            let params = &self.params;
            let Collapse::Recursion {
                gamma0,
                gamma1,
                gamma2,
            } = params.collapse()
            else {
                panic!("Recursion respond requires Collapse::Recursion parameters");
            };
            (
                self.recursion_state().src_infos.size(),
                tau(params),
                self.database.t(),
                gamma0,
                gamma1,
                gamma2,
                params.base2k(),
                qtilde_bits(params),
            )
        };
        let mut timings = OnlineTimings::default();
        let started = Instant::now();
        let key0 = self.prepare_query_key(&query.keys.gamma0);
        let key1 = self.prepare_query_key(&query.keys.gamma1);
        let key2 = self.prepare_query_key(&query.keys.gamma2);
        timings.add_key_precompute("recursion.key_precompute", started.elapsed());
        let off = self.recursion_precomputation();
        let module = self.params.module();

        // Level-1 body: D·b0, packed with the offline mask precomputes → resp0.
        let mut sc = ScratchOwned::<BE>::alloc(self.scratch_for_pack());
        let mut bodies: Vec<VecZnx<BE::OwnedBuf>> = Vec::with_capacity(t);
        let rows_per_group = self.database.rows_per_physical_group();
        let mut l1_body_product = Duration::default();
        for row_group in 0..self.database.physical_rows() {
            let started = Instant::now();
            let mut res_body = module.vec_znx_alloc(1, size);
            full_torus_f64_body_product::<BE>(
                &mut res_body,
                base2k,
                &off.db_prep[row_group],
                &query.src0,
                base2k,
                self.params.k(),
            );
            for local in 0..rows_per_group {
                let batch = row_group * rows_per_group + local;
                if batch >= t {
                    break;
                }
                let mut body = module.vec_znx_alloc(1, size);
                body.zero();
                copy_vec_znx_rows(&mut body, 0, &res_body, local * gamma0, gamma0);
                bodies.push(body);
            }
            l1_body_product += started.elapsed();
        }
        timings.add_body_product("recursion.l1.body_product", l1_body_product);
        let started = Instant::now();
        let inputs: Vec<_> = off.l1_precompute.iter().zip(bodies.iter()).collect();
        let resp0 = partial_pack_batch(
            module,
            &self.recursion_state().src_infos,
            qtilde_bits,
            &inputs,
            &key0.precomp,
            &mut sc.borrow(),
        );
        timings.add_pack("recursion.l1.pack", started.elapsed());

        // Decompose resp0 → body digit DB (the mask digits came from offline).
        let started = Instant::now();
        let mut body_data: Vec<Vec<i16>> = vec![vec![0i16; t]; gamma0 * tau];
        for (k, glwe) in resp0.iter().enumerate() {
            let data = glwe.data();
            for c in 0..gamma0 {
                for l in 0..tau {
                    body_data[c * tau + l][k] = data.at(0, l)[c] as i16;
                }
            }
        }
        timings.add_decompose("recursion.resp0.decompose_body", started.elapsed());

        // resp1: online body for the offline mask-digit precomputes.
        let resp1 = self.pack_bodies_timed(
            &off.resp1_prep,
            &off.resp1_precompute,
            &query.src1,
            gamma1,
            &key1.precomp,
            &mut timings,
            PackBodyPhaseNames {
                body_product: "recursion.resp1.body_product",
                pack: "recursion.resp1.pack",
            },
        );
        // resp2: the digit DB is query-dependent, but the second-level query
        // mask A1 is fixed by the CRS and was materialized in generate_query_mask().
        let q1_masks = &self.recursion_state().q1_masks;
        let resp2 = self.pack_digits_timed(
            &body_data,
            q1_masks,
            &query.src1,
            gamma2,
            &key2,
            &mut timings,
        );
        (
            Response::Recursion(RecursionResponse::new(resp1, resp2)),
            timings,
        )
    }

    fn prepare_query_key<'q>(&mut self, ck: &'q CompressedKey<BE>) -> KeyBundle<'q, BE> {
        let precomp = self.params.module().pack_partial_keys_precompute(
            &ck.key,
            ck.stride,
            self.params.baby_size(),
            &mut self.scratch.borrow(),
        );
        KeyBundle {
            key: &ck.key,
            precomp,
            stride: ck.stride,
        }
    }
}
