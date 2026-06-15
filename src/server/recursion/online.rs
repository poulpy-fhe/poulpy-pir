//! The light per-query ONLINE work: level-1 body select `D·b0` packed into
//! `resp0`, decompose into body digits, and the `resp1`/`resp2` responses.

use std::time::Instant;

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
    packing::{Packing, recursion::partial_pack_batch_pooled},
    parallel::{assign_panels, num_threads},
    payload::Payload,
    server::{
        OnlineTimings, Server, ServerCollapse, ServerPrecomputation,
        api::RecursionServerModule,
        common::{copy_vec_znx_rows, full_torus_f64_body_product},
    },
};

use super::{CompressedKey, KeyBundle, RecursionQuery, packing::pack_bodies_pooled, qtilde_bits, tau};

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
        // Field-level borrows (not the `&self` accessors) so the per-query pack
        // below can take `&mut self.scratch_pool` disjointly.
        let ServerPrecomputation::Recursion(off) = &self.precomputation else {
            panic!("recursion respond requested for non-recursion server");
        };
        assert!(
            !off.l1_precompute.is_empty(),
            "call Server::offline() before respond()"
        );
        let ServerCollapse::Recursion(state) = &self.collapse else {
            panic!("recursion respond requested for non-recursion server");
        };
        let module = self.params.module();

        // Level-1 body: D·b0, packed with the offline mask precomputes → resp0.
        let rows_per_group = self.database.rows_per_physical_group();
        let physical_rows = self.database.physical_rows();
        let torus_bits = self.params.k();

        // Body product `D·b0` + row split is scratch-free, so parallelize across
        // row groups directly (each writes a disjoint output slab; no pool needed).
        let nthreads = num_threads(physical_rows);
        let work = assign_panels(physical_rows, 1, nthreads);
        type GroupBodies<BE> = Option<Vec<VecZnx<<BE as Backend>::OwnedBuf>>>;
        let mut group_bodies: Vec<GroupBodies<BE>> = (0..physical_rows).map(|_| None).collect();

        let started = Instant::now();
        {
            let db_prep = &off.db_prep;
            let src0 = &query.src0;
            let mut slabs: Vec<&mut [GroupBodies<BE>]> = Vec::with_capacity(work.len());
            let mut rest = group_bodies.as_mut_slice();
            for grp in &work {
                let (head, tail) = rest.split_at_mut(grp.len());
                slabs.push(head);
                rest = tail;
            }
            std::thread::scope(|scope| {
                for (slab, grp) in slabs.into_iter().zip(work.iter()) {
                    scope.spawn(move || {
                        for (slot, w) in slab.iter_mut().zip(grp.iter()) {
                            let row_group = w.panel;
                            let mut res_body = module.vec_znx_alloc(1, size);
                            full_torus_f64_body_product::<BE>(
                                &mut res_body,
                                base2k,
                                &db_prep[row_group],
                                src0,
                                base2k,
                                torus_bits,
                            );
                            let mut out = Vec::with_capacity(rows_per_group);
                            for local in 0..rows_per_group {
                                let batch = row_group * rows_per_group + local;
                                if batch >= t {
                                    break;
                                }
                                let mut body = module.vec_znx_alloc(1, size);
                                body.zero();
                                copy_vec_znx_rows(&mut body, 0, &res_body, local * gamma0, gamma0);
                                out.push(body);
                            }
                            *slot = Some(out);
                        }
                    });
                }
            });
        }
        let mut bodies: Vec<VecZnx<BE::OwnedBuf>> = Vec::with_capacity(t);
        for g in group_bodies {
            bodies.extend(g.expect("l1 body worker did not fill its slot"));
        }
        timings.add_body_product("recursion.l1.body_product", started.elapsed());
        let started = Instant::now();
        let inputs: Vec<_> = off.l1_precompute.iter().zip(bodies.iter()).collect();
        let resp0 = partial_pack_batch_pooled(
            module,
            &state.src_infos,
            qtilde_bits,
            &inputs,
            &key0.precomp,
            &mut self.scratch_pool,
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

        // resp1: online body for the offline mask-digit precomputes, packed via
        // the pooled parallel partial pack.
        let (resp1, resp1_body, resp1_pack) = pack_bodies_pooled(
            module,
            &state.src_infos,
            qtilde_bits,
            base2k,
            torus_bits,
            gamma1,
            &off.resp1_prep,
            &off.resp1_precompute,
            &query.src1,
            &key1.precomp,
            &mut self.scratch_pool,
        );
        timings.add_body_product("recursion.resp1.body_product", resp1_body);
        timings.add_pack("recursion.resp1.pack", resp1_pack);

        // resp2: the digit DB is query-dependent (mask precompute runs online,
        // sequentially); the second-level query mask A1 is fixed by the CRS.
        let q1_masks = &state.q1_masks;
        let (resp2_prepared, resp2_precomputes) =
            self.precompute_pack_mask_online(&body_data, q1_masks, gamma2, &key2, &mut timings);
        let (resp2, resp2_body, resp2_pack) = pack_bodies_pooled(
            module,
            &state.src_infos,
            qtilde_bits,
            base2k,
            torus_bits,
            gamma2,
            &resp2_prepared,
            &resp2_precomputes,
            &query.src1,
            &key2.precomp,
            &mut self.scratch_pool,
        );
        timings.add_body_product("recursion.resp2.body_product", resp2_body);
        timings.add_pack("recursion.resp2.pack", resp2_pack);
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
