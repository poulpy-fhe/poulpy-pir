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
        Gemm, OnlineTimings, Server, ServerCollapse, ServerPrecomputation,
        api::RecursionServerModule,
        common::{PreparedF64, copy_vec_znx_rows, full_torus_f64_body_product_batch},
    },
};

use super::{
    CompressedKey, KeyBundle, RecursionQuery, packing::pack_bodies_pooled, qtilde_bits, tau,
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
    /// Answers `query` ONLINE (a batch of one — see
    /// [`respond_recursion_batch`](Self::respond_recursion_batch)).
    pub(crate) fn respond_recursion(
        &mut self,
        query: &RecursionQuery<BE>,
    ) -> (Response<BE>, OnlineTimings) {
        let (responses, timings) = self.respond_recursion_batch(std::slice::from_ref(&query));
        let response = responses
            .into_iter()
            .next()
            .expect("single-query recursion batch returned no response");
        (response, timings)
    }

    /// Answers a batch of recursion queries ONLINE. The level-1 body select `D·b0`
    /// is the dominant, memory-bound step: it is computed as a **single i16×f64
    /// GEMM over the whole batch** ([`recursion_l1_bodies`](Self::recursion_l1_bodies)),
    /// so the plaintext DB is streamed once for all `nq` queries instead of once
    /// per query. The remaining (FHE) packing pipeline — `resp0` pack, decompose,
    /// `resp1`, `resp2` — stays per query. Timings are summed over the batch.
    pub(crate) fn respond_recursion_batch(
        &mut self,
        queries: &[&RecursionQuery<BE>],
    ) -> (Vec<Response<BE>>, OnlineTimings) {
        assert!(!queries.is_empty(), "empty recursion batch");
        let (size, base2k, torus_bits, t, gamma0, n) = {
            let params = &self.params;
            let Collapse::Recursion { gamma0, .. } = params.collapse() else {
                panic!("Recursion respond requires Collapse::Recursion parameters");
            };
            (
                self.recursion_state().src_infos.size(),
                params.base2k(),
                params.k(),
                self.database.t(),
                gamma0,
                params.n(),
            )
        };

        // The batched level-1 body select holds `chunk × t` bodies
        // (`VecZnx(1, size)`) at once, so cap `chunk` to keep that working set
        // within a memory budget: a bigger DB (larger `t`) simply uses a smaller
        // chunk. Each chunk still streams the DB once for its GEMM — the batch win
        // — so `nq/chunk` DB passes instead of `nq`. Materializing every query's
        // bodies at once (`nq × t`) is what OOMs on large batches.
        let bytes_per_body = n.saturating_mul(size).saturating_mul(8);
        let per_query_bytes = t.saturating_mul(bytes_per_body).max(1);
        const BODY_BUDGET: usize = 2 << 30; // ~2 GiB working set for level-1 bodies
        let chunk = (BODY_BUDGET / per_query_bytes).clamp(1, queries.len());

        let mut timings = OnlineTimings::default();
        let mut responses = Vec::with_capacity(queries.len());
        for chunk_queries in queries.chunks(chunk) {
            let mut chunk_timings = OnlineTimings::default();

            // Level-1 body `D·b0` for this chunk, as one GEMM per DB panel.
            let started = Instant::now();
            let all_bodies =
                self.recursion_l1_bodies(chunk_queries, size, base2k, torus_bits, t, gamma0);
            chunk_timings.add_body_product("recursion.l1.body_product", started.elapsed());

            // Per-query FHE finish (pack / decompose / resp1 / resp2).
            for (query, bodies) in chunk_queries.iter().zip(all_bodies) {
                let mut per_query = OnlineTimings::default();
                let response = self.recursion_finish_from_bodies(query, bodies, &mut per_query);
                chunk_timings.accumulate(&per_query);
                responses.push(response);
            }
            timings.accumulate(&chunk_timings);
        }
        (responses, timings)
    }

    /// Batched level-1 body select: `bodies[q] = split(D · b0^q)` for every query
    /// `q`. Each DB row-group runs one i16×f64 GEMM whose RHS stacks the `nq`
    /// queries' `src0` bodies as columns, so each `U` panel is read once and
    /// amortized over the batch. Scratch-free, so it parallelizes across row groups
    /// (each writes a disjoint output slab). Returns per-query `Vec`s of length `t`.
    fn recursion_l1_bodies(
        &self,
        queries: &[&RecursionQuery<BE>],
        size: usize,
        base2k: usize,
        torus_bits: usize,
        t: usize,
        gamma0: usize,
    ) -> Vec<Vec<VecZnx<BE::OwnedBuf>>> {
        let module = self.params.module();
        let gemm: &dyn Gemm = &*self.gemm;
        let nq = queries.len();
        let rows_per_group = self.database.rows_per_physical_group();
        let physical_rows = self.database.physical_rows();
        let column_blocks = self.database.column_blocks();

        // Zero-copy `PreparedF64` views over the contiguous plaintext DB.
        let db_views: Vec<Vec<PreparedF64<'_>>> = (0..physical_rows)
            .map(|rg| {
                (0..column_blocks)
                    .map(|block| PreparedF64::from_matrix(self.database.physical_block(rg, block)))
                    .collect()
            })
            .collect();

        // Each query's `src0` blocks (shared across row groups) as GEMM columns.
        let src0s: Vec<&[_]> = queries.iter().map(|q| q.src0.as_slice()).collect();

        let nthreads = num_threads(physical_rows);
        let work = assign_panels(physical_rows, 1, nthreads);
        // Per row group: `nq` queries × up-to-`rows_per_group` split bodies.
        type GroupOut<BE> = Option<Vec<Vec<VecZnx<<BE as Backend>::OwnedBuf>>>>;
        let mut group_out: Vec<GroupOut<BE>> = (0..physical_rows).map(|_| None).collect();

        {
            let db_views = &db_views;
            let src0s = &src0s;
            let mut slabs: Vec<&mut [GroupOut<BE>]> = Vec::with_capacity(work.len());
            let mut rest = group_out.as_mut_slice();
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
                            // One GEMM for the whole batch against this panel.
                            let mut res_bodies: Vec<VecZnx<BE::OwnedBuf>> =
                                (0..nq).map(|_| module.vec_znx_alloc(1, size)).collect();
                            full_torus_f64_body_product_batch::<BE>(
                                &mut res_bodies,
                                base2k,
                                &db_views[row_group],
                                src0s,
                                base2k,
                                torus_bits,
                                gemm,
                            );
                            // Per-query row split into γ0-tall bodies.
                            let mut per_query: Vec<Vec<VecZnx<BE::OwnedBuf>>> =
                                (0..nq).map(|_| Vec::with_capacity(rows_per_group)).collect();
                            for local in 0..rows_per_group {
                                let batch_idx = row_group * rows_per_group + local;
                                if batch_idx >= t {
                                    break;
                                }
                                for (qi, res_body) in res_bodies.iter().enumerate() {
                                    let mut body = module.vec_znx_alloc(1, size);
                                    body.zero();
                                    copy_vec_znx_rows(
                                        &mut body,
                                        0,
                                        res_body,
                                        local * gamma0,
                                        gamma0,
                                    );
                                    per_query[qi].push(body);
                                }
                            }
                            *slot = Some(per_query);
                        }
                    });
                }
            });
        }

        // Assemble per-query bodies in row-group order.
        let mut all_bodies: Vec<Vec<VecZnx<BE::OwnedBuf>>> =
            (0..nq).map(|_| Vec::with_capacity(t)).collect();
        for g in group_out {
            let per_query = g.expect("l1 body worker did not fill its slot");
            for (qi, bodies) in per_query.into_iter().enumerate() {
                all_bodies[qi].extend(bodies);
            }
        }
        all_bodies
    }

    /// The per-query FHE finish for one recursion query, given its level-1 `bodies`
    /// (`D·b0` split): `resp0` pack → decompose → `resp1` → `resp2`. Records the
    /// key-precompute, pack, decompose, and (small) `resp1`/`resp2` body-product
    /// phases into `timings`.
    fn recursion_finish_from_bodies(
        &mut self,
        query: &RecursionQuery<BE>,
        bodies: Vec<VecZnx<BE::OwnedBuf>>,
        timings: &mut OnlineTimings,
    ) -> Response<BE> {
        let t = self.database.t();
        let (tau, gamma0, gamma1, gamma2, base2k, qtilde_bits) = {
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
                tau(params),
                gamma0,
                gamma1,
                gamma2,
                params.base2k(),
                qtilde_bits(params),
            )
        };

        let started = Instant::now();
        let key0 = self.prepare_query_key(&query.keys.gamma0);
        let key1 = self.prepare_query_key(&query.keys.gamma1);
        let key2 = self.prepare_query_key(&query.keys.gamma2);
        timings.add_key_precompute("recursion.key_precompute", started.elapsed());

        // Field-level borrows (not the `&self` accessors) so the pooled packs below
        // can take `&mut self.scratch_pool` disjointly.
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
        let gemm: &dyn Gemm = &*self.gemm;
        let torus_bits = self.params.k();

        // resp0: pack the level-1 bodies with the offline mask precomputes.
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

        // resp1: online body for the offline mask-digit precomputes.
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
            gemm,
            &mut self.scratch_pool,
        );
        timings.add_body_product("recursion.resp1.body_product", resp1_body);
        timings.add_pack("recursion.resp1.pack", resp1_pack);

        // resp2: the digit DB is query-dependent (mask precompute runs online).
        let q1_masks = &state.q1_masks;
        let (resp2_prepared, resp2_precomputes) =
            self.precompute_pack_mask_online(&body_data, q1_masks, gamma2, &key2, timings);
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
            gemm,
            &mut self.scratch_pool,
        );
        timings.add_body_product("recursion.resp2.body_product", resp2_body);
        timings.add_pack("recursion.resp2.pack", resp2_pack);

        Response::Recursion(RecursionResponse::new(resp1, resp2))
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
