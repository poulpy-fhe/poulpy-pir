use crate::circuit::AggregateLWE;
use poulpy_core::{
    LWEMatrixMul,
    layouts::{
        Base2K, CoeffMatrix, CoeffMatrixLayout, Degree, GLWECompressedSeed, GLWECompressedSeedMut,
        GLWECompressedToBackendRef, GLWEInfos, LWEInfos, LWEMatrix, LWEMatrixInfos,
        LWEMatrixLayout, LWEMatrixToBackendMut, ModuleCoreAlloc, TorusPrecision,
    },
};
use poulpy_hal::{
    api::{
        ScratchArenaTakeBasic, VecZnxCopyBackend, VecZnxCopyRangeBackend,
        VecZnxFillUniformSourceBackend, VecZnxRotateBackend, VecZnxZeroBackend,
    },
    layouts::{
        Backend, HostDataMut, Module, ScratchArena, VecZnx, VecZnxToBackendMut, VecZnxToBackendRef,
        ZnxViewMut,
    },
    source::Source,
};

pub struct Database<BE: Backend> {
    entries: usize,
    db: Vec<CoeffMatrix<BE::OwnedBuf, i16>>,
    precomp: Vec<LWEMatrix<BE::OwnedBuf>>,
    precomp_seed: Option<[u8; 32]>,
}

impl<BE: Backend> Database<BE> {
    pub fn new(module: Module<BE>, db_entries: usize, base2k: usize) -> Self {
        assert!(base2k >= 16);

        let n = module.n();
        let nb_matrices = db_entries.div_ceil(n * n);

        let u_infos = CoeffMatrixLayout {
            n: n.into(),
            rows_out: n.into(),
            base2k: Base2K(base2k as u32),
            k: TorusPrecision(base2k as u32),
        };
        let res_infos = LWEMatrixLayout {
            rows: n,
            n: Degree(n as u32),
            base2k: Base2K(base2k as u32),
            k: TorusPrecision(base2k as u32),
        };

        let mut db = Vec::with_capacity(nb_matrices);
        let mut precomp = Vec::with_capacity(nb_matrices);

        for _ in 0..nb_matrices {
            db.push(module.coeff_matrix_alloc_from_infos(&u_infos));
            precomp.push(module.lwe_matrix_alloc_from_infos(&res_infos));
        }

        Self {
            entries: db_entries,
            db,
            precomp,
            precomp_seed: None,
        }
    }

    pub fn encode_shard(&mut self, module: Module<BE>, start_idx: usize, shard: &[i16])
    where
        BE::OwnedBuf: HostDataMut,
    {
        let n = module.n();
        let shard_end = start_idx
            .checked_add(shard.len())
            .expect("database shard index overflow");

        assert!(
            shard_end <= self.entries,
            "database shard writes past the configured number of entries"
        );

        let n_squared = n * n;

        for (offset, &value) in shard.iter().enumerate() {
            let idx = start_idx + offset;
            let mat_idx = idx / n_squared;
            let local_idx = idx % n_squared;
            let row_out = local_idx / n;
            let row_in = local_idx % n;

            self.db[mat_idx].data_mut().at_mut(row_out, 0)[row_in] = value as i64;
        }

        self.precomp_seed = None;
    }

    pub fn preprocess_query_mask_tmp_bytes<Q>(&self, module: &Module<BE>, query: &Q) -> usize
    where
        Module<BE>: AggregateLWE<BE> + LWEMatrixMul<BE>,
        Q: GLWEInfos,
    {
        let mask_infos = LWEMatrixLayout {
            rows: module.n(),
            n: Degree((module.n() * query.rank().as_usize()) as u32),
            base2k: query.base2k(),
            k: query.max_k(),
        };
        let default_mask_bytes = default_query_mask_tmp_bytes(module, &mask_infos, query);

        self.db
            .iter()
            .zip(self.precomp.iter())
            .map(|(u, res)| {
                let mask_mul_bytes = module.lwe_matrix_mul_tmp_bytes(res, u, &mask_infos);
                let aggregate_bytes =
                    VecZnx::<Vec<u8>>::bytes_of(module.n(), res.mask().cols(), res.mask().size())
                        + module.aggregate_lwe_tmp_bytes(res.mask().size());

                default_mask_bytes.max(mask_mul_bytes).max(aggregate_bytes)
            })
            .max()
            .unwrap_or(0)
    }

    pub fn preprocess_query_mask<Q>(
        &mut self,
        module: &Module<BE>,
        query: &Q,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        Module<BE>: AggregateLWE<BE>
            + LWEMatrixMul<BE>
            + VecZnxCopyBackend<BE>
            + VecZnxCopyRangeBackend<BE>
            + VecZnxFillUniformSourceBackend<BE>
            + VecZnxRotateBackend<BE>
            + VecZnxZeroBackend<BE>,
        Q: GLWECompressedToBackendRef<BE> + GLWECompressedSeed + GLWEInfos,
    {
        self.preprocess_query_mask_seed(module, *query.seed(), query, scratch);
    }

    pub fn preprocess_query_mask_seed<Q>(
        &mut self,
        module: &Module<BE>,
        seed: [u8; 32],
        query_infos: &Q,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        Module<BE>: AggregateLWE<BE>
            + LWEMatrixMul<BE>
            + VecZnxCopyBackend<BE>
            + VecZnxCopyRangeBackend<BE>
            + VecZnxFillUniformSourceBackend<BE>
            + VecZnxRotateBackend<BE>
            + VecZnxZeroBackend<BE>,
        Q: GLWEInfos,
    {
        let mask_infos = LWEMatrixLayout {
            rows: module.n(),
            n: Degree((module.n() * query_infos.rank().as_usize()) as u32),
            base2k: query_infos.base2k(),
            k: query_infos.max_k(),
        };
        let mut default_mask = module.lwe_matrix_alloc_from_infos(&mask_infos);
        fill_default_query_mask(module, &mut default_mask, seed, query_infos, scratch);

        let mut mask_product = module.lwe_matrix_alloc_from_infos(&mask_infos);
        for (u, res) in self.db.iter().zip(self.precomp.iter_mut()) {
            module.lwe_matrix_mul(&mut mask_product, u, &default_mask, scratch);
            aggregate_lwe_mask(module, res, &mask_product, scratch);
        }

        self.precomp_seed = Some(seed);
    }

    pub fn query_tmp_bytes<Q>(&self, module: &Module<BE>, query: &Q) -> usize
    where
        Module<BE>: AggregateLWE<BE> + LWEMatrixMul<BE>,
        Q: GLWEInfos,
    {
        let mask_bytes = self.preprocess_query_mask_tmp_bytes(module, query);
        let body_bytes = self
            .db
            .iter()
            .zip(self.precomp.iter())
            .map(|(u, res)| module.lwe_matrix_mul_body_tmp_bytes(res, u, query))
            .max()
            .unwrap_or(0);

        mask_bytes.max(body_bytes)
    }

    pub fn query<Q>(
        &mut self,
        module: &Module<BE>,
        query: &Q,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> &[LWEMatrix<BE::OwnedBuf>]
    where
        Module<BE>: AggregateLWE<BE>
            + LWEMatrixMul<BE>
            + VecZnxCopyBackend<BE>
            + VecZnxCopyRangeBackend<BE>
            + VecZnxFillUniformSourceBackend<BE>
            + VecZnxRotateBackend<BE>
            + VecZnxZeroBackend<BE>,
        Q: GLWECompressedToBackendRef<BE> + GLWECompressedSeed + GLWEInfos,
    {
        if self.precomp_seed != Some(*query.seed()) {
            self.preprocess_query_mask(module, query, scratch);
        }

        for (u, res) in self.db.iter().zip(self.precomp.iter_mut()) {
            module.lwe_matrix_mul_body(res, u, query, scratch);
        }

        &self.precomp
    }

    pub fn matrices(&self) -> &[CoeffMatrix<BE::OwnedBuf, i16>] {
        &self.db
    }

    pub fn precomputed(&self) -> &[LWEMatrix<BE::OwnedBuf>] {
        &self.precomp
    }
}

pub fn set_default_query_mask_seed<Q>(query: &mut Q, seed: [u8; 32])
where
    Q: GLWECompressedSeedMut,
{
    *query.seed_mut() = seed;
}

pub fn default_query_mask_tmp_bytes<BE, R, G>(
    module: &Module<BE>,
    _dst_infos: &R,
    glwe_infos: &G,
) -> usize
where
    BE: Backend,
    R: LWEMatrixInfos,
    G: GLWEInfos,
{
    VecZnx::<Vec<u8>>::bytes_of(module.n(), glwe_infos.rank().as_usize(), glwe_infos.size())
        + VecZnx::<Vec<u8>>::bytes_of(module.n(), 1, glwe_infos.size())
}

pub fn fill_default_query_mask<BE, R, G>(
    module: &Module<BE>,
    dst: &mut R,
    seed: [u8; 32],
    glwe_infos: &G,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    Module<BE>: VecZnxCopyRangeBackend<BE>
        + VecZnxFillUniformSourceBackend<BE>
        + VecZnxRotateBackend<BE>
        + VecZnxZeroBackend<BE>,
    R: LWEMatrixToBackendMut<BE> + LWEMatrixInfos,
    G: GLWEInfos,
{
    let mut dst = dst.to_backend_mut();
    let n = module.n();
    let rank = glwe_infos.rank().as_usize();
    let lwe_n = rank * n;
    let rows = dst.rows();
    let size = glwe_infos.size();

    assert_eq!(
        glwe_infos.n().as_usize(),
        n,
        "fill_default_query_mask: GLWE.n() != module.n()"
    );
    assert_eq!(
        dst.n().as_usize(),
        lwe_n,
        "fill_default_query_mask: destination LWE dimension != rank * module.n()"
    );
    assert!(rows <= n, "fill_default_query_mask: rows > module.n()");
    assert_eq!(
        dst.base2k(),
        glwe_infos.base2k(),
        "fill_default_query_mask: base2k mismatch"
    );

    module.vec_znx_zero_backend(dst.body_mut(), 0);
    for col in 0..lwe_n {
        module.vec_znx_zero_backend(dst.mask_mut(), col);
    }

    let arena = scratch.borrow();
    let (mut mask_polys, arena) = arena.take_vec_znx_scratch(n, rank, size);
    let (mut rotate_tmp, _) = arena.take_vec_znx_scratch(n, 1, size);

    {
        let mut source = Source::new(seed);
        for col in 0..rank {
            module.vec_znx_fill_uniform_source_backend(
                glwe_infos.base2k().as_usize(),
                &mut mask_polys,
                col,
                &mut source,
            );
        }
    }

    let mask_polys_ref = mask_polys.to_backend_ref();
    for row in 0..rows {
        for glwe_col in 0..rank {
            {
                let mut rotate_mut = rotate_tmp.to_backend_mut();
                module.vec_znx_rotate_backend(
                    -(row as i64),
                    &mut rotate_mut,
                    0,
                    &mask_polys_ref,
                    glwe_col,
                );
            }
            let rotate_ref = rotate_tmp.to_backend_ref();
            for limb in 0..size {
                for coeff in 0..n {
                    module.vec_znx_copy_range_backend(
                        dst.mask_mut(),
                        glwe_col * n + coeff,
                        limb,
                        row,
                        &rotate_ref,
                        0,
                        limb,
                        coeff,
                        1,
                    );
                }
            }
        }
    }
}

fn aggregate_lwe_mask<BE>(
    module: &Module<BE>,
    dst: &mut LWEMatrix<BE::OwnedBuf>,
    src: &LWEMatrix<BE::OwnedBuf>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    Module<BE>: AggregateLWE<BE> + VecZnxCopyBackend<BE>,
{
    let n = module.n();
    let size = src.mask().size();

    let (mut aggregate, mut arena_1) = scratch.borrow().take_vec_znx_scratch(n, n, size);

    module.aggregate_lwe(
        &mut aggregate,
        src.base2k().as_usize(),
        src.mask(),
        &mut arena_1,
    );

    let mut mask_mut =
        <VecZnx<BE::OwnedBuf> as VecZnxToBackendMut<BE>>::to_backend_mut(dst.mask_mut());
    let aggregate_ref = aggregate.to_backend_ref();
    for col in 0..n {
        module.vec_znx_copy_backend(&mut mask_mut, col, &aggregate_ref, col);
    }
}
