use poulpy_hal::{
    api::{
        ScratchArenaTakeBasic, VecZnxAddAssignBackend, VecZnxAutomorphismBackend,
        VecZnxCopyBackend, VecZnxRotateAssignBackend, VecZnxRotateAssignTmpBytes,
        VecZnxRshAssignBackend, VecZnxRshTmpBytes, VecZnxTransposeBackend,
    },
    layouts::{
        Backend, GaloisElement, Module, ScratchArena, VecZnx, VecZnxBackendMut,
        VecZnxReborrowBackendRef, VecZnxToBackendMut, VecZnxToBackendRef, ZnxInfos,
    },
};

pub trait AggregateLWE<BE: Backend> {
    fn aggregate_lwe_tmp_bytes(&self, size: usize) -> usize;

    fn aggregate_lwe<R, A>(
        &self,
        dst: &mut R,
        base2k: usize,
        a: &A,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: VecZnxToBackendMut<BE> + ZnxInfos,
        A: VecZnxToBackendRef<BE> + ZnxInfos;
}

impl<BE: Backend> AggregateLWE<BE> for Module<BE>
where
    Self: VecZnxCopyBackend<BE>
        + VecZnxTransposeBackend<BE>
        + VecZnxAutomorphismBackend<BE>
        + VecZnxAddAssignBackend<BE>
        + VecZnxRotateAssignBackend<BE>
        + VecZnxRotateAssignTmpBytes
        + VecZnxRshAssignBackend<BE>
        + VecZnxRshTmpBytes
        + GaloisElement,
{
    fn aggregate_lwe_tmp_bytes(&self, size: usize) -> usize {
        const ALIGN: usize = 64;
        let round = |x: usize| x.next_multiple_of(ALIGN);
        let n = self.n();
        let log_n = n.trailing_zeros() as usize;
        let transposed = round(VecZnx::<Vec<u8>>::bytes_of(n, n, size));
        let one_col = round(VecZnx::<Vec<u8>>::bytes_of(n, 1, size));
        let tree = round(VecZnx::<Vec<u8>>::bytes_of(n, log_n, size));
        let internal = round(
            self.vec_znx_rsh_tmp_bytes()
                .max(self.vec_znx_rotate_assign_tmp_bytes()),
        );
        transposed + 2 * one_col + 2 * tree + internal
    }

    fn aggregate_lwe<R, A>(
        &self,
        dst: &mut R,
        base2k: usize,
        a: &A,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: VecZnxToBackendMut<BE> + ZnxInfos,
        A: VecZnxToBackendRef<BE> + ZnxInfos,
    {
        let n = self.n();
        assert!(
            n.is_power_of_two(),
            "InspiRING requires a power-of-two ring degree"
        );
        assert!(n >= 2, "InspiRING requires ring degree d >= 2");

        let n_half = n >> 1;
        let log_n = n.trailing_zeros() as usize;
        let size = dst.size();

        assert_eq!(
            dst.n(),
            n,
            "destination VecZnx degree must match module degree"
        );
        assert_eq!(dst.cols(), n, "destination VecZnx must have d columns");
        assert_eq!(
            dst.size(),
            a.size(),
            "destination size must match input A size"
        );
        assert_eq!(a.n(), n, "input A must have d rows");
        assert_eq!(a.cols(), n, "input A must have d columns");

        let h_list: Vec<i64> = (0..n_half).map(|i| self.galois_element(i as i64)).collect();

        let arena = scratch.borrow();
        let (mut transposed, arena) = arena.take_vec_znx_scratch(n, n, size);
        let (mut shared, arena) = arena.take_vec_znx_scratch(n, 1, size);
        let (mut stage_a, arena) = arena.take_vec_znx_scratch(n, 1, size);
        let (mut tree_a, arena) = arena.take_vec_znx_scratch(n, log_n, size);
        let (mut tree_b, mut arena) = arena.take_vec_znx_scratch(n, log_n, size);

        {
            let a_ref = a.to_backend_ref();
            let mut t_mut = transposed.to_backend_mut();
            self.vec_znx_transpose_backend(&mut t_mut, &a_ref);
        }

        {
            let mut t_mut = transposed.to_backend_mut();
            for col in 0..n {
                self.vec_znx_rsh_assign_backend(
                    base2k,
                    log_n,
                    &mut t_mut,
                    col,
                    &mut arena.borrow(),
                );
            }
        }

        let t_ref = transposed.to_backend_ref();
        let mut shared_mut = shared.to_backend_mut();
        let mut stage_a_mut = stage_a.to_backend_mut();
        let mut tree_a_mut = tree_a.to_backend_mut();
        let mut tree_b_mut = tree_b.to_backend_mut();
        let mut dst_mut = dst.to_backend_mut();

        let mut occupied_a = vec![false; log_n];
        let mut occupied_b = vec![false; log_n];

        for (j, &h) in h_list.iter().enumerate() {
            let col_a = j;
            let col_b = j + n_half;
            occupied_a.iter_mut().for_each(|x| *x = false);
            occupied_b.iter_mut().for_each(|x| *x = false);

            for k in 0..n {
                self.vec_znx_automorphism_backend(h, &mut shared_mut, 0, &t_ref, k);
                {
                    let shared_ref =
                        VecZnxReborrowBackendRef::<BE>::reborrow_backend_ref(&shared_mut);
                    self.vec_znx_automorphism_backend(-1, &mut stage_a_mut, 0, &shared_ref, 0);
                }

                if k != 0 {
                    self.vec_znx_rotate_assign_backend(
                        k as i64,
                        &mut shared_mut,
                        0,
                        &mut arena.borrow(),
                    );
                    self.vec_znx_rotate_assign_backend(
                        k as i64,
                        &mut stage_a_mut,
                        0,
                        &mut arena.borrow(),
                    );
                }

                binary_tree_step(
                    self,
                    &mut stage_a_mut,
                    &mut tree_a_mut,
                    &mut occupied_a,
                    &mut dst_mut,
                    col_a,
                );
                binary_tree_step(
                    self,
                    &mut shared_mut,
                    &mut tree_b_mut,
                    &mut occupied_b,
                    &mut dst_mut,
                    col_b,
                );
            }

            debug_assert!(
                occupied_a.iter().all(|&x| !x) && occupied_b.iter().all(|&x| !x),
                "after d streamed leaves, both trees must be flushed into dst"
            );
        }
    }
}

#[inline]
fn binary_tree_step<BE, M>(
    module: &M,
    stage: &mut VecZnxBackendMut<'_, BE>,
    tree: &mut VecZnxBackendMut<'_, BE>,
    occupied: &mut [bool],
    dst: &mut VecZnxBackendMut<'_, BE>,
    dst_col: usize,
) where
    BE: Backend,
    M: VecZnxAddAssignBackend<BE> + VecZnxCopyBackend<BE>,
{
    let log_n = occupied.len();
    let mut level = 0;
    while level < log_n && occupied[level] {
        let tree_ref = VecZnxReborrowBackendRef::<BE>::reborrow_backend_ref(tree);
        module.vec_znx_add_assign_backend(stage, 0, &tree_ref, level);
        occupied[level] = false;
        level += 1;
    }
    let stage_ref = VecZnxReborrowBackendRef::<BE>::reborrow_backend_ref(stage);
    if level == log_n {
        module.vec_znx_copy_backend(dst, dst_col, &stage_ref, 0);
    } else {
        module.vec_znx_copy_backend(tree, level, &stage_ref, 0);
        occupied[level] = true;
    }
}
