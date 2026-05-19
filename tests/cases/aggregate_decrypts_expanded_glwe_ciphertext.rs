use crate::common::{scalar_mut, scalar_ref, vec_mut, vec_ref};
use poulpy_core::{
    EncryptionLayout, GLWEDecrypt, GLWEEncryptSk, GLWEExpandLWEMatrix,
    layouts::{
        Base2K, Degree, GLWELayout, GLWESecretPreparedFactory, LWEInfos, LWEMatrixLayout,
        ModuleCoreAlloc, Rank, SecretConversion, TorusPrecision,
    },
};
use poulpy_cpu_avx::FFT64Avx;
use poulpy_hal::{
    api::{
        ModuleNew, ScalarZnxAutomorphismBackend, ScratchOwnedAlloc, ScratchOwnedBorrow,
        SvpApplyDft, SvpPPolAlloc, SvpPrepare, VecZnxBigAddAssign, VecZnxBigAlloc,
        VecZnxBigFromSmallBackend, VecZnxBigNormalize, VecZnxDftAddAssign, VecZnxDftAlloc,
        VecZnxIdftApplyTmpA, VecZnxNormalizeAssignBackend, VecZnxSubAssignBackend,
    },
    layouts::{
        Backend, GaloisElement, HostDataMut, HostDataRef, Module, ScalarZnx, ScalarZnxToBackendMut,
        ScalarZnxToBackendRef, ScratchOwned, SvpPPolToBackendMut, SvpPPolToBackendRef, VecZnx,
        VecZnxBigToBackendMut, VecZnxBigToBackendRef, VecZnxDftToBackendMut, VecZnxDftToBackendRef,
        VecZnxToBackendMut, VecZnxToBackendRef,
    },
    source::Source,
};
use poulpy_pir::circuit::AggregateLWE;
use std::time::Instant;

fn run<BE>()
where
    BE: Backend,
    BE::OwnedBuf: HostDataMut + HostDataRef,
    Module<BE>: AggregateLWE<BE>
        + GLWEDecrypt<BE>
        + GLWEEncryptSk<BE>
        + GLWEExpandLWEMatrix<BE>
        + GLWESecretPreparedFactory<BE>
        + GaloisElement
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>
        + ModuleNew<BE>
        + ScalarZnxAutomorphismBackend<BE>
        + SecretConversion<BE>
        + SvpApplyDft<BE>
        + SvpPPolAlloc<BE>
        + SvpPrepare<BE>
        + VecZnxBigAddAssign<BE>
        + VecZnxBigAlloc<BE>
        + VecZnxBigFromSmallBackend<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxDftAddAssign<BE>
        + VecZnxDftAlloc<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxNormalizeAssignBackend<BE>
        + VecZnxSubAssignBackend<BE>,
    ScalarZnx<BE::OwnedBuf>: ScalarZnxToBackendMut<BE> + ScalarZnxToBackendRef<BE>,
    VecZnx<BE::OwnedBuf>: VecZnxToBackendMut<BE> + VecZnxToBackendRef<BE>,
{
    let n = 1024usize;
    let module = Module::<BE>::new(n as u64);
    let base2k = Base2K(18);
    let k_ct = TorusPrecision(36);
    let k_pt = TorusPrecision(8);
    let rank = Rank(1);

    let glwe_infos = EncryptionLayout::new_from_default_sigma(GLWELayout {
        n: Degree(n as u32),
        base2k,
        k: k_ct,
        rank,
    })
    .unwrap();
    let matrix_infos = LWEMatrixLayout {
        rows: n,
        n: Degree(n as u32),
        base2k: glwe_infos.base2k(),
        k: glwe_infos.max_k(),
    };

    let mut source_xs = Source::new([42u8; 32]);
    let mut source_xe = Source::new([2u8; 32]);
    let mut source_xa = Source::new([3u8; 32]);

    let mut sk_glwe = module.glwe_secret_alloc(rank);
    sk_glwe.fill_ternary_prob(0.5, &mut source_xs);
    let sk_lwe = module.lwe_secret_from_glwe_secret(&sk_glwe);

    let mut sk_prepared = module.glwe_secret_prepared_alloc_from_infos(&sk_glwe);
    module.glwe_secret_prepare(&mut sk_prepared, &sk_glwe);

    let message: Vec<i64> = (0..n)
        .map(|_| (source_xa.next_i32() & 15) as i64 - 8)
        .collect();
    let mut plaintext = module.glwe_plaintext_alloc_from_infos(&glwe_infos);
    plaintext.encode_vec_i64(&message, k_pt);

    let scratch_bytes = module
        .glwe_encrypt_sk_tmp_bytes(&glwe_infos)
        .max(module.glwe_decrypt_tmp_bytes(&glwe_infos))
        .max(module.glwe_expand_lwe_matrix_tmp_bytes(&matrix_infos, &glwe_infos))
        .max(module.aggregate_lwe_tmp_bytes(matrix_infos.size()))
        .max(1 << 20);
    let mut scratch = ScratchOwned::<BE>::alloc(scratch_bytes);

    let mut glwe = module.glwe_alloc_from_infos(&glwe_infos);
    module.glwe_encrypt_sk(
        &mut glwe,
        &plaintext,
        &sk_prepared,
        &glwe_infos,
        &mut source_xe,
        &mut source_xa,
        &mut scratch.borrow(),
    );
    let mut decrypted_glwe_pt = module.glwe_plaintext_alloc_from_infos(&glwe_infos);
    module.glwe_decrypt(
        &glwe,
        &mut decrypted_glwe_pt,
        &sk_prepared,
        &mut scratch.borrow(),
    );
    let mut decrypted_glwe_msg = vec![0; n];
    decrypted_glwe_pt.decode_vec_i64(&mut decrypted_glwe_msg, k_pt);
    assert_eq!(decrypted_glwe_msg, message);

    let mut lwe_matrix = module.lwe_matrix_alloc_from_infos(&matrix_infos);
    print!("start: glwe_expand_lwe_matrix");
    let now = Instant::now();
    module.glwe_expand_lwe_matrix(&mut lwe_matrix, &glwe, &mut scratch.borrow());
    println!(": {:?}", now.elapsed());

    let size = lwe_matrix.mask().size();
    let n_half = n >> 1;
    let h_list = (0..n_half)
        .map(|i| module.galois_element(i as i64))
        .collect::<Vec<_>>();

    let decrypt_and_measure_noise =
        |aggregate: &VecZnx<BE::OwnedBuf>, scratch: &mut ScratchOwned<BE>| -> (Vec<i64>, f64) {
            let mut acc_big = module.vec_znx_big_alloc(1, size);
            {
                let mut acc_big_mut = acc_big.to_backend_mut();
                let body_ref = vec_ref::<BE>(lwe_matrix.body());
                module.vec_znx_big_from_small_backend(&mut acc_big_mut, 0, &body_ref, 0);
            }

            let mut svp = module.svp_ppol_alloc(1);
            let mut acc_dft = module.vec_znx_dft_alloc(1, size);
            let mut tmp_dft = module.vec_znx_dft_alloc(1, size);
            let aggregate_ref = vec_ref::<BE>(aggregate);

            for col in 0..n {
                let p = if col < n_half {
                    h_list[col]
                } else {
                    -h_list[col - n_half]
                };
                let mut auto_secret = module.scalar_znx_alloc(1);
                {
                    let secret_ref = scalar_ref::<BE>(sk_lwe.data());
                    let mut auto_secret_mut = scalar_mut::<BE>(&mut auto_secret);
                    module.scalar_znx_automorphism_backend(
                        module.galois_element_inv(p),
                        &mut auto_secret_mut,
                        0,
                        &secret_ref,
                        0,
                    );
                }
                {
                    let mut svp_mut = svp.to_backend_mut();
                    let auto_secret_ref = scalar_ref::<BE>(&auto_secret);
                    module.svp_prepare(&mut svp_mut, 0, &auto_secret_ref, 0);
                }
                {
                    let mut tmp_dft_mut = tmp_dft.to_backend_mut();
                    let svp_ref = svp.to_backend_ref();
                    module.svp_apply_dft(&mut tmp_dft_mut, 0, &svp_ref, 0, &aggregate_ref, col);
                }
                {
                    let mut acc_dft_mut = acc_dft.to_backend_mut();
                    let tmp_dft_ref = tmp_dft.to_backend_ref();
                    module.vec_znx_dft_add_assign(&mut acc_dft_mut, 0, &tmp_dft_ref, 0);
                }
            }

            let mut product_big = module.vec_znx_big_alloc(1, size);
            {
                let mut product_big_mut = product_big.to_backend_mut();
                let mut acc_dft_mut = acc_dft.to_backend_mut();
                module.vec_znx_idft_apply_tmpa(&mut product_big_mut, 0, &mut acc_dft_mut, 0);
            }
            {
                let product_big_ref = product_big.to_backend_ref();
                let mut acc_big_mut = acc_big.to_backend_mut();
                module.vec_znx_big_add_assign(&mut acc_big_mut, 0, &product_big_ref, 0);
            }

            let mut decrypted_pt = module.vec_znx_alloc(1, size);
            {
                let acc_big_ref = acc_big.to_backend_ref();
                let mut decrypted_pt_mut = vec_mut::<BE>(&mut decrypted_pt);
                module.vec_znx_big_normalize(
                    &mut decrypted_pt_mut,
                    base2k.as_usize(),
                    0,
                    0,
                    &acc_big_ref,
                    base2k.as_usize(),
                    0,
                    &mut scratch.borrow(),
                );
            }

            let mut decoded = vec![0; n];
            decrypted_pt.decode_vec_i64(base2k.as_usize(), 0, k_pt.as_usize(), &mut decoded);

            {
                let plaintext_ref = vec_ref::<BE>(plaintext.data());
                let mut decrypted_pt_mut = vec_mut::<BE>(&mut decrypted_pt);
                module.vec_znx_sub_assign_backend(&mut decrypted_pt_mut, 0, &plaintext_ref, 0);
                module.vec_znx_normalize_assign_backend(
                    base2k.as_usize(),
                    &mut decrypted_pt_mut,
                    0,
                    &mut scratch.borrow(),
                );
            }
            let noise_log2 = decrypted_pt.stats(base2k.as_usize(), 0).std().log2();
            (decoded, noise_log2)
        };

    let mut aggregate = module.vec_znx_alloc(n, size);

    print!("start: aggregate_lwe");
    let now = Instant::now();
    module.aggregate_lwe(
        &mut aggregate,
        base2k.as_usize(),
        lwe_matrix.mask(),
        &mut scratch.borrow(),
    );
    println!(": {:?}", now.elapsed());
    let (decoded, noise_log2) = decrypt_and_measure_noise(&aggregate, &mut scratch);
    println!("noise log2(std) = {noise_log2:.3}");
    assert_eq!(decoded, message);
}

#[test]
fn aggregate_lwe_decrypts_expanded_glwe_ciphertext() {
    run::<FFT64Avx>();
}
