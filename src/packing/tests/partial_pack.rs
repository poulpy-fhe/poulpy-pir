//! Verifies partial packing (Algorithm 2 `PartialInspiRING`): packing `γ ≤ n/2`
//! LWEs into one RLWE with a single `key_g` collapse (no `key_h`) recovers the
//! first `γ` coefficients. Mirrors the full-pack test but uses the partial
//! precompute path; the online `pack` branches on `precomputations.partial()`.

use crate::packing::{
    Packing, PackingKeysGenerate, PackingMaskAggregation, PackingPrecomputeInfos,
};
use poulpy_core::{
    EncryptionLayout, GLWECompressedEncryptSk, GLWEDecrypt, GLWEExpandLWEMatrix,
    layouts::{
        Base2K, Degree, GGLWEPreparedFactory, GLWEAutomorphismKeyLayout, GLWEDecompress,
        GLWELayout, GLWESecret, GLWESecretPreparedFactory, LWEInfos, LWEMatrixLayout, LWESecret,
        ModuleCoreAlloc, ModuleCoreCompressedAlloc, Rank, SecretConversion, TorusPrecision,
    },
};
use poulpy_cpu_avx::FFT64Avx;
use poulpy_hal::{
    api::{ScalarZnxAutomorphismBackend, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{
        Backend, Module, ScalarZnxToBackendMut, ScalarZnxToBackendRef, ScratchOwned, ZnxView,
        ZnxViewMut, ZnxZero,
    },
    source::Source,
};

fn run(gamma: usize) {
    type BE = FFT64Avx;
    let n = 64usize; // γ ≤ d/2; stride e = (d/2)/γ, generator g_γ = 5^e of order γ
    let module = Module::<BE>::new(n as u64);
    let base2k = 18usize;
    let k_ct = 2 * base2k;
    let k_pt = 16usize;
    let dsize = 1usize;
    let dnum = k_ct.div_ceil(base2k * dsize);
    let k_ksk = k_ct + base2k * dsize;
    let baby_size = 8usize;
    let query_seed = [17u8; 32];
    let packing_key_seed = [21u8; 32];

    let src_infos = EncryptionLayout::new_from_default_sigma(GLWELayout {
        n: Degree(n as u32),
        base2k: Base2K(base2k as u32),
        k: TorusPrecision(k_ct as u32),
        rank: Rank(1),
    })
    .unwrap();
    // Partial packing consumes exactly γ LWEs (one IRCtx component per LWE), so
    // the LWE matrix and the aggregate hold γ rows/columns, not n.
    let matrix_infos = LWEMatrixLayout {
        rows: gamma,
        n: Degree(n as u32),
        base2k: src_infos.base2k(),
        k: src_infos.max_k(),
    };
    let key_infos = EncryptionLayout::new_from_default_sigma(GLWEAutomorphismKeyLayout {
        n: Degree(n as u32),
        base2k: Base2K(base2k as u32),
        k: TorusPrecision(k_ksk as u32),
        rank: Rank(1),
        dnum: dnum.into(),
        dsize: dsize.into(),
    })
    .unwrap();
    // Full metadata + an n-column dummy aggregate over-size the scratch (an upper
    // bound on the γ-column partial path).
    let full_metadata = PackingPrecomputeInfos::new(n - 1, matrix_infos.size(), base2k, baby_size);
    let scratch_aggregate = module.vec_znx_alloc(n, matrix_infos.size());
    let mut aggregate = module.vec_znx_alloc(gamma, matrix_infos.size());

    let mut scratch = ScratchOwned::<BE>::alloc(
        module
            .glwe_compressed_encrypt_sk_tmp_bytes(&src_infos)
            .max(module.glwe_decrypt_tmp_bytes(&src_infos))
            .max(module.glwe_expand_lwe_matrix_tmp_bytes(&matrix_infos, &src_infos))
            .max(module.pack_partial_mask_preprocessing_tmp_bytes(gamma, matrix_infos.size()))
            .max(module.pack_keys_generate_tmp_bytes(&key_infos))
            .max(module.gglwe_prepare_tmp_bytes(&key_infos))
            .max(module.pack_keys_precompute_tmp_bytes(&key_infos, &key_infos, baby_size))
            .max(module.pack_precompute_tmp_bytes(full_metadata, &scratch_aggregate, &key_infos)),
    );

    let mut source_xs = Source::new([11u8; 32]);
    let mut source_xe = Source::new([12u8; 32]);

    let mut sk_lwe = module.lwe_secret_alloc(Degree(n as u32));
    sk_lwe.fill_ternary_prob(0.5, &mut source_xs);
    let sk_src = module.glwe_secret_from_lwe_secret(&sk_lwe);
    let sk_base = glwe_secret_wrap_lwe(&module, &sk_lwe);

    let mut sk_src_prep = module.glwe_secret_prepared_alloc_from_infos(&sk_src);
    module.glwe_secret_prepare(&mut sk_src_prep, &sk_src);
    let mut sk_dst_prep = module.glwe_secret_prepared_alloc_from_infos(&sk_base);
    module.glwe_secret_prepare(&mut sk_dst_prep, &sk_base);

    // Partial packing uses the order-γ generator g_γ = 5^stride (stride=(d/2)/γ);
    // generate its dedicated key K_{g_γ} and precompute its baby bodies (no key_h).
    let stride = (n / 2) / gamma;
    let key_partial = module.pack_partial_key_generate(
        &key_infos,
        &sk_lwe,
        packing_key_seed,
        stride,
        &mut source_xe,
        &mut scratch.borrow(),
    );
    let key_precomputations =
        module.pack_partial_keys_precompute(&key_partial, stride, baby_size, &mut scratch.borrow());

    let data: Vec<i64> = (0..n).map(|i| (i as i64 % 7) - 3).collect();
    let mut pt = module.glwe_plaintext_alloc_from_infos(&src_infos);
    pt.encode_vec_i64(&data, TorusPrecision(k_pt as u32));

    let mut src = module.glwe_compressed_alloc_from_infos(&src_infos);
    module.glwe_compressed_encrypt_sk(
        &mut src,
        &pt,
        &sk_src_prep,
        query_seed,
        &src_infos,
        &mut source_xe,
        &mut scratch.borrow(),
    );

    let mut src_glwe = module.glwe_alloc_from_infos(&src_infos);
    module.decompress_glwe(&mut src_glwe, &src);
    // γ-row LWE matrix: expands only the first γ LWEs (coefficients 0..γ).
    let mut lwe_matrix = module.lwe_matrix_alloc_from_infos(&matrix_infos);
    module.glwe_expand_lwe_matrix(&mut lwe_matrix, &src_glwe, &mut scratch.borrow());

    // Partial mask aggregation (TransformPartial): γ-row mask -> γ-column IRCtx.
    module.packing_partial_mask_preprocessing(
        &mut aggregate,
        base2k,
        gamma,
        lwe_matrix.mask(),
        &mut scratch.borrow(),
    );

    // The LWE-matrix body is a degree-γ polynomial (b_k at coefficient k); embed
    // it into a degree-n polynomial (zero above γ) for the degree-n pack output.
    let size = matrix_infos.size();
    let mut body = module.vec_znx_alloc(1, size);
    body.zero();
    for limb in 0..size {
        let src_limb = lwe_matrix.body().at(0, limb);
        body.at_mut(0, limb)[..gamma].copy_from_slice(src_limb);
    }

    // Partial precompute: γ-1 key_g collapse steps, no key_h, generator stride e.
    let mut precompute =
        module.pack_partial_precompute_alloc(gamma - 1, size, base2k, baby_size, stride);
    module.pack_partial_precompute(
        &mut precompute,
        &aggregate,
        &key_partial,
        &mut scratch.borrow(),
    );

    let mut res = module.glwe_alloc_from_infos(&src_infos);
    module.pack(
        &mut res,
        &body,
        &precompute,
        &key_precomputations,
        1,
        &mut scratch.borrow(),
    );

    let mut decoded_pt = module.glwe_plaintext_alloc_from_infos(&src_infos);
    module.glwe_decrypt(&res, &mut decoded_pt, &sk_dst_prep, &mut scratch.borrow());
    let mut decoded = vec![0; n];
    decoded_pt.decode_vec_i64(&mut decoded, TorusPrecision(k_pt as u32));

    // Partial packing recovers exactly the first γ coefficients.
    assert_eq!(
        decoded[..gamma],
        data[..gamma],
        "partial pack did not recover the first γ coefficients"
    );
}

fn glwe_secret_wrap_lwe(
    module: &Module<FFT64Avx>,
    sk_lwe: &LWESecret<<FFT64Avx as Backend>::OwnedBuf>,
) -> GLWESecret<<FFT64Avx as Backend>::OwnedBuf> {
    let mut sk_glwe = module.glwe_secret_alloc(Rank(1));
    sk_glwe.fill_zero();
    {
        let src_ref = ScalarZnxToBackendRef::<FFT64Avx>::to_backend_ref(sk_lwe.data());
        let mut dst_mut = ScalarZnxToBackendMut::<FFT64Avx>::to_backend_mut(sk_glwe.data_mut());
        module.scalar_znx_automorphism_backend(1, &mut dst_mut, 0, &src_ref, 0);
    }
    sk_glwe
}

// Partial packing (Algorithm 2) uses the order-γ generator g_γ = 5^{(d/2)/γ}, so
// the partial trace π_γ extracts coefficients at multiples of γ (Lemma 5) and the
// first γ coefficients are clean. Exercises several γ ≤ d/2 (powers of two),
// including the γ=d/2 edge case (stride 1, g_γ = 5 = the full pack's first half).
#[test]
fn partial_pack_recovers_first_gamma_coeffs() {
    for gamma in [2usize, 4, 8, 16, 32] {
        run(gamma);
    }
}
