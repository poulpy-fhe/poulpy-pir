//! Fused reduced-precision pack vs. the two-step reference.
//!
//! Packs the same batches through the old path (`pack` at full precision +
//! `modulus_switch_to_digits`) and the fused production path
//! (`partial_pack_batch_pooled` → `pack_to_qtilde`, accumulation at the
//! precision that survives the switch). Checks, at the production regime
//! (`n = 2048`, `base2k = 18`, `k = 54`, `qtilde = 2^32`) for `γ = 32`,
//! `γ = 64`, and the worst preset `γ = 1024`:
//! 1. both paths decrypt to the identical expected messages;
//! 2. the fused path's noise is within ~1 bit of the two-step path's (the
//!    dropped third limb rounds each product at bit 36, ≤ 1/16 qtilde-unit per
//!    product — an order of magnitude below the switch's own rounding noise);
//! 3. the fused output digits are balanced valid `i16` decomposition digits.

use crate::packing::{
    Packing, PackingKeysGenerate, PackingMaskAggregation,
    recursion::{
        decompose_digits, partial_pack_batch, partial_pack_batch_pooled, qtilde_glwe_layout,
        switch_final_mask_to_qtilde,
    },
};
use poulpy_core::{
    EncryptionLayout, GLWECompressedEncryptSk, GLWEDecrypt, GLWEExpandLWEMatrix, GLWENormalize,
    layouts::{
        Base2K, Degree, GGLWEPreparedFactory, GLWEAutomorphismKeyLayout, GLWEDecompress,
        GLWELayout, GLWESecretPreparedFactory, LWEInfos, LWEMatrixLayout, ModuleCoreAlloc,
        ModuleCoreCompressedAlloc, Rank, SecretConversion, TorusPrecision,
    },
};
use poulpy_cpu_avx::FFT64Avx;
use poulpy_cpu_ref::FFT64Ref;
use poulpy_hal::{
    api::{ScalarZnxAutomorphismBackend, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{
        Backend, Module, ScalarZnxToBackendMut, ScalarZnxToBackendRef, ScratchOwned, VecZnx,
        ZnxView, ZnxViewMut, ZnxZero,
    },
    source::Source,
};

macro_rules! run {
    ($backend:ty, $gamma:expr) => {{
        type BE = $backend;
        let gamma = $gamma;
        // The production InsPIRe² regime: 3×18 pack limbs, qtilde = 2^32.
        let n = 2048usize;
        let module = Module::<BE>::new(n as u64);
        let base2k = 18usize;
        let k_ct = 54usize;
        let k_pt = 16usize;
        let qtilde_bits = 32usize;
        let dsize = 1usize;
        let dnum = 3usize;
        let k_ksk = k_ct;
        let baby_size = 8usize;
        let stride = (n / 2) / gamma;
        let batches = 2usize;

        let src_infos = EncryptionLayout::new_from_default_sigma(GLWELayout {
            n: Degree(n as u32),
            base2k: Base2K(base2k as u32),
            k: TorusPrecision(k_ct as u32),
            rank: Rank(1),
        })
        .unwrap();
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
        let qtilde_infos = qtilde_glwe_layout(Degree(n as u32), qtilde_bits);
        let size = matrix_infos.size();
        let scratch_aggregate = module.vec_znx_alloc(n, size);

        let scratch_bytes = module
            .glwe_compressed_encrypt_sk_tmp_bytes(&src_infos)
            .max(module.glwe_decrypt_tmp_bytes(&qtilde_infos))
            .max(module.glwe_expand_lwe_matrix_tmp_bytes(&matrix_infos, &src_infos))
            .max(module.pack_partial_mask_preprocessing_tmp_bytes(gamma, size))
            .max(module.pack_keys_generate_tmp_bytes(&key_infos))
            .max(module.gglwe_prepare_tmp_bytes(&key_infos))
            .max(module.pack_keys_precompute_tmp_bytes(&key_infos, &key_infos, baby_size))
            .max(module.glwe_normalize_tmp_bytes())
            .max(module.pack_precompute_tmp_bytes(
                crate::packing::PackingPrecomputeInfos::new(n - 1, size, base2k, baby_size),
                &scratch_aggregate,
                &key_infos,
            ));
        let mut scratch = ScratchOwned::<BE>::alloc(scratch_bytes);

        let mut source_xs = Source::new([7u8; 32]);
        let mut source_xe = Source::new([8u8; 32]);

        let mut sk_lwe = module.lwe_secret_alloc(Degree(n as u32));
        sk_lwe.fill_ternary_prob(0.5, &mut source_xs);
        let sk_src = module.glwe_secret_from_lwe_secret(&sk_lwe);
        let sk_base = {
            let mut sk_glwe = module.glwe_secret_alloc(Rank(1));
            sk_glwe.fill_zero();
            {
                let src_ref = ScalarZnxToBackendRef::<BE>::to_backend_ref(sk_lwe.data());
                let mut dst_mut = ScalarZnxToBackendMut::<BE>::to_backend_mut(sk_glwe.data_mut());
                module.scalar_znx_automorphism_backend(1, &mut dst_mut, 0, &src_ref, 0);
            }
            sk_glwe
        };
        let mut sk_src_prep = module.glwe_secret_prepared_alloc_from_infos(&sk_src);
        module.glwe_secret_prepare(&mut sk_src_prep, &sk_src);
        let mut sk_dst_prep = module.glwe_secret_prepared_alloc_from_infos(&sk_base);
        module.glwe_secret_prepare(&mut sk_dst_prep, &sk_base);

        let key_partial = module.pack_partial_key_generate(
            &key_infos,
            &sk_lwe,
            [21u8; 32],
            stride,
            &mut source_xe,
            &mut scratch.borrow(),
        );
        let mut key_precomputations = module.pack_partial_keys_precompute(
            &key_partial,
            stride,
            baby_size,
            &mut scratch.borrow(),
        );
        key_precomputations.build_qtilde_keys(&module, qtilde_bits);
        // The 3 → 2 limb reduction below assumes the key products span 3 limbs.
        assert_eq!(
            key_precomputations.key_size(),
            3,
            "production regime key products must have 3 limbs"
        );

        let mut precomputes = Vec::with_capacity(batches);
        let mut bodies: Vec<VecZnx<<BE as Backend>::OwnedBuf>> = Vec::with_capacity(batches);
        let mut expected: Vec<Vec<i64>> = Vec::with_capacity(batches);

        for b in 0..batches {
            let data: Vec<i64> = (0..n).map(|i| ((i + 2 * b) as i64 % 7) - 3).collect();
            expected.push(data[..gamma].to_vec());

            let mut pt = module.glwe_plaintext_alloc_from_infos(&src_infos);
            pt.encode_vec_i64(&data, TorusPrecision(k_pt as u32));
            let mut src = module.glwe_compressed_alloc_from_infos(&src_infos);
            module.glwe_compressed_encrypt_sk(
                &mut src,
                &pt,
                &sk_src_prep,
                [30u8 + b as u8; 32],
                &src_infos,
                &mut source_xe,
                &mut scratch.borrow(),
            );
            let mut src_glwe = module.glwe_alloc_from_infos(&src_infos);
            module.decompress_glwe(&mut src_glwe, &src);
            let mut lwe_matrix = module.lwe_matrix_alloc_from_infos(&matrix_infos);
            module.glwe_expand_lwe_matrix(&mut lwe_matrix, &src_glwe, &mut scratch.borrow());

            let mut aggregate = module.vec_znx_alloc(gamma, size);
            module.packing_partial_mask_preprocessing(
                &mut aggregate,
                base2k,
                gamma,
                lwe_matrix.mask(),
                &mut scratch.borrow(),
            );

            let mut body = module.vec_znx_alloc(1, size);
            body.zero();
            for limb in 0..size {
                let src_limb = lwe_matrix.body().at(0, limb);
                body.at_mut(0, limb)[..gamma].copy_from_slice(src_limb);
            }

            let mut precompute =
                module.pack_partial_precompute_alloc(gamma - 1, size, base2k, baby_size, stride);
            module.pack_partial_precompute(
                &mut precompute,
                &aggregate,
                &key_partial,
                &mut scratch.borrow(),
            );
            switch_final_mask_to_qtilde(
                &module,
                &mut precompute,
                qtilde_bits,
                &mut scratch.borrow(),
            );

            precomputes.push(precompute);
            bodies.push(body);
        }

        let inputs: Vec<_> = precomputes.iter().zip(bodies.iter()).collect();

        // Reference: full-precision pack, then a separate modulus switch.
        let packed_ref = partial_pack_batch(
            &module,
            &src_infos,
            qtilde_bits,
            &inputs,
            &key_precomputations,
            &mut scratch.borrow(),
        );

        // Production: the fused reduced-precision pack (through the pooled entry).
        let mut pool: Vec<ScratchOwned<BE>> = (0..2)
            .map(|_| ScratchOwned::<BE>::alloc(scratch_bytes))
            .collect();
        let packed_fused = partial_pack_batch_pooled(
            &module,
            &src_infos,
            qtilde_bits,
            &inputs,
            &key_precomputations,
            &mut pool,
        );
        assert_eq!(packed_ref.len(), batches);
        assert_eq!(packed_fused.len(), batches);

        let tau = decompose_digits(qtilde_bits);
        let msg_shift = (qtilde_bits - k_pt) as u32;
        let mut max_noise_ref = 0i64;
        let mut max_noise_fused = 0i64;
        for b in 0..batches {
            for (which, glwe, max_noise) in [
                ("two-step", &packed_ref[b], &mut max_noise_ref),
                ("fused", &packed_fused[b], &mut max_noise_fused),
            ] {
                let mut pt = module.glwe_plaintext_alloc_from_infos(&qtilde_infos);
                module.glwe_decrypt(glwe, &mut pt, &sk_dst_prep, &mut scratch.borrow());

                // Message equality at k_pt.
                let mut decoded = vec![0i64; n];
                pt.decode_vec_i64(&mut decoded, TorusPrecision(k_pt as u32));
                assert_eq!(
                    &decoded[..gamma],
                    expected[b].as_slice(),
                    "batch {b}: {which} path lost the first γ messages (γ = {gamma})"
                );

                // Noise below the message, in qtilde units (2^-32 of the torus).
                let mut full = vec![0i64; n];
                pt.decode_vec_i64(&mut full, TorusPrecision(qtilde_bits as u32));
                for c in 0..gamma {
                    let e = full[c] - (expected[b][c] << msg_shift);
                    *max_noise = (*max_noise).max(e.abs());
                }

                // Digits stay valid i16 decomposition digits.
                let data = glwe.data();
                assert_eq!(data.size(), tau, "{which}: switched RLWE must have τ limbs");
                for col in 0..data.cols() {
                    for limb in 0..tau {
                        for &v in data.at(col, limb) {
                            assert!(
                                (i16::MIN as i64..=i16::MAX as i64).contains(&v),
                                "{which}: digit out of i16 range after balancing: {v}"
                            );
                        }
                    }
                }
            }
        }

        // The truncation error (~sqrt(γ)/16 qtilde units) must stay within ~1 bit
        // of the reference noise, which is dominated by the switch's own rounding.
        assert!(
            max_noise_fused <= 2 * max_noise_ref + 8,
            "γ = {gamma}: fused-path noise {max_noise_fused} exceeds ~1 bit over the \
         two-step path's {max_noise_ref} (qtilde units)"
        );
        println!(
            "γ = {gamma}: max noise (qtilde units): two-step = {max_noise_ref}, \
         fused = {max_noise_fused}"
        );
    }};
}

#[test]
fn fused_pack_matches_two_step_ref_gamma_32() {
    run!(FFT64Ref, 32);
}

#[test]
fn fused_pack_matches_two_step_ref_gamma_64() {
    run!(FFT64Ref, 64);
}

#[test]
fn fused_pack_matches_two_step_ref_gamma_1024() {
    run!(FFT64Ref, 1024);
}

#[test]
fn fused_pack_matches_two_step_avx_gamma_32() {
    run!(FFT64Avx, 32);
}

#[test]
fn fused_pack_matches_two_step_avx_gamma_64() {
    run!(FFT64Avx, 64);
}

#[test]
fn fused_pack_matches_two_step_avx_gamma_1024() {
    run!(FFT64Avx, 1024);
}

#[cfg(feature = "avx512-fhe")]
#[test]
fn fused_pack_matches_two_step_avx512_gamma_32() {
    run!(poulpy_cpu_avx512::FFT64Avx512, 32);
}

#[cfg(feature = "avx512-fhe")]
#[test]
fn fused_pack_matches_two_step_avx512_gamma_64() {
    run!(poulpy_cpu_avx512::FFT64Avx512, 64);
}

#[cfg(feature = "avx512-fhe")]
#[test]
fn fused_pack_matches_two_step_avx512_gamma_1024() {
    run!(poulpy_cpu_avx512::FFT64Avx512, 1024);
}
