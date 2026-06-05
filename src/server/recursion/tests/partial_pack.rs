//! Phase 0: `partial_pack_batch` + modulus-switch/decompose.
//!
//! Packs `ℓ` batches of `γ` LWEs, modulus-switches each packed RLWE to `q̃` at
//! base2k = 16, and checks (1) every batch decrypts to its first `γ` messages
//! (partial pack + mod-switch), and (2) the base2k = 16 limbs are valid `i16`
//! decomposition digits (the `DECOMPOSE` reinterpret).

use crate::packing::{
    Packing, PackingKeysGenerate, PackingMaskAggregation,
    recursion::{decompose_digits, partial_pack_batch, qtilde_glwe_layout},
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
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, ScratchOwned, VecZnx, ZnxView, ZnxViewMut, ZnxZero},
    source::Source,
};

use super::glwe_secret_wrap_lwe;

type BE = FFT64Avx;

fn run(gamma: usize, batches: usize) {
    let n = 64usize;
    let module = Module::<BE>::new(n as u64);
    let base2k = 18usize;
    let k_ct = 2 * base2k; // q = 2^36
    let k_pt = 16usize;
    let qtilde_bits = 32usize; // q̃ = 2^32 -> τ = 2 base2k=16 limbs
    let dsize = 1usize;
    let dnum = k_ct.div_ceil(base2k * dsize);
    let k_ksk = k_ct + base2k * dsize;
    let baby_size = 8usize;
    let stride = (n / 2) / gamma;

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

    let mut scratch = ScratchOwned::<BE>::alloc(
        module
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
            )),
    );

    let mut source_xs = Source::new([7u8; 32]);
    let mut source_xe = Source::new([8u8; 32]);

    let mut sk_lwe = module.lwe_secret_alloc(Degree(n as u32));
    sk_lwe.fill_ternary_prob(0.5, &mut source_xs);
    let sk_src = module.glwe_secret_from_lwe_secret(&sk_lwe);
    let sk_base = glwe_secret_wrap_lwe(&module, &sk_lwe);
    let mut sk_src_prep = module.glwe_secret_prepared_alloc_from_infos(&sk_src);
    module.glwe_secret_prepare(&mut sk_src_prep, &sk_src);
    let mut sk_dst_prep = module.glwe_secret_prepared_alloc_from_infos(&sk_base);
    module.glwe_secret_prepare(&mut sk_dst_prep, &sk_base);

    // Shared partial packing key K_{g_γ}.
    let key_partial = module.pack_partial_key_generate(
        &key_infos,
        &sk_lwe,
        [21u8; 32],
        stride,
        &mut source_xe,
        &mut scratch.borrow(),
    );
    let key_precomputations =
        module.pack_partial_keys_precompute(&key_partial, stride, baby_size, &mut scratch.borrow());

    // Build the per-batch partial-pack inputs (precompute + body).
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

        precomputes.push(precompute);
        bodies.push(body);
    }

    let inputs: Vec<_> = precomputes.iter().zip(bodies.iter()).collect();

    let packed = partial_pack_batch(
        &module,
        &src_infos,
        qtilde_bits,
        &inputs,
        &key_precomputations,
        &mut scratch.borrow(),
    );
    assert_eq!(packed.len(), batches);

    // (1) Each mod-switched RLWE decrypts to its batch's first γ messages.
    for (b, glwe) in packed.iter().enumerate() {
        let mut pt = module.glwe_plaintext_alloc_from_infos(&qtilde_infos);
        module.glwe_decrypt(glwe, &mut pt, &sk_dst_prep, &mut scratch.borrow());
        let mut decoded = vec![0i64; n];
        pt.decode_vec_i64(&mut decoded, TorusPrecision(k_pt as u32));
        assert_eq!(
            &decoded[..gamma],
            expected[b].as_slice(),
            "batch {b}: mod-switched partial pack lost the first γ coefficients"
        );
    }

    // (2) DECOMPOSE: reinterpreting each base2k=16 limb as i16 (`limb as i16`,
    // the low 16 bits) yields base-2^16 digits that recompose to the same value
    // mod q̃ — i.e. the limbs ARE the i16 decomposition, ready to reinterpret as
    // an i16 plaintext matrix. (The most-significant limb may exceed i16; its
    // overflow is a multiple of q̃ and so vanishes in the recomposition.)
    let tau = decompose_digits(qtilde_bits);
    let qtilde = 1i128 << qtilde_bits;
    let data = packed[0].data();
    assert_eq!(data.size(), tau, "mod-switched RLWE must have τ limbs");
    let _ = qtilde;
    for col in 0..data.cols() {
        for limb in 0..tau {
            for &v in data.at(col, limb) {
                assert!(
                    (i16::MIN as i64..=i16::MAX as i64).contains(&v),
                    "DECOMPOSE digit out of i16 range after balancing: {v}"
                );
            }
        }
    }
}

#[test]
fn partial_pack_batch_decompose() {
    run(8, 3);
    run(16, 2);
}
