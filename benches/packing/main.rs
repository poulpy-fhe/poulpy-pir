//! Benchmarks the currently wired packing strategy through the public
//! [`poulpy_pir::packing::Packing`] API.
//!
//! The setup mirrors the local packing test: build compressed automorphism
//! keys, prepare the key bodies, expand one query to an LWE matrix, aggregate
//! the mask, then benchmark the fixed-mask precompute and online packing phases
//! separately.

#[path = "../../src/circuit/aggregate.rs"]
mod aggregate;

use aggregate::AggregateLWE;
use poulpy_core::{
    EncryptionLayout, GLWEAutomorphismKeyCompressedEncryptSk, GLWEEncryptSk, GLWEExpandLWEMatrix,
    layouts::{
        Base2K, Degree, GGLWEPreparedFactory, GLWEAutomorphismKeyCompressed,
        GLWEAutomorphismKeyLayout, GLWELayout, GLWESecret, GLWESecretPreparedFactory, LWEInfos,
        LWEMatrixLayout, LWESecret, ModuleCoreAlloc, ModuleCoreCompressedAlloc, Rank,
        SecretConversion, TorusPrecision,
    },
};
use poulpy_cpu_avx::FFT64Avx;
use poulpy_hal::{
    api::{ScalarZnxAutomorphismBackend, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{
        Backend, GaloisElement, Module, ScalarZnxToBackendMut, ScalarZnxToBackendRef, ScratchOwned,
    },
    source::Source,
};
use poulpy_pir::packing::{Packing, PackingPrecomputeInfos};
use std::{
    hint::black_box,
    time::{Duration, Instant},
};

const RING_DEGREE: usize = 1024;
const BASE2K: usize = 12;
const CT_BITS: usize = 24;
const KSK_BITS: usize = 36;
const PT_BITS: usize = 16;
const DSIZE: usize = 2;
const BABY_SIZE: usize = 8;
const CHUNK_SIZE: usize = 1;
const PRECOMPUTE_ITERS: usize = 64;
const PACK_ITERS: usize = 512;

fn main() {
    let mut setup = Setup::new();

    let precompute_avg = time_average(PRECOMPUTE_ITERS, || {
        setup.module.pack_precompute(
            &mut setup.precompute,
            &setup.aggregate,
            &setup.key_g,
            &mut setup.scratch.borrow(),
        );
        black_box(&setup.precompute);
    });

    setup.module.pack_precompute(
        &mut setup.precompute,
        &setup.aggregate,
        &setup.key_g,
        &mut setup.scratch.borrow(),
    );

    let pack_avg = time_average(PACK_ITERS, || {
        setup.module.pack(
            &mut setup.packed,
            setup.lwe_matrix.body(),
            &setup.precompute,
            &setup.key_precomputations,
            CHUNK_SIZE,
            &mut setup.scratch.borrow(),
        );
        black_box(&setup.packed);
    });

    println!(
        "packing bench (n={}, k_ct={}, k_ksk={}, baby_size={}, chunk_size={})",
        RING_DEGREE, CT_BITS, KSK_BITS, BABY_SIZE, CHUNK_SIZE
    );
    println!("  {:<24}{:>12}", "phase", "ms");
    println!(
        "  {:<24}{:>12.3}",
        "pack_precompute",
        millis(precompute_avg)
    );
    println!("  {:<24}{:>12.3}", "pack", millis(pack_avg));
}

struct Setup {
    module: Module<FFT64Avx>,
    scratch: ScratchOwned<FFT64Avx>,
    aggregate: poulpy_hal::layouts::VecZnx<<FFT64Avx as Backend>::OwnedBuf>,
    lwe_matrix: poulpy_core::layouts::LWEMatrix<<FFT64Avx as Backend>::OwnedBuf>,
    key_g: GLWEAutomorphismKeyCompressed<<FFT64Avx as Backend>::OwnedBuf>,
    key_precomputations: poulpy_pir::packing::PackingKeyPrecomputations<
        poulpy_core::layouts::prepared::GGLWEPrepared<<FFT64Avx as Backend>::OwnedBuf, FFT64Avx>,
    >,
    precompute: poulpy_pir::packing::PackingPrecomputations<FFT64Avx>,
    packed: poulpy_core::layouts::GLWE<<FFT64Avx as Backend>::OwnedBuf>,
}

impl Setup {
    fn new() -> Self {
        let n = RING_DEGREE;
        let module = Module::<FFT64Avx>::new(n as u64);
        let dnum = CT_BITS.div_ceil(BASE2K * DSIZE);

        let src_infos = EncryptionLayout::new_from_default_sigma(GLWELayout {
            n: Degree(n as u32),
            base2k: Base2K(BASE2K as u32),
            k: TorusPrecision(CT_BITS as u32),
            rank: Rank(1),
        })
        .unwrap();
        let matrix_infos = LWEMatrixLayout {
            rows: n,
            n: Degree(n as u32),
            base2k: src_infos.base2k(),
            k: src_infos.max_k(),
        };
        let key_infos = EncryptionLayout::new_from_default_sigma(GLWEAutomorphismKeyLayout {
            n: Degree(n as u32),
            base2k: Base2K(BASE2K as u32),
            k: TorusPrecision(KSK_BITS as u32),
            rank: Rank(1),
            dnum: dnum.into(),
            dsize: DSIZE.into(),
        })
        .unwrap();
        let precompute_infos =
            PackingPrecomputeInfos::new(n - 1, matrix_infos.size(), BASE2K, BABY_SIZE);

        let mut aggregate = module.vec_znx_alloc(n, matrix_infos.size());
        let mut scratch = ScratchOwned::<FFT64Avx>::alloc(
            module
                .glwe_encrypt_sk_tmp_bytes(&src_infos)
                .max(module.glwe_expand_lwe_matrix_tmp_bytes(&matrix_infos, &src_infos))
                .max(module.aggregate_lwe_tmp_bytes(matrix_infos.size()))
                .max(module.glwe_automorphism_key_compressed_encrypt_sk_tmp_bytes(&key_infos))
                .max(module.gglwe_prepare_tmp_bytes(&key_infos))
                .max(module.pack_keys_precompute_tmp_bytes(&key_infos, &key_infos, BABY_SIZE))
                .max(module.pack_precompute_tmp_bytes(precompute_infos, &aggregate, &key_infos))
                .max(1 << 28),
        );

        let mut source_xs = Source::new([11u8; 32]);
        let mut source_xe = Source::new([12u8; 32]);
        let mut source_xa = Source::new([13u8; 32]);

        let mut sk_lwe = module.lwe_secret_alloc(Degree(n as u32));
        sk_lwe.fill_ternary_prob(0.5, &mut source_xs);
        let sk_src = module.glwe_secret_from_lwe_secret(&sk_lwe);
        let sk_base = glwe_secret_wrap_lwe(&module, &sk_lwe);

        let mut sk_src_prep = module.glwe_secret_prepared_alloc_from_infos(&sk_src);
        module.glwe_secret_prepare(&mut sk_src_prep, &sk_src);

        let (key_g, key_h) = encrypt_packing_keys(
            &module,
            &key_infos,
            &sk_base,
            [21u8; 32],
            &mut source_xe,
            &mut scratch,
        );
        let key_precomputations =
            module.pack_keys_precompute(&key_g, &key_h, BABY_SIZE, &mut scratch.borrow());

        let data: Vec<i64> = (0..n).map(|i| (i as i64 % 7) - 3).collect();
        let mut pt = module.glwe_plaintext_alloc_from_infos(&src_infos);
        pt.encode_vec_i64(&data, TorusPrecision(PT_BITS as u32));

        let mut src = module.glwe_alloc_from_infos(&src_infos);
        module.glwe_encrypt_sk(
            &mut src,
            &pt,
            &sk_src_prep,
            &src_infos,
            &mut source_xe,
            &mut source_xa,
            &mut scratch.borrow(),
        );

        let mut lwe_matrix = module.lwe_matrix_alloc_from_infos(&matrix_infos);
        module.glwe_expand_lwe_matrix(&mut lwe_matrix, &src, &mut scratch.borrow());
        module.aggregate_lwe(
            &mut aggregate,
            BASE2K,
            lwe_matrix.mask(),
            &mut scratch.borrow(),
        );

        let mut precompute = module.pack_precompute_alloc(
            precompute_infos.steps(),
            precompute_infos.size(),
            precompute_infos.base2k(),
            precompute_infos.baby_size(),
        );
        module.pack_precompute(&mut precompute, &aggregate, &key_g, &mut scratch.borrow());

        let packed = module.glwe_alloc_from_infos(&src_infos);

        Self {
            module,
            scratch,
            aggregate,
            lwe_matrix,
            key_g,
            key_precomputations,
            precompute,
            packed,
        }
    }
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

fn encrypt_packing_keys(
    module: &Module<FFT64Avx>,
    key_infos: &EncryptionLayout<GLWEAutomorphismKeyLayout>,
    sk_base: &GLWESecret<<FFT64Avx as Backend>::OwnedBuf>,
    key_seed: [u8; 32],
    source_xe: &mut Source,
    scratch: &mut ScratchOwned<FFT64Avx>,
) -> (
    GLWEAutomorphismKeyCompressed<<FFT64Avx as Backend>::OwnedBuf>,
    GLWEAutomorphismKeyCompressed<<FFT64Avx as Backend>::OwnedBuf>,
) {
    let mut key_g = module.glwe_automorphism_key_compressed_alloc_from_infos(key_infos);
    module.glwe_automorphism_key_compressed_encrypt_sk(
        &mut key_g,
        module.galois_element_inv(module.galois_element(1)),
        sk_base,
        key_seed,
        key_infos,
        source_xe,
        &mut scratch.borrow(),
    );
    let mut key_h = module.glwe_automorphism_key_compressed_alloc_from_infos(key_infos);
    module.glwe_automorphism_key_compressed_encrypt_sk(
        &mut key_h,
        -1,
        sk_base,
        key_seed,
        key_infos,
        source_xe,
        &mut scratch.borrow(),
    );
    (key_g, key_h)
}

fn time_average(iterations: usize, mut f: impl FnMut()) -> Duration {
    let start = Instant::now();
    for _ in 0..iterations {
        f();
    }
    start.elapsed() / iterations as u32
}

fn millis(d: Duration) -> f64 {
    d.as_secs_f64() * 1_000.0
}
