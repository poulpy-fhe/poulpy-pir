//! Benchmarks encrypted Horner evaluation for the two relevant polynomial
//! degrees over `R = Z[X] / (X^N + 1)`: `d = N` and `d = 2N - 1`.
//!
//! The coefficients are encrypted GLWEs and the evaluation point is an encrypted
//! unit monomial `w^i = X^i` as a prepared GGSW/RGSW-style ciphertext. The timed
//! path is therefore the hot homomorphic polynomial evaluation: repeated
//! `GLWE x GGSW` external products plus GLWE additions.

use poulpy_core::{
    EncryptionLayout, GGSWEncryptSk, GLWEAdd, GLWECopy, GLWEEncryptSk, GLWEExternalProduct,
    ScratchArenaTakeCore,
    layouts::{
        Base2K, Degree, Dnum, Dsize, GGSWLayout, GGSWPreparedFactory, GLWE, GLWELayout,
        GLWEPlaintext, GLWESecret, GLWESecretPreparedFactory, LWEInfos, ModuleCoreAlloc, Rank,
        TorusPrecision, prepared::GGSWPrepared,
    },
};
use poulpy_cpu_avx::FFT64Avx;
use poulpy_hal::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{
        Backend, HostDataMut, HostDataRef, Module, ScalarZnx, ScratchArena, ScratchOwned,
        ZnxViewMut,
    },
    source::Source,
};
use poulpy_pir::encoding::ModPEncoder;
use std::time::{Duration, Instant};

const RING_DEGREE: usize = 1024;
const ITERATIONS: usize = 3;
const P: i64 = 65537;
const GGSW_TORUS_BITS: usize = 36;
const GLWE_TORUS_BITS: usize = 36;
const GGSW_DNUM: usize = 2;
const BASE2K: usize = 18;

fn main() {
    bench::<FFT64Avx>();
}

fn root_monomial<BE>(module: &Module<BE>, exponent: usize) -> ScalarZnx<BE::OwnedBuf>
where
    BE: Backend,
    BE::OwnedBuf: HostDataMut,
    Module<BE>: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
{
    let n = module.n();
    let exponent = exponent % (2 * n);
    let mut root = module.scalar_znx_alloc(1);
    if exponent < n {
        root.at_mut(0, 0)[exponent] = 1;
    } else {
        root.at_mut(0, 0)[exponent - n] = -1;
    }
    root
}

fn encrypted_horner_at_root<BE>(
    module: &Module<BE>,
    coeffs: &[GLWE<BE::OwnedBuf>],
    root: &GGSWPrepared<BE::OwnedBuf, BE>,
    acc: &mut GLWE<BE::OwnedBuf>,
    product: &mut GLWE<BE::OwnedBuf>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    Module<BE>: GLWEAdd<BE> + GLWECopy<BE> + GLWEExternalProduct<BE>,
{
    module.glwe_copy(acc, coeffs.last().unwrap());
    for coeff in coeffs[..coeffs.len() - 1].iter().rev() {
        module.glwe_external_product(product, acc, root, root.size(), scratch);
        module.glwe_add_into(acc, product, coeff);
    }
}

fn bench<BE>()
where
    BE: Backend,
    BE::OwnedBuf: HostDataMut + HostDataRef,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: GGSWEncryptSk<BE>
        + GGSWPreparedFactory<BE>
        + GLWEAdd<BE>
        + GLWECopy<BE>
        + GLWEEncryptSk<BE>
        + GLWEExternalProduct<BE>
        + GLWESecretPreparedFactory<BE>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>
        + ModuleNew<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    ScalarZnx<BE::OwnedBuf>: poulpy_hal::layouts::ScalarZnxToBackendRef<BE>,
{
    let n = RING_DEGREE;
    let max_degree = 2 * n - 1;
    let max_coeff_count = max_degree + 1;

    let module = Module::<BE>::new(n as u64);
    let encoder = ModPEncoder::new(P, GLWE_TORUS_BITS);
    let rank = Rank(1);

    let glwe_infos = EncryptionLayout::new_from_default_sigma(GLWELayout {
        n: Degree(n as u32),
        base2k: Base2K(BASE2K as u32),
        k: TorusPrecision(GLWE_TORUS_BITS as u32),
        rank,
    })
    .unwrap();
    let ggsw_infos = EncryptionLayout::new_from_default_sigma(GGSWLayout {
        n: Degree(n as u32),
        base2k: Base2K(BASE2K as u32),
        k: TorusPrecision(GGSW_TORUS_BITS as u32),
        dnum: Dnum(GGSW_DNUM as u32),
        dsize: Dsize(1),
        rank,
    })
    .unwrap();

    let scratch_bytes = module
        .glwe_encrypt_sk_tmp_bytes(&glwe_infos)
        .max(module.ggsw_encrypt_sk_tmp_bytes(&ggsw_infos))
        .max(module.ggsw_prepare_tmp_bytes(&ggsw_infos))
        .max(module.glwe_external_product_tmp_bytes(&glwe_infos, &glwe_infos, &ggsw_infos));
    let mut scratch = ScratchOwned::<BE>::alloc(scratch_bytes);

    let mut source_xs = Source::new([21u8; 32]);
    let mut source_xe = Source::new([22u8; 32]);
    let mut source_xa = Source::new([23u8; 32]);

    let mut sk: GLWESecret<BE::OwnedBuf> = module.glwe_secret_alloc(rank);
    sk.fill_ternary_prob(0.5, &mut source_xs);
    let mut sk_prepared = module.glwe_secret_prepared_alloc(rank);
    module.glwe_secret_prepare(&mut sk_prepared, &sk);

    let root_pt = root_monomial(&module, 17);
    let mut root_ct = module.ggsw_alloc_from_infos(&ggsw_infos);
    module.ggsw_encrypt_sk(
        &mut root_ct,
        &root_pt,
        &sk_prepared,
        &ggsw_infos,
        &mut source_xe,
        &mut source_xa,
        &mut scratch.borrow(),
    );
    let mut root_prepared = module.ggsw_prepared_alloc_from_infos(&root_ct);
    module.ggsw_prepare(&mut root_prepared, &root_ct, &mut scratch.borrow());

    let coeffs = encrypted_coefficients(
        &module,
        &glwe_infos,
        &encoder,
        max_coeff_count,
        &sk_prepared,
        &mut source_xe,
        &mut source_xa,
        &mut scratch,
    );

    let mut acc = module.glwe_alloc_from_infos(&glwe_infos);
    let mut product = module.glwe_alloc_from_infos(&glwe_infos);

    println!(
        "encrypted Horner bench (N={}, iterations={}, backend=FFT64Avx)",
        n, ITERATIONS
    );
    println!(
        "  {:<12}{:>12}{:>12}{:>16}",
        "degree", "coeffs", "ext-prod", "ms/eval"
    );

    for degree in [n, 2 * n - 1] {
        let coeff_count = degree + 1;
        let avg = time_average(ITERATIONS, || {
            encrypted_horner_at_root(
                &module,
                &coeffs[..coeff_count],
                &root_prepared,
                &mut acc,
                &mut product,
                &mut scratch.borrow(),
            );
        });
        println!(
            "  d={:<9}{:>12}{:>12}{:>16.3}",
            degree,
            coeff_count,
            degree,
            millis(avg)
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn encrypted_coefficients<BE>(
    module: &Module<BE>,
    glwe_infos: &EncryptionLayout<GLWELayout>,
    encoder: &ModPEncoder,
    count: usize,
    sk_prepared: &impl poulpy_core::layouts::prepared::GLWESecretPreparedToBackendRef<BE>,
    source_xe: &mut Source,
    source_xa: &mut Source,
    scratch: &mut ScratchOwned<BE>,
) -> Vec<GLWE<BE::OwnedBuf>>
where
    BE: Backend,
    BE::OwnedBuf: HostDataMut,
    Module<BE>: GLWEEncryptSk<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
{
    let n = module.n();
    let mut coeffs = Vec::with_capacity(count);
    for j in 0..count {
        let mut values = vec![0i64; n];
        for (i, value) in values.iter_mut().enumerate() {
            let raw = ((j as i64 * 4099 + i as i64 * 917 + (j * i) as i64 * 37) % P) - (P / 2);
            *value = encoder.normalize(raw);
        }

        let mut plaintext: GLWEPlaintext<BE::OwnedBuf> =
            module.glwe_plaintext_alloc_from_infos(glwe_infos);
        encoder.encode_vec_i64(&mut plaintext.data, BASE2K, 0, &values);

        let mut ct = module.glwe_alloc_from_infos(glwe_infos);
        module.glwe_encrypt_sk(
            &mut ct,
            &plaintext,
            sk_prepared,
            glwe_infos,
            source_xe,
            source_xa,
            &mut scratch.borrow(),
        );
        coeffs.push(ct);
    }
    coeffs
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
