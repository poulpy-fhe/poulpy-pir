//! Parameter-derived helpers shared by the InsPIRe² client and server:
//! modulus-switch precision, digit count, source layout, and validity checks.

use poulpy_core::{
    EncryptionLayout,
    layouts::{Base2K, Degree, GLWELayout, Rank, TorusPrecision},
};
use poulpy_hal::layouts::Backend;

use crate::{config::Collapse, parameters::Parameters, payload::Payload};

/// Modulus-switch precision after packing (`qtilde = 2^qtilde_bits`).
pub(crate) fn qtilde_bits<BE: Backend, P: Payload<[u8; 32]>>(
    params: &Parameters<BE, [u8; 32], P>,
) -> usize {
    2 * params.matmul_base2k()
}

/// Base2k=16 decomposition digits for one packed RLWE.
pub(crate) fn tau<BE: Backend, P: Payload<[u8; 32]>>(
    params: &Parameters<BE, [u8; 32], P>,
) -> usize {
    crate::packing::recursion::decompose_digits(qtilde_bits(params))
}

/// The source ciphertext layout (the query / `resp0` regime).
pub(crate) fn src_infos_for<BE: Backend, P: Payload<[u8; 32]>>(
    params: &Parameters<BE, [u8; 32], P>,
) -> EncryptionLayout<GLWELayout> {
    EncryptionLayout::new_from_default_sigma(GLWELayout {
        n: Degree(params.n() as u32),
        base2k: Base2K(params.base2k() as u32),
        k: TorusPrecision(params.k() as u32),
        rank: Rank(1),
    })
    .unwrap()
}

/// Parameter validity shared by InsPIRe² client and server construction.
pub(crate) fn assert_params_valid<BE: Backend, P: Payload<[u8; 32]>>(
    params: &Parameters<BE, [u8; 32], P>,
    t: usize,
    cols: usize,
) {
    let half = params.n() / 2;
    assert!(
        params.n().is_power_of_two() && params.n() >= 4,
        "n must be a power of two >= 4"
    );
    let Collapse::Recursion {
        gamma0,
        gamma1,
        gamma2,
    } = params.collapse()
    else {
        panic!("Recursion parameters must use Collapse::Recursion");
    };
    for (name, g) in [("gamma0", gamma0), ("gamma1", gamma1), ("gamma2", gamma2)] {
        assert!(
            g.is_power_of_two() && g >= 1 && g <= half,
            "{name} must be a power of two in 1..=n/2"
        );
    }
    assert!(t >= 1, "t (batches) must be non-zero");
    assert!(cols >= 1, "cols (= N/t) must be non-zero");
    assert!(
        params.matmul_base2k() == 16,
        "k_pt must be 16: decompose digits are base-2^16 (i16)"
    );
    assert!(
        params.k() > params.matmul_base2k() && params.k().is_multiple_of(params.base2k()),
        "k_ct must be a multiple of base2k and > k_pt"
    );
    assert!(
        qtilde_bits(params).is_multiple_of(16) && qtilde_bits(params) <= params.k(),
        "qtilde bits must be a multiple of 16 and <= k_ct"
    );
}
