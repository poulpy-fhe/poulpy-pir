//! Plaintext test oracle for the (now-removed) homomorphic coefficient-matrix
//! product `res = U · query`.
//!
//! `U · query` is a *linear* map on the ciphertext's torus components (the mask
//! `a`-parts and the body `b`-part), so it needs no secret key: each output limb
//! column is `sum_in U[out, in] * query[in, col]` reduced mod `2^torus_bits`. This
//! reimplements that product in straightforward `i128` arithmetic, independent of
//! the production `f64` GEMM path, so packing/PIR tests can keep using `U · query`
//! as a trusted oracle.

use poulpy_core::layouts::{LWEInfos, LWEMatrix};
use poulpy_hal::layouts::VecZnx;

use crate::database::CoeffMatrix;

/// `res[col] = sum_in U[out, in] * q[in, col] mod 2^torus_bits` for one operand
/// plane (a mask or a body), written column by column.
#[allow(clippy::too_many_arguments)]
fn matmul_plane(
    res: &mut VecZnx<Vec<u8>>,
    res_base2k: usize,
    u_mat: &[i64],
    rows_out: usize,
    rows_in: usize,
    q: &VecZnx<Vec<u8>>,
    q_base2k: usize,
    torus_bits: usize,
) {
    assert_eq!(res.n(), rows_out, "res rows mismatch");
    assert_eq!(q.n(), rows_in, "query rows mismatch");
    let modulus: i128 = 1i128 << torus_bits;
    let half: i128 = modulus >> 1;
    let mut qcol = vec![0i64; rows_in];
    let mut outcol = vec![0i64; rows_out];
    for col in 0..q.cols() {
        q.decode_vec_i64(q_base2k, col, torus_bits, &mut qcol);
        for out in 0..rows_out {
            let mut acc: i128 = 0;
            for in_ in 0..rows_in {
                acc += (u_mat[out * rows_in + in_] as i128) * (qcol[in_] as i128);
            }
            let mut r = acc.rem_euclid(modulus);
            if r >= half {
                r -= modulus;
            }
            outcol[out] = r as i64;
        }
        res.encode_vec_i64(res_base2k, col, torus_bits, &outcol);
    }
}

/// Decodes `U` (a [`CoeffMatrix`]) into a dense row-major `i64` matrix.
fn decode_u(u: &CoeffMatrix) -> (Vec<i64>, usize, usize) {
    let rows_out = u.rows_out();
    let rows_in = u.rows_in();
    let mut mat = vec![0i64; rows_out * rows_in];
    for out in 0..rows_out {
        let row = u.row(out);
        for in_ in 0..rows_in {
            mat[out * rows_in + in_] = row[in_] as i64;
        }
    }
    (mat, rows_out, rows_in)
}

/// Oracle for `lwe_matrix_mul_mask`: `res.mask = U · query.mask`. The torus
/// precision is taken from `query` (both operands share the regime).
pub(crate) fn lwe_matrix_mul_mask(
    res: &mut LWEMatrix<Vec<u8>>,
    u: &CoeffMatrix,
    query: &LWEMatrix<Vec<u8>>,
) {
    let (mat, rows_out, rows_in) = decode_u(u);
    let res_base2k = res.base2k().as_usize();
    let q_base2k = query.base2k().as_usize();
    let torus_bits = query.max_k().as_usize();
    matmul_plane(
        res.mask_mut(),
        res_base2k,
        &mat,
        rows_out,
        rows_in,
        query.mask(),
        q_base2k,
        torus_bits,
    );
}

/// Oracle for `lwe_matrix_mul_body`: `res.body = U · query.body`.
pub(crate) fn lwe_matrix_mul_body(
    res: &mut LWEMatrix<Vec<u8>>,
    u: &CoeffMatrix,
    query: &LWEMatrix<Vec<u8>>,
) {
    let (mat, rows_out, rows_in) = decode_u(u);
    let res_base2k = res.base2k().as_usize();
    let q_base2k = query.base2k().as_usize();
    let torus_bits = query.max_k().as_usize();
    matmul_plane(
        res.body_mut(),
        res_base2k,
        &mat,
        rows_out,
        rows_in,
        query.body(),
        q_base2k,
        torus_bits,
    );
}

/// Oracle for `lwe_matrix_mul`: `res = U · query` (both mask and body).
pub(crate) fn lwe_matrix_mul(
    res: &mut LWEMatrix<Vec<u8>>,
    u: &CoeffMatrix,
    query: &LWEMatrix<Vec<u8>>,
) {
    lwe_matrix_mul_mask(res, u, query);
    lwe_matrix_mul_body(res, u, query);
}
