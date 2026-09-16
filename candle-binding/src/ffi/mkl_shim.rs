//! MKL `hgemm_` shim (half-precision BLAS GEMM).
//!
//! candle-core's `mkl` feature routes every f16 matmul through the MKL
//! `hgemm_` Fortran entry point (`candle-core/src/mkl.rs`). The static MKL
//! 2020.1 package that `intel-mkl-src` fetches via ocipkg
//! (`mkl-static-lp64-iomp`) predates MKL's f16 GEMM support, so the symbol
//! does not exist there and loading this binding fails at runtime with
//! `undefined symbol: hgemm_`.
//!
//! This module provides `hgemm_` on top of the `gemm` crate — the same
//! SIMD-optimized kernels candle uses when built *without* MKL — so the
//! mkl build keeps working (f32/f64 matmuls still go through MKL).

use gemm::{gemm as gemm_kernel, Parallelism};
use half::f16;
use libc::{c_char, c_int};

/// Fortran BLAS interface: `C := alpha*op(A)*op(B) + beta*C`, column-major.
///
/// `op(X) = X` for `trans == 'N'`, `X^T` for `trans == 'T'`; A is m×k with
/// leading dimension `lda`, B is k×n with `ldb`, C is m×n with `ldc`.
#[no_mangle]
pub unsafe extern "C" fn hgemm_(
    transa: *const c_char,
    transb: *const c_char,
    m: *const c_int,
    n: *const c_int,
    k: *const c_int,
    alpha: *const f16,
    a: *const f16,
    lda: *const c_int,
    b: *const f16,
    ldb: *const c_int,
    beta: *const f16,
    c: *mut f16,
    ldc: *const c_int,
) {
    let transa = *transa as u8;
    let transb = *transb as u8;
    let m = *m as usize;
    let n = *n as usize;
    let k = *k as usize;
    let lda = *lda as isize;
    let ldb = *ldb as isize;
    let ldc = *ldc as isize;
    let alpha = *alpha;
    let beta = *beta;

    // Map BLAS column-major leading dimensions onto the gemm crate's
    // (col_stride, row_stride) convention. Column-major storage with
    // leading dimension L keeps element (row r, col c) at buffer[r + c*L],
    // so both operands follow the same pattern:
    //   'N': stride pattern (cs=L, rs=1)
    //   'T': stride pattern (cs=1, rs=L)
    let (lhs_cs, lhs_rs) = if transa == b'N' { (lda, 1) } else { (1, lda) };
    let (rhs_cs, rhs_rs) = if transb == b'N' { (ldb, 1) } else { (1, ldb) };

    // NOTE: the gemm crate computes dst = alpha*dst + beta*(lhs*rhs) — the
    // OPPOSITE of BLAS (C = alpha*op(A)*op(B) + beta*C). Map accordingly:
    // gemm-alpha scales the existing destination (BLAS beta), gemm-beta
    // scales the product (BLAS alpha).
    gemm_kernel(
        m,
        n,
        k,
        c,
        ldc,               // dst_cs
        1,                 // dst_rs
        beta != f16::ZERO, // read_dst: skip loading C when beta == 0
        a,
        lhs_cs,
        lhs_rs,
        b,
        rhs_cs,
        rhs_rs,
        beta,  // gemm-crate alpha: scales existing dst
        alpha, // gemm-crate beta: scales the product
        false, // conj_dst
        false, // conj_lhs
        false, // conj_rhs
        Parallelism::Rayon(rayon::current_num_threads()),
    );
}
