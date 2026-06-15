#pragma once

#include <hip/hip_runtime.h>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * CK Flash Attention forward pass.
 *
 * Computes:  O = softmax(mask(Q @ K^T / scale) + bias) @ V
 *
 * Sliding-window masking is applied via window_size_left/right (set to -1 to
 * disable, i.e. full / global attention).  An optional additive bias (e.g. a
 * 1-D padding mask broadcast along the Q dimension) is added after scaling.
 *
 * All pointer arguments are HIP device pointers.
 * Tensors are laid out as [batch, nhead, seqlen, hdim] (row-major, contiguous).
 *
 * @param stream            HIP stream to launch on
 * @param q                 [B, H, Sq, D]  FP16
 * @param k                 [B, H, Sk, D]  FP16
 * @param v                 [B, H, Sk, D]  FP16  (row-major)
 * @param mask_bias         Additive bias, or nullptr.  Shape is either
 *                          [B, H_b, Sq, Sk] (2-D) or [B, H_b, 1, Sk] (1-D broadcast).
 * @param out               [B, H, Sq, D]  FP16  output
 * @param batch             batch size
 * @param nhead             number of Q/K/V heads (MHA, not GQA)
 * @param seqlen_q          query sequence length
 * @param seqlen_k          key/value sequence length
 * @param hdim              head dimension (must be 32, 64, or 128)
 * @param nhead_bias        head dim of bias (0 = no bias, 1 = broadcast, nhead = per-head)
 * @param scale             softmax scale, typically 1/sqrt(hdim)
 * @param window_size_left  sliding-window left size (-1 = unlimited)
 * @param window_size_right sliding-window right size (-1 = unlimited)
 * @param mask_type         0 = no built-in mask, 1 = top-left reference
 * @param bias_broadcast_sq if nonzero, bias Q-dim is 1 and broadcasts over Sq
 * @return                  0 on success, non-zero on error
 */
int ck_flash_attn_fwd(
    hipStream_t stream,
    const void* q,
    const void* k,
    const void* v,
    const void* mask_bias,
    void* out,
    int32_t batch,
    int32_t nhead,
    int32_t seqlen_q,
    int32_t seqlen_k,
    int32_t hdim,
    int32_t nhead_bias,
    float   scale,
    int32_t window_size_left,
    int32_t window_size_right,
    int32_t mask_type,
    int32_t bias_broadcast_sq);

#ifdef __cplusplus
}
#endif
