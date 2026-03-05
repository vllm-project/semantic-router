// CK-tile FMHA forward kernel instantiation for ORT custom op.
// Compiles with hipcc; instantiates FP16 batch-mode kernels for hdim {32,64,128}.
// Supports an optional additive attention mask (elementwise bias) and
// built-in sliding-window masking via GenericAttentionMask<true, true>.

#include <hip/hip_runtime.h>
#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/fmha.hpp"
#include "ck_tile/ops/fmha_fwd.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/mask.hpp"
#include "ck_tile/ops/bias.hpp"

#include "ck_flash_attn.h"

// ── Kernel type builder ────────────────────────────────────────
// Uses GenericAttentionMask<true, true> so the kernel always evaluates the
// window-based mask.  For "global" (full) attention, pass window_size_left=-1
// and window_size_right=-1 at runtime — the CK mask helper treats negative
// window sizes as "unlimited", making the mask a no-op.
template <int kHeadDim, ck_tile::BlockAttentionBiasEnum kBiasEnum>
struct FmhaFwdFp16Kernel {
    using TypeConfig = FmhaFwdTypeConfig<FmhaFwdFp16>;

    using QDataType   = typename TypeConfig::QDataType;
    using KDataType   = typename TypeConfig::KDataType;
    using VDataType   = typename TypeConfig::VDataType;
    using BiasDataType = typename TypeConfig::BiasDataType;
    using RandValOutputDataType = typename TypeConfig::RandValOutputDataType;
    using LSEDataType  = typename TypeConfig::LSEDataType;
    using SaccDataType = typename TypeConfig::SaccDataType;
    using SMPLComputeDataType = typename TypeConfig::SMPLComputeDataType;
    using PDataType   = typename TypeConfig::PDataType;
    using OaccDataType = typename TypeConfig::OaccDataType;
    using ODataType   = typename TypeConfig::ODataType;

    static constexpr int kK0 = (kHeadDim <= 32) ? 16 : 32;

    using Shape = ck_tile::TileFmhaShape<
        ck_tile::sequence<128, 64, kK0, kHeadDim, 32, kHeadDim>,
        ck_tile::sequence<4, 1, 1>,
        ck_tile::sequence<32, 32, 16>,
        ck_tile::sequence<4, 1, 1>,
        ck_tile::sequence<32, 32, 16>,
        true
    >;

    // IsMasking=true, IsLocal=true: enables sliding-window masking controlled
    // at runtime by window_size_left / window_size_right.
    using FmhaMask = ck_tile::GenericAttentionMask<true, true>;

    struct Traits {
        static constexpr bool kPadSeqLenQ       = true;
        static constexpr bool kPadSeqLenK       = true;
        static constexpr bool kPadHeadDimQ      = false;
        static constexpr bool kPadHeadDimV      = false;
        static constexpr bool kHasLogitsSoftCap = false;
        static constexpr bool kSkipMinSeqlenQ   = false;
        static constexpr auto BiasEnum          = kBiasEnum;
        static constexpr bool kStoreLSE         = false;
        static constexpr bool kHasDropout       = false;
        static constexpr bool kDoFp8StaticQuant = false;
        static constexpr ck_tile::index_t kBlockPerCu = -1;
    };

    using Pipeline = ck_tile::BlockFmhaPipelineQRKSVS<
        ck_tile::BlockFmhaPipelineProblem<
            QDataType, KDataType, VDataType,
            SaccDataType, SMPLComputeDataType,
            BiasDataType, RandValOutputDataType,
            LSEDataType, PDataType,
            OaccDataType, ODataType,
            Shape,
            false,
            ck_tile::StandardAttention,
            FmhaMask,
            Traits
        >,
        ck_tile::BlockFmhaPipelineQRKSVSDefaultPolicy
    >;

    using Epilogue = ck_tile::Default2DEpilogue<
        ck_tile::Default2DEpilogueProblem<OaccDataType, ODataType, false, false>
    >;

    using Kernel = ck_tile::FmhaFwdKernel<Pipeline, Epilogue>;
};

// ── Launch helper ──────────────────────────────────────────────
template <int kHeadDim, ck_tile::BlockAttentionBiasEnum kBiasEnum>
static int launch_fmha_fwd_impl(
    hipStream_t stream,
    const void* q, const void* k, const void* v,
    const void* mask_bias, void* out,
    int32_t batch, int32_t nhead,
    int32_t seqlen_q, int32_t seqlen_k,
    int32_t nhead_bias,
    float scale,
    int32_t window_size_left,
    int32_t window_size_right,
    int32_t mask_type,
    int32_t bias_broadcast_sq)
{
    using Kernel = typename FmhaFwdFp16Kernel<kHeadDim, kBiasEnum>::Kernel;

    fmha_fwd_args args{};
    args.q_ptr        = q;
    args.k_ptr        = k;
    args.v_ptr        = v;
    args.bias_ptr     = mask_bias;
    args.rand_val_ptr = nullptr;
    args.lse_ptr      = nullptr;
    args.o_ptr        = out;

    args.seqstart_q_ptr = nullptr;
    args.seqstart_k_ptr = nullptr;
    args.seqlen_k_ptr   = nullptr;

    args.seqlen_q     = seqlen_q;
    args.seqlen_k     = seqlen_k;
    args.batch        = batch;
    args.max_seqlen_q = seqlen_q;
    args.hdim_q       = kHeadDim;
    args.hdim_v       = kHeadDim;
    args.nhead_q      = nhead;
    args.nhead_k      = nhead;

    args.scale_s = scale;
    args.scale_p = 1.0f;
    args.scale_o = 1.0f;
    args.logits_soft_cap = 0.0f;

    args.stride_q       = kHeadDim;
    args.stride_k       = kHeadDim;
    args.stride_v       = kHeadDim;
    args.stride_randval = 0;
    args.stride_o       = kHeadDim;

    args.nhead_stride_q       = seqlen_q * kHeadDim;
    args.nhead_stride_k       = seqlen_k * kHeadDim;
    args.nhead_stride_v       = seqlen_k * kHeadDim;
    args.nhead_stride_randval = 0;
    args.nhead_stride_lse     = 0;
    args.nhead_stride_o       = seqlen_q * kHeadDim;

    args.batch_stride_q       = nhead * seqlen_q * kHeadDim;
    args.batch_stride_k       = nhead * seqlen_k * kHeadDim;
    args.batch_stride_v       = nhead * seqlen_k * kHeadDim;
    args.batch_stride_randval = 0;
    args.batch_stride_lse     = 0;
    args.batch_stride_o       = nhead * seqlen_q * kHeadDim;

    if (mask_bias) {
        // stride_bias = row stride along Q dimension.
        // For a 2-D bias [B, H_b, Sq, Sk]: stride = Sk
        // For a 1-D bias [B, H_b, 1,  Sk]: stride = 0 (broadcast over Sq)
        args.stride_bias       = bias_broadcast_sq ? 0 : seqlen_k;
        args.nhead_stride_bias = (nhead_bias == 1) ? 0
            : (bias_broadcast_sq ? seqlen_k : seqlen_q * seqlen_k);
        args.batch_stride_bias = nhead_bias *
            (bias_broadcast_sq ? seqlen_k : seqlen_q * seqlen_k);
    } else {
        args.stride_bias       = 0;
        args.nhead_stride_bias = 0;
        args.batch_stride_bias = 0;
    }

    args.window_size_left  = window_size_left;
    args.window_size_right = window_size_right;
    args.mask_type         = mask_type;
    args.min_seqlen_q      = 0;

    args.p_drop     = 0.0f;
    args.s_randval  = false;
    args.drop_seed_offset = std::pair<uint64_t, uint64_t>{0, 0};

    auto [kargs, grids] = fmha_fwd_create_kargs_and_grids<Kernel>(args);
    constexpr dim3 blocks(Kernel::kBlockSize);

    const ck_tile::stream_config sc{stream, false, 0, 0, 1, true, false, 1};

    ck_tile::launch_kernel(
        sc,
        ck_tile::make_kernel<Kernel::kBlockSize, Kernel::kBlockPerCu>(
            Kernel{}, grids, blocks, 0, kargs));

    return 0;
}

template <int kHeadDim>
static int launch_fmha_fwd(
    hipStream_t stream,
    const void* q, const void* k, const void* v,
    const void* mask_bias, void* out,
    int32_t batch, int32_t nhead,
    int32_t seqlen_q, int32_t seqlen_k,
    int32_t nhead_bias,
    float scale,
    int32_t window_size_left,
    int32_t window_size_right,
    int32_t mask_type,
    int32_t bias_broadcast_sq)
{
    if (mask_bias) {
        return launch_fmha_fwd_impl<kHeadDim,
            ck_tile::BlockAttentionBiasEnum::ELEMENTWISE_BIAS>(
            stream, q, k, v, mask_bias, out,
            batch, nhead, seqlen_q, seqlen_k, nhead_bias, scale,
            window_size_left, window_size_right, mask_type, bias_broadcast_sq);
    } else {
        return launch_fmha_fwd_impl<kHeadDim,
            ck_tile::BlockAttentionBiasEnum::NO_BIAS>(
            stream, q, k, v, nullptr, out,
            batch, nhead, seqlen_q, seqlen_k, 0, scale,
            window_size_left, window_size_right, mask_type, 0);
    }
}

// ── C entry point ──────────────────────────────────────────────
extern "C" int ck_flash_attn_fwd(
    hipStream_t stream,
    const void* q, const void* k, const void* v,
    const void* mask_bias, void* out,
    int32_t batch, int32_t nhead,
    int32_t seqlen_q, int32_t seqlen_k,
    int32_t hdim, int32_t nhead_bias,
    float scale,
    int32_t window_size_left,
    int32_t window_size_right,
    int32_t mask_type,
    int32_t bias_broadcast_sq)
{
    switch (hdim) {
    case 32:
        return launch_fmha_fwd<32>(stream, q, k, v, mask_bias, out,
                                   batch, nhead, seqlen_q, seqlen_k, nhead_bias, scale,
                                   window_size_left, window_size_right, mask_type, bias_broadcast_sq);
    case 64:
        return launch_fmha_fwd<64>(stream, q, k, v, mask_bias, out,
                                   batch, nhead, seqlen_q, seqlen_k, nhead_bias, scale,
                                   window_size_left, window_size_right, mask_type, bias_broadcast_sq);
    case 128:
        return launch_fmha_fwd<128>(stream, q, k, v, mask_bias, out,
                                    batch, nhead, seqlen_q, seqlen_k, nhead_bias, scale,
                                    window_size_left, window_size_right, mask_type, bias_broadcast_sq);
    default:
        return -1;
    }
}
