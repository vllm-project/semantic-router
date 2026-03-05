// ORT custom-op shared library: registers com.ck::CKFlashAttention.
// Uses the raw ORT C API only -- no C++ wrappers that reference OrtGetApiBase.

#include "onnxruntime_c_api.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "ck_flash_attn.h"

static const OrtApi* g_api = nullptr;

// ── Helpers ────────────────────────────────────────────────────
#define CHECK_ORT(expr)                                   \
    do {                                                  \
        OrtStatus* _s = (expr);                           \
        if (_s) return _s;                                \
    } while (0)

// ── CKFlashAttention kernel ────────────────────────────────────
struct CKFlashAttentionKernel {
    float scale;
    int32_t window_size_left;
    int32_t window_size_right;
};

static void* CreateKernel(const OrtApi* api, const OrtKernelInfo* info) {
    auto* k = new CKFlashAttentionKernel();
    api->KernelInfoGetAttribute_float(info, "scale", &k->scale);

    int64_t wl = -1, wr = -1;
    if (api->KernelInfoGetAttribute_int64(info, "window_size_left", &wl) != nullptr)
        wl = -1;
    if (api->KernelInfoGetAttribute_int64(info, "window_size_right", &wr) != nullptr)
        wr = -1;

    k->window_size_left  = static_cast<int32_t>(wl);
    k->window_size_right = static_cast<int32_t>(wr);
    return k;
}

static void KernelCompute(void* op_kernel, OrtKernelContext* context) {
    auto* kern = static_cast<CKFlashAttentionKernel*>(op_kernel);
    const OrtApi* api = g_api;

    // Get inputs
    const OrtValue* q_val = nullptr;
    const OrtValue* k_val = nullptr;
    const OrtValue* v_val = nullptr;
    const OrtValue* mask_val = nullptr;
    api->KernelContext_GetInput(context, 0, &q_val);
    api->KernelContext_GetInput(context, 1, &k_val);
    api->KernelContext_GetInput(context, 2, &v_val);
    api->KernelContext_GetInput(context, 3, &mask_val);

    // Q shape: [B, H, Sq, D]
    OrtTensorTypeAndShapeInfo* q_info = nullptr;
    api->GetTensorTypeAndShape(q_val, &q_info);
    size_t q_ndim = 0;
    api->GetDimensionsCount(q_info, &q_ndim);
    std::vector<int64_t> q_shape(q_ndim);
    api->GetDimensions(q_info, q_shape.data(), q_ndim);
    api->ReleaseTensorTypeAndShapeInfo(q_info);

    int64_t batch    = q_shape[0];
    int64_t nhead    = q_shape[1];
    int64_t seqlen_q = q_shape[2];
    int64_t hdim     = q_shape[3];

    // K shape for seqlen_k
    OrtTensorTypeAndShapeInfo* k_info = nullptr;
    api->GetTensorTypeAndShape(k_val, &k_info);
    size_t k_ndim = 0;
    api->GetDimensionsCount(k_info, &k_ndim);
    std::vector<int64_t> k_shape(k_ndim);
    api->GetDimensions(k_info, k_shape.data(), k_ndim);
    api->ReleaseTensorTypeAndShapeInfo(k_info);
    int64_t seqlen_k = k_shape[2];

    // Allocate output
    OrtValue* out_val = nullptr;
    api->KernelContext_GetOutput(context, 0, q_shape.data(), q_ndim, &out_val);

    // Get raw data pointers
    void* q_data = nullptr;
    void* k_data = nullptr;
    void* v_data = nullptr;
    void* o_data = nullptr;
    api->GetTensorMutableData(const_cast<OrtValue*>(q_val), &q_data);
    api->GetTensorMutableData(const_cast<OrtValue*>(k_val), &k_data);
    api->GetTensorMutableData(const_cast<OrtValue*>(v_val), &v_data);
    api->GetTensorMutableData(out_val, &o_data);

    // Optional mask / additive bias.
    // Shape can be [B, H_b, Sq, Sk] (2-D) or [B, H_b, 1, Sk] (1-D broadcast).
    const void* mask_data = nullptr;
    int32_t nhead_bias = 0;
    int32_t bias_broadcast_sq = 0;
    if (mask_val) {
        OrtTensorTypeAndShapeInfo* m_info = nullptr;
        api->GetTensorTypeAndShape(mask_val, &m_info);
        size_t m_count = 0;
        api->GetTensorShapeElementCount(m_info, &m_count);
        size_t m_ndim = 0;
        api->GetDimensionsCount(m_info, &m_ndim);
        std::vector<int64_t> m_shape(m_ndim);
        api->GetDimensions(m_info, m_shape.data(), m_ndim);
        api->ReleaseTensorTypeAndShapeInfo(m_info);
        if (m_count > 0) {
            void* tmp = nullptr;
            api->GetTensorMutableData(const_cast<OrtValue*>(mask_val), &tmp);
            mask_data = tmp;
            nhead_bias = (m_ndim >= 2) ? static_cast<int32_t>(m_shape[1]) : 1;
            // Detect 1-D broadcast: Q dimension == 1 while actual seqlen_q > 1
            if (m_ndim >= 3 && m_shape[2] == 1 && seqlen_q > 1)
                bias_broadcast_sq = 1;
        }
    }

    // Determine mask_type: use MASK_FROM_TOP_LEFT (1) when windowing is active.
    int32_t mask_type = 0;
    bool has_window = (kern->window_size_left >= 0 || kern->window_size_right >= 0);
    if (has_window)
        mask_type = 1;

    void* stream_ptr = nullptr;
    api->KernelContext_GetGPUComputeStream(context, &stream_ptr);
    hipStream_t hip_stream = static_cast<hipStream_t>(stream_ptr);

    int rc = ck_flash_attn_fwd(
        hip_stream,
        q_data, k_data, v_data,
        mask_data, o_data,
        static_cast<int32_t>(batch),
        static_cast<int32_t>(nhead),
        static_cast<int32_t>(seqlen_q),
        static_cast<int32_t>(seqlen_k),
        static_cast<int32_t>(hdim),
        nhead_bias,
        kern->scale,
        kern->window_size_left,
        kern->window_size_right,
        mask_type,
        bias_broadcast_sq);

    if (rc != 0) {
        fprintf(stderr, "CKFlashAttention: ck_flash_attn_fwd returned %d "
                "(B=%ld H=%ld Sq=%ld Sk=%ld D=%ld wl=%d wr=%d)\n",
                rc, batch, nhead, seqlen_q, seqlen_k, hdim,
                kern->window_size_left, kern->window_size_right);
    }
}

static void KernelDestroy(void* op_kernel) {
    delete static_cast<CKFlashAttentionKernel*>(op_kernel);
}

// ── OrtCustomOp vtable ─────────────────────────────────────────
static const char* GetOpName(const OrtCustomOp*) { return "CKFlashAttention"; }
static const char* GetEPType(const OrtCustomOp*) { return "ROCMExecutionProvider"; }

static size_t GetInputTypeCount(const OrtCustomOp*) { return 4; }
static ONNXTensorElementDataType GetInputType(const OrtCustomOp*, size_t) {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
}
static OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(const OrtCustomOp*, size_t i) {
    if (i == 3) return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_OPTIONAL;
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
}

static size_t GetOutputTypeCount(const OrtCustomOp*) { return 1; }
static ONNXTensorElementDataType GetOutputType(const OrtCustomOp*, size_t) {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
}
static OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(const OrtCustomOp*, size_t) {
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
}

static OrtStatus* CreateKernelV2(const OrtCustomOp*, const OrtApi* api, const OrtKernelInfo* info, void** kernel) {
    *kernel = CreateKernel(api, info);
    return nullptr;
}

static OrtStatus* KernelComputeV2(void* op_kernel, OrtKernelContext* context) {
    KernelCompute(op_kernel, context);
    return nullptr;
}

static OrtStatus* InferOutputShape(const OrtCustomOp*, OrtShapeInferContext* ctx) {
    const OrtTensorTypeAndShapeInfo* input_info = nullptr;
    auto status = g_api->ShapeInferContext_GetInputTypeShape(ctx, 0, const_cast<OrtTensorTypeAndShapeInfo**>(&input_info));
    if (status) return status;
    status = g_api->ShapeInferContext_SetOutputTypeShape(ctx, 0, input_info);
    return status;
}

static int GetStartVersion(const OrtCustomOp*) { return 1; }
static int GetEndVersion(const OrtCustomOp*) { return 999; }

static OrtCustomOp g_ck_flash_attn_op = {};

static void* LegacyCreateKernel(const OrtCustomOp* op, const OrtApi* api, const OrtKernelInfo* info) {
    return CreateKernel(api, info);
}

static void LegacyKernelCompute(void* op_kernel, OrtKernelContext* context) {
    KernelCompute(op_kernel, context);
}

static OrtMemType GetInputMemType(const OrtCustomOp*, size_t) {
    return OrtMemTypeDefault;
}

static int ReturnZero(const OrtCustomOp*) { return 0; }

static void InitOp() {
    memset(&g_ck_flash_attn_op, 0, sizeof(g_ck_flash_attn_op));
    g_ck_flash_attn_op.version = ORT_API_VERSION;
    g_ck_flash_attn_op.CreateKernel = LegacyCreateKernel;
    g_ck_flash_attn_op.GetName = GetOpName;
    g_ck_flash_attn_op.GetExecutionProviderType = GetEPType;
    g_ck_flash_attn_op.GetInputType = GetInputType;
    g_ck_flash_attn_op.GetInputTypeCount = GetInputTypeCount;
    g_ck_flash_attn_op.GetOutputType = GetOutputType;
    g_ck_flash_attn_op.GetOutputTypeCount = GetOutputTypeCount;
    g_ck_flash_attn_op.KernelCompute = LegacyKernelCompute;
    g_ck_flash_attn_op.KernelDestroy = KernelDestroy;
    g_ck_flash_attn_op.GetInputCharacteristic = GetInputCharacteristic;
    g_ck_flash_attn_op.GetOutputCharacteristic = GetOutputCharacteristic;
    g_ck_flash_attn_op.GetInputMemoryType = GetInputMemType;
    g_ck_flash_attn_op.GetVariadicInputMinArity = ReturnZero;
    g_ck_flash_attn_op.GetVariadicInputHomogeneity = ReturnZero;
    g_ck_flash_attn_op.GetVariadicOutputMinArity = ReturnZero;
    g_ck_flash_attn_op.GetVariadicOutputHomogeneity = ReturnZero;
    g_ck_flash_attn_op.CreateKernelV2 = CreateKernelV2;
    g_ck_flash_attn_op.KernelComputeV2 = KernelComputeV2;
    g_ck_flash_attn_op.InferOutputShapeFn = InferOutputShape;
    g_ck_flash_attn_op.GetStartVersion = GetStartVersion;
    g_ck_flash_attn_op.GetEndVersion = GetEndVersion;
}

// ── Library entry point ────────────────────────────────────────
extern "C" {

ORT_EXPORT OrtStatus* ORT_API_CALL RegisterCustomOps(
    OrtSessionOptions* options,
    const OrtApiBase* api_base)
{
    const OrtApi* api = api_base->GetApi(ORT_API_VERSION);
    if (!api) return nullptr;
    g_api = api;

    InitOp();

    OrtCustomOpDomain* domain = nullptr;
    CHECK_ORT(api->CreateCustomOpDomain("com.ck", &domain));
    CHECK_ORT(api->CustomOpDomain_Add(domain, &g_ck_flash_attn_op));
    CHECK_ORT(api->AddCustomOpDomain(options, domain));

    return nullptr;
}

} // extern "C"
