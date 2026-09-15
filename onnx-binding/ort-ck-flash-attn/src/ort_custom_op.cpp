// ORT custom-op library for com.ck::CKFlashAttention. All failures return an
// OrtStatus to the owning session; this library never selects another EP.
#include "onnxruntime_c_api.h"
#include "ck_flash_attn.h"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#define CHECK_ORT(expr) do { OrtStatus* status = (expr); if (status) return status; } while (0)

namespace {
std::atomic<const OrtApi*> shape_api{nullptr};

struct Kernel {
    const OrtApi* api;
    float scale;
    int32_t left;
    int32_t right;
};

OrtStatus* Invalid(const OrtApi* api, const char* message) {
    return api->CreateStatus(ORT_INVALID_ARGUMENT, message);
}

struct ShapeInfo {
    const OrtApi* api;
    OrtTensorTypeAndShapeInfo* value = nullptr;
    ~ShapeInfo() { api->ReleaseTensorTypeAndShapeInfo(value); }
};

struct Tensor {
    std::vector<int64_t> shape;
    size_t count = 0;
    void* data = nullptr;
};

OrtStatus* ReadTensor(const OrtApi* api, const OrtValue* value, Tensor& tensor) {
    if (!value) return Invalid(api, "CKFlashAttention: required tensor is absent");
    ShapeInfo info{api};
    CHECK_ORT(api->GetTensorTypeAndShape(value, &info.value));
    ONNXTensorElementDataType type;
    CHECK_ORT(api->GetTensorElementType(info.value, &type));
    if (type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
        return Invalid(api, "CKFlashAttention: tensors must be float16");
    size_t rank = 0;
    CHECK_ORT(api->GetDimensionsCount(info.value, &rank));
    if (rank > 4) return Invalid(api, "CKFlashAttention: tensor rank exceeds four");
    tensor.shape.resize(rank);
    CHECK_ORT(api->GetDimensions(info.value, tensor.shape.data(), rank));
    CHECK_ORT(api->GetTensorShapeElementCount(info.value, &tensor.count));
    // CK's dispatch strides are signed int32. Reject overflow before any cast
    // or multiplication, including the optional dense additive bias.
    if (tensor.count > static_cast<size_t>(std::numeric_limits<int32_t>::max()))
        return Invalid(api, "CKFlashAttention: tensor exceeds int32 dispatch indexing");
    for (int64_t dimension : tensor.shape) {
        if (dimension < 0 || dimension > std::numeric_limits<int32_t>::max())
            return Invalid(api, "CKFlashAttention: invalid tensor dimension");
    }
    if (tensor.count) {
        CHECK_ORT(api->GetTensorMutableData(const_cast<OrtValue*>(value), &tensor.data));
        if (!tensor.data) return Invalid(api, "CKFlashAttention: tensor data is null");
    }
    return nullptr;
}

OrtStatus* CreateKernelV2(const OrtCustomOp*, const OrtApi* api,
                          const OrtKernelInfo* info, void** output) {
    *output = nullptr;
    try {
        auto kernel = std::make_unique<Kernel>();
        kernel->api = api;
        CHECK_ORT(api->KernelInfoGetAttribute_float(info, "scale", &kernel->scale));
        if (!std::isfinite(kernel->scale) || kernel->scale <= 0)
            return Invalid(api, "CKFlashAttention: scale must be finite and positive");
        // Exporters specify both attributes, including -1 for global attention.
        // Missing or wrongly typed attributes must not silently change masking.
        int64_t left = 0, right = 0;
        CHECK_ORT(api->KernelInfoGetAttribute_int64(info, "window_size_left", &left));
        CHECK_ORT(api->KernelInfoGetAttribute_int64(info, "window_size_right", &right));
        if (left < -1 || right < -1 || left > INT32_MAX || right > INT32_MAX)
            return Invalid(api, "CKFlashAttention: window bounds must be -1 or nonnegative int32");
        kernel->left = static_cast<int32_t>(left);
        kernel->right = static_cast<int32_t>(right);
        *output = kernel.release();
        return nullptr;
    } catch (const std::exception& error) {
        return api->CreateStatus(ORT_RUNTIME_EXCEPTION, error.what());
    } catch (...) {
        return api->CreateStatus(ORT_RUNTIME_EXCEPTION, "CKFlashAttention: kernel creation failed");
    }
}

OrtStatus* Compute(Kernel& kernel, OrtKernelContext* context) {
    const OrtApi* api = kernel.api;
    const OrtValue *q_value = nullptr, *k_value = nullptr, *v_value = nullptr, *bias_value = nullptr;
    size_t count = 0;
    CHECK_ORT(api->KernelContext_GetInputCount(context, &count));
    if (count < 3 || count > 4) return Invalid(api, "CKFlashAttention: expected Q, K, V and optional bias");
    CHECK_ORT(api->KernelContext_GetInput(context, 0, &q_value));
    CHECK_ORT(api->KernelContext_GetInput(context, 1, &k_value));
    CHECK_ORT(api->KernelContext_GetInput(context, 2, &v_value));
    if (count == 4) CHECK_ORT(api->KernelContext_GetInput(context, 3, &bias_value));
    Tensor q, k, v, bias;
    CHECK_ORT(ReadTensor(api, q_value, q));
    CHECK_ORT(ReadTensor(api, k_value, k));
    CHECK_ORT(ReadTensor(api, v_value, v));
    if (q.shape.size() != 4 || k.shape.size() != 4 || v.shape.size() != 4 ||
        !q.count || !k.count || !v.count)
        return Invalid(api, "CKFlashAttention: Q/K/V must be nonempty rank-four tensors");
    if (q.shape[0] != k.shape[0] || q.shape[1] != k.shape[1] ||
        q.shape[3] != k.shape[3] || k.shape != v.shape)
        return Invalid(api, "CKFlashAttention: incompatible batch, heads, sequence or head dimensions");
    const int64_t dimension = q.shape[3];
    if (dimension != 32 && dimension != 64 && dimension != 128)
        return Invalid(api, "CKFlashAttention: head dimension must be 32, 64 or 128");
    int32_t bias_heads = 0, broadcast_query = 0;
    if (bias_value) {
        CHECK_ORT(ReadTensor(api, bias_value, bias));
        if (bias.count) {
            if (bias.shape.size() != 4 || bias.shape[0] != q.shape[0] ||
                (bias.shape[1] != 1 && bias.shape[1] != q.shape[1]) ||
                (bias.shape[2] != 1 && bias.shape[2] != q.shape[2]) ||
                bias.shape[3] != k.shape[2])
                return Invalid(api, "CKFlashAttention: bias must be [B,1|H,1|Sq,Sk]");
            bias_heads = static_cast<int32_t>(bias.shape[1]);
            broadcast_query = bias.shape[2] == 1 && q.shape[2] != 1;
        }
    }
    OrtValue* output = nullptr;
    CHECK_ORT(api->KernelContext_GetOutput(context, 0, q.shape.data(), q.shape.size(), &output));
    if (!output) return Invalid(api, "CKFlashAttention: output allocation returned null");
    void* output_data = nullptr;
    CHECK_ORT(api->GetTensorMutableData(output, &output_data));
    if (!output_data) return Invalid(api, "CKFlashAttention: output data is null");
    void* stream = nullptr;
    CHECK_ORT(api->KernelContext_GetGPUComputeStream(context, &stream));
    const int result = ck_flash_attn_fwd(
        static_cast<hipStream_t>(stream), q.data, k.data, v.data, bias.data, output_data,
        static_cast<int32_t>(q.shape[0]), static_cast<int32_t>(q.shape[1]),
        static_cast<int32_t>(q.shape[2]), static_cast<int32_t>(k.shape[2]),
        static_cast<int32_t>(dimension), bias_heads, kernel.scale, kernel.left,
        kernel.right, kernel.left >= 0 || kernel.right >= 0, broadcast_query);
    if (result != 0) {
        const std::string message = "CKFlashAttention: HIP/CK dispatch failed with code " + std::to_string(result);
        return api->CreateStatus(ORT_EP_FAIL, message.c_str());
    }
    return nullptr;
}

OrtStatus* KernelComputeV2(void* state, OrtKernelContext* context) {
    auto& kernel = *static_cast<Kernel*>(state);
    try { return Compute(kernel, context); }
    catch (const std::exception& error) { return kernel.api->CreateStatus(ORT_RUNTIME_EXCEPTION, error.what()); }
    catch (...) { return kernel.api->CreateStatus(ORT_RUNTIME_EXCEPTION, "CKFlashAttention: compute failed"); }
}

void KernelDestroy(void* state) { delete static_cast<Kernel*>(state); }
const char* GetName(const OrtCustomOp*) { return "CKFlashAttention"; }
const char* GetEP(const OrtCustomOp*) { return "ROCMExecutionProvider"; }
size_t InputCount(const OrtCustomOp*) { return 4; }
size_t OutputCount(const OrtCustomOp*) { return 1; }
ONNXTensorElementDataType TensorType(const OrtCustomOp*, size_t) { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16; }
OrtCustomOpInputOutputCharacteristic InputCharacteristic(const OrtCustomOp*, size_t i) {
    return i == 3 ? INPUT_OUTPUT_OPTIONAL : INPUT_OUTPUT_REQUIRED;
}
OrtCustomOpInputOutputCharacteristic OutputCharacteristic(const OrtCustomOp*, size_t) { return INPUT_OUTPUT_REQUIRED; }
OrtMemType InputMemory(const OrtCustomOp*, size_t) { return OrtMemTypeDefault; }
int Zero(const OrtCustomOp*) { return 0; }
int StartVersion(const OrtCustomOp*) { return 1; }
int EndVersion(const OrtCustomOp*) { return 999; }
OrtStatus* InferOutputShape(const OrtCustomOp*, OrtShapeInferContext* context) {
    const OrtApi* api = shape_api.load(std::memory_order_acquire);
    OrtTensorTypeAndShapeInfo* info = nullptr;
    CHECK_ORT(api->ShapeInferContext_GetInputTypeShape(context, 0, &info));
    return api->ShapeInferContext_SetOutputTypeShape(context, 0, info);
}

struct Registration {
    std::mutex mutex;
    const OrtApi* api = nullptr;
    OrtCustomOp op{};
    OrtCustomOpDomain* domain = nullptr;
    // RegisterCustomOpsLibrary_V2 retains the library for all sessions using it.
    // The immutable shared domain must live equally long (ORT C API contract).
    ~Registration() { if (domain) api->ReleaseCustomOpDomain(domain); }
};
Registration registration;

OrtCustomOp MakeOp() {
    OrtCustomOp op{};
    op.version = ORT_API_VERSION;
    // ORT >= 1.16 selects V2 callbacks. No legacy callback can swallow errors.
    op.CreateKernelV2 = CreateKernelV2;
    op.KernelComputeV2 = KernelComputeV2;
    op.KernelDestroy = KernelDestroy;
    op.GetName = GetName;
    op.GetExecutionProviderType = GetEP;
    op.GetInputTypeCount = InputCount;
    op.GetOutputTypeCount = OutputCount;
    op.GetInputType = TensorType;
    op.GetOutputType = TensorType;
    op.GetInputCharacteristic = InputCharacteristic;
    op.GetOutputCharacteristic = OutputCharacteristic;
    op.GetInputMemoryType = InputMemory;
    op.GetVariadicInputMinArity = Zero;
    op.GetVariadicInputHomogeneity = Zero;
    op.GetVariadicOutputMinArity = Zero;
    op.GetVariadicOutputHomogeneity = Zero;
    op.InferOutputShapeFn = InferOutputShape;
    op.GetStartVersion = StartVersion;
    op.GetEndVersion = EndVersion;
    return op;
}
}  // namespace

extern "C" ORT_EXPORT OrtStatus* ORT_API_CALL RegisterCustomOps(
    OrtSessionOptions* options, const OrtApiBase* api_base) {
    const OrtApi* api = api_base->GetApi(ORT_API_VERSION);
    if (!api) {
        // Every valid ORT ApiBase supports v1's CreateStatus. Never report
        // success when this library's required API version is unavailable.
        return api_base->GetApi(1)->CreateStatus(ORT_FAIL, "CKFlashAttention: incompatible ORT API version");
    }
    try {
        std::lock_guard<std::mutex> lock(registration.mutex);
        if (registration.api && registration.api != api)
            return Invalid(api, "CKFlashAttention: one library cannot mix ORT API instances");
        if (!registration.domain) {
            OrtCustomOpDomain* domain = nullptr;
            CHECK_ORT(api->CreateCustomOpDomain("com.ck", &domain));
            registration.op = MakeOp();
            OrtStatus* status = api->CustomOpDomain_Add(domain, &registration.op);
            if (status) { api->ReleaseCustomOpDomain(domain); return status; }
            registration.api = api;
            registration.domain = domain;
            shape_api.store(api, std::memory_order_release);
        }
        return api->AddCustomOpDomain(options, registration.domain);
    } catch (const std::exception& error) {
        return api->CreateStatus(ORT_RUNTIME_EXCEPTION, error.what());
    } catch (...) {
        return api->CreateStatus(ORT_RUNTIME_EXCEPTION, "CKFlashAttention: registration failed");
    }
}
