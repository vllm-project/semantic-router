// Host-only tests of the real custom-op callbacks. CK is replaced by a dispatch
// stub; these tests prove failure propagation and lifetime, not GPU numerics.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "../src/ort_custom_op.cpp"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <thread>

struct OrtStatus { OrtErrorCode code; std::string message; };
struct OrtTensorTypeAndShapeInfo { std::vector<int64_t> shape; ONNXTensorElementDataType type; };
struct OrtValue { std::vector<int64_t> shape; ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16; };
struct OrtKernelInfo { float scale = 0.125f; int64_t left = -1; int64_t right = -1; bool missing = false; };
struct OrtKernelContext { std::vector<const OrtValue*> inputs; OrtValue output; bool fail_stream = false; bool fail_output = false; };
struct OrtSessionOptions {};
struct OrtCustomOpDomain {};

static OrtApi test_api{};
static OrtApi other_api{};
static int dispatch_result = 0, dispatch_calls = 0, live_shapes = 0;
static int domains_created = 0, ops_added = 0, sessions_added = 0;
static bool incompatible_version = false, other_instance = false;

static OrtStatus* Status(OrtErrorCode code, const char* text) noexcept { return new OrtStatus{code, text}; }
static void ReleaseStatus(OrtStatus* value) noexcept { delete value; }
static const OrtApi* Api(uint32_t version) noexcept {
    if (incompatible_version && version != 1) return nullptr;
    return other_instance ? &other_api : &test_api;
}
static OrtStatus* FloatAttribute(const OrtKernelInfo* info, const char*, float* output) noexcept { *output = info->scale; return nullptr; }
static OrtStatus* IntAttribute(const OrtKernelInfo* info, const char* name, int64_t* output) noexcept {
    if (info->missing) return Status(ORT_FAIL, "missing attribute");
    *output = std::string(name) == "window_size_left" ? info->left : info->right; return nullptr;
}
static OrtStatus* Info(const OrtValue* input, OrtTensorTypeAndShapeInfo** output) noexcept {
    *output = new OrtTensorTypeAndShapeInfo{input->shape, input->type}; ++live_shapes; return nullptr;
}
static void ReleaseInfo(OrtTensorTypeAndShapeInfo* input) noexcept { if (input) { --live_shapes; delete input; } }
static OrtStatus* Type(const OrtTensorTypeAndShapeInfo* info, ONNXTensorElementDataType* type) noexcept { *type = info->type; return nullptr; }
static OrtStatus* Rank(const OrtTensorTypeAndShapeInfo* info, size_t* rank) noexcept { *rank = info->shape.size(); return nullptr; }
static OrtStatus* Dims(const OrtTensorTypeAndShapeInfo* info, int64_t* dims, size_t) noexcept { std::copy(info->shape.begin(), info->shape.end(), dims); return nullptr; }
static OrtStatus* Elements(const OrtTensorTypeAndShapeInfo* info, size_t* count) noexcept { *count = 1; for (auto d : info->shape) *count *= d; return nullptr; }
static OrtStatus* Data(OrtValue* value, void** data) noexcept { *data = value; return nullptr; }
static OrtStatus* Inputs(const OrtKernelContext* context, size_t* count) noexcept { *count = context->inputs.size(); return nullptr; }
static OrtStatus* Input(const OrtKernelContext* context, size_t index, const OrtValue** input) noexcept { *input = context->inputs.at(index); return nullptr; }
static OrtStatus* Output(OrtKernelContext* context, size_t, const int64_t* dims, size_t rank, OrtValue** output) noexcept {
    if (context->fail_output) return Status(ORT_FAIL, "allocation failure");
    context->output.shape.assign(dims, dims+rank); *output = &context->output; return nullptr;
}
static OrtStatus* Stream(const OrtKernelContext* context, void** stream) noexcept {
    if (context->fail_stream) return Status(ORT_EP_FAIL, "stream failure");
    *stream = nullptr; return nullptr;
}
static OrtStatus* CreateDomain(const char*, OrtCustomOpDomain** output) noexcept { *output = new OrtCustomOpDomain(); ++domains_created; return nullptr; }
static void ReleaseDomain(OrtCustomOpDomain* domain) noexcept { delete domain; }
static OrtStatus* AddOp(OrtCustomOpDomain*, const OrtCustomOp* op) noexcept {
    assert(op->CreateKernelV2 && op->KernelComputeV2 && !op->CreateKernel && !op->KernelCompute);
    assert(op->GetInputType(op, 0) == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
    ++ops_added; return nullptr;
}
static OrtStatus* AddDomain(OrtSessionOptions*, OrtCustomOpDomain*) noexcept { ++sessions_added; return nullptr; }
extern "C" int ck_flash_attn_fwd(hipStream_t, const void*, const void*, const void*, const void*, void*,
    int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, float, int32_t, int32_t, int32_t, int32_t) {
    ++dispatch_calls; return dispatch_result;
}

static void ExpectError(OrtStatus* status, OrtErrorCode code) {
    assert(status && status->code == code); ReleaseStatus(status); assert(live_shapes == 0);
}

int main() {
    test_api.CreateStatus = Status; test_api.ReleaseStatus = ReleaseStatus;
    test_api.KernelInfoGetAttribute_float = FloatAttribute; test_api.KernelInfoGetAttribute_int64 = IntAttribute;
    test_api.GetTensorTypeAndShape = Info; test_api.ReleaseTensorTypeAndShapeInfo = ReleaseInfo;
    test_api.GetTensorElementType = Type; test_api.GetDimensionsCount = Rank; test_api.GetDimensions = Dims;
    test_api.GetTensorShapeElementCount = Elements; test_api.GetTensorMutableData = Data;
    test_api.KernelContext_GetInputCount = Inputs; test_api.KernelContext_GetInput = Input;
    test_api.KernelContext_GetOutput = Output; test_api.KernelContext_GetGPUComputeStream = Stream;
    test_api.CreateCustomOpDomain = CreateDomain; test_api.ReleaseCustomOpDomain = ReleaseDomain;
    test_api.CustomOpDomain_Add = AddOp; test_api.AddCustomOpDomain = AddDomain;
    other_api = test_api;
    OrtApiBase base{}; base.GetApi = Api;
    std::vector<std::thread> threads;
    for (int i=0; i<16; ++i) threads.emplace_back([&] { OrtSessionOptions opts; assert(!RegisterCustomOps(&opts, &base)); });
    for (auto& thread : threads) thread.join();
    assert(domains_created == 1 && ops_added == 1 && sessions_added == 16);
    OrtSessionOptions opts;
    other_instance = true; ExpectError(RegisterCustomOps(&opts, &base), ORT_INVALID_ARGUMENT); other_instance = false;
    incompatible_version = true; ExpectError(RegisterCustomOps(&opts, &base), ORT_FAIL); incompatible_version = false;
    OrtKernelInfo attributes;
    void* kernel = nullptr;
    assert(!CreateKernelV2(nullptr, &test_api, &attributes, &kernel));
    OrtValue q{{1,12,16,64}}, k=q, v=q, bias{{1,1,1,16}};
    OrtKernelContext context{{&q,&k,&v,&bias}, {}, false, false};
    assert(!KernelComputeV2(kernel, &context)); assert(live_shapes == 0 && dispatch_calls == 1);
    assert(context.output.shape == q.shape);
    dispatch_result = 17; ExpectError(KernelComputeV2(kernel, &context), ORT_EP_FAIL); dispatch_result = 0;
    context.fail_stream = true; ExpectError(KernelComputeV2(kernel, &context), ORT_EP_FAIL); context.fail_stream = false;
    context.fail_output = true; ExpectError(KernelComputeV2(kernel, &context), ORT_FAIL); context.fail_output = false;
    int calls = dispatch_calls;
    q.shape = {1,12,16}; ExpectError(KernelComputeV2(kernel, &context), ORT_INVALID_ARGUMENT); q=k;
    q.type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; ExpectError(KernelComputeV2(kernel, &context), ORT_INVALID_ARGUMENT); q=k;
    q.shape[0] = 0; ExpectError(KernelComputeV2(kernel, &context), ORT_INVALID_ARGUMENT); q=k;
    q.shape[2] = INT32_MAX; ExpectError(KernelComputeV2(kernel, &context), ORT_INVALID_ARGUMENT); q=k;
    v.shape[2] = 15; ExpectError(KernelComputeV2(kernel, &context), ORT_INVALID_ARGUMENT); v=k;
    bias.shape = {1,2,1,16}; ExpectError(KernelComputeV2(kernel, &context), ORT_INVALID_ARGUMENT); bias.shape={1,1,1,16};
    bias.shape = {1,1,2,16}; ExpectError(KernelComputeV2(kernel, &context), ORT_INVALID_ARGUMENT); bias.shape={1,1,1,16};
    context.inputs[0] = nullptr; ExpectError(KernelComputeV2(kernel, &context), ORT_INVALID_ARGUMENT); context.inputs[0]=&q;
    assert(calls == dispatch_calls);
    bias.shape={0}; assert(!KernelComputeV2(kernel, &context));
    KernelDestroy(kernel);
    for (float scale : {0.0f, -0.1f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
        attributes.scale=scale; ExpectError(CreateKernelV2(nullptr,&test_api,&attributes,&kernel),ORT_INVALID_ARGUMENT); assert(!kernel);
    }
    attributes.scale=0.125f; attributes.left=-2;
    ExpectError(CreateKernelV2(nullptr,&test_api,&attributes,&kernel),ORT_INVALID_ARGUMENT);
    attributes.left=static_cast<int64_t>(INT32_MAX)+1;
    ExpectError(CreateKernelV2(nullptr,&test_api,&attributes,&kernel),ORT_INVALID_ARGUMENT);
    attributes.left=-1; attributes.missing=true;
    ExpectError(CreateKernelV2(nullptr,&test_api,&attributes,&kernel),ORT_FAIL);
    assert(!kernel && live_shapes==0);
    std::cout << "Custom-op callbacks: validation, status propagation, 16-session registration PASS\n";
}
