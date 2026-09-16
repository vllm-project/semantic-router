#include "../../include/owned_model.h"
#include "../../include/embeddings/embedding_generator.h"
#include "../../include/classifiers/text_classifier.h"
#include <algorithm>
#include <memory>
#include <vector>

struct OVEmbeddingHandle {
    openvino_sr::embeddings::EmbeddingGenerator model;
};

struct OVClassifierHandle {
    openvino_sr::classifiers::TextClassifier model;
};

namespace {
OVOwnedResult resultFrom(const std::vector<float>& values, int original,
                         int maximum, bool reject) {
    OVOwnedResult result{nullptr, 0, original, 0, 2};
    if (reject && original > maximum) {
        result.status = 1;
        return result;
    }
    if (values.empty()) return result;
    auto buffer = std::make_unique<float[]>(values.size());
    std::copy(values.begin(), values.end(), buffer.get());
    result.length = static_cast<int>(values.size());
    result.values = buffer.release();
    result.processed_tokens = std::min(original, maximum);
    result.status = 0;
    return result;
}
}  // namespace

extern "C" {
OVEmbeddingHandle* ov_embedding_open(const char* path, const char* device) {
    if (!path || !device) return nullptr;
    try {
        auto handle = std::make_unique<OVEmbeddingHandle>();
        if (!handle->model.initialize(path, device)) return nullptr;
        return handle.release();
    } catch (...) { return nullptr; }
}

OVClassifierHandle* ov_classifier_open(const char* path, const char* device, int classes) {
    if (!path || !device || classes <= 0) return nullptr;
    try {
        auto handle = std::make_unique<OVClassifierHandle>();
        if (!handle->model.initialize(path, classes, device)) return nullptr;
        return handle.release();
    } catch (...) { return nullptr; }
}

OVOwnedResult ov_embedding_run(OVEmbeddingHandle* handle, const char* text,
                               int max_tokens, bool reject_overflow) {
    if (!handle || !text || max_tokens <= 0) return {nullptr, 0, 0, 0, 2};
    try {
        int original = 0;
        auto values = handle->model.generateEmbedding(text, max_tokens, reject_overflow, &original);
        return resultFrom(values, original, max_tokens, reject_overflow);
    } catch (...) { return {nullptr, 0, 0, 0, 2}; }
}

OVOwnedResult ov_classifier_run(OVClassifierHandle* handle, const char* text,
                                int max_tokens, bool reject_overflow) {
    if (!handle || !text || max_tokens <= 0) return {nullptr, 0, 0, 0, 2};
    try {
        int original = 0;
        auto value = handle->model.classifyWithProbabilities(text, max_tokens, reject_overflow, &original);
        return resultFrom(value.probabilities, original, max_tokens, reject_overflow);
    } catch (...) { return {nullptr, 0, 0, 0, 2}; }
}

void ov_embedding_close(OVEmbeddingHandle* handle) { delete handle; }
void ov_classifier_close(OVClassifierHandle* handle) { delete handle; }
void ov_owned_result_free(OVOwnedResult result) { delete[] result.values; }
}
