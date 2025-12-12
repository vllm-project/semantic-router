#pragma once

#include <vector>
#include <string>
#include <memory>
#include <mutex>
#include <atomic>
#include <openvino/openvino.hpp>

namespace openvino_sr {
namespace core {

// Constants
constexpr int MAX_VOCAB_SIZE = 30522;  // BERT base vocab size
constexpr int CLS_TOKEN_ID = 101;
constexpr int SEP_TOKEN_ID = 102;
constexpr int PAD_TOKEN_ID = 0;

// InferRequest pool slot for thread-safe concurrent inference
struct InferRequestSlot {
    ov::InferRequest request;
    std::mutex mutex;
};

// Model instance with compiled model and metadata
struct ModelInstance {
    std::shared_ptr<ov::CompiledModel> compiled_model;
    std::shared_ptr<ov::CompiledModel> tokenizer_model;
    int max_length = 512;
    int num_classes = 0;
    std::string model_path;
    
    // InferRequest pool for concurrent execution
    std::vector<std::unique_ptr<InferRequestSlot>> infer_pool;
    std::atomic<size_t> pool_index{0};
    
    ModelInstance() = default;
    ModelInstance(const ModelInstance&) = delete;
    ModelInstance& operator=(const ModelInstance&) = delete;
};

// Classification result
struct ClassificationResult {
    int predicted_class = -1;
    float confidence = 0.0f;
};

// Classification result with all probabilities
struct ClassificationResultWithProbs {
    int predicted_class = -1;
    float confidence = 0.0f;
    std::vector<float> probabilities;
};

// Entity span (intermediate representation for BIO tagging)
struct EntitySpan {
    std::string entity_type;
    int start = 0;
    int end = 0;
    float confidence = 0.0f;
};

// Token classification entity (final result)
struct TokenEntity {
    std::string entity_type;
    int start = 0;
    int end = 0;
    std::string text;
    float confidence = 0.0f;
};

// Token classification result
struct TokenClassificationResult {
    std::vector<TokenEntity> entities;
};

// Similarity result
struct SimilarityResult {
    int index = -1;
    float score = -1.0f;
};

// Similarity match (for batch operations)
struct SimilarityMatch {
    int index;
    float similarity;
};

} // namespace core
} // namespace openvino_sr

