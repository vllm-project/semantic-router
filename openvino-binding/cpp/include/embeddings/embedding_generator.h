#pragma once

#include "../core/types.h"
#include "../core/tokenizer.h"
#include <string>
#include <vector>
#include <memory>
#include <mutex>

namespace openvino_sr {
namespace embeddings {

/**
 * @brief EmbeddingGenerator creates dense vector embeddings from text
 */
class EmbeddingGenerator {
public:
    EmbeddingGenerator() = default;
    
    // Initialize embedding model
    bool initialize(
        const std::string& model_path,
        const std::string& device = "CPU"
    );
    
    // Generate embedding for text
    std::vector<float> generateEmbedding(const std::string& text, int max_length = 512);
    
    // Compute similarity between two texts
    float computeSimilarity(const std::string& text1, const std::string& text2, int max_length = 512);
    
    // Find most similar candidate
    core::SimilarityResult findMostSimilar(
        const std::string& query,
        const std::vector<std::string>& candidates,
        int max_length = 512
    );
    
    // Find top-K similar candidates
    std::vector<core::SimilarityMatch> findTopKSimilar(
        const std::string& query,
        const std::vector<std::string>& candidates,
        int top_k,
        int max_length = 512
    );
    
    // Check if initialized
    bool isInitialized() const { return model_ && model_->compiled_model != nullptr; }
    
private:
    std::shared_ptr<core::ModelInstance> model_;
    core::OVNativeTokenizer tokenizer_;
    std::mutex mutex_;
};

} // namespace embeddings
} // namespace openvino_sr

