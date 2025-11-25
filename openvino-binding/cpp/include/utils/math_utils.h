#pragma once

#include <vector>
#include <string>

namespace openvino_sr {
namespace utils {

/**
 * @brief Compute cosine similarity between two vectors
 */
float cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b);

/**
 * @brief Apply softmax to a vector of logits
 */
std::vector<float> softmax(const std::vector<float>& logits);

/**
 * @brief Perform mean pooling over token embeddings with attention mask
 */
std::vector<float> meanPooling(
    const float* embeddings,
    const int64_t* attention_mask,
    size_t sequence_length,
    size_t embedding_dim
);

} // namespace utils
} // namespace openvino_sr

