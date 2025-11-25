#include "../../include/utils/math_utils.h"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace openvino_sr {
namespace utils {

float cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) {
        return -1.0f;
    }

    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    norm_a = std::sqrt(norm_a);
    norm_b = std::sqrt(norm_b);

    if (norm_a < 1e-9f || norm_b < 1e-9f) {
        return 0.0f;
    }

    return dot / (norm_a * norm_b);
}

std::vector<float> softmax(const std::vector<float>& logits) {
    std::vector<float> exp_values;
    float max_val = *std::max_element(logits.begin(), logits.end());
    float sum = 0.0f;

    for (float val : logits) {
        float exp_val = std::exp(val - max_val);
        exp_values.push_back(exp_val);
        sum += exp_val;
    }

    for (auto& val : exp_values) {
        val /= sum;
    }

    return exp_values;
}

std::vector<float> meanPooling(
    const float* embeddings,
    const int64_t* attention_mask,
    size_t sequence_length,
    size_t embedding_dim
) {
    std::vector<float> pooled(embedding_dim, 0.0f);
    int valid_token_count = 0;

    for (size_t seq_idx = 0; seq_idx < sequence_length; ++seq_idx) {
        if (attention_mask[seq_idx] > 0) {
            for (size_t h = 0; h < embedding_dim; ++h) {
                size_t idx = seq_idx * embedding_dim + h;
                pooled[h] += embeddings[idx];
            }
            valid_token_count++;
        }
    }

    // Average
    if (valid_token_count > 0) {
        for (size_t h = 0; h < embedding_dim; ++h) {
            pooled[h] /= valid_token_count;
        }
    }

    return pooled;
}

} // namespace utils
} // namespace openvino_sr

