#pragma once

#include "types.h"
#include <openvino/openvino.hpp>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <atomic>

namespace openvino_sr {
namespace core {

/**
 * @brief Tokenization result with input_ids, attention_mask, and token_type_ids
 */
struct TokenizationResult {
    std::vector<int64_t> input_ids;
    std::vector<int64_t> attention_mask;
    std::vector<int64_t> token_type_ids;
    bool success = false;
};

/**
 * @brief Native OpenVINO Tokenizer using openvino_tokenizers extension
 * 
 * Thread-safe: CompiledModel is shared, each thread creates its own InferRequest
 */
class OVNativeTokenizer {
public:
    OVNativeTokenizer() = default;
    
    // Load/initialize tokenizer with model directory
    bool loadVocab(const std::string& model_dir);
    
    // Tokenize text to input_ids only
    std::vector<int> tokenize(const std::string& text, int max_length);
    
    // Full tokenization with attention_mask and token_type_ids
    TokenizationResult tokenizeFull(const std::string& text, int max_length);
    
    // Check if tokenizer is initialized
    bool isInitialized() const { return initialized_.load(std::memory_order_acquire); }
    
private:
    bool ensureInitialized();
    
    std::shared_ptr<ov::CompiledModel> compiled_tokenizer_;
    std::string tokenizer_path_;
    mutable std::mutex init_mutex_;
    std::atomic<bool> initialized_{false};
    bool auto_init_attempted_ = false;
};

} // namespace core
} // namespace openvino_sr

