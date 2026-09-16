#pragma once

#include "../core/types.h"
#include "../core/tokenizer.h"
#include <string>
#include <memory>
#include <mutex>
#include <vector>

namespace openvino_sr {
namespace classifiers {

/**
 * @brief TextClassifier handles text classification using BERT-based models
 */
class TextClassifier {
public:
    TextClassifier() = default;
    
    // Initialize classifier
    bool initialize(
        const std::string& model_path,
        int num_classes,
        const std::string& device = "CPU"
    );
    
    // Classify text
    core::ClassificationResult classify(const std::string& text);
    
    // Classify with all class probabilities
    core::ClassificationResultWithProbs classifyWithProbabilities(const std::string& text, int max_length = 8192,
                                                                    bool reject_overflow = false, int* original_tokens = nullptr,
                                         const std::vector<int>& end_tokens = {});
    
    // Check if initialized
    bool isInitialized() const { return model_ && model_->compiled_model != nullptr; }
    
private:
    std::shared_ptr<core::ModelInstance> model_;
    core::OVNativeTokenizer tokenizer_;
    std::mutex mutex_;
    std::string attention_mask_name_;  // Detected at init time
};

} // namespace classifiers
} // namespace openvino_sr

