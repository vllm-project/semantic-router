#pragma once

#include "../core/types.h"
#include "../core/tokenizer.h"
#include <string>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace openvino_sr {
namespace classifiers {

/**
 * @brief TokenClassifier handles token-level classification (NER, PII detection)
 */
class TokenClassifier {
public:
    TokenClassifier() = default;
    
    // Initialize token classifier
    bool initialize(
        const std::string& model_path,
        int num_classes,
        const std::string& device = "CPU"
    );
    
    // Classify tokens with BIO tagging
    core::TokenClassificationResult classifyTokens(
        const std::string& text,
        const std::string& id2label_json
    );
    
    // Check if initialized
    bool isInitialized() const { return model_ && model_->compiled_model != nullptr; }
    
private:
    std::shared_ptr<core::ModelInstance> model_;
    core::OVNativeTokenizer tokenizer_;
    std::mutex mutex_;
};

} // namespace classifiers
} // namespace openvino_sr

