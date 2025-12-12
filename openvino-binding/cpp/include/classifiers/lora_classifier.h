#pragma once

#include "../core/types.h"
#include "../core/tokenizer.h"
#include "lora_adapter.h"
#include <string>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace openvino_sr {
namespace classifiers {

/**
 * @brief Task types for LoRA multi-task classification
 */
enum class TaskType {
    Intent,
    PII,
    Security,
    Classification
};

/**
 * @brief Token-level prediction for token classification models
 */
struct TokenPrediction {
    std::string token;          // The token text
    int class_id;               // Predicted class ID
    float confidence;           // Confidence score (0.0 to 1.0)
};

/**
 * @brief Detected entity from BIO tagging
 */
struct DetectedEntity {
    std::string type;           // Entity type (e.g., "EMAIL_ADDRESS", "PERSON")
    std::string text;           // The detected entity text
    int start_token;            // Start token index
    int end_token;              // End token index (inclusive)
    float confidence;           // Average confidence of tokens in entity
};

/**
 * @brief Token classification result
 */
struct TokenClassificationResult {
    std::vector<TokenPrediction> token_predictions;  // Per-token predictions
    std::vector<DetectedEntity> entities;            // Detected entities (aggregated from BIO tags)
    float processing_time_ms;                        // Processing time in milliseconds
};

/**
 * @brief LoRA-enabled classifier for BERT and ModernBERT
 * 
 * Supports multi-task classification with parameter-efficient LoRA adapters.
 * Each task has its own LoRA adapter and classification head.
 */
class LoRAClassifier {
public:
    LoRAClassifier() = default;
    
    /**
     * @brief Initialize LoRA classifier with base model and adapters
     * @param base_model_path Path to base BERT/ModernBERT model (.xml file)
     * @param lora_adapters_path Path to directory containing LoRA adapter models
     * @param task_configs Map of task types to number of classes
     * @param device Device name ("CPU", "GPU", etc.)
     * @param model_type "bert" or "modernbert"
     * @return true if successful
     */
    bool initialize(
        const std::string& base_model_path,
        const std::string& lora_adapters_path,
        const std::unordered_map<TaskType, int>& task_configs,
        const std::string& device = "CPU",
        const std::string& model_type = "bert"
    );
    
    /**
     * @brief Classify text for a specific task (sequence classification)
     * @param text Input text
     * @param task Task type
     * @return Classification result
     */
    core::ClassificationResult classifyTask(const std::string& text, TaskType task);
    
    /**
     * @brief Classify tokens for token-level classification (e.g., NER, PII detection)
     * @param text Input text
     * @param task Task type (should be PII or similar token classification task)
     * @return Token classification result with per-token predictions and detected entities
     */
    TokenClassificationResult classifyTokens(const std::string& text, TaskType task);
    
    /**
     * @brief Check if initialized
     */
    bool isInitialized() const { 
        return base_model_ && base_model_->compiled_model != nullptr; 
    }
    
    /**
     * @brief Get supported tasks
     */
    std::vector<TaskType> getSupportedTasks() const;
    
private:
    /**
     * @brief Get pooled output from base model
     */
    ov::Tensor getPooledOutput(const std::string& text);
    
    /**
     * @brief Apply task-specific LoRA adapter and classification head
     */
    core::ClassificationResult applyLoRAAndClassify(
        const ov::Tensor& pooled_output,
        TaskType task
    );
    
    /**
     * @brief Load task-specific LoRA adapter and classification head
     */
    bool loadTaskAdapter(
        const std::string& lora_adapters_path,
        TaskType task,
        int num_classes,
        const std::string& device
    );
    
    /**
     * @brief Get task name as string
     */
    std::string getTaskName(TaskType task) const;
    
    /**
     * @brief Get maximum sequence length for the model type
     * @return Max sequence length (8192 for ModernBERT, 512 for BERT)
     */
    int getMaxSequenceLength() const;
    
    /**
     * @brief Aggregate BIO tags into detected entities
     * @param original_text The original input text
     * @param tokens Vector of token strings
     * @param predictions Vector of token predictions
     * @param labels Map of class IDs to label names
     * @return Vector of detected entities
     */
    std::vector<DetectedEntity> aggregateBIOTags(
        const std::string& original_text,
        const std::vector<std::string>& tokens,
        const std::vector<TokenPrediction>& predictions,
        const std::unordered_map<int, std::string>& labels
    ) const;
    
    /**
     * @brief Load label mapping from JSON file
     * @param adapters_path Path to adapters directory containing label_mapping.json
     * @return Map of class IDs to label names
     */
    std::unordered_map<int, std::string> loadLabelMapping(const std::string& adapters_path) const;
    
    std::shared_ptr<core::ModelInstance> base_model_;  // Frozen base model
    std::unordered_map<TaskType, LoRAAdapter> lora_adapters_;  // Task-specific LoRA adapters
    std::unordered_map<TaskType, std::shared_ptr<ov::CompiledModel>> task_heads_;  // Classification heads
    std::unordered_map<TaskType, int> task_num_classes_;  // Number of classes per task
    std::string adapters_path_;  // Path to adapters directory
    core::OVNativeTokenizer tokenizer_;
    std::mutex mutex_;
    std::string model_type_;  // "bert" or "modernbert"
};

} // namespace classifiers
} // namespace openvino_sr

