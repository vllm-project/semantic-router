#include "../../include/classifiers/token_classifier.h"
#include "../../include/core/model_manager.h"
#include <iostream>
#include <algorithm>
#include <cstring>
#include <regex>
#include <cmath>

namespace openvino_sr {
namespace classifiers {

// Constants for special tokens (ModernBERT-specific)
static const int MODERNBERT_PAD = 50283;
static const int MODERNBERT_SEP = 50282;

// Helper function to parse id2label JSON mapping
static std::unordered_map<int, std::string> parseId2Label(const std::string& json_str) {
    std::unordered_map<int, std::string> id2label;
    
    try {
        // Simple JSON parsing for id2label format: {"0": "O", "1": "B-PER", ...}
        // Pattern: "(\d+)"\s*:\s*"([^"]+)"
        std::regex entry_regex("\"(\\d+)\"\\s*:\\s*\"([^\"]+)\"");
        std::smatch match;
        
        std::string::const_iterator search_start(json_str.cbegin());
        while (std::regex_search(search_start, json_str.cend(), match, entry_regex)) {
            int id = std::stoi(match[1]);
            std::string label = match[2];
            id2label[id] = label;
            search_start = match.suffix().first;
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to parse id2label JSON: " << e.what() << std::endl;
    }
    
    return id2label;
}

// Extract entities from BIO-tagged tokens (for ModernBERT and other token classifiers)
static std::vector<core::EntitySpan> extractBioEntities(
    const std::vector<int>& predictions,
    const std::vector<float>& confidences,
    const std::unordered_map<int, std::string>& id2label,
    const std::vector<int>& token_ids
) {
    std::vector<core::EntitySpan> entities;
    
    std::string current_entity_type;
    int current_start = -1;
    float current_confidence = 0.0f;
    int token_count = 0;
    
    for (size_t i = 0; i < predictions.size(); ++i) {
        // Skip special tokens ([CLS], [SEP]) and padding
        if (i == 0 || token_ids[i] == MODERNBERT_SEP || token_ids[i] == MODERNBERT_PAD) {
            // End current entity if any
            if (current_start != -1) {
                core::EntitySpan entity;
                entity.entity_type = current_entity_type;
                entity.start = current_start;
                entity.end = static_cast<int>(i);
                entity.confidence = current_confidence / token_count;
                entities.push_back(entity);
                
                current_start = -1;
                token_count = 0;
            }
            continue;
        }
        
        int pred_id = predictions[i];
        auto label_it = id2label.find(pred_id);
        if (label_it == id2label.end()) continue;
        
        std::string label = label_it->second;
        
        // Parse BIO tags
        if (label == "O") {
            // Outside - end current entity
            if (current_start != -1) {
                core::EntitySpan entity;
                entity.entity_type = current_entity_type;
                entity.start = current_start;
                entity.end = static_cast<int>(i);
                entity.confidence = current_confidence / token_count;
                entities.push_back(entity);
                
                current_start = -1;
                token_count = 0;
            }
        } else if (label.size() >= 2 && label[0] == 'B' && label[1] == '-') {
            // Begin new entity
            if (current_start != -1) {
                // End previous entity
                core::EntitySpan entity;
                entity.entity_type = current_entity_type;
                entity.start = current_start;
                entity.end = static_cast<int>(i);
                entity.confidence = current_confidence / token_count;
                entities.push_back(entity);
            }
            // Start new entity
            current_entity_type = label.substr(2);  // Extract entity type (e.g., "PER" from "B-PER")
            current_start = static_cast<int>(i);
            current_confidence = confidences[i];
            token_count = 1;
        } else if (label.size() >= 2 && label[0] == 'I' && label[1] == '-') {
            // Inside entity - continue current entity
            std::string entity_type = label.substr(2);
            if (current_start != -1 && entity_type == current_entity_type) {
                current_confidence += confidences[i];
                token_count++;
            } else {
                // Type mismatch or no current entity - treat as new entity
                if (current_start != -1) {
                    core::EntitySpan entity;
                    entity.entity_type = current_entity_type;
                    entity.start = current_start;
                    entity.end = static_cast<int>(i);
                    entity.confidence = current_confidence / token_count;
                    entities.push_back(entity);
                }
                current_entity_type = entity_type;
                current_start = static_cast<int>(i);
                current_confidence = confidences[i];
                token_count = 1;
            }
        }
    }
    
    // End final entity if any
    if (current_start != -1) {
        core::EntitySpan entity;
        entity.entity_type = current_entity_type;
        entity.start = current_start;
        entity.end = static_cast<int>(predictions.size());
        entity.confidence = current_confidence / token_count;
        entities.push_back(entity);
    }
    
    return entities;
}

bool TokenClassifier::initialize(
    const std::string& model_path,
    int num_classes,
    const std::string& device
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    try {
        auto& manager = core::ModelManager::getInstance();
        manager.ensureCoreInitialized();
        
        // Create model instance
        model_ = std::make_shared<core::ModelInstance>();
        model_->num_classes = num_classes;
        model_->model_path = model_path;
        
        // Load and compile model (no special config needed for token classification)
        model_->compiled_model = manager.loadModel(model_path, device);
        if (!model_->compiled_model) {
            return false;
        }
        
        // Load tokenizer vocabulary
        std::string model_dir = model_path;
        auto last_slash = model_dir.find_last_of("/\\");
        if (last_slash != std::string::npos) {
            model_dir = model_dir.substr(0, last_slash);
        }
        tokenizer_.loadVocab(model_dir);
        
        std::cout << "OpenVINO token classifier initialized: " << model_path 
                  << " on " << device << " with " << num_classes << " classes" << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize token classifier: " << e.what() << std::endl;
        return false;
    }
}

core::TokenClassificationResult TokenClassifier::classifyTokens(
    const std::string& text,
    const std::string& id2label_json
) {
    core::TokenClassificationResult result;
    
    if (!model_ || !model_->compiled_model) {
        std::cerr << "Token classifier not initialized" << std::endl;
        return result;
    }
    
    try {
        // Parse id2label mapping
        auto id2label = parseId2Label(id2label_json);
        if (id2label.empty()) {
            // Default BIO labels for NER (similar to ModernBERT PII classifier)
            id2label = {
                {0, "O"},
                {1, "B-PER"}, {2, "I-PER"},
                {3, "B-ORG"}, {4, "I-ORG"},
                {5, "B-LOC"}, {6, "I-LOC"},
                {7, "B-MISC"}, {8, "I-MISC"}
            };
        }
        
        // Tokenize input
        std::vector<int> token_ids = tokenizer_.tokenize(text, 512);
        
        if (token_ids.empty()) {
            std::cerr << "Tokenization failed or returned empty" << std::endl;
            return result;
        }
        
        // Create attention mask (1 for real tokens, 0 for padding)
        std::vector<int64_t> attention_mask(token_ids.size());
        for (size_t i = 0; i < token_ids.size(); ++i) {
            attention_mask[i] = (token_ids[i] != MODERNBERT_PAD) ? 1 : 0;
        }
        
        // Convert token_ids to int64 for ModernBERT
        std::vector<int64_t> token_ids_i64(token_ids.begin(), token_ids.end());
        
        // Create input tensors
        ov::Tensor input_ids_tensor(ov::element::i64, {1, token_ids_i64.size()});
        std::memcpy(input_ids_tensor.data<int64_t>(), token_ids_i64.data(), 
                    token_ids_i64.size() * sizeof(int64_t));
        
        ov::Tensor attention_mask_tensor(ov::element::i64, {1, attention_mask.size()});
        std::memcpy(attention_mask_tensor.data<int64_t>(), attention_mask.data(), 
                    attention_mask.size() * sizeof(int64_t));
        
        // Create infer request (thread-safe per-request)
        auto infer_request = model_->compiled_model->create_infer_request();
        
        // Set input tensors
        infer_request.set_input_tensor(0, input_ids_tensor);
        infer_request.set_input_tensor(1, attention_mask_tensor);
        
        // Run inference
        infer_request.infer();
        
        // Get output tensor (logits shape: [batch, seq_len, num_classes])
        auto output_tensor = infer_request.get_output_tensor();
        const float* logits = output_tensor.data<const float>();
        
        auto shape = output_tensor.get_shape();
        size_t seq_len = shape[1];
        size_t num_classes = shape[2];
        
        // Get predictions and confidences
        std::vector<int> predictions;
        std::vector<float> confidences;
        
        for (size_t i = 0; i < seq_len && i < token_ids_i64.size(); ++i) {
            // Skip padding tokens
            if (token_ids_i64[i] == MODERNBERT_PAD) break;
            
            // Find class with maximum logit
            size_t max_class = 0;
            float max_logit = logits[i * num_classes];
            
            for (size_t c = 1; c < num_classes; ++c) {
                float logit = logits[i * num_classes + c];
                if (logit > max_logit) {
                    max_logit = logit;
                    max_class = c;
                }
            }
            
            // Apply softmax to get confidence
            float sum_exp = 0.0f;
            for (size_t c = 0; c < num_classes; ++c) {
                sum_exp += std::exp(logits[i * num_classes + c]);
            }
            float confidence = std::exp(max_logit) / sum_exp;
            
            predictions.push_back(static_cast<int>(max_class));
            confidences.push_back(confidence);
        }
        
        // Extract entities using BIO tagging (ModernBERT-compatible)
        auto entity_spans = extractBioEntities(predictions, confidences, id2label, token_ids);
        
        // Convert EntitySpan to TokenEntity and filter by confidence
        // ModernBERT token classifiers often have lower per-token confidence
        result.entities.clear();
        for (const auto& span : entity_spans) {
            if (span.confidence > 0.3f) {
                core::TokenEntity entity;
                entity.entity_type = span.entity_type;
                entity.start = span.start;
                entity.end = span.end;
                entity.text = span.entity_type;  // Simplified - in full implementation use character offsets
                entity.confidence = span.confidence;
                result.entities.push_back(entity);
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Token classification error: " << e.what() << std::endl;
    }
    
    return result;
}

} // namespace classifiers
} // namespace openvino_sr

