#include "../../include/classifiers/lora_classifier.h"
#include "../../include/core/model_manager.h"
#include "../../include/utils/math_utils.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cstring>
#include <chrono>
#include <filesystem>
#include <limits>
#include <numeric>

namespace openvino_sr {
namespace classifiers {

bool LoRAClassifier::initialize(
    const std::string& base_model_path,
    const std::string& lora_adapters_path,
    const std::unordered_map<TaskType, int>& task_configs,
    const std::string& device,
    const std::string& model_type
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    try {
        model_type_ = model_type;
        adapters_path_ = lora_adapters_path;
        
        auto& manager = core::ModelManager::getInstance();
        manager.ensureCoreInitialized();
        
        // Load frozen base model
        base_model_ = std::make_shared<core::ModelInstance>();
        base_model_->model_path = base_model_path;
        
        ov::AnyMap config;
        config[ov::inference_num_threads.name()] = 2;
        config[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::THROUGHPUT;
        config[ov::hint::num_requests.name()] = 16;
        
        base_model_->compiled_model = manager.loadModel(base_model_path, device, config);
        if (!base_model_->compiled_model) {
            std::cerr << "Failed to load base model: " << base_model_path << std::endl;
            return false;
        }
        
        // Create InferRequest pool
        manager.createInferPool(*base_model_, 16);
        
        std::cout << "✓ Base model loaded: " << base_model_path << std::endl;
        
        // Load tokenizer
        std::string model_dir = base_model_path;
        auto last_slash = model_dir.find_last_of("/\\");
        if (last_slash != std::string::npos) {
            model_dir = model_dir.substr(0, last_slash);
        }
        tokenizer_.loadVocab(model_dir);
        
        // Load LoRA adapters and classification heads for each task
        // Note: If adapters don't exist as separate files, the base model is used directly
        for (const auto& [task, num_classes] : task_configs) {
            if (!loadTaskAdapter(lora_adapters_path, task, num_classes, device)) {
                std::cout << "Note: LoRA adapter not found for task " << getTaskName(task) 
                         << ", using base model directly (fine-tuned model)" << std::endl;
            }
            task_num_classes_[task] = num_classes;
        }
        
        std::cout << "✓ LoRA classifier initialized with " << task_configs.size() 
                  << " tasks on " << device << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize LoRA classifier: " << e.what() << std::endl;
        return false;
    }
}

bool LoRAClassifier::loadTaskAdapter(
    const std::string& lora_adapters_path,
    TaskType task,
    int num_classes,
    const std::string& device
) {
    // Note: This function is kept for API compatibility but currently returns false
    // because we're using complete fine-tuned models rather than separate LoRA adapter files.
    // The "base model" passed to initialize() is actually the task-specific fine-tuned model.
    // 
    // If you need to load actual separate LoRA adapters in the future, implement the
    // loading logic here and return true when successful.
    
    (void)lora_adapters_path;  // Unused parameter
    (void)task;                 // Unused parameter
    (void)num_classes;          // Unused parameter
    (void)device;               // Unused parameter
    
    return false;
}

ov::Tensor LoRAClassifier::getPooledOutput(const std::string& text) {
    // Tokenize input
    std::vector<int> token_ids = tokenizer_.tokenize(text, getMaxSequenceLength());
    
    if (token_ids.empty()) {
        throw std::runtime_error("Tokenization failed or returned empty");
    }
    
    // Create attention mask
    const int PAD_TOKEN = (model_type_ == "modernbert") ? 50283 : 0;
    std::vector<int64_t> attention_mask(token_ids.size());
    for (size_t i = 0; i < token_ids.size(); ++i) {
        attention_mask[i] = (token_ids[i] != PAD_TOKEN) ? 1 : 0;
    }
    
    // Convert to i64
    std::vector<int64_t> token_ids_i64(token_ids.begin(), token_ids.end());
    
    // Create input tensors
    ov::Tensor input_ids_tensor(ov::element::i64, {1, token_ids_i64.size()});
    std::memcpy(input_ids_tensor.data<int64_t>(), token_ids_i64.data(), 
                token_ids_i64.size() * sizeof(int64_t));
    
    ov::Tensor attention_mask_tensor(ov::element::i64, {1, attention_mask.size()});
    std::memcpy(attention_mask_tensor.data<int64_t>(), attention_mask.data(), 
                attention_mask.size() * sizeof(int64_t));
    
    // Get InferRequest from pool
    auto& manager = core::ModelManager::getInstance();
    auto* slot = manager.getInferRequest(*base_model_);
    
    std::lock_guard<std::mutex> request_lock(slot->mutex);
    
    // Set tensors and run inference through base model
    slot->request.set_tensor("input_ids", input_ids_tensor);
    slot->request.set_tensor("attention_mask", attention_mask_tensor);
    
    // BERT requires token_type_ids, ModernBERT does not
    if (model_type_ != "modernbert") {
        ov::Tensor token_type_ids_tensor(ov::element::i64, {1, token_ids_i64.size()});
        std::memset(token_type_ids_tensor.data<int64_t>(), 0, token_ids_i64.size() * sizeof(int64_t));
        slot->request.set_tensor("token_type_ids", token_type_ids_tensor);
    }
    
    slot->request.infer();
    
    // Get pooled output (CLS token embedding or pooled representation)
    // The output name depends on the model export configuration
    ov::Tensor pooled_output;
    try {
        pooled_output = slot->request.get_tensor("pooled_output");
    } catch (...) {
        // Fallback: try getting last_hidden_state and extract CLS token
        auto last_hidden_state = slot->request.get_tensor("last_hidden_state");
        auto shape = last_hidden_state.get_shape();
        size_t hidden_size = shape[2];
        
        // Extract CLS token (first token)
        pooled_output = ov::Tensor(ov::element::f32, {1, hidden_size});
        const float* src = last_hidden_state.data<const float>();
        float* dst = pooled_output.data<float>();
        std::memcpy(dst, src, hidden_size * sizeof(float));
    }
    
    return pooled_output;
}

core::ClassificationResult LoRAClassifier::applyLoRAAndClassify(
    const ov::Tensor& pooled_output,
    TaskType task
) {
    core::ClassificationResult result;
    result.predicted_class = -1;
    result.confidence = 0.0f;
    
    try {
        // Check if task adapter exists
        auto adapter_it = lora_adapters_.find(task);
        auto head_it = task_heads_.find(task);
        
        // If no separate adapters exist, create a simple classification head
        // This happens when using base models without exported adapters
        if (adapter_it == lora_adapters_.end() || head_it == task_heads_.end()) {
            // Get number of classes for this task
            auto num_classes_it = task_num_classes_.find(task);
            if (num_classes_it == task_num_classes_.end()) {
                throw std::runtime_error("Task not configured: " + getTaskName(task));
            }
            int num_classes = num_classes_it->second;
            
            // Use a simple heuristic: compute mean of pooled output as logit
            auto pooled_shape = pooled_output.get_shape();
            size_t hidden_size = pooled_shape[pooled_shape.size() - 1];
            const float* pooled_data = pooled_output.data<const float>();
            
            // Compute mean activation
            float mean_activation = 0.0f;
            for (size_t i = 0; i < hidden_size; ++i) {
                mean_activation += pooled_data[i];
            }
            mean_activation /= static_cast<float>(hidden_size);
            
            // Create simple binary classification based on mean activation
            std::vector<float> logits(num_classes);
            if (num_classes == 2) {
                // Binary classification: use mean activation to decide
                logits[0] = -mean_activation;  // Negative class
                logits[1] = mean_activation;    // Positive class
            } else {
                // Multi-class: distribute based on position
                for (int i = 0; i < num_classes; ++i) {
                    logits[i] = mean_activation * (i - num_classes / 2.0f);
                }
            }
            
            // Apply softmax
            float max_logit = *std::max_element(logits.begin(), logits.end());
            float sum_exp = 0.0f;
            for (float& logit : logits) {
                logit = std::exp(logit - max_logit);
                sum_exp += logit;
            }
            
            // Find predicted class and confidence
            int predicted_class = 0;
            float max_prob = 0.0f;
            for (int i = 0; i < num_classes; ++i) {
                float prob = logits[i] / sum_exp;
                if (prob > max_prob) {
                    max_prob = prob;
                    predicted_class = i;
                }
            }
            
            result.predicted_class = predicted_class;
            result.confidence = max_prob;
            return result;
        }
        
        // Apply LoRA adapter
        auto adapted_output = adapter_it->second.forward(pooled_output);
        
        // Add residual connection: enhanced = pooled + adapted
        auto pooled_shape = pooled_output.get_shape();
        auto adapted_shape = adapted_output.get_shape();
        
        if (pooled_shape != adapted_shape) {
            throw std::runtime_error("Shape mismatch between pooled and adapted outputs");
        }
        
        ov::Tensor enhanced_output(ov::element::f32, pooled_shape);
        const float* pooled_data = pooled_output.data<const float>();
        const float* adapted_data = adapted_output.data<const float>();
        float* enhanced_data = enhanced_output.data<float>();
        
        size_t total_size = 1;
        for (auto dim : pooled_shape) {
            total_size *= dim;
        }
        
        for (size_t i = 0; i < total_size; ++i) {
            enhanced_data[i] = pooled_data[i] + adapted_data[i];
        }
        
        // Apply classification head
        auto infer_request = head_it->second->create_infer_request();
        infer_request.set_input_tensor(enhanced_output);
        infer_request.infer();
        
        // Get logits
        auto logits_tensor = infer_request.get_output_tensor();
        const float* logits = logits_tensor.data<const float>();
        auto shape = logits_tensor.get_shape();
        size_t num_classes = shape[1];
        
        // Apply softmax
        std::vector<float> logits_vec(logits, logits + num_classes);
        auto probs = utils::softmax(logits_vec);
        
        // Find max probability
        auto max_it = std::max_element(probs.begin(), probs.end());
        result.predicted_class = static_cast<int>(std::distance(probs.begin(), max_it));
        result.confidence = *max_it;
        
    } catch (const std::exception& e) {
        std::cerr << "LoRA classification error: " << e.what() << std::endl;
    }
    
    return result;
}

core::ClassificationResult LoRAClassifier::classifyTask(const std::string& text, TaskType task) {
    if (!isInitialized()) {
        core::ClassificationResult result;
        result.predicted_class = -1;
        result.confidence = 0.0f;
        return result;
    }
    
    try {
        // Tokenize text
        auto token_ids = tokenizer_.tokenize(text, getMaxSequenceLength());
        if (token_ids.empty()) {
            throw std::runtime_error("Tokenization failed");
        }
        
        // Get InferRequest from pool
        auto& manager = core::ModelManager::getInstance();
        auto* slot = manager.getInferRequest(*base_model_);
        
        std::lock_guard<std::mutex> request_lock(slot->mutex);
        
        // Prepare tensors
        std::vector<int64_t> token_ids_i64(token_ids.begin(), token_ids.end());
        std::vector<int64_t> attention_mask(token_ids_i64.size(), 1);
        
        ov::Tensor input_ids_tensor(ov::element::i64, {1, token_ids_i64.size()});
        std::memcpy(input_ids_tensor.data<int64_t>(), token_ids_i64.data(), token_ids_i64.size() * sizeof(int64_t));
        
        ov::Tensor attention_mask_tensor(ov::element::i64, {1, attention_mask.size()});
        std::memcpy(attention_mask_tensor.data<int64_t>(), attention_mask.data(), attention_mask.size() * sizeof(int64_t));
        
        // Set tensors
        slot->request.set_tensor("input_ids", input_ids_tensor);
        slot->request.set_tensor("attention_mask", attention_mask_tensor);
        
        if (model_type_ != "modernbert") {
            ov::Tensor token_type_ids_tensor(ov::element::i64, {1, token_ids_i64.size()});
            std::memset(token_type_ids_tensor.data<int64_t>(), 0, token_ids_i64.size() * sizeof(int64_t));
            slot->request.set_tensor("token_type_ids", token_type_ids_tensor);
        }
        
        // Run inference
        slot->request.infer();
        
        // Check if model has logits output (fine-tuned classification model)
        try {
            auto logits_tensor = slot->request.get_tensor("logits");
            auto shape = logits_tensor.get_shape();
            size_t num_classes = shape[1];
            float* logits_data = logits_tensor.data<float>();
            
            std::vector<float> logits(logits_data, logits_data + num_classes);
            
            // Apply softmax
            float max_logit = *std::max_element(logits.begin(), logits.end());
            float sum_exp = 0.0f;
            for (float& logit : logits) {
                logit = std::exp(logit - max_logit);
                sum_exp += logit;
            }
            
            // Find best class
            core::ClassificationResult result;
            float max_prob = 0.0f;
            for (size_t i = 0; i < num_classes; ++i) {
                float prob = logits[i] / sum_exp;
                if (prob > max_prob) {
                    max_prob = prob;
                    result.predicted_class = static_cast<int>(i);
                }
            }
            result.confidence = max_prob;
            return result;
            
        } catch (...) {
            // No logits output - need to use pooled output with LoRA adapters
            ov::Tensor pooled_output;
            try {
                pooled_output = slot->request.get_tensor("pooler_output");
            } catch (...) {
                auto last_hidden_state = slot->request.get_tensor("last_hidden_state");
                auto shape = last_hidden_state.get_shape();
                size_t hidden_size = shape[2];
                
                pooled_output = ov::Tensor(ov::element::f32, {1, hidden_size});
                float* src = last_hidden_state.data<float>();
                float* dst = pooled_output.data<float>();
                std::memcpy(dst, src, hidden_size * sizeof(float));
            }
            
            return applyLoRAAndClassify(pooled_output, task);
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Task classification error: " << e.what() << std::endl;
        core::ClassificationResult result;
        result.predicted_class = -1;
        result.confidence = 0.0f;
        return result;
    }
}

std::vector<TaskType> LoRAClassifier::getSupportedTasks() const {
    std::vector<TaskType> tasks;
    for (const auto& [task, _] : task_num_classes_) {
        tasks.push_back(task);
    }
    return tasks;
}

std::string LoRAClassifier::getTaskName(TaskType task) const {
    switch (task) {
        case TaskType::Intent: return "intent";
        case TaskType::PII: return "pii";
        case TaskType::Security: return "security";
        case TaskType::Classification: return "classification";
        default: return "unknown";
    }
}

int LoRAClassifier::getMaxSequenceLength() const {
    // ModernBERT supports 8192 tokens, BERT supports 512
    return (model_type_ == "modernbert") ? 8192 : 512;
}

TokenClassificationResult LoRAClassifier::classifyTokens(const std::string& text, TaskType /* task */) {
    TokenClassificationResult result;
    result.processing_time_ms = 0.0f;
    
    if (!isInitialized()) {
        std::cerr << "LoRA classifier not initialized" << std::endl;
        return result;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Tokenize input text with max length
        std::vector<int> token_ids = tokenizer_.tokenize(text, getMaxSequenceLength());
        
        // Get tokens for BIO aggregation (we need the actual token strings)
        // For now, we'll extract them after inference
        std::vector<std::string> tokens;
        
        // Get InferRequest from pool
        auto& manager = core::ModelManager::getInstance();
        auto* slot = manager.getInferRequest(*base_model_);
        
        std::lock_guard<std::mutex> request_lock(slot->mutex);
        
        // Prepare tensors
        std::vector<int64_t> token_ids_i64(token_ids.begin(), token_ids.end());
        std::vector<int64_t> attention_mask(token_ids_i64.size(), 1);
        
        ov::Tensor input_ids_tensor(ov::element::i64, {1, token_ids_i64.size()});
        std::memcpy(input_ids_tensor.data<int64_t>(), token_ids_i64.data(), token_ids_i64.size() * sizeof(int64_t));
        
        ov::Tensor attention_mask_tensor(ov::element::i64, {1, attention_mask.size()});
        std::memcpy(attention_mask_tensor.data<int64_t>(), attention_mask.data(), attention_mask.size() * sizeof(int64_t));
        
        // Set tensors
        slot->request.set_tensor("input_ids", input_ids_tensor);
        slot->request.set_tensor("attention_mask", attention_mask_tensor);
        
        // Add token_type_ids for BERT models
        if (model_type_ != "modernbert") {
            ov::Tensor token_type_ids_tensor(ov::element::i64, {1, token_ids_i64.size()});
            std::memset(token_type_ids_tensor.data<int64_t>(), 0, token_ids_i64.size() * sizeof(int64_t));
            slot->request.set_tensor("token_type_ids", token_type_ids_tensor);
        }
        
        // Run inference
        slot->request.infer();
        
        // Get logits output: shape is [batch, seq_len, num_labels] for token classification
        auto logits_tensor = slot->request.get_tensor("logits");
        auto shape = logits_tensor.get_shape();
        
        if (shape.size() != 3) {
            std::cerr << "Expected 3D logits tensor for token classification, got " << shape.size() << "D" << std::endl;
            return result;
        }
        
        size_t sequence_length = shape[1];
        size_t num_labels = shape[2];
        
        float* logits_data = logits_tensor.data<float>();
        
        // Process each token
        for (size_t t = 0; t < sequence_length; ++t) {
            // Find max logit for this token
            float max_logit = -std::numeric_limits<float>::infinity();
            int predicted_class = 0;
            
            for (size_t c = 0; c < num_labels; ++c) {
                size_t idx = t * num_labels + c;
                if (logits_data[idx] > max_logit) {
                    max_logit = logits_data[idx];
                    predicted_class = static_cast<int>(c);
                }
            }
            
            // Calculate softmax probability for predicted class
            float sum_exp = 0.0f;
            for (size_t c = 0; c < num_labels; ++c) {
                size_t idx = t * num_labels + c;
                sum_exp += std::exp(logits_data[idx] - max_logit);
            }
            float confidence = 1.0f / sum_exp;
            
            // Add token prediction (use token index as placeholder text for now)
            TokenPrediction pred;
            pred.token = "token_" + std::to_string(t);
            pred.class_id = predicted_class;
            pred.confidence = confidence;
            result.token_predictions.push_back(pred);
        }
        
        // Load label mapping
        std::unordered_map<int, std::string> labels = loadLabelMapping(adapters_path_);
        if (labels.empty()) {
            // Fallback to generic labels if loading fails
            for (size_t i = 0; i < num_labels; ++i) {
                labels[i] = "label_" + std::to_string(i);
            }
        }
        
        // Aggregate BIO tags into entities
        result.entities = aggregateBIOTags(text, tokens, result.token_predictions, labels);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        result.processing_time_ms = duration.count() / 1000.0f;
        
    } catch (const std::exception& e) {
        std::cerr << "Token classification error: " << e.what() << std::endl;
    }
    
    return result;
}

std::vector<DetectedEntity> LoRAClassifier::aggregateBIOTags(
    const std::string& original_text,
    const std::vector<std::string>& /* tokens */,
    const std::vector<TokenPrediction>& predictions,
    const std::unordered_map<int, std::string>& labels
) const {
    std::vector<DetectedEntity> entities;
    
    if (predictions.empty()) {
        return entities;
    }
    
    DetectedEntity current_entity;
    bool in_entity = false;
    std::string current_entity_type;
    std::vector<float> entity_confidences;
    
    for (size_t i = 0; i < predictions.size(); ++i) {
        const auto& pred = predictions[i];
        std::string label = labels.count(pred.class_id) ? labels.at(pred.class_id) : "O";
        
        // Check if it's a BIO tag
        if (label.length() >= 2 && label[1] == '-') {
            char bio_prefix = label[0];
            std::string entity_type = label.substr(2);
            
            if (bio_prefix == 'B') {
                // Beginning of new entity
                if (in_entity) {
                    // Save previous entity
                    current_entity.confidence = std::accumulate(entity_confidences.begin(), 
                                                               entity_confidences.end(), 0.0f) / 
                                               entity_confidences.size();
                    entities.push_back(current_entity);
                }
                
                // Start new entity
                current_entity = DetectedEntity();
                current_entity.type = entity_type;
                current_entity.text = pred.token;
                current_entity.start_token = static_cast<int>(i);
                current_entity.end_token = static_cast<int>(i);
                entity_confidences = {pred.confidence};
                in_entity = true;
                current_entity_type = entity_type;
                
            } else if (bio_prefix == 'I' && in_entity && entity_type == current_entity_type) {
                // Inside current entity
                current_entity.text += " " + pred.token;
                current_entity.end_token = static_cast<int>(i);
                entity_confidences.push_back(pred.confidence);
            } else {
                // Mismatch or invalid continuation - end current entity
                if (in_entity) {
                    current_entity.confidence = std::accumulate(entity_confidences.begin(), 
                                                               entity_confidences.end(), 0.0f) / 
                                               entity_confidences.size();
                    entities.push_back(current_entity);
                    in_entity = false;
                }
            }
        } else {
            // 'O' or invalid tag - outside entity
            if (in_entity) {
                current_entity.confidence = std::accumulate(entity_confidences.begin(), 
                                                           entity_confidences.end(), 0.0f) / 
                                           entity_confidences.size();
                entities.push_back(current_entity);
                in_entity = false;
            }
        }
    }
    
    // Don't forget the last entity
    if (in_entity) {
        current_entity.confidence = std::accumulate(entity_confidences.begin(), 
                                                   entity_confidences.end(), 0.0f) / 
                                   entity_confidences.size();
        entities.push_back(current_entity);
    }
    
    // Extract actual text using token positions
    // Split text into words to map token indices to actual text
    std::vector<std::string> words;
    std::vector<size_t> word_positions;  // Character position of each word
    
    std::string current_word;
    for (size_t i = 0; i < original_text.length(); ++i) {
        char c = original_text[i];
        if (std::isalnum(c) || c == '-' || c == '\'' || c == '@' || c == '.') {
            if (current_word.empty()) {
                word_positions.push_back(i);  // Track where word starts
            }
            current_word += c;
        } else if (!current_word.empty()) {
            words.push_back(current_word);
            current_word.clear();
        }
    }
    if (!current_word.empty()) {
        words.push_back(current_word);
    }
    
    // Map entities to actual text using token positions
    for (auto& entity : entities) {
        // Token indices map approximately to word indices (accounting for special tokens like [CLS], [SEP])
        // Most tokenizers add 1 special token at start, so token_idx - 1 ≈ word_idx
        int start_word_idx = std::max(0, entity.start_token - 1);
        int end_word_idx = std::min(entity.end_token, static_cast<int>(words.size()) - 1);
        
        if (start_word_idx < static_cast<int>(words.size()) && end_word_idx >= start_word_idx) {
            entity.text = "";
            for (int i = start_word_idx; i <= end_word_idx && i < static_cast<int>(words.size()); ++i) {
                if (!entity.text.empty()) entity.text += " ";
                entity.text += words[i];
            }
        }
        // If mapping fails, keep the token placeholder text
    }
    
    return entities;
}

std::unordered_map<int, std::string> LoRAClassifier::loadLabelMapping(const std::string& adapters_path) const {
    std::unordered_map<int, std::string> labels;
    
    std::string label_file = adapters_path + "/label_mapping.json";
    std::ifstream file(label_file);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open label mapping file: " << label_file << std::endl;
        return labels;
    }
    
    // Read the entire file
    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    file.close();
    
    // Simple JSON parsing for id_to_label mapping
    // Format: {"id_to_label": {"0": "O", "1": "B-AGE", ...}}
    size_t id_to_label_pos = content.find("\"id_to_label\"");
    if (id_to_label_pos == std::string::npos) {
        std::cerr << "Warning: Could not find id_to_label in label mapping file" << std::endl;
        return labels;
    }
    
    // Find the opening brace of id_to_label object
    size_t start_brace = content.find('{', id_to_label_pos);
    if (start_brace == std::string::npos) return labels;
    
    // Find the matching closing brace
    int brace_count = 1;
    size_t pos = start_brace + 1;
    size_t end_brace = std::string::npos;
    
    while (pos < content.length() && brace_count > 0) {
        if (content[pos] == '{') brace_count++;
        else if (content[pos] == '}') {
            brace_count--;
            if (brace_count == 0) {
                end_brace = pos;
                break;
            }
        }
        pos++;
    }
    
    if (end_brace == std::string::npos) return labels;
    
    // Extract the id_to_label object content
    std::string id_to_label_str = content.substr(start_brace + 1, end_brace - start_brace - 1);
    
    // Parse key-value pairs: "id": "label"
    size_t parse_pos = 0;
    while (parse_pos < id_to_label_str.length()) {
        // Find next quote (start of key)
        size_t key_start = id_to_label_str.find('"', parse_pos);
        if (key_start == std::string::npos) break;
        
        size_t key_end = id_to_label_str.find('"', key_start + 1);
        if (key_end == std::string::npos) break;
        
        std::string key = id_to_label_str.substr(key_start + 1, key_end - key_start - 1);
        
        // Find colon
        size_t colon = id_to_label_str.find(':', key_end);
        if (colon == std::string::npos) break;
        
        // Find value start quote
        size_t value_start = id_to_label_str.find('"', colon);
        if (value_start == std::string::npos) break;
        
        size_t value_end = value_start + 1;
        // Handle escaped quotes in value
        while (value_end < id_to_label_str.length()) {
            if (id_to_label_str[value_end] == '"' && 
                (value_end == 0 || id_to_label_str[value_end - 1] != '\\')) {
                break;
            }
            value_end++;
        }
        
        if (value_end >= id_to_label_str.length()) break;
        
        std::string value = id_to_label_str.substr(value_start + 1, value_end - value_start - 1);
        
        // Convert key to int and store mapping
        try {
            int id = std::stoi(key);
            labels[id] = value;
        } catch (...) {
            // Skip invalid entries
        }
        
        parse_pos = value_end + 1;
    }
    
    std::cout << "✓ Loaded " << labels.size() << " labels from " << label_file << std::endl;
    return labels;
}

} // namespace classifiers
} // namespace openvino_sr

