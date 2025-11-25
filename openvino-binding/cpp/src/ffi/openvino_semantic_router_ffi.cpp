/**
 * Foreign Function Interface (FFI) Layer for OpenVINO Semantic Router
 * 
 * This file provides C-compatible wrappers around the C++ implementation.
 * All functions are exposed with C linkage for Go CGO bindings.
 */

#include "../../include/openvino_semantic_router.h"
#include "../../include/core/model_manager.h"
#include "../../include/classifiers/text_classifier.h"
#include "../../include/classifiers/token_classifier.h"
#include "../../include/classifiers/lora_classifier.h"
#include "../../include/embeddings/embedding_generator.h"
#include "../../include/utils/preprocessing.h"

#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <cstring>

using namespace openvino_sr;

// ================================================================================================
// GLOBAL INSTANCES (Singleton Pattern)
// ================================================================================================

static std::unique_ptr<classifiers::TextClassifier> g_text_classifier;
static std::unique_ptr<classifiers::TokenClassifier> g_token_classifier;
static std::unique_ptr<embeddings::EmbeddingGenerator> g_embedding_generator;
static std::unique_ptr<embeddings::EmbeddingGenerator> g_similarity_generator;
static std::unique_ptr<classifiers::LoRAClassifier> g_bert_lora_classifier;
static std::unique_ptr<classifiers::LoRAClassifier> g_modernbert_lora_classifier;

// ================================================================================================
// INITIALIZATION FUNCTIONS
// ================================================================================================

bool ov_init_similarity_model(const char* model_path, const char* device) {
    try {
        if (!g_similarity_generator) {
            g_similarity_generator = std::make_unique<embeddings::EmbeddingGenerator>();
        }
        return g_similarity_generator->initialize(model_path, device);
    } catch (const std::exception& e) {
        std::cerr << "Error initializing similarity model: " << e.what() << std::endl;
        return false;
    }
}

bool ov_is_similarity_model_initialized() {
    return g_similarity_generator != nullptr;
}

bool ov_init_classifier(const char* model_path, int num_classes, const char* device) {
    try {
        if (!g_text_classifier) {
            g_text_classifier = std::make_unique<classifiers::TextClassifier>();
        }
        return g_text_classifier->initialize(model_path, num_classes, device);
    } catch (const std::exception& e) {
        std::cerr << "Error initializing classifier: " << e.what() << std::endl;
        return false;
    }
}

bool ov_init_embedding_model(const char* model_path, const char* device) {
    try {
        if (!g_embedding_generator) {
            g_embedding_generator = std::make_unique<embeddings::EmbeddingGenerator>();
        }
        return g_embedding_generator->initialize(model_path, device);
    } catch (const std::exception& e) {
        std::cerr << "Error initializing embedding model: " << e.what() << std::endl;
        return false;
    }
}

bool ov_is_embedding_model_initialized() {
    return g_embedding_generator != nullptr;
}

bool ov_init_token_classifier(const char* model_path, int num_classes, const char* device) {
    try {
        if (!g_token_classifier) {
            g_token_classifier = std::make_unique<classifiers::TokenClassifier>();
        }
        return g_token_classifier->initialize(model_path, num_classes, device);
    } catch (const std::exception& e) {
        std::cerr << "Error initializing token classifier: " << e.what() << std::endl;
        return false;
    }
}

// ================================================================================================
// TOKENIZATION FUNCTIONS
// ================================================================================================

OVTokenizationResult ov_tokenize_text(const char* text, int max_length) {
    OVTokenizationResult result{};
    result.error = true;
    
    // This is a simple wrapper - full tokenization is handled internally
    // For debugging/testing purposes only
    result.token_count = 0;
    result.token_ids = nullptr;
    result.tokens = nullptr;
    result.error = false;
    
    return result;
}

void ov_free_tokenization_result(OVTokenizationResult result) {
    if (result.token_ids) {
        delete[] result.token_ids;
    }
    if (result.tokens) {
        for (int i = 0; i < result.token_count; ++i) {
            if (result.tokens[i]) {
                delete[] result.tokens[i];
            }
        }
        delete[] result.tokens;
    }
}

// ================================================================================================
// EMBEDDING FUNCTIONS
// ================================================================================================

OVEmbeddingResult ov_get_text_embedding(const char* text, int max_length) {
    OVEmbeddingResult result{};
    result.error = true;
    
    if (!g_embedding_generator) {
        std::cerr << "Embedding model not initialized" << std::endl;
        return result;
    }
    
    try {
        auto start = std::chrono::high_resolution_clock::now();
        
        std::string text_str(text);
        auto embedding = g_embedding_generator->generateEmbedding(text_str, max_length);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        result.processing_time_ms = duration.count() / 1000.0f;
        
        if (embedding.empty()) {
            return result;
        }
        
        result.length = static_cast<int>(embedding.size());
        result.data = new float[result.length];
        std::copy(embedding.begin(), embedding.end(), result.data);
        result.error = false;
        
    } catch (const std::exception& e) {
        std::cerr << "Embedding error: " << e.what() << std::endl;
    }
    
    return result;
}

void ov_free_embedding(float* data, int /* length */) {
    if (data) {
        delete[] data;
    }
}

// ================================================================================================
// SIMILARITY FUNCTIONS
// ================================================================================================

float ov_calculate_similarity(const char* text1, const char* text2, int max_length) {
    auto* generator = g_similarity_generator ? g_similarity_generator.get() : g_embedding_generator.get();
    if (!generator) {
        std::cerr << "No model initialized for similarity calculation" << std::endl;
        return -1.0f;
    }
    
    return generator->computeSimilarity(text1, text2, max_length);
}

OVSimilarityResult ov_find_most_similar(const char* query, const char** candidates, 
                                         int num_candidates, int max_length) {
    OVSimilarityResult result{-1, -1.0f};
    
    auto* generator = g_similarity_generator ? g_similarity_generator.get() : g_embedding_generator.get();
    if (!generator) {
        std::cerr << "No model initialized for similarity search" << std::endl;
        return result;
    }
    
    try {
        std::vector<std::string> candidates_vec;
        for (int i = 0; i < num_candidates; ++i) {
            candidates_vec.push_back(candidates[i]);
        }
        
        auto cpp_result = generator->findMostSimilar(query, candidates_vec, max_length);
        result.index = cpp_result.index;
        result.score = cpp_result.score;
        
    } catch (const std::exception& e) {
        std::cerr << "Find most similar error: " << e.what() << std::endl;
    }
    
    return result;
}

int ov_calculate_embedding_similarity(const char* text1, const char* text2, 
                                       int max_length, OVEmbeddingSimilarityResult* result) {
    if (!result) {
        return -1;
    }
    
    result->error = true;
    
    try {
        auto start = std::chrono::high_resolution_clock::now();
        
        float similarity = ov_calculate_similarity(text1, text2, max_length);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        result->similarity = similarity;
        result->processing_time_ms = duration.count() / 1000.0f;
        result->error = (similarity < -0.5f);
        
        return result->error ? -1 : 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Embedding similarity error: " << e.what() << std::endl;
        return -1;
    }
}

int ov_calculate_similarity_batch(const char* query, const char** candidates, 
                                   int num_candidates, int top_k, int max_length,
                                   OVBatchSimilarityResult* result) {
    if (!result) {
        return -1;
    }
    
    result->error = true;
    result->matches = nullptr;
    result->num_matches = 0;
    
    auto* generator = g_similarity_generator ? g_similarity_generator.get() : g_embedding_generator.get();
    if (!generator) {
        std::cerr << "No model initialized for batch similarity" << std::endl;
        return -1;
    }
    
    if (num_candidates == 0) {
        return -1;
    }
    
    try {
        auto start = std::chrono::high_resolution_clock::now();
        
        std::vector<std::string> candidates_vec;
        for (int i = 0; i < num_candidates; ++i) {
            candidates_vec.push_back(candidates[i]);
        }
        
        auto matches = generator->findTopKSimilar(query, candidates_vec, top_k, max_length);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        result->processing_time_ms = duration.count() / 1000.0f;
        
        result->num_matches = static_cast<int>(matches.size());
        result->matches = new OVSimilarityMatch[result->num_matches];
        for (size_t i = 0; i < matches.size(); ++i) {
            result->matches[i].index = matches[i].index;
            result->matches[i].similarity = matches[i].similarity;
        }
        
        result->error = false;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Batch similarity error: " << e.what() << std::endl;
        return -1;
    }
}

void ov_free_batch_similarity_result(OVBatchSimilarityResult* result) {
    if (result && result->matches) {
        delete[] result->matches;
        result->matches = nullptr;
        result->num_matches = 0;
    }
}

// ================================================================================================
// CLASSIFICATION FUNCTIONS
// ================================================================================================

OVClassificationResult ov_classify_text(const char* text) {
    OVClassificationResult result{};
    result.predicted_class = -1;
    result.confidence = 0.0f;
    
    if (!g_text_classifier) {
        std::cerr << "Classifier not initialized" << std::endl;
        return result;
    }
    
    try {
        auto cpp_result = g_text_classifier->classify(text);
        result.predicted_class = cpp_result.predicted_class;
        result.confidence = cpp_result.confidence;
        
    } catch (const std::exception& e) {
        std::cerr << "Classification error: " << e.what() << std::endl;
    }
    
    return result;
}

OVClassificationResultWithProbs ov_classify_text_with_probabilities(const char* text) {
    OVClassificationResultWithProbs result{};
    result.predicted_class = -1;
    result.confidence = 0.0f;
    result.probabilities = nullptr;
    result.num_classes = 0;
    
    if (!g_text_classifier) {
        std::cerr << "Classifier not initialized" << std::endl;
        return result;
    }
    
    try {
        auto cpp_result = g_text_classifier->classifyWithProbabilities(text);
        result.predicted_class = cpp_result.predicted_class;
        result.confidence = cpp_result.confidence;
        result.num_classes = static_cast<int>(cpp_result.probabilities.size());
        result.probabilities = new float[result.num_classes];
        std::copy(cpp_result.probabilities.begin(), cpp_result.probabilities.end(), result.probabilities);
        
    } catch (const std::exception& e) {
        std::cerr << "Classification with probabilities error: " << e.what() << std::endl;
    }
    
    return result;
}

void ov_free_probabilities(float* probabilities, int /* num_classes */) {
    if (probabilities) {
        delete[] probabilities;
    }
}

// ================================================================================================
// TOKEN CLASSIFICATION FUNCTIONS
// ================================================================================================

OVTokenClassificationResult ov_classify_tokens(const char* text, const char* id2label_json) {
    OVTokenClassificationResult result{};
    result.entities = nullptr;
    result.num_entities = 0;
    
    if (!g_token_classifier) {
        std::cerr << "Token classifier not initialized" << std::endl;
        result.num_entities = -1;
        return result;
    }
    
    try {
        std::string text_str(text);
        std::string json_str(id2label_json ? id2label_json : "{}");
        
        auto cpp_result = g_token_classifier->classifyTokens(text_str, json_str);
        
        if (!cpp_result.entities.empty()) {
            result.num_entities = static_cast<int>(cpp_result.entities.size());
            result.entities = new OVTokenEntity[result.num_entities];
            
            for (size_t i = 0; i < cpp_result.entities.size(); ++i) {
                const auto& entity = cpp_result.entities[i];
                
                result.entities[i].entity_type = utils::strDup(entity.entity_type.c_str());
                result.entities[i].start = entity.start;
                result.entities[i].end = entity.end;
                result.entities[i].text = utils::strDup(entity.entity_type.c_str());  // Simplified
                result.entities[i].confidence = entity.confidence;
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Token classification error: " << e.what() << std::endl;
        result.num_entities = -1;
    }
    
    return result;
}

void ov_free_token_result(OVTokenClassificationResult result) {
    if (result.entities) {
        for (int i = 0; i < result.num_entities; ++i) {
            if (result.entities[i].entity_type) {
                delete[] result.entities[i].entity_type;
            }
            if (result.entities[i].text) {
                delete[] result.entities[i].text;
            }
        }
        delete[] result.entities;
    }
}

// ================================================================================================
// UTILITY FUNCTIONS
// ================================================================================================

void ov_free_cstring(char* s) {
    if (s) {
        delete[] s;
    }
}

const char* ov_get_version() {
    static std::string version;
    try {
        auto& manager = core::ModelManager::getInstance();
        manager.ensureCoreInitialized();
        version = manager.getCore().get_versions("CPU").begin()->second.buildNumber;
        return version.c_str();
    } catch (...) {
        return "unknown";
    }
}

char* ov_get_available_devices() {
    try {
        auto& manager = core::ModelManager::getInstance();
        manager.ensureCoreInitialized();
        auto devices = manager.getCore().get_available_devices();
        
        std::string devices_str;
        for (size_t i = 0; i < devices.size(); ++i) {
            devices_str += devices[i];
            if (i < devices.size() - 1) {
                devices_str += ",";
            }
        }
        
        char* result = new char[devices_str.length() + 1];
        std::strcpy(result, devices_str.c_str());
        return result;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to get available devices: " << e.what() << std::endl;
        return nullptr;
    }
}

// ================================================================================================
// MODERNBERT SUPPORT (Convenience Aliases)
// ================================================================================================

bool ov_init_modernbert_embedding(const char* model_path, const char* device) {
    std::cout << "Initializing ModernBERT embedding model (optimized BERT)..." << std::endl;
    return ov_init_embedding_model(model_path, device);
}

bool ov_is_modernbert_embedding_initialized() {
    return ov_is_embedding_model_initialized();
}

bool ov_init_modernbert_classifier(const char* model_path, int num_classes, const char* device) {
    std::cout << "Initializing ModernBERT classifier model (optimized BERT)..." << std::endl;
    return ov_init_classifier(model_path, num_classes, device);
}

bool ov_is_modernbert_classifier_initialized() {
    return g_text_classifier != nullptr;
}

bool ov_init_modernbert_token_classifier(const char* model_path, int num_classes, const char* device) {
    std::cout << "Initializing ModernBERT token classifier (optimized BERT with BIO tagging)..." << std::endl;
    return ov_init_token_classifier(model_path, num_classes, device);
}

bool ov_is_modernbert_token_classifier_initialized() {
    return g_token_classifier != nullptr;
}

OVClassificationResult ov_classify_modernbert(const char* text) {
    return ov_classify_text(text);
}

OVTokenClassificationResult ov_classify_modernbert_tokens(const char* text, const char* id2label_json) {
    return ov_classify_tokens(text, id2label_json);
}

OVEmbeddingResult ov_get_modernbert_embedding(const char* text, int max_length) {
    return ov_get_text_embedding(text, max_length);
}

OVClassificationResultWithProbs ov_classify_modernbert_text_with_probabilities(const char* text) {
    return ov_classify_text_with_probabilities(text);
}

// ================================================================================================
// LORA ADAPTER SUPPORT (BERT AND MODERNBERT)
// ================================================================================================

bool ov_init_bert_lora_classifier(
    const char* base_model_path,
    const char* lora_adapters_path,
    const char* device
) {
    try {
        // Validate input parameters
        if (!base_model_path || !lora_adapters_path || !device ||
            strlen(base_model_path) == 0 || strlen(lora_adapters_path) == 0) {
            std::cerr << "Error: Invalid input parameters (empty or null)" << std::endl;
            return false;
        }
        
        // Check if model file exists
        if (!std::filesystem::exists(base_model_path)) {
            std::cerr << "Error: Model file not found: " << base_model_path << std::endl;
            return false;
        }
        
        if (!g_bert_lora_classifier) {
            g_bert_lora_classifier = std::make_unique<classifiers::LoRAClassifier>();
        }
        
        // Default task configuration: Intent, PII, Security
        std::unordered_map<classifiers::TaskType, int> task_configs = {
            {classifiers::TaskType::Intent, 2},      // Binary classification
            {classifiers::TaskType::PII, 2},         // Binary classification
            {classifiers::TaskType::Security, 2}     // Binary classification
        };
        
        return g_bert_lora_classifier->initialize(
            base_model_path,
            lora_adapters_path,
            task_configs,
            device,
            "bert"
        );
    } catch (const std::exception& e) {
        std::cerr << "Error initializing BERT LoRA classifier: " << e.what() << std::endl;
        return false;
    }
}

bool ov_is_bert_lora_classifier_initialized() {
    return g_bert_lora_classifier != nullptr && g_bert_lora_classifier->isInitialized();
}

bool ov_init_modernbert_lora_classifier(
    const char* base_model_path,
    const char* lora_adapters_path,
    const char* device
) {
    try {
        // Validate input parameters
        if (!base_model_path || !lora_adapters_path || !device ||
            strlen(base_model_path) == 0 || strlen(lora_adapters_path) == 0) {
            std::cerr << "Error: Invalid input parameters (empty or null)" << std::endl;
            return false;
        }
        
        // Check if model file exists
        if (!std::filesystem::exists(base_model_path)) {
            std::cerr << "Error: Model file not found: " << base_model_path << std::endl;
            return false;
        }
        
        if (!g_modernbert_lora_classifier) {
            g_modernbert_lora_classifier = std::make_unique<classifiers::LoRAClassifier>();
        }
        
        // Default task configuration: Intent, PII, Security
        std::unordered_map<classifiers::TaskType, int> task_configs = {
            {classifiers::TaskType::Intent, 2},      // Binary classification
            {classifiers::TaskType::PII, 2},         // Binary classification
            {classifiers::TaskType::Security, 2}     // Binary classification
        };
        
        return g_modernbert_lora_classifier->initialize(
            base_model_path,
            lora_adapters_path,
            task_configs,
            device,
            "modernbert"
        );
    } catch (const std::exception& e) {
        std::cerr << "Error initializing ModernBERT LoRA classifier: " << e.what() << std::endl;
        return false;
    }
}

bool ov_is_modernbert_lora_classifier_initialized() {
    return g_modernbert_lora_classifier != nullptr && g_modernbert_lora_classifier->isInitialized();
}

// Helper function to convert OVTaskType to TaskType
static classifiers::TaskType convertTaskType(OVTaskType task) {
    switch (task) {
        case OV_TASK_INTENT: return classifiers::TaskType::Intent;
        case OV_TASK_PII: return classifiers::TaskType::PII;
        case OV_TASK_SECURITY: return classifiers::TaskType::Security;
        case OV_TASK_CLASSIFICATION: return classifiers::TaskType::Classification;
        default: return classifiers::TaskType::Classification;
    }
}

OVClassificationResult ov_classify_bert_lora_task(const char* text, OVTaskType task) {
    OVClassificationResult result{};
    result.predicted_class = -1;
    result.confidence = 0.0f;
    
    if (!g_bert_lora_classifier || !g_bert_lora_classifier->isInitialized()) {
        std::cerr << "BERT LoRA classifier not initialized" << std::endl;
        return result;
    }
    
    try {
        auto cpp_task = convertTaskType(task);
        auto cpp_result = g_bert_lora_classifier->classifyTask(text, cpp_task);
        
        result.predicted_class = cpp_result.predicted_class;
        result.confidence = cpp_result.confidence;
        
    } catch (const std::exception& e) {
        std::cerr << "Error in BERT LoRA task classification: " << e.what() << std::endl;
    }
    
    return result;
}

OVClassificationResult ov_classify_modernbert_lora_task(const char* text, OVTaskType task) {
    OVClassificationResult result{};
    result.predicted_class = -1;
    result.confidence = 0.0f;
    
    if (!g_modernbert_lora_classifier || !g_modernbert_lora_classifier->isInitialized()) {
        std::cerr << "ModernBERT LoRA classifier not initialized" << std::endl;
        return result;
    }
    
    try {
        auto cpp_task = convertTaskType(task);
        auto cpp_result = g_modernbert_lora_classifier->classifyTask(text, cpp_task);
        
        result.predicted_class = cpp_result.predicted_class;
        result.confidence = cpp_result.confidence;
        
    } catch (const std::exception& e) {
        std::cerr << "Error in ModernBERT LoRA task classification: " << e.what() << std::endl;
    }
    
    return result;
}

OVTokenClassificationResult ov_classify_bert_lora_tokens(const char* text, OVTaskType task) {
    OVTokenClassificationResult result{};
    result.entities = nullptr;
    result.num_entities = 0;
    
    if (!g_bert_lora_classifier || !g_bert_lora_classifier->isInitialized()) {
        std::cerr << "BERT LoRA classifier not initialized" << std::endl;
        return result;
    }
    
    try {
        classifiers::TaskType cpp_task = static_cast<classifiers::TaskType>(task);
        auto cpp_result = g_bert_lora_classifier->classifyTokens(text, cpp_task);
        
        // Convert entities to OVTokenEntity format
        if (!cpp_result.entities.empty()) {
            result.num_entities = static_cast<int>(cpp_result.entities.size());
            result.entities = new OVTokenEntity[result.num_entities];
            
            for (int i = 0; i < result.num_entities; ++i) {
                const auto& entity = cpp_result.entities[i];
                result.entities[i].entity_type = strdup(entity.type.c_str());
                result.entities[i].text = strdup(entity.text.c_str());
                result.entities[i].start = entity.start_token;
                result.entities[i].end = entity.end_token;
                result.entities[i].confidence = entity.confidence;
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error in BERT LoRA token classification: " << e.what() << std::endl;
    }
    
    return result;
}

OVTokenClassificationResult ov_classify_modernbert_lora_tokens(const char* text, OVTaskType task) {
    OVTokenClassificationResult result{};
    result.entities = nullptr;
    result.num_entities = 0;
    
    if (!g_modernbert_lora_classifier || !g_modernbert_lora_classifier->isInitialized()) {
        std::cerr << "ModernBERT LoRA classifier not initialized" << std::endl;
        return result;
    }
    
    try {
        classifiers::TaskType cpp_task = static_cast<classifiers::TaskType>(task);
        auto cpp_result = g_modernbert_lora_classifier->classifyTokens(text, cpp_task);
        
        // Convert entities to OVTokenEntity format
        if (!cpp_result.entities.empty()) {
            result.num_entities = static_cast<int>(cpp_result.entities.size());
            result.entities = new OVTokenEntity[result.num_entities];
            
            for (int i = 0; i < result.num_entities; ++i) {
                const auto& entity = cpp_result.entities[i];
                result.entities[i].entity_type = strdup(entity.type.c_str());
                result.entities[i].text = strdup(entity.text.c_str());
                result.entities[i].start = entity.start_token;
                result.entities[i].end = entity.end_token;
                result.entities[i].confidence = entity.confidence;
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error in ModernBERT LoRA token classification: " << e.what() << std::endl;
    }
    
    return result;
}
