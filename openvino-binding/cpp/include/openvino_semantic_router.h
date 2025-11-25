#ifndef OPENVINO_SEMANTIC_ROUTER_H
#define OPENVINO_SEMANTIC_ROUTER_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ================================================================================================
// INITIALIZATION FUNCTIONS
// ================================================================================================

/**
 * @brief Initialize BERT similarity model for semantic routing
 * @param model_path Path to OpenVINO IR model (.xml file)
 * @param device Device name ("CPU", "GPU", "AUTO", etc.)
 * @return true if initialization succeeded, false otherwise
 */
bool ov_init_similarity_model(const char* model_path, const char* device);

/**
 * @brief Check if similarity model is initialized
 * @return true if initialized, false otherwise
 */
bool ov_is_similarity_model_initialized();

/**
 * @brief Initialize BERT classifier model
 * @param model_path Path to OpenVINO IR model (.xml file)
 * @param num_classes Number of classification classes
 * @param device Device name ("CPU", "GPU", "AUTO", etc.)
 * @return true if initialization succeeded, false otherwise
 */
bool ov_init_classifier(const char* model_path, int num_classes, const char* device);

/**
 * @brief Initialize embedding model (BERT-based)
 * @param model_path Path to OpenVINO IR model (.xml file)
 * @param device Device name ("CPU", "GPU", "AUTO", etc.)
 * @return true if initialization succeeded, false otherwise
 */
bool ov_init_embedding_model(const char* model_path, const char* device);

/**
 * @brief Check if embedding model is initialized
 * @return true if initialized, false otherwise
 */
bool ov_is_embedding_model_initialized();

// ================================================================================================
// TOKENIZATION STRUCTURES AND FUNCTIONS
// ================================================================================================

/**
 * @brief Tokenization result structure
 */
typedef struct {
    int* token_ids;      // Array of token IDs
    int token_count;     // Number of tokens
    char** tokens;       // Array of token strings
    bool error;          // Error flag
} OVTokenizationResult;

/**
 * @brief Tokenize text using the BERT tokenizer
 * @param text Input text to tokenize
 * @param max_length Maximum sequence length
 * @return Tokenization result (caller must free using ov_free_tokenization_result)
 */
OVTokenizationResult ov_tokenize_text(const char* text, int max_length);

/**
 * @brief Free tokenization result memory
 * @param result Tokenization result to free
 */
void ov_free_tokenization_result(OVTokenizationResult result);

// ================================================================================================
// EMBEDDING STRUCTURES AND FUNCTIONS
// ================================================================================================

/**
 * @brief Embedding result structure
 */
typedef struct {
    float* data;                // Embedding vector data
    int length;                 // Length of embedding vector
    float processing_time_ms;   // Processing time in milliseconds
    bool error;                 // Error flag
} OVEmbeddingResult;

/**
 * @brief Generate embedding for input text
 * @param text Input text
 * @param max_length Maximum sequence length
 * @return Embedding result (caller must free using ov_free_embedding)
 */
OVEmbeddingResult ov_get_text_embedding(const char* text, int max_length);

/**
 * @brief Free embedding memory
 * @param data Embedding data pointer
 * @param length Length of embedding vector
 */
void ov_free_embedding(float* data, int length);

// ================================================================================================
// SIMILARITY STRUCTURES AND FUNCTIONS
// ================================================================================================

/**
 * @brief Similarity result structure for single comparison
 */
typedef struct {
    int index;       // Index of the most similar candidate
    float score;     // Similarity score (0.0 to 1.0)
} OVSimilarityResult;

/**
 * @brief Embedding similarity result structure
 */
typedef struct {
    float similarity;         // Cosine similarity score (-1.0 to 1.0)
    float processing_time_ms; // Processing time in milliseconds
    bool error;               // Error flag
} OVEmbeddingSimilarityResult;

/**
 * @brief Batch similarity match structure
 */
typedef struct {
    int index;        // Index of the candidate in the input array
    float similarity; // Cosine similarity score
} OVSimilarityMatch;

/**
 * @brief Batch similarity result structure
 */
typedef struct {
    OVSimilarityMatch* matches; // Array of top-k matches, sorted by similarity (descending)
    int num_matches;            // Number of matches returned (≤ top_k)
    float processing_time_ms;   // Processing time in milliseconds
    bool error;                 // Error flag
} OVBatchSimilarityResult;

/**
 * @brief Calculate similarity between two texts
 * @param text1 First text
 * @param text2 Second text
 * @param max_length Maximum sequence length
 * @return Similarity score (0.0 to 1.0), -1.0 on error
 */
float ov_calculate_similarity(const char* text1, const char* text2, int max_length);

/**
 * @brief Find the most similar text from candidates
 * @param query Query text
 * @param candidates Array of candidate texts
 * @param num_candidates Number of candidates
 * @param max_length Maximum sequence length
 * @return Similarity result with index and score
 */
OVSimilarityResult ov_find_most_similar(const char* query, const char** candidates, 
                                         int num_candidates, int max_length);

/**
 * @brief Calculate embedding similarity between two texts
 * @param text1 First text
 * @param text2 Second text
 * @param max_length Maximum sequence length
 * @param result Pointer to result structure
 * @return 0 on success, -1 on error
 */
int ov_calculate_embedding_similarity(const char* text1, const char* text2, 
                                       int max_length, OVEmbeddingSimilarityResult* result);

/**
 * @brief Calculate batch similarity for multiple candidates
 * @param query Query text
 * @param candidates Array of candidate texts
 * @param num_candidates Number of candidates
 * @param top_k Number of top matches to return (0 = return all)
 * @param max_length Maximum sequence length
 * @param result Pointer to result structure
 * @return 0 on success, -1 on error
 */
int ov_calculate_similarity_batch(const char* query, const char** candidates, 
                                   int num_candidates, int top_k, int max_length,
                                   OVBatchSimilarityResult* result);

/**
 * @brief Free batch similarity result memory
 * @param result Pointer to result structure
 */
void ov_free_batch_similarity_result(OVBatchSimilarityResult* result);

// ================================================================================================
// CLASSIFICATION STRUCTURES AND FUNCTIONS
// ================================================================================================

/**
 * @brief Classification result structure
 */
typedef struct {
    int predicted_class;     // Predicted class index
    float confidence;        // Confidence score (0.0 to 1.0)
} OVClassificationResult;

/**
 * @brief Classification result with full probability distribution
 */
typedef struct {
    int predicted_class;         // Predicted class index
    float confidence;            // Confidence score (0.0 to 1.0)
    float* probabilities;        // Full probability distribution
    int num_classes;             // Number of classes
} OVClassificationResultWithProbs;

/**
 * @brief Classify text using BERT classifier
 * @param text Input text
 * @return Classification result
 */
OVClassificationResult ov_classify_text(const char* text);

/**
 * @brief Classify text with full probability distribution
 * @param text Input text
 * @return Classification result with probabilities (caller must free using ov_free_probabilities)
 */
OVClassificationResultWithProbs ov_classify_text_with_probabilities(const char* text);

/**
 * @brief Free probabilities array
 * @param probabilities Probabilities array
 * @param num_classes Number of classes
 */
void ov_free_probabilities(float* probabilities, int num_classes);

// ================================================================================================
// TOKEN CLASSIFICATION STRUCTURES AND FUNCTIONS
// ================================================================================================

/**
 * @brief Token entity structure for token classification
 */
typedef struct {
    char* entity_type;      // Entity type (e.g., "PERSON", "EMAIL", "PHONE")
    int start;              // Start character position
    int end;                // End character position
    char* text;             // Entity text
    float confidence;       // Confidence score (0.0 to 1.0)
} OVTokenEntity;

/**
 * @brief Token classification result structure
 */
typedef struct {
    OVTokenEntity* entities;    // Array of detected entities
    int num_entities;           // Number of entities
} OVTokenClassificationResult;

/**
 * @brief Initialize BERT token classifier
 * @param model_path Path to OpenVINO IR model (.xml file)
 * @param num_classes Number of token classes
 * @param device Device name ("CPU", "GPU", "AUTO", etc.)
 * @return true if initialization succeeded, false otherwise
 */
bool ov_init_token_classifier(const char* model_path, int num_classes, const char* device);

/**
 * @brief Classify tokens in text (e.g., PII detection)
 * @param text Input text
 * @param id2label_json JSON mapping of class IDs to labels
 * @return Token classification result (caller must free using ov_free_token_result)
 */
OVTokenClassificationResult ov_classify_tokens(const char* text, const char* id2label_json);

/**
 * @brief Free token classification result memory
 * @param result Token classification result
 */
void ov_free_token_result(OVTokenClassificationResult result);

// ================================================================================================
// MODERNBERT SUPPORT
// ================================================================================================

/**
 * @brief Initialize ModernBERT embedding model (supports ModernBERT-base and ModernBERT-large)
 * @param model_path Path to OpenVINO IR model (.xml file)
 * @param device Device name ("CPU", "GPU", "AUTO", etc.)
 * @return true if initialization succeeded, false otherwise
 */
bool ov_init_modernbert_embedding(const char* model_path, const char* device);

/**
 * @brief Check if ModernBERT embedding model is initialized
 * @return true if initialized, false otherwise
 */
bool ov_is_modernbert_embedding_initialized();

/**
 * @brief Initialize ModernBERT classification model
 * @param model_path Path to OpenVINO IR model (.xml file)
 * @param num_classes Number of classification classes
 * @param device Device name ("CPU", "GPU", "AUTO", etc.)
 * @return true if initialization succeeded, false otherwise
 */
bool ov_init_modernbert_classifier(const char* model_path, int num_classes, const char* device);

/**
 * @brief Check if ModernBERT classifier is initialized
 * @return true if initialized, false otherwise
 */
bool ov_is_modernbert_classifier_initialized();

/**
 * @brief Initialize ModernBERT token classification model (for PII, NER, etc.)
 * @param model_path Path to OpenVINO IR model (.xml file)
 * @param num_classes Number of token classes
 * @param device Device name ("CPU", "GPU", "AUTO", etc.)
 * @return true if initialization succeeded, false otherwise
 */
bool ov_init_modernbert_token_classifier(const char* model_path, int num_classes, const char* device);

/**
 * @brief Check if ModernBERT token classifier is initialized
 * @return true if initialized, false otherwise
 */
bool ov_is_modernbert_token_classifier_initialized();

/**
 * @brief ModernBERT classification (returns class index and confidence)
 * @param text Input text
 * @return Classification result
 */
OVClassificationResult ov_classify_modernbert(const char* text);

/**
 * @brief ModernBERT token classification with BIO tagging
 * @param text Input text
 * @param id2label_json JSON mapping of class IDs to labels
 * @return Token classification result (caller must free using ov_free_token_result)
 */
OVTokenClassificationResult ov_classify_modernbert_tokens(const char* text, const char* id2label_json);

/**
 * @brief Get ModernBERT embedding for text
 * @param text Input text
 * @param max_length Maximum sequence length
 * @return Embedding result (caller must free using ov_free_embedding)
 */
OVEmbeddingResult ov_get_modernbert_embedding(const char* text, int max_length);

// ================================================================================================
// UTILITY FUNCTIONS
// ================================================================================================

/**
 * @brief Free C string allocated by library
 * @param s String to free
 */
void ov_free_cstring(char* s);

/**
 * @brief Get OpenVINO version
 * @return Version string (do not free)
 */
const char* ov_get_version();

/**
 * @brief Get available devices
 * @return Comma-separated list of devices (caller must free using ov_free_cstring)
 */
char* ov_get_available_devices();

#ifdef __cplusplus
}
#endif

#endif // OPENVINO_SEMANTIC_ROUTER_H

