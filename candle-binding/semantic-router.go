//go:build !windows && cgo
// +build !windows,cgo

package candle_binding

import (
	"fmt"
	"log"
	"runtime"
	"sync"
	"unsafe"
)

/*
#cgo LDFLAGS: -L${SRCDIR}/target/release -lcandle_semantic_router -ldl -lm
#include <stdlib.h>
#include <stdbool.h>

extern bool init_similarity_model(const char* model_id, bool use_cpu);

extern float calculate_similarity(const char* text1, const char* text2, int max_length);

extern bool init_classifier(const char* model_id, int num_classes, bool use_cpu);

extern bool init_pii_classifier(const char* model_id, int num_classes, bool use_cpu);

extern bool init_jailbreak_classifier(const char* model_id, int num_classes, bool use_cpu);

extern bool init_modernbert_classifier(const char* model_id, bool use_cpu);

extern bool init_modernbert_pii_classifier(const char* model_id, bool use_cpu);

extern bool init_modernbert_jailbreak_classifier(const char* model_id, bool use_cpu);

extern bool init_modernbert_pii_token_classifier(const char* model_id, bool use_cpu);

// Token classification structures
typedef struct {
    char* entity_type;
    int start;
    int end;
    char* text;
    float confidence;
} ModernBertTokenEntity;

typedef struct {
    ModernBertTokenEntity* entities;
    int num_entities;
} ModernBertTokenClassificationResult;

extern ModernBertTokenClassificationResult classify_modernbert_pii_tokens(const char* text, const char* model_config_path);
extern void free_modernbert_token_result(ModernBertTokenClassificationResult result);

// BERT token classification structures (compatible with ModernBERT)
typedef struct {
    char* entity_type;
    int start;
    int end;
    char* text;
    float confidence;
} BertTokenEntity;

typedef struct {
    BertTokenEntity* entities;
    int num_entities;
} BertTokenClassificationResult;

extern bool init_bert_token_classifier(const char* model_path, int num_classes, bool use_cpu);
extern BertTokenClassificationResult classify_bert_pii_tokens(const char* text, const char* id2label_json);
extern void free_bert_token_classification_result(BertTokenClassificationResult result);

// Similarity result structure
typedef struct {
    int index;
    float score;
} SimilarityResult;

// Embedding result structure
typedef struct {
    float* data;
    int length;
    bool error;
} EmbeddingResult;

// Tokenization result structure
typedef struct {
    int* token_ids;
    int token_count;
    char** tokens;
    bool error;
} TokenizationResult;

// Classification result structure
typedef struct {
    int class;
    float confidence;
} ClassificationResult;

// Classification result with full probability distribution structure
typedef struct {
    int class;
    float confidence;
    float* probabilities;
    int num_classes;
} ClassificationResultWithProbs;

// ModernBERT Classification result structure
typedef struct {
    int class;
    float confidence;
} ModernBertClassificationResult;

// ModernBERT Classification result with full probability distribution structure
typedef struct {
    int class;
    float confidence;
    float* probabilities;
    int num_classes;
} ModernBertClassificationResultWithProbs;

extern SimilarityResult find_most_similar(const char* query, const char** candidates, int num_candidates, int max_length);
extern EmbeddingResult get_text_embedding(const char* text, int max_length);
extern TokenizationResult tokenize_text(const char* text, int max_length);
extern void free_cstring(char* s);
extern void free_embedding(float* data, int length);
extern void free_tokenization_result(TokenizationResult result);
extern ClassificationResult classify_text(const char* text);
extern ClassificationResultWithProbs classify_text_with_probabilities(const char* text);
extern void free_probabilities(float* probabilities, int num_classes);
extern ClassificationResult classify_pii_text(const char* text);
extern ClassificationResult classify_jailbreak_text(const char* text);
extern ClassificationResult classify_bert_text(const char* text);
extern ModernBertClassificationResult classify_modernbert_text(const char* text);
extern ModernBertClassificationResultWithProbs classify_modernbert_text_with_probabilities(const char* text);
extern void free_modernbert_probabilities(float* probabilities, int num_classes);
extern ModernBertClassificationResult classify_modernbert_pii_text(const char* text);
extern ModernBertClassificationResult classify_modernbert_jailbreak_text(const char* text);

// New official Candle BERT functions
extern bool init_candle_bert_classifier(const char* model_path, int num_classes, bool use_cpu);
extern bool init_candle_bert_token_classifier(const char* model_path, int num_classes, bool use_cpu);
extern ClassificationResult classify_candle_bert_text(const char* text);
extern BertTokenClassificationResult classify_candle_bert_tokens(const char* text);
extern BertTokenClassificationResult classify_candle_bert_tokens_with_labels(const char* text, const char* id2label_json);

// LoRA Unified Classifier C structures
typedef struct {
    char* category;
    float confidence;
} LoRAIntentResult;

typedef struct {
    bool has_pii;
    char** pii_types;
    int num_pii_types;
    float confidence;
} LoRAPIIResult;

typedef struct {
    bool is_jailbreak;
    char* threat_type;
    float confidence;
} LoRASecurityResult;

typedef struct {
    LoRAIntentResult* intent_results;
    LoRAPIIResult* pii_results;
    LoRASecurityResult* security_results;
    int batch_size;
    float avg_confidence;
} LoRABatchResult;

// LoRA Unified Classifier C declarations
extern bool init_lora_unified_classifier(const char* intent_model_path, const char* pii_model_path, const char* security_model_path, const char* architecture, bool use_cpu);
extern LoRABatchResult classify_batch_with_lora(const char** texts, int num_texts);
extern void free_lora_batch_result(LoRABatchResult result);
*/
import "C"

var (
	initOnce                              sync.Once
	initErr                               error
	modelInitialized                      bool
	classifierInitOnce                    sync.Once
	classifierInitErr                     error
	piiClassifierInitOnce                 sync.Once
	piiClassifierInitErr                  error
	jailbreakClassifierInitOnce           sync.Once
	jailbreakClassifierInitErr            error
	modernbertClassifierInitOnce          sync.Once
	modernbertClassifierInitErr           error
	modernbertPiiClassifierInitOnce       sync.Once
	modernbertPiiClassifierInitErr        error
	modernbertJailbreakClassifierInitOnce sync.Once
	modernbertJailbreakClassifierInitErr  error
	modernbertPiiTokenClassifierInitOnce  sync.Once
	modernbertPiiTokenClassifierInitErr   error
	bertTokenClassifierInitOnce           sync.Once
	bertTokenClassifierInitErr            error
)

// TokenizeResult represents the result of tokenization
type TokenizeResult struct {
	TokenIDs []int32  // Token IDs
	Tokens   []string // String representation of tokens
}

// SimResult represents the result of a similarity search
type SimResult struct {
	Index int     // Index of the most similar text
	Score float32 // Similarity score
}

// ClassResult represents the result of a text classification
type ClassResult struct {
	Class      int     // Class index
	Confidence float32 // Confidence score
}

// ClassResultWithProbs represents the result of a text classification with full probability distribution
type ClassResultWithProbs struct {
	Class         int       // Class index
	Confidence    float32   // Confidence score
	Probabilities []float32 // Full probability distribution
	NumClasses    int       // Number of classes
}

// TokenEntity represents a single detected entity in token classification
type TokenEntity struct {
	EntityType string  // Type of entity (e.g., "PERSON", "EMAIL", "PHONE")
	Start      int     // Start character position in original text
	End        int     // End character position in original text
	Text       string  // Actual entity text
	Confidence float32 // Confidence score (0.0 to 1.0)
}

// TokenClassificationResult represents the result of token classification
type TokenClassificationResult struct {
	Entities []TokenEntity // Array of detected entities
}

// LoRA Unified Classifier structures
type LoRAIntentResult struct {
	Category   string
	Confidence float32
}

type LoRAPIIResult struct {
	HasPII     bool
	PIITypes   []string
	Confidence float32
}

type LoRASecurityResult struct {
	IsJailbreak bool
	ThreatType  string
	Confidence  float32
}

type LoRABatchResult struct {
	IntentResults   []LoRAIntentResult
	PIIResults      []LoRAPIIResult
	SecurityResults []LoRASecurityResult
	BatchSize       int
	AvgConfidence   float32
}

// InitModel initializes the BERT model with the specified model ID
func InitModel(modelID string, useCPU bool) error {
	var err error
	initOnce.Do(func() {
		if modelID == "" {
			modelID = "sentence-transformers/all-MiniLM-L6-v2"
		}

		log.Printf("Initializing BERT similarity model: %s", modelID)

		// Initialize BERT directly using CGO
		cModelID := C.CString(modelID)
		defer C.free(unsafe.Pointer(cModelID))

		success := C.init_similarity_model(cModelID, C.bool(useCPU))
		if !bool(success) {
			err = fmt.Errorf("failed to initialize BERT similarity model")
			return
		}

		modelInitialized = true
	})

	// Reset the once so we can try again with a different model ID if needed
	if err != nil {
		initOnce = sync.Once{}
		modelInitialized = false
	}

	return err
}

// TokenizeText tokenizes the given text into tokens and their IDs with maxLength parameter
func TokenizeText(text string, maxLength int) (TokenizeResult, error) {
	if !modelInitialized {
		return TokenizeResult{}, fmt.Errorf("BERT model not initialized")
	}

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	// Pass maxLength parameter to C function to ensure consistent tokenization with Python
	result := C.tokenize_text(cText, C.int(maxLength))

	// Make sure we free the memory allocated by Rust when we're done
	defer C.free_tokenization_result(result)

	if bool(result.error) {
		return TokenizeResult{}, fmt.Errorf("failed to tokenize text")
	}

	// Convert C array of token IDs to Go slice
	tokenCount := int(result.token_count)
	tokenIDs := make([]int32, tokenCount)

	if tokenCount > 0 && result.token_ids != nil {
		// Create a slice that refers to the C array
		cTokenIDs := (*[1 << 30]C.int)(unsafe.Pointer(result.token_ids))[:tokenCount:tokenCount]

		// Copy values
		for i := 0; i < tokenCount; i++ {
			tokenIDs[i] = int32(cTokenIDs[i])
		}
	}

	// Convert C array of token strings to Go slice
	tokens := make([]string, tokenCount)

	if tokenCount > 0 && result.tokens != nil {
		// Create a slice that refers to the C array of char pointers
		cTokens := (*[1 << 30]*C.char)(unsafe.Pointer(result.tokens))[:tokenCount:tokenCount]

		// Convert each C string to Go string
		for i := 0; i < tokenCount; i++ {
			tokens[i] = C.GoString(cTokens[i])
		}
	}

	tokResult := TokenizeResult{
		TokenIDs: tokenIDs,
		Tokens:   tokens,
	}

	return tokResult, nil
}

// TokenizeTextDefault tokenizes text with default max length (512)
func TokenizeTextDefault(text string) (TokenizeResult, error) {
	return TokenizeText(text, 512)
}

// GetEmbedding gets the embedding vector for a text
func GetEmbedding(text string, maxLength int) ([]float32, error) {
	if !modelInitialized {
		return nil, fmt.Errorf("BERT model not initialized")
	}

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	result := C.get_text_embedding(cText, C.int(maxLength))

	if bool(result.error) {
		return nil, fmt.Errorf("failed to generate embedding")
	}

	// Convert the C array to a Go slice
	length := int(result.length)
	embedding := make([]float32, length)

	if length > 0 {
		// Create a slice that refers to the C array
		cFloats := (*[1 << 30]C.float)(unsafe.Pointer(result.data))[:length:length]

		// Copy and convert each value
		for i := 0; i < length; i++ {
			embedding[i] = float32(cFloats[i])
		}

		// Free the memory allocated in Rust
		C.free_embedding(result.data, result.length)
	}

	return embedding, nil
}

// GetEmbeddingDefault gets the embedding vector for a text with default max length (512)
func GetEmbeddingDefault(text string) ([]float32, error) {
	return GetEmbedding(text, 512)
}

// CalculateSimilarity calculates the similarity between two texts with maxLength parameter
func CalculateSimilarity(text1, text2 string, maxLength int) float32 {
	if !modelInitialized {
		log.Printf("BERT model not initialized")
		return -1.0
	}

	cText1 := C.CString(text1)
	defer C.free(unsafe.Pointer(cText1))

	cText2 := C.CString(text2)
	defer C.free(unsafe.Pointer(cText2))

	result := C.calculate_similarity(cText1, cText2, C.int(maxLength))
	return float32(result)
}

// CalculateSimilarityDefault calculates the similarity between two texts with default max length (512)
func CalculateSimilarityDefault(text1, text2 string) float32 {
	return CalculateSimilarity(text1, text2, 512)
}

// FindMostSimilar finds the most similar text from a list of candidates with maxLength parameter
func FindMostSimilar(query string, candidates []string, maxLength int) SimResult {
	if !modelInitialized {
		log.Printf("BERT model not initialized")
		return SimResult{Index: -1, Score: -1.0}
	}

	if len(candidates) == 0 {
		return SimResult{Index: -1, Score: -1.0}
	}

	cQuery := C.CString(query)
	defer C.free(unsafe.Pointer(cQuery))

	// Convert the candidates to C strings
	cCandidates := make([]*C.char, len(candidates))
	for i, candidate := range candidates {
		cCandidates[i] = C.CString(candidate)
		defer C.free(unsafe.Pointer(cCandidates[i]))
	}

	// Create a C array of C strings
	cCandidatesPtr := (**C.char)(unsafe.Pointer(&cCandidates[0]))

	result := C.find_most_similar(cQuery, cCandidatesPtr, C.int(len(candidates)), C.int(maxLength))

	return SimResult{
		Index: int(result.index),
		Score: float32(result.score),
	}
}

// FindMostSimilarDefault finds the most similar text with default max length (512)
func FindMostSimilarDefault(query string, candidates []string) SimResult {
	return FindMostSimilar(query, candidates, 512)
}

// SetMemoryCleanupHandler sets up a finalizer to clean up memory when the Go GC runs
func SetMemoryCleanupHandler() {
	runtime.GC()
}

// IsModelInitialized returns whether the model has been successfully initialized
func IsModelInitialized() bool {
	return modelInitialized
}

// InitClassifier initializes the BERT classifier with the specified model path and number of classes
func InitClassifier(modelPath string, numClasses int, useCPU bool) error {
	var err error
	classifierInitOnce.Do(func() {
		if modelPath == "" {
			// Default to BERT base model if path is empty
			modelPath = "bert-base-uncased"
		}

		if numClasses < 2 {
			err = fmt.Errorf("number of classes must be at least 2, got %d", numClasses)
			return
		}

		log.Printf("Initializing classifier model: %s", modelPath)

		// Initialize classifier directly using CGO
		cModelID := C.CString(modelPath)
		defer C.free(unsafe.Pointer(cModelID))

		success := C.init_classifier(cModelID, C.int(numClasses), C.bool(useCPU))
		if !bool(success) {
			err = fmt.Errorf("failed to initialize classifier model")
		}
	})
	return err
}

// InitPIIClassifier initializes the BERT PII classifier with the specified model path and number of classes
func InitPIIClassifier(modelPath string, numClasses int, useCPU bool) error {
	var err error
	piiClassifierInitOnce.Do(func() {
		if modelPath == "" {
			// Default to a suitable PII classification model if path is empty
			modelPath = "./models/pii_classifier_modernbert-base_presidio_token_model"
		}

		if numClasses < 2 {
			err = fmt.Errorf("number of classes must be at least 2, got %d", numClasses)
			return
		}

		log.Printf("Initializing PII classifier model: %s", modelPath)

		// Initialize PII classifier directly using CGO
		cModelID := C.CString(modelPath)
		defer C.free(unsafe.Pointer(cModelID))

		success := C.init_pii_classifier(cModelID, C.int(numClasses), C.bool(useCPU))
		if !bool(success) {
			err = fmt.Errorf("failed to initialize PII classifier model")
		}
	})
	return err
}

// InitJailbreakClassifier initializes the BERT jailbreak classifier with the specified model path and number of classes
func InitJailbreakClassifier(modelPath string, numClasses int, useCPU bool) error {
	var err error
	jailbreakClassifierInitOnce.Do(func() {
		if modelPath == "" {
			// Default to the jailbreak classification model if path is empty
			modelPath = "./models/jailbreak_classifier_modernbert-base_model"
		}

		if numClasses < 2 {
			err = fmt.Errorf("number of classes must be at least 2, got %d", numClasses)
			return
		}

		log.Printf("Initializing jailbreak classifier model: %s", modelPath)

		// Initialize jailbreak classifier directly using CGO
		cModelID := C.CString(modelPath)
		defer C.free(unsafe.Pointer(cModelID))

		success := C.init_jailbreak_classifier(cModelID, C.int(numClasses), C.bool(useCPU))
		if !bool(success) {
			err = fmt.Errorf("failed to initialize jailbreak classifier model")
		}
	})
	return err
}

// ClassifyText classifies the provided text and returns the predicted class and confidence
func ClassifyText(text string) (ClassResult, error) {
	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	result := C.classify_text(cText)

	if result.class < 0 {
		return ClassResult{}, fmt.Errorf("failed to classify text")
	}

	return ClassResult{
		Class:      int(result.class),
		Confidence: float32(result.confidence),
	}, nil
}

// ClassifyTextWithProbabilities classifies the provided text and returns the predicted class, confidence, and full probability distribution
func ClassifyTextWithProbabilities(text string) (ClassResultWithProbs, error) {
	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	result := C.classify_text_with_probabilities(cText)

	if result.class < 0 {
		return ClassResultWithProbs{}, fmt.Errorf("failed to classify text with probabilities")
	}

	// Convert C array to Go slice
	probabilities := make([]float32, int(result.num_classes))
	if result.probabilities != nil && result.num_classes > 0 {
		probsSlice := (*[1 << 30]C.float)(unsafe.Pointer(result.probabilities))[:result.num_classes:result.num_classes]
		for i, prob := range probsSlice {
			probabilities[i] = float32(prob)
		}
		// Free the C-allocated memory
		C.free_probabilities(result.probabilities, result.num_classes)
	}

	return ClassResultWithProbs{
		Class:         int(result.class),
		Confidence:    float32(result.confidence),
		Probabilities: probabilities,
		NumClasses:    int(result.num_classes),
	}, nil
}

// ClassifyPIIText classifies the provided text for PII detection and returns the predicted class and confidence
func ClassifyPIIText(text string) (ClassResult, error) {
	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	result := C.classify_pii_text(cText)

	if result.class < 0 {
		return ClassResult{}, fmt.Errorf("failed to classify PII text")
	}

	return ClassResult{
		Class:      int(result.class),
		Confidence: float32(result.confidence),
	}, nil
}

// ClassifyJailbreakText classifies the provided text for jailbreak detection and returns the predicted class and confidence
func ClassifyJailbreakText(text string) (ClassResult, error) {
	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	result := C.classify_jailbreak_text(cText)

	if result.class < 0 {
		return ClassResult{}, fmt.Errorf("failed to classify jailbreak text")
	}

	return ClassResult{
		Class:      int(result.class),
		Confidence: float32(result.confidence),
	}, nil
}

// InitModernBertClassifier initializes the ModernBERT classifier with the specified model path
// Number of classes is automatically inferred from the model weights
func InitModernBertClassifier(modelPath string, useCPU bool) error {
	var err error
	modernbertClassifierInitOnce.Do(func() {
		if modelPath == "" {
			// Default to ModernBERT base model if path is empty
			modelPath = "answerdotai/ModernBERT-base"
		}

		log.Printf("Initializing ModernBERT classifier model: %s", modelPath)

		// Initialize ModernBERT classifier directly using CGO
		cModelID := C.CString(modelPath)
		defer C.free(unsafe.Pointer(cModelID))

		success := C.init_modernbert_classifier(cModelID, C.bool(useCPU))
		if !bool(success) {
			err = fmt.Errorf("failed to initialize ModernBERT classifier model")
		}
	})
	return err
}

// InitModernBertPIIClassifier initializes the ModernBERT PII classifier with the specified model path
// Number of classes is automatically inferred from the model weights
func InitModernBertPIIClassifier(modelPath string, useCPU bool) error {
	var err error
	modernbertPiiClassifierInitOnce.Do(func() {
		if modelPath == "" {
			// Default to a suitable ModernBERT PII classification model if path is empty
			modelPath = "./pii_classifier_modernbert_model"
		}

		log.Printf("Initializing ModernBERT PII classifier model: %s", modelPath)

		// Initialize ModernBERT PII classifier directly using CGO
		cModelID := C.CString(modelPath)
		defer C.free(unsafe.Pointer(cModelID))

		success := C.init_modernbert_pii_classifier(cModelID, C.bool(useCPU))
		if !bool(success) {
			err = fmt.Errorf("failed to initialize ModernBERT PII classifier model")
		}
	})
	return err
}

// InitModernBertJailbreakClassifier initializes the ModernBERT jailbreak classifier with the specified model path
// Number of classes is automatically inferred from the model weights
func InitModernBertJailbreakClassifier(modelPath string, useCPU bool) error {
	var err error
	modernbertJailbreakClassifierInitOnce.Do(func() {
		if modelPath == "" {
			// Default to the ModernBERT jailbreak classification model if path is empty
			modelPath = "./jailbreak_classifier_modernbert_model"
		}

		log.Printf("Initializing ModernBERT jailbreak classifier model: %s", modelPath)

		// Initialize ModernBERT jailbreak classifier directly using CGO
		cModelID := C.CString(modelPath)
		defer C.free(unsafe.Pointer(cModelID))

		success := C.init_modernbert_jailbreak_classifier(cModelID, C.bool(useCPU))
		if !bool(success) {
			err = fmt.Errorf("failed to initialize ModernBERT jailbreak classifier model")
		}
	})
	return err
}

// InitModernBertPIITokenClassifier initializes the ModernBERT PII token classifier with the specified model path
// This is used for token-level entity extraction (e.g., finding specific PII entities and their locations)
func InitModernBertPIITokenClassifier(modelPath string, useCPU bool) error {
	var err error
	modernbertPiiTokenClassifierInitOnce.Do(func() {
		if modelPath == "" {
			// Default to a suitable ModernBERT PII token classification model if path is empty
			modelPath = "./pii_classifier_modernbert_ai4privacy_token_model"
		}

		log.Printf("Initializing ModernBERT PII token classifier model: %s", modelPath)

		// Initialize ModernBERT PII token classifier directly using CGO
		cModelID := C.CString(modelPath)
		defer C.free(unsafe.Pointer(cModelID))

		success := C.init_modernbert_pii_token_classifier(cModelID, C.bool(useCPU))
		if !bool(success) {
			err = fmt.Errorf("failed to initialize ModernBERT PII token classifier model")
		}
	})
	return err
}

// ClassifyModernBertText classifies the provided text using ModernBERT and returns the predicted class and confidence
func ClassifyModernBertText(text string) (ClassResult, error) {
	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	result := C.classify_modernbert_text(cText)

	if result.class < 0 {
		return ClassResult{}, fmt.Errorf("failed to classify text with ModernBERT")
	}

	return ClassResult{
		Class:      int(result.class),
		Confidence: float32(result.confidence),
	}, nil
}

// ClassifyModernBertTextWithProbabilities classifies the provided text using ModernBERT and returns the predicted class, confidence, and full probability distribution
func ClassifyModernBertTextWithProbabilities(text string) (ClassResultWithProbs, error) {
	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	result := C.classify_modernbert_text_with_probabilities(cText)

	if result.class < 0 {
		return ClassResultWithProbs{}, fmt.Errorf("failed to classify text with probabilities using ModernBERT")
	}

	// Convert C array to Go slice
	probabilities := make([]float32, int(result.num_classes))
	if result.probabilities != nil && result.num_classes > 0 {
		probsSlice := (*[1 << 30]C.float)(unsafe.Pointer(result.probabilities))[:result.num_classes:result.num_classes]
		for i, prob := range probsSlice {
			probabilities[i] = float32(prob)
		}
		// Free the C-allocated memory
		C.free_modernbert_probabilities(result.probabilities, result.num_classes)
	}

	return ClassResultWithProbs{
		Class:         int(result.class),
		Confidence:    float32(result.confidence),
		Probabilities: probabilities,
		NumClasses:    int(result.num_classes),
	}, nil
}

// ClassifyModernBertPIIText classifies the provided text for PII detection using ModernBERT and returns the predicted class and confidence
func ClassifyModernBertPIIText(text string) (ClassResult, error) {
	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	result := C.classify_modernbert_pii_text(cText)

	if result.class < 0 {
		return ClassResult{}, fmt.Errorf("failed to classify PII text with ModernBERT")
	}

	return ClassResult{
		Class:      int(result.class),
		Confidence: float32(result.confidence),
	}, nil
}

// ClassifyModernBertJailbreakText classifies the provided text for jailbreak detection using ModernBERT and returns the predicted class and confidence
func ClassifyModernBertJailbreakText(text string) (ClassResult, error) {
	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	result := C.classify_modernbert_jailbreak_text(cText)

	if result.class < 0 {
		return ClassResult{}, fmt.Errorf("failed to classify jailbreak text with ModernBERT")
	}

	return ClassResult{
		Class:      int(result.class),
		Confidence: float32(result.confidence),
	}, nil
}

// ClassifyModernBertPIITokens performs token-level PII classification using ModernBERT
// and returns detected entities with their positions and confidence scores
func ClassifyModernBertPIITokens(text string, modelConfigPath string) (TokenClassificationResult, error) {
	// Validate inputs
	if text == "" {
		return TokenClassificationResult{}, fmt.Errorf("text cannot be empty")
	}
	if modelConfigPath == "" {
		return TokenClassificationResult{}, fmt.Errorf("model config path cannot be empty")
	}

	// Convert Go strings to C strings
	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	cConfigPath := C.CString(modelConfigPath)
	defer C.free(unsafe.Pointer(cConfigPath))

	// Call the Rust function
	result := C.classify_modernbert_pii_tokens(cText, cConfigPath)

	// Defer memory cleanup - this is crucial to prevent memory leaks
	defer C.free_modernbert_token_result(result)

	// Check for errors
	if result.num_entities < 0 {
		return TokenClassificationResult{}, fmt.Errorf("failed to classify PII tokens with ModernBERT")
	}

	// Handle empty result (no entities found)
	if result.num_entities == 0 {
		return TokenClassificationResult{Entities: []TokenEntity{}}, nil
	}

	// Convert C result to Go structures
	numEntities := int(result.num_entities)
	entities := make([]TokenEntity, numEntities)

	// Create a slice that refers to the C array
	cEntities := (*[1 << 30]C.ModernBertTokenEntity)(unsafe.Pointer(result.entities))[:numEntities:numEntities]

	// Convert each C entity to Go entity
	for i := 0; i < numEntities; i++ {
		cEntity := &cEntities[i]

		entities[i] = TokenEntity{
			EntityType: C.GoString(cEntity.entity_type),
			Start:      int(cEntity.start),
			End:        int(cEntity.end),
			Text:       C.GoString(cEntity.text),
			Confidence: float32(cEntity.confidence),
		}
	}

	return TokenClassificationResult{
		Entities: entities,
	}, nil
}

// ================================================================================================
// BERT TOKEN CLASSIFICATION GO BINDINGS
// ================================================================================================

// InitBertTokenClassifier initializes the BERT token classifier
func InitBertTokenClassifier(modelPath string, numClasses int, useCPU bool) error {
	var err error
	bertTokenClassifierInitOnce.Do(func() {
		log.Printf("Initializing BERT token classifier: %s", modelPath)

		cModelPath := C.CString(modelPath)
		defer C.free(unsafe.Pointer(cModelPath))

		success := C.init_bert_token_classifier(cModelPath, C.int(numClasses), C.bool(useCPU))
		if !bool(success) {
			err = fmt.Errorf("failed to initialize BERT token classifier")
			return
		}

		log.Printf("BERT token classifier initialized successfully")
	})

	// Reset the once so we can try again with a different model if needed
	if err != nil {
		bertTokenClassifierInitOnce = sync.Once{}
	}

	bertTokenClassifierInitErr = err
	return err
}

// ClassifyBertPIITokens performs token classification for PII detection using BERT
func ClassifyBertPIITokens(text string, id2labelJson string) (TokenClassificationResult, error) {
	if bertTokenClassifierInitErr != nil {
		return TokenClassificationResult{}, fmt.Errorf("BERT token classifier not initialized: %v", bertTokenClassifierInitErr)
	}

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	cId2Label := C.CString(id2labelJson)
	defer C.free(unsafe.Pointer(cId2Label))

	// Call the Rust function
	result := C.classify_bert_pii_tokens(cText, cId2Label)
	defer C.free_bert_token_classification_result(result)

	// Check for errors
	if result.num_entities < 0 {
		return TokenClassificationResult{}, fmt.Errorf("failed to classify PII tokens with BERT")
	}

	// Handle empty result (no entities found)
	if result.num_entities == 0 {
		return TokenClassificationResult{Entities: []TokenEntity{}}, nil
	}

	// Convert C result to Go structures
	numEntities := int(result.num_entities)
	entities := make([]TokenEntity, numEntities)

	// Access the C array safely
	cEntities := (*[1 << 20]C.BertTokenEntity)(unsafe.Pointer(result.entities))[:numEntities:numEntities]

	for i := 0; i < numEntities; i++ {
		entities[i] = TokenEntity{
			EntityType: C.GoString(cEntities[i].entity_type),
			Start:      int(cEntities[i].start),
			End:        int(cEntities[i].end),
			Text:       C.GoString(cEntities[i].text),
			Confidence: float32(cEntities[i].confidence),
		}
	}

	return TokenClassificationResult{
		Entities: entities,
	}, nil
}

// ClassifyBertText performs sequence classification using BERT
func ClassifyBertText(text string) (ClassResult, error) {
	if bertTokenClassifierInitErr != nil {
		return ClassResult{}, fmt.Errorf("BERT classifier not initialized: %v", bertTokenClassifierInitErr)
	}

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	result := C.classify_bert_text(cText)

	if result.class < 0 {
		return ClassResult{}, fmt.Errorf("failed to classify text with BERT")
	}

	return ClassResult{
		Class:      int(result.class),
		Confidence: float32(result.confidence),
	}, nil
}

// ================================================================================================
// END OF BERT TOKEN CLASSIFICATION GO BINDINGS
// ================================================================================================

// ================================================================================================
// NEW OFFICIAL CANDLE BERT GO BINDINGS
// ================================================================================================

// InitCandleBertClassifier initializes a BERT sequence classifier using official Candle implementation
func InitCandleBertClassifier(modelPath string, numClasses int, useCPU bool) bool {
	cModelPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cModelPath))

	return bool(C.init_candle_bert_classifier(cModelPath, C.int(numClasses), C.bool(useCPU)))
}

// InitCandleBertTokenClassifier initializes a BERT token classifier using official Candle implementation
func InitCandleBertTokenClassifier(modelPath string, numClasses int, useCPU bool) bool {
	cModelPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cModelPath))

	return bool(C.init_candle_bert_token_classifier(cModelPath, C.int(numClasses), C.bool(useCPU)))
}

// ClassifyCandleBertText classifies text using official Candle BERT implementation
func ClassifyCandleBertText(text string) (ClassResult, error) {
	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	result := C.classify_candle_bert_text(cText)

	if result.class < 0 {
		return ClassResult{}, fmt.Errorf("failed to classify text with Candle BERT")
	}

	return ClassResult{
		Class:      int(result.class),
		Confidence: float32(result.confidence),
	}, nil
}

// ClassifyCandleBertTokens classifies tokens using official Candle BERT token classifier
func ClassifyCandleBertTokens(text string) (TokenClassificationResult, error) {
	if text == "" {
		return TokenClassificationResult{}, fmt.Errorf("text cannot be empty")
	}

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	result := C.classify_candle_bert_tokens(cText)
	defer C.free_bert_token_classification_result(result)

	if result.num_entities < 0 {
		return TokenClassificationResult{}, fmt.Errorf("failed to classify tokens with Candle BERT")
	}

	if result.num_entities == 0 {
		return TokenClassificationResult{Entities: []TokenEntity{}}, nil
	}

	// Convert C result to Go
	entities := make([]TokenEntity, result.num_entities)
	cEntities := (*[1000]C.BertTokenEntity)(unsafe.Pointer(result.entities))[:result.num_entities:result.num_entities]

	for i, cEntity := range cEntities {
		entities[i] = TokenEntity{
			EntityType: C.GoString(cEntity.entity_type),
			Start:      int(cEntity.start),
			End:        int(cEntity.end),
			Text:       C.GoString(cEntity.text),
			Confidence: float32(cEntity.confidence),
		}
	}

	return TokenClassificationResult{
		Entities: entities,
	}, nil
}

// ClassifyCandleBertTokensWithLabels classifies tokens using official Candle BERT with proper label mapping
func ClassifyCandleBertTokensWithLabels(text string, id2labelJSON string) (TokenClassificationResult, error) {
	if text == "" {
		return TokenClassificationResult{}, fmt.Errorf("text cannot be empty")
	}
	if id2labelJSON == "" {
		return TokenClassificationResult{}, fmt.Errorf("id2label mapping cannot be empty")
	}

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	cLabels := C.CString(id2labelJSON)
	defer C.free(unsafe.Pointer(cLabels))

	result := C.classify_candle_bert_tokens_with_labels(cText, cLabels)
	defer C.free_bert_token_classification_result(result)

	if result.num_entities < 0 {
		return TokenClassificationResult{}, fmt.Errorf("failed to classify tokens with Candle BERT")
	}

	if result.num_entities == 0 {
		return TokenClassificationResult{Entities: []TokenEntity{}}, nil
	}

	// Convert C result to Go
	entities := make([]TokenEntity, result.num_entities)
	cEntities := (*[1000]C.BertTokenEntity)(unsafe.Pointer(result.entities))[:result.num_entities:result.num_entities]

	for i, cEntity := range cEntities {
		entities[i] = TokenEntity{
			EntityType: C.GoString(cEntity.entity_type),
			Start:      int(cEntity.start),
			End:        int(cEntity.end),
			Text:       C.GoString(cEntity.text),
			Confidence: float32(cEntity.confidence),
		}
	}

	return TokenClassificationResult{
		Entities: entities,
	}, nil
}

// ================================================================================================
// END OF NEW OFFICIAL CANDLE BERT GO BINDINGS
// ================================================================================================
// LORA UNIFIED CLASSIFIER GO BINDINGS
// ================================================================================================

// InitLoRAUnifiedClassifier initializes the LoRA Unified Classifier
func InitLoRAUnifiedClassifier(intentModelPath, piiModelPath, securityModelPath, architecture string, useCPU bool) error {
	cIntentPath := C.CString(intentModelPath)
	defer C.free(unsafe.Pointer(cIntentPath))

	cPIIPath := C.CString(piiModelPath)
	defer C.free(unsafe.Pointer(cPIIPath))

	cSecurityPath := C.CString(securityModelPath)
	defer C.free(unsafe.Pointer(cSecurityPath))

	cArch := C.CString(architecture)
	defer C.free(unsafe.Pointer(cArch))

	log.Printf("Initializing LoRA Unified Classifier with architecture: %s", architecture)

	success := C.init_lora_unified_classifier(cIntentPath, cPIIPath, cSecurityPath, cArch, C.bool(useCPU))
	if !success {
		return fmt.Errorf("failed to initialize LoRA Unified Classifier")
	}

	log.Printf("LoRA Unified Classifier initialized successfully")
	return nil
}

// ClassifyBatchWithLoRA performs batch classification using LoRA models
func ClassifyBatchWithLoRA(texts []string) (LoRABatchResult, error) {
	if len(texts) == 0 {
		return LoRABatchResult{}, fmt.Errorf("empty text batch")
	}

	// Convert Go strings to C strings
	cTexts := make([]*C.char, len(texts))
	for i, text := range texts {
		cTexts[i] = C.CString(text)
		defer C.free(unsafe.Pointer(cTexts[i]))
	}

	log.Printf("Processing batch with LoRA models, batch size: %d", len(texts))

	// Call C function
	cResult := C.classify_batch_with_lora((**C.char)(unsafe.Pointer(&cTexts[0])), C.int(len(texts)))
	defer C.free_lora_batch_result(cResult)

	if cResult.batch_size <= 0 {
		return LoRABatchResult{}, fmt.Errorf("batch classification failed")
	}

	// Convert C results to Go
	result := LoRABatchResult{
		BatchSize:     int(cResult.batch_size),
		AvgConfidence: float32(cResult.avg_confidence),
	}

	// Convert intent results
	if cResult.intent_results != nil {
		intentSlice := (*[1000]C.LoRAIntentResult)(unsafe.Pointer(cResult.intent_results))[:cResult.batch_size:cResult.batch_size]
		for _, cIntent := range intentSlice {
			result.IntentResults = append(result.IntentResults, LoRAIntentResult{
				Category:   C.GoString(cIntent.category),
				Confidence: float32(cIntent.confidence),
			})
		}
	}

	// Convert PII results
	if cResult.pii_results != nil {
		piiSlice := (*[1000]C.LoRAPIIResult)(unsafe.Pointer(cResult.pii_results))[:cResult.batch_size:cResult.batch_size]
		for _, cPII := range piiSlice {
			piiResult := LoRAPIIResult{
				HasPII:     bool(cPII.has_pii),
				Confidence: float32(cPII.confidence),
			}

			// Convert PII types
			if cPII.pii_types != nil && cPII.num_pii_types > 0 {
				piiTypesSlice := (*[1000]*C.char)(unsafe.Pointer(cPII.pii_types))[:cPII.num_pii_types:cPII.num_pii_types]
				for _, cType := range piiTypesSlice {
					piiResult.PIITypes = append(piiResult.PIITypes, C.GoString(cType))
				}
			}

			result.PIIResults = append(result.PIIResults, piiResult)
		}
	}

	// Convert security results
	if cResult.security_results != nil {
		securitySlice := (*[1000]C.LoRASecurityResult)(unsafe.Pointer(cResult.security_results))[:cResult.batch_size:cResult.batch_size]
		for _, cSecurity := range securitySlice {
			result.SecurityResults = append(result.SecurityResults, LoRASecurityResult{
				IsJailbreak: bool(cSecurity.is_jailbreak),
				ThreatType:  C.GoString(cSecurity.threat_type),
				Confidence:  float32(cSecurity.confidence),
			})
		}
	}

	return result, nil
}

// ================================================================================================
// END OF LORA UNIFIED CLASSIFIER GO BINDINGS
// ================================================================================================
