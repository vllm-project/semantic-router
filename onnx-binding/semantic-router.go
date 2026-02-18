//go:build !windows && cgo && (amd64 || arm64)
// +build !windows
// +build cgo
// +build amd64 arm64

// Package onnx_binding provides Go bindings for mmBERT ONNX Runtime inference.
// This mirrors the candle_binding API for drop-in compatibility.
package onnx_binding

/*
#cgo LDFLAGS: -L${SRCDIR}/target/release -lonnx_semantic_router -ldl -lm -lpthread
#include <stdlib.h>
#include <stdbool.h>

// ============================================================================
// Embedding Types
// ============================================================================

typedef struct {
    float* data;
    int length;
    bool error;
    int model_type;
    int sequence_length;
    float processing_time_ms;
} EmbeddingResult;

typedef struct {
    float similarity;
    int model_type;
    float processing_time_ms;
    bool error;
} EmbeddingSimilarityResult;

typedef struct {
    int index;
    float similarity;
} SimilarityMatch;

typedef struct {
    SimilarityMatch* matches;
    int num_matches;
    int model_type;
    float processing_time_ms;
    bool error;
} BatchSimilarityResult;

typedef struct {
    char* model_name;
    bool is_loaded;
    int max_sequence_length;
    int default_dimension;
    char* model_path;
    bool supports_layer_exit;
    char* available_layers;
} EmbeddingModelInfo;

typedef struct {
    EmbeddingModelInfo* models;
    int num_models;
    bool error;
} EmbeddingModelsInfoResult;

// ============================================================================
// Classification Types
// ============================================================================

typedef struct {
    char* label;
    int class_id;
    float confidence;
    int num_classes;
    float* probabilities;
    float processing_time_ms;
    bool error;
} ClassificationResultFFI;

typedef struct {
    char* text;
    char* entity_type;
    int start;
    int end;
    float confidence;
} PIIEntityFFI;

typedef struct {
    PIIEntityFFI* entities;
    int num_entities;
    float processing_time_ms;
    bool error;
} PIIResultFFI;

// ============================================================================
// Embedding Functions
// ============================================================================

extern bool init_mmbert_embedding_model(const char* model_path, bool use_cpu);
extern bool is_mmbert_model_initialized();
extern int get_embedding(const char* text, EmbeddingResult* result);
extern int get_embedding_with_dim(const char* text, int target_dim, EmbeddingResult* result);
extern int get_embedding_2d_matryoshka(const char* text, int target_layer, int target_dim, EmbeddingResult* result);
extern int get_embeddings_batch(const char** texts, int num_texts, int target_layer, int target_dim, EmbeddingResult* results);
extern int calculate_embedding_similarity(const char* text1, const char* text2, int target_layer, int target_dim, EmbeddingSimilarityResult* result);
extern int calculate_similarity_batch(const char* query, const char** candidates, int num_candidates, int top_k, int target_layer, int target_dim, BatchSimilarityResult* result);
extern int get_embedding_models_info(EmbeddingModelsInfoResult* result);
extern void free_embedding(float* data, int length);
extern void free_batch_similarity_result(BatchSimilarityResult* result);
extern void free_embedding_models_info(EmbeddingModelsInfoResult* result);

// ============================================================================
// Classification Functions
// ============================================================================

extern bool init_sequence_classifier(const char* name, const char* model_path, bool use_gpu);
extern bool init_token_classifier(const char* name, const char* model_path, bool use_gpu);
extern bool is_classifier_loaded(const char* name);
extern int classify_text(const char* classifier_name, const char* text, ClassificationResultFFI* result);
extern int classify_batch(const char* classifier_name, const char** texts, int num_texts, ClassificationResultFFI* results);
extern int detect_pii(const char* classifier_name, const char* text, PIIResultFFI* result);
extern void free_classification_result(ClassificationResultFFI* result);
extern void free_pii_result(PIIResultFFI* result);
*/
import "C"

import (
	"errors"
	"fmt"
	"sync"
	"unsafe"
)

// ============================================================================
// Go Types (matching candle_binding)
// ============================================================================

// EmbeddingOutput contains embedding result with metadata
type EmbeddingOutput struct {
	Embedding        []float32
	ModelType        string // "mmbert", "unknown"
	SequenceLength   int
	ProcessingTimeMs float32
}

// SimilarityOutput contains similarity result with metadata
type SimilarityOutput struct {
	Similarity       float32
	ModelType        string // "mmbert", "unknown"
	ProcessingTimeMs float32
}

// SimilarityMatchResult represents a single similarity match
type SimilarityMatchResult struct {
	Index      int
	Similarity float32
}

// BatchSimilarityOutput contains batch similarity results
type BatchSimilarityOutput struct {
	Matches          []SimilarityMatchResult
	ModelType        string // "mmbert", "unknown"
	ProcessingTimeMs float32
}

// ClassResult contains classification result (candle_binding compatible)
type ClassResult struct {
	Class      int
	Confidence float32
	Categories []string // Optional: categories for jailbreak detection
}

// ClassResultWithProbs contains classification result with probabilities
type ClassResultWithProbs struct {
	Class         int
	Confidence    float32
	Probabilities []float32
}

// TokenEntity represents a detected PII entity (candle_binding compatible)
type TokenEntity struct {
	Text       string
	EntityType string
	Start      int
	End        int
	Confidence float32
}

// TokenClassificationResult contains token classification results
type TokenClassificationResult struct {
	Entities []TokenEntity
}

// SimResult contains similarity result (legacy API)
type SimResult struct {
	Index int
	Score float32
}

// ModelsInfoOutput contains model information
type ModelsInfoOutput struct {
	Models []ModelInfo
}

// ModelInfo contains info about a single model
type ModelInfo struct {
	ModelName         string
	IsLoaded          bool
	MaxSequenceLength int
	DefaultDimension  int
	ModelPath         string
	SupportsLayerExit bool
	AvailableLayers   string
}

// ============================================================================
// Initialization Functions
// ============================================================================

var (
	initMu sync.Mutex
)

// InitMmBertEmbeddingModel initializes the mmBERT embedding model
// This is the ONNX Runtime equivalent of candle_binding.InitMmBertEmbeddingModel
func InitMmBertEmbeddingModel(modelPath string, useCPU bool) error {
	initMu.Lock()
	defer initMu.Unlock()

	cPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cPath))

	if !C.init_mmbert_embedding_model(cPath, C.bool(useCPU)) {
		return fmt.Errorf("failed to initialize mmBERT embedding model from %s", modelPath)
	}
	return nil
}

// IsMmBertModelInitialized checks if the embedding model is loaded
func IsMmBertModelInitialized() bool {
	return bool(C.is_mmbert_model_initialized())
}

// InitEmbeddingModels initializes embedding models (candle_binding compatible API)
// For onnx_binding, only mmBERT is supported. qwen3 and gemma paths are ignored.
func InitEmbeddingModels(qwen3ModelPath, gemmaModelPath, mmBertModelPath string, useCPU bool) error {
	// Only initialize mmBERT - qwen3 and gemma are not supported in onnx_binding
	if mmBertModelPath == "" {
		return fmt.Errorf("mmBERT model path is required for onnx_binding")
	}
	return InitMmBertEmbeddingModel(mmBertModelPath, useCPU)
}

// InitEmbeddingModelsWithMmBert is an alias for InitEmbeddingModels
func InitEmbeddingModelsWithMmBert(qwen3ModelPath, gemmaModelPath, mmBertModelPath string, useCPU bool) error {
	return InitEmbeddingModels(qwen3ModelPath, gemmaModelPath, mmBertModelPath, useCPU)
}

// InitEmbeddingModelsBatched initializes batched embedding (candle_binding compatible API)
// For onnx_binding, batching is handled internally. This uses mmBERT.
func InitEmbeddingModelsBatched(qwen3ModelPath string, maxBatchSize int, maxWaitMs uint64, useCPU bool) error {
	// onnx_binding handles batching internally, so we just initialize mmBERT
	// Use qwen3ModelPath as the model path for compatibility (caller should pass mmBERT path)
	if qwen3ModelPath == "" {
		return fmt.Errorf("model path is required")
	}
	return InitMmBertEmbeddingModel(qwen3ModelPath, useCPU)
}

// InitModel initializes the similarity model (legacy API)
func InitModel(modelID string, useCPU bool) error {
	return InitMmBertEmbeddingModel(modelID, useCPU)
}

// IsModelInitialized returns whether the model is initialized (rust state, go state)
func IsModelInitialized() (bool, bool) {
	initialized := IsMmBertModelInitialized()
	return initialized, initialized
}

// InitMmBert32KIntentClassifier initializes the intent classifier
func InitMmBert32KIntentClassifier(modelPath string, useCPU bool) error {
	return initClassifier("intent", modelPath, !useCPU)
}

// InitMmBert32KFactcheckClassifier initializes the factcheck classifier
func InitMmBert32KFactcheckClassifier(modelPath string, useCPU bool) error {
	return initClassifier("factcheck", modelPath, !useCPU)
}

// InitMmBert32KJailbreakClassifier initializes the jailbreak classifier
func InitMmBert32KJailbreakClassifier(modelPath string, useCPU bool) error {
	return initClassifier("jailbreak", modelPath, !useCPU)
}

// InitMmBert32KFeedbackClassifier initializes the feedback classifier
func InitMmBert32KFeedbackClassifier(modelPath string, useCPU bool) error {
	return initClassifier("feedback", modelPath, !useCPU)
}

// InitMmBert32KPIIClassifier initializes the PII token classifier
func InitMmBert32KPIIClassifier(modelPath string, useCPU bool) error {
	return initTokenClassifier("pii", modelPath, !useCPU)
}

func initClassifier(name, modelPath string, useGPU bool) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	cPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cPath))

	if !C.init_sequence_classifier(cName, cPath, C.bool(useGPU)) {
		return fmt.Errorf("failed to initialize %s classifier from %s", name, modelPath)
	}
	return nil
}

func initTokenClassifier(name, modelPath string, useGPU bool) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	cPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cPath))

	if !C.init_token_classifier(cName, cPath, C.bool(useGPU)) {
		return fmt.Errorf("failed to initialize %s token classifier from %s", name, modelPath)
	}
	return nil
}

// ============================================================================
// Embedding Functions
// ============================================================================

// GetEmbedding generates an embedding for the given text
func GetEmbedding(text string, maxLength int) ([]float32, error) {
	output, err := GetEmbeddingWithMetadata(text, 0, 0, 0)
	if err != nil {
		return nil, err
	}
	return output.Embedding, nil
}

// GetEmbeddingDefault generates an embedding with default settings
func GetEmbeddingDefault(text string) ([]float32, error) {
	return GetEmbedding(text, 0)
}

// GetEmbeddingWithDim generates an embedding with target dimension (Matryoshka)
func GetEmbeddingWithDim(text string, qualityPriority, latencyPriority float32, targetDim int) ([]float32, error) {
	output, err := GetEmbeddingWithMetadata(text, qualityPriority, latencyPriority, targetDim)
	if err != nil {
		return nil, err
	}
	return output.Embedding, nil
}

// GetEmbeddingWithMetadata generates embedding with full metadata
func GetEmbeddingWithMetadata(text string, qualityPriority, latencyPriority float32, targetDim int) (*EmbeddingOutput, error) {
	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	var result C.EmbeddingResult
	status := C.get_embedding_with_dim(cText, C.int(targetDim), &result)

	if status != 0 || result.error {
		return nil, errors.New("embedding generation failed")
	}

	defer C.free_embedding(result.data, result.length)

	// Copy embedding data
	embedding := make([]float32, int(result.length))
	cData := (*[1 << 30]float32)(unsafe.Pointer(result.data))[:result.length:result.length]
	copy(embedding, cData)

	return &EmbeddingOutput{
		Embedding:        embedding,
		ModelType:        modelTypeToString(int(result.model_type)),
		SequenceLength:   int(result.sequence_length),
		ProcessingTimeMs: float32(result.processing_time_ms),
	}, nil
}

// modelTypeToString converts model type int to string
func modelTypeToString(modelType int) string {
	switch modelType {
	case 0:
		return "mmbert"
	default:
		return "unknown"
	}
}

// GetEmbedding2DMatryoshka generates embedding with 2D Matryoshka (layer + dimension)
func GetEmbedding2DMatryoshka(text string, modelType string, targetLayer int, targetDim int) (*EmbeddingOutput, error) {
	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	var result C.EmbeddingResult
	status := C.get_embedding_2d_matryoshka(cText, C.int(targetLayer), C.int(targetDim), &result)

	if status != 0 || result.error {
		return nil, errors.New("2D matryoshka embedding generation failed")
	}

	defer C.free_embedding(result.data, result.length)

	embedding := make([]float32, int(result.length))
	cData := (*[1 << 30]float32)(unsafe.Pointer(result.data))[:result.length:result.length]
	copy(embedding, cData)

	return &EmbeddingOutput{
		Embedding:        embedding,
		ModelType:        modelTypeToString(int(result.model_type)),
		SequenceLength:   int(result.sequence_length),
		ProcessingTimeMs: float32(result.processing_time_ms),
	}, nil
}

// GetEmbeddingWithModelType generates embedding with specific model type
func GetEmbeddingWithModelType(text string, modelType string, targetDim int) (*EmbeddingOutput, error) {
	// ORT binding only supports mmbert, ignore modelType
	return GetEmbeddingWithMetadata(text, 0, 0, targetDim)
}

// ============================================================================
// Similarity Functions
// ============================================================================

// CalculateEmbeddingSimilarity calculates cosine similarity between two texts
func CalculateEmbeddingSimilarity(text1, text2 string, modelType string, targetDim int) (*SimilarityOutput, error) {
	cText1 := C.CString(text1)
	defer C.free(unsafe.Pointer(cText1))
	cText2 := C.CString(text2)
	defer C.free(unsafe.Pointer(cText2))

	var result C.EmbeddingSimilarityResult
	status := C.calculate_embedding_similarity(cText1, cText2, 0, C.int(targetDim), &result)

	if status != 0 || result.error {
		return nil, errors.New("similarity calculation failed")
	}

	return &SimilarityOutput{
		Similarity:       float32(result.similarity),
		ModelType:        modelTypeToString(int(result.model_type)),
		ProcessingTimeMs: float32(result.processing_time_ms),
	}, nil
}

// CalculateSimilarityBatch finds top-k most similar candidates for a query
func CalculateSimilarityBatch(query string, candidates []string, topK int, modelType string, targetDim int) (*BatchSimilarityOutput, error) {
	if len(candidates) == 0 {
		return nil, errors.New("no candidates provided")
	}

	cQuery := C.CString(query)
	defer C.free(unsafe.Pointer(cQuery))

	// Convert candidates to C strings
	cCandidates := make([]*C.char, len(candidates))
	for i, c := range candidates {
		cCandidates[i] = C.CString(c)
		defer C.free(unsafe.Pointer(cCandidates[i]))
	}

	var result C.BatchSimilarityResult
	status := C.calculate_similarity_batch(
		cQuery,
		(**C.char)(unsafe.Pointer(&cCandidates[0])),
		C.int(len(candidates)),
		C.int(topK),
		0, // target_layer
		C.int(targetDim),
		&result,
	)

	if status != 0 || result.error {
		return nil, errors.New("batch similarity calculation failed")
	}

	defer C.free_batch_similarity_result(&result)

	// Copy matches
	matches := make([]SimilarityMatchResult, int(result.num_matches))
	if result.num_matches > 0 {
		cMatches := (*[1 << 20]C.SimilarityMatch)(unsafe.Pointer(result.matches))[:result.num_matches:result.num_matches]
		for i, m := range cMatches {
			matches[i] = SimilarityMatchResult{
				Index:      int(m.index),
				Similarity: float32(m.similarity),
			}
		}
	}

	return &BatchSimilarityOutput{
		Matches:          matches,
		ModelType:        modelTypeToString(int(result.model_type)),
		ProcessingTimeMs: float32(result.processing_time_ms),
	}, nil
}

// FindMostSimilar finds the most similar candidate (legacy API)
func FindMostSimilar(query string, candidates []string, maxLength int) (int, float32) {
	result, err := CalculateSimilarityBatch(query, candidates, 1, "mmbert", 0)
	if err != nil || len(result.Matches) == 0 {
		return -1, 0.0
	}
	return result.Matches[0].Index, result.Matches[0].Similarity
}

// ============================================================================
// Classification Functions
// ============================================================================

// ClassifyMmBert32KIntent classifies text for intent
func ClassifyMmBert32KIntent(text string) (ClassResult, error) {
	return classifyWithClassifier("intent", text)
}

// ClassifyMmBert32KFactcheck classifies text for factcheck
func ClassifyMmBert32KFactcheck(text string) (ClassResult, error) {
	return classifyWithClassifier("factcheck", text)
}

// ClassifyMmBert32KJailbreak classifies text for jailbreak detection
func ClassifyMmBert32KJailbreak(text string) (ClassResult, error) {
	return classifyWithClassifier("jailbreak", text)
}

// ClassifyMmBert32KFeedback classifies text for feedback detection
func ClassifyMmBert32KFeedback(text string) (ClassResult, error) {
	return classifyWithClassifier("feedback", text)
}

// ClassifyMmBert32KPII detects PII entities in text
func ClassifyMmBert32KPII(text string, modelConfigPath string) ([]TokenEntity, error) {
	cName := C.CString("pii")
	defer C.free(unsafe.Pointer(cName))
	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	var result C.PIIResultFFI
	status := C.detect_pii(cName, cText, &result)

	if status != 0 || result.error {
		return nil, errors.New("PII detection failed")
	}

	defer C.free_pii_result(&result)

	// Copy entities
	entities := make([]TokenEntity, int(result.num_entities))
	if result.num_entities > 0 {
		cEntities := (*[1 << 20]C.PIIEntityFFI)(unsafe.Pointer(result.entities))[:result.num_entities:result.num_entities]
		for i, e := range cEntities {
			entities[i] = TokenEntity{
				Text:       C.GoString(e.text),
				EntityType: C.GoString(e.entity_type),
				Start:      int(e.start),
				End:        int(e.end),
				Confidence: float32(e.confidence),
			}
		}
	}

	return entities, nil
}

func classifyWithClassifier(name, text string) (ClassResult, error) {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	var result C.ClassificationResultFFI
	status := C.classify_text(cName, cText, &result)

	if status != 0 || result.error {
		return ClassResult{Class: -1, Confidence: 0}, fmt.Errorf("%s classification failed", name)
	}

	defer C.free_classification_result(&result)

	return ClassResult{
		Class:      int(result.class_id),
		Confidence: float32(result.confidence),
	}, nil
}

// ============================================================================
// Model Info Functions
// ============================================================================

// GetEmbeddingModelsInfo returns information about loaded embedding models
func GetEmbeddingModelsInfo() (*ModelsInfoOutput, error) {
	var result C.EmbeddingModelsInfoResult
	status := C.get_embedding_models_info(&result)

	if status != 0 || result.error {
		return nil, errors.New("failed to get model info")
	}

	defer C.free_embedding_models_info(&result)

	models := make([]ModelInfo, int(result.num_models))
	if result.num_models > 0 {
		cModels := (*[1 << 10]C.EmbeddingModelInfo)(unsafe.Pointer(result.models))[:result.num_models:result.num_models]
		for i, m := range cModels {
			models[i] = ModelInfo{
				ModelName:         C.GoString(m.model_name),
				IsLoaded:          bool(m.is_loaded),
				MaxSequenceLength: int(m.max_sequence_length),
				DefaultDimension:  int(m.default_dimension),
				ModelPath:         C.GoString(m.model_path),
				SupportsLayerExit: bool(m.supports_layer_exit),
				AvailableLayers:   C.GoString(m.available_layers),
			}
		}
	}

	return &ModelsInfoOutput{Models: models}, nil
}

// IsClassifierLoaded checks if a classifier is loaded
func IsClassifierLoaded(name string) bool {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return bool(C.is_classifier_loaded(cName))
}

// ============================================================================
// Batched Embedding Functions (candle_binding compatible)
// ============================================================================

// GetEmbeddingBatched generates embedding using batched inference
// In onnx_binding, batching is handled transparently
func GetEmbeddingBatched(text string, modelType string, targetDim int) (*EmbeddingOutput, error) {
	return GetEmbeddingWithModelType(text, modelType, targetDim)
}

// ============================================================================
// Additional candle_binding Compatible Functions
// ============================================================================

// InitCandleBertClassifier initializes a BERT classifier (stub for compatibility)
func InitCandleBertClassifier(modelPath string, numClasses int, useCPU bool) bool {
	// Map to mmBERT intent classifier
	err := initClassifier("bert", modelPath, !useCPU)
	return err == nil
}

// InitModernBertClassifier initializes ModernBERT classifier
func InitModernBertClassifier(modelPath string, useCPU bool) error {
	return initClassifier("modernbert", modelPath, !useCPU)
}

// InitModernBertJailbreakClassifier initializes ModernBERT jailbreak classifier
func InitModernBertJailbreakClassifier(modelPath string, useCPU bool) error {
	return initClassifier("jailbreak", modelPath, !useCPU)
}

// InitModernBertPIITokenClassifier initializes ModernBERT PII classifier
func InitModernBertPIITokenClassifier(modelPath string, useCPU bool) error {
	return initTokenClassifier("pii", modelPath, !useCPU)
}

// InitJailbreakClassifier initializes a jailbreak classifier
func InitJailbreakClassifier(modelPath string, numClasses int, useCPU bool) error {
	return initClassifier("jailbreak", modelPath, !useCPU)
}

// InitCandleBertTokenClassifier initializes token classifier
func InitCandleBertTokenClassifier(modelPath string, numClasses int, useCPU bool) bool {
	err := initTokenClassifier("bert_token", modelPath, !useCPU)
	return err == nil
}

// ClassifyCandleBertText classifies text using BERT
func ClassifyCandleBertText(text string) (ClassResult, error) {
	return classifyWithClassifier("bert", text)
}

// ClassifyModernBertText classifies text using ModernBERT
func ClassifyModernBertText(text string) (ClassResult, error) {
	return classifyWithClassifier("modernbert", text)
}

// ClassifyModernBertTextWithProbabilities classifies with probabilities
func ClassifyModernBertTextWithProbabilities(text string) (ClassResultWithProbs, error) {
	result, err := classifyWithClassifier("modernbert", text)
	if err != nil {
		return ClassResultWithProbs{}, err
	}
	return ClassResultWithProbs{
		Class:         result.Class,
		Confidence:    result.Confidence,
		Probabilities: []float32{}, // TODO: implement probability extraction
	}, nil
}

// ClassifyModernBertJailbreakText classifies for jailbreak
func ClassifyModernBertJailbreakText(text string) (ClassResult, error) {
	return classifyWithClassifier("jailbreak", text)
}

// ClassifyJailbreakText classifies for jailbreak (legacy)
func ClassifyJailbreakText(text string) (ClassResult, error) {
	return classifyWithClassifier("jailbreak", text)
}

// ClassifyCandleBertTokens classifies tokens
func ClassifyCandleBertTokens(text string) (TokenClassificationResult, error) {
	entities, err := ClassifyMmBert32KPII(text, "")
	if err != nil {
		return TokenClassificationResult{}, err
	}
	return TokenClassificationResult{Entities: entities}, nil
}

// CalculateSimilarity calculates similarity between two texts (legacy)
func CalculateSimilarity(text1, text2 string, maxLength int) float32 {
	result, err := CalculateEmbeddingSimilarity(text1, text2, "mmbert", 0)
	if err != nil {
		return 0.0
	}
	return result.Similarity
}

// CalculateSimilarityDefault calculates similarity with defaults
func CalculateSimilarityDefault(text1, text2 string) float32 {
	return CalculateSimilarity(text1, text2, 0)
}

// FindMostSimilarDefault finds most similar with default settings
func FindMostSimilarDefault(query string, candidates []string) SimResult {
	idx, score := FindMostSimilar(query, candidates, 0)
	return SimResult{Index: idx, Score: score}
}

// ============================================================================
// NLI Types and Constants (candle_binding compatible)
// ============================================================================

// NLILabel represents NLI classification label
type NLILabel int

const (
	// NLIEntailment means the premise supports the hypothesis
	NLIEntailment NLILabel = 0
	// NLINeutral means the premise neither supports nor contradicts
	NLINeutral NLILabel = 1
	// NLIContradiction means the premise contradicts the hypothesis
	NLIContradiction NLILabel = 2
	// NLIError means an error occurred during classification
	NLIError NLILabel = -1
)

// ============================================================================
// Hallucination Detection (stub - not implemented in onnx_binding)
// ============================================================================

// HallucinationDetectionResult represents hallucination detection output
type HallucinationDetectionResult struct {
	HasHallucination bool
	Confidence       float32
	Spans            []HallucinationSpan
}

// HallucinationSpan represents a detected hallucination span
type HallucinationSpan struct {
	Text       string
	Start      int
	End        int
	Confidence float32
	Label      string
}

// EnhancedHallucinationDetectionResult with NLI explanations
type EnhancedHallucinationDetectionResult struct {
	HasHallucination bool
	Confidence       float32
	Spans            []EnhancedHallucinationSpan
}

// EnhancedHallucinationSpan with NLI info
type EnhancedHallucinationSpan struct {
	Text                    string
	Start                   int
	End                     int
	HallucinationConfidence float32
	NLILabel                NLILabel
	NLILabelStr             string
	NLIConfidence           float32
	Severity                int
	Explanation             string
}

// NLIResult represents NLI classification result
type NLIResult struct {
	Label             NLILabel
	LabelStr          string
	Confidence        float32
	EntailmentProb    float32
	NeutralProb       float32
	ContradictionProb float32
	ContradictProb    float32 // alias for ContradictionProb
}

// InitHallucinationModel initializes the hallucination detection model
// Note: Not yet implemented in onnx_binding
func InitHallucinationModel(modelPath string, useCPU bool) error {
	return fmt.Errorf("hallucination model not yet implemented in onnx_binding")
}

// InitNLIModel initializes the NLI model
// Note: Not yet implemented in onnx_binding
func InitNLIModel(modelPath string, useCPU bool) error {
	return fmt.Errorf("NLI model not yet implemented in onnx_binding")
}

// DetectHallucinations detects hallucinations in text
// Note: Not yet implemented in onnx_binding
func DetectHallucinations(context, question, answer string, threshold float32) (*HallucinationDetectionResult, error) {
	return nil, fmt.Errorf("hallucination detection not yet implemented in onnx_binding")
}

// DetectHallucinationsWithNLI detects hallucinations with NLI explanations
// Note: Not yet implemented in onnx_binding
func DetectHallucinationsWithNLI(context, question, answer string, threshold float32) (*EnhancedHallucinationDetectionResult, error) {
	return nil, fmt.Errorf("enhanced hallucination detection not yet implemented in onnx_binding")
}

// ClassifyNLI performs NLI classification
// Note: Not yet implemented in onnx_binding
func ClassifyNLI(premise, hypothesis string) (*NLIResult, error) {
	return nil, fmt.Errorf("NLI classification not yet implemented in onnx_binding")
}

// ============================================================================
// FactCheck Classifier Functions
// ============================================================================

// InitFactCheckClassifier initializes the fact-check classifier
func InitFactCheckClassifier(modelPath string, useCPU bool) error {
	return initClassifier("factcheck", modelPath, !useCPU)
}

// ClassifyFactCheckText classifies text for fact-checking
func ClassifyFactCheckText(text string) (ClassResult, error) {
	return classifyWithClassifier("factcheck", text)
}

// ============================================================================
// Feedback Detector Functions
// ============================================================================

// InitFeedbackDetector initializes the feedback detector
func InitFeedbackDetector(modelPath string, useCPU bool) error {
	return initClassifier("feedback", modelPath, !useCPU)
}

// ClassifyFeedbackText classifies text for feedback detection
func ClassifyFeedbackText(text string) (ClassResult, error) {
	return classifyWithClassifier("feedback", text)
}

// InitQwen3PreferenceClassifier is not supported in ONNX binding (Qwen3 is Candle-only).
// Returns an error so callers can disable or fall back when using ONNX config.
func InitQwen3PreferenceClassifier(modelPath string, useCPU bool) error {
	return errors.New("Qwen3 preference classifier is not supported in ONNX binding; use Candle binding or disable preference routing")
}

// ClassifyQwen3Preference is not supported in ONNX binding (Qwen3 is Candle-only).
// Returns an error so callers can disable or fall back when using ONNX config.
func ClassifyQwen3Preference(text string, labels []string) (ClassResult, error) {
	return ClassResult{}, errors.New("Qwen3 preference classifier is not supported in ONNX binding; use Candle binding or disable preference routing")
}
