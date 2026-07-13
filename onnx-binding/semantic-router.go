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

typedef struct {
    char* dimensions;
    char* layers;
    bool supports_2d;
} MatryoshkaInfo;

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
    char* error_message;
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
extern int get_matryoshka_info(MatryoshkaInfo* result);
extern void free_embedding(float* data, int length);
extern void free_batch_similarity_result(BatchSimilarityResult* result);
extern void free_embedding_models_info(EmbeddingModelsInfoResult* result);
extern void free_matryoshka_info(MatryoshkaInfo* result);

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

// ============================================================================
// Multi-Modal Embedding Types & Functions
// ============================================================================

typedef struct {
    float* data;
    int length;
    bool error;
    int modality;
    float processing_time_ms;
} MultiModalEmbeddingResult;

extern bool init_multimodal_embedding_model(const char* model_path, bool use_cpu);
extern int multimodal_encode_text(const char* text, int target_dim, MultiModalEmbeddingResult* result);
extern int multimodal_encode_image(const float* pixel_data, int height, int width, int target_dim, MultiModalEmbeddingResult* result);
extern int multimodal_encode_audio(const float* mel_data, int n_mels, int time_frames, int target_dim, MultiModalEmbeddingResult* result);
extern void free_multimodal_embedding(float* data, int length);
*/
import "C"

import (
	"errors"
	"fmt"
	"strings"
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

// BatchSimilarityMatch represents a single match in batch similarity matching.
// Its name and fields intentionally match candle_binding so the ONNX module can
// remain a drop-in replacement selected by go.onnx.mod.
type BatchSimilarityMatch struct {
	Index      int
	Similarity float32
}

// SimilarityMatchResult preserves the original ONNX binding name.
type SimilarityMatchResult = BatchSimilarityMatch

// BatchSimilarityOutput contains batch similarity results
type BatchSimilarityOutput struct {
	Matches          []BatchSimilarityMatch
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
	NumClasses    int
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

// MatryoshkaConfig exposes the layer and dimension combinations supported by
// the ONNX mmBERT model.
type MatryoshkaConfig struct {
	Dimensions string
	Layers     string
	Supports2D bool
}

// ============================================================================
// Initialization Functions
// ============================================================================

var (
	initMu             sync.Mutex
	errEmbeddedNULByte = errors.New("input contains an embedded NUL byte")
)

func checkedCString(value, field string) (*C.char, error) {
	if strings.IndexByte(value, 0) >= 0 {
		return nil, fmt.Errorf("%w: %s", errEmbeddedNULByte, field)
	}
	return C.CString(value), nil
}

// InitMmBertEmbeddingModel initializes the mmBERT embedding model
// This is the ONNX Runtime equivalent of candle_binding.InitMmBertEmbeddingModel
func InitMmBertEmbeddingModel(modelPath string, useCPU bool) error {
	if strings.TrimSpace(modelPath) == "" {
		return errors.New("mmBERT model path is required")
	}
	initMu.Lock()
	defer initMu.Unlock()

	cPath, err := checkedCString(modelPath, "mmBERT model path")
	if err != nil {
		return err
	}
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

// InitEmbeddingModels initializes embedding models (candle_binding compatible API).
// ONNX supports only mmBERT, so unsupported requested models must fail closed
// instead of returning success for a partially initialized model union.
func InitEmbeddingModels(qwen3ModelPath, gemmaModelPath, mmBertModelPath string, useCPU bool) error {
	if strings.TrimSpace(qwen3ModelPath) != "" || strings.TrimSpace(gemmaModelPath) != "" {
		return fmt.Errorf("onnx_binding supports only mmBERT embedding models; qwen3 and gemma are unsupported")
	}
	if mmBertModelPath == "" {
		return fmt.Errorf("mmBERT model path is required for onnx_binding")
	}
	return InitMmBertEmbeddingModel(mmBertModelPath, useCPU)
}

// InitEmbeddingModelsWithMmBert is an alias for InitEmbeddingModels
func InitEmbeddingModelsWithMmBert(qwen3ModelPath, gemmaModelPath, mmBertModelPath string, useCPU bool) error {
	return InitEmbeddingModels(qwen3ModelPath, gemmaModelPath, mmBertModelPath, useCPU)
}

// InitEmbeddingModelsBatched initializes batched embedding (candle_binding compatible API).
// Its qwen3-only signature cannot express the ONNX mmBERT capability, so it is
// rejected rather than loading a qwen3 path as though it were mmBERT.
func InitEmbeddingModelsBatched(qwen3ModelPath string, maxBatchSize int, maxWaitMs uint64, useCPU bool) error {
	return fmt.Errorf("onnx_binding does not support the qwen3 batched embedding initializer")
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
	cName, err := checkedCString(name, "classifier name")
	if err != nil {
		return err
	}
	defer C.free(unsafe.Pointer(cName))
	cPath, err := checkedCString(modelPath, "classifier model path")
	if err != nil {
		return err
	}
	defer C.free(unsafe.Pointer(cPath))

	if !C.init_sequence_classifier(cName, cPath, C.bool(useGPU)) {
		return fmt.Errorf("failed to initialize %s classifier from %s", name, modelPath)
	}
	return nil
}

func initTokenClassifier(name, modelPath string, useGPU bool) error {
	cName, err := checkedCString(name, "token classifier name")
	if err != nil {
		return err
	}
	defer C.free(unsafe.Pointer(cName))
	cPath, err := checkedCString(modelPath, "token classifier model path")
	if err != nil {
		return err
	}
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
	if err := validateONNXNonNegativeCInt("maximum token length", maxLength); err != nil {
		return nil, err
	}
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
	if err := validateONNXEmbeddingPriorities(qualityPriority, latencyPriority); err != nil {
		return nil, err
	}
	if err := validateONNXNonNegativeCInt("target dimension", targetDim); err != nil {
		return nil, err
	}
	cText, err := checkedCString(text, "embedding text")
	if err != nil {
		return nil, err
	}
	defer C.free(unsafe.Pointer(cText))

	var result C.EmbeddingResult
	status := C.get_embedding_with_dim(cText, C.int(targetDim), &result)

	if status != 0 {
		return nil, embeddingStatusError("embedding generation failed", int(status))
	}
	if result.error {
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
	case 3:
		return "multimodal"
	default:
		return "unknown"
	}
}

func validateONNXEmbeddingModelType(modelType string, allowMultimodal bool) (string, error) {
	normalized := strings.ToLower(strings.TrimSpace(modelType))
	if normalized == "mmbert" {
		return normalized, nil
	}
	if allowMultimodal && normalized == "multimodal" {
		return normalized, nil
	}
	return "", fmt.Errorf("unsupported ONNX embedding model type %q", modelType)
}

// GetEmbedding2DMatryoshka generates embedding with 2D Matryoshka (layer + dimension)
func GetEmbedding2DMatryoshka(text string, modelType string, targetLayer int, targetDim int) (*EmbeddingOutput, error) {
	if err := validateONNXEmbeddingControls(targetLayer, targetDim); err != nil {
		return nil, err
	}
	if _, err := validateONNXEmbeddingModelType(modelType, false); err != nil {
		return nil, err
	}
	cText, err := checkedCString(text, "2D matryoshka embedding text")
	if err != nil {
		return nil, err
	}
	defer C.free(unsafe.Pointer(cText))

	var result C.EmbeddingResult
	status := C.get_embedding_2d_matryoshka(cText, C.int(targetLayer), C.int(targetDim), &result)

	if status != 0 {
		return nil, embeddingStatusError("2D matryoshka embedding generation failed", int(status))
	}
	if result.error {
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
	normalized, err := validateONNXEmbeddingModelType(modelType, true)
	if err != nil {
		return nil, err
	}
	switch normalized {
	case "multimodal":
		output, err := MultiModalEncodeText(text, targetDim)
		if err != nil {
			return nil, err
		}
		return &EmbeddingOutput{
			Embedding:        output.Embedding,
			ModelType:        "multimodal",
			SequenceLength:   0,
			ProcessingTimeMs: output.ProcessingTimeMs,
		}, nil
	case "mmbert":
		return GetEmbeddingWithMetadata(text, 0, 0, targetDim)
	default:
		panic("validated ONNX embedding model type became unreachable")
	}
}

// GetEmbeddingsBatch generates embeddings for multiple texts in one native
// batch while preserving the 2D Matryoshka layer and dimension controls.
func GetEmbeddingsBatch(texts []string, targetLayer, targetDim int) ([]*EmbeddingOutput, error) {
	if err := validateONNXEmbeddingControls(targetLayer, targetDim); err != nil {
		return nil, err
	}
	if len(texts) == 0 {
		return nil, errors.New("no texts provided")
	}
	if err := validateONNXNonNegativeCInt("embedding batch size", len(texts)); err != nil {
		return nil, err
	}

	cTexts := make([]*C.char, len(texts))
	for i, value := range texts {
		var err error
		cTexts[i], err = checkedCString(value, fmt.Sprintf("embedding texts[%d]", i))
		if err != nil {
			return nil, err
		}
		defer C.free(unsafe.Pointer(cTexts[i]))
	}
	results := make([]C.EmbeddingResult, len(texts))
	status := C.get_embeddings_batch(
		(**C.char)(unsafe.Pointer(&cTexts[0])),
		C.int(len(cTexts)),
		C.int(targetLayer),
		C.int(targetDim),
		&results[0],
	)
	defer freeEmbeddingResults(results)
	if status != 0 {
		return nil, embeddingStatusError("batch embedding generation failed", int(status))
	}

	outputs := make([]*EmbeddingOutput, len(results))
	for i := range results {
		result := &results[i]
		if result.error || result.data == nil || result.length <= 0 {
			return nil, fmt.Errorf("batch embedding generation failed at index %d", i)
		}
		length := int(result.length)
		embedding := make([]float32, length)
		copy(embedding, unsafe.Slice((*float32)(unsafe.Pointer(result.data)), length))
		outputs[i] = &EmbeddingOutput{
			Embedding:        embedding,
			ModelType:        modelTypeToString(int(result.model_type)),
			SequenceLength:   int(result.sequence_length),
			ProcessingTimeMs: float32(result.processing_time_ms),
		}
	}
	return outputs, nil
}

func freeEmbeddingResults(results []C.EmbeddingResult) {
	for i := range results {
		if results[i].data != nil && results[i].length > 0 {
			C.free_embedding(results[i].data, results[i].length)
		}
	}
}

// ============================================================================
// Similarity Functions
// ============================================================================

// CalculateEmbeddingSimilarity calculates cosine similarity between two texts
// with the legacy parameter surface.
func CalculateEmbeddingSimilarity(text1, text2 string, modelType string, targetDim int) (*SimilarityOutput, error) {
	return CalculateEmbeddingSimilarityWithOptions(text1, text2, SimilarityOptions{
		ModelType:       modelType,
		TargetDim:       targetDim,
		QualityPriority: 0.5,
		LatencyPriority: 0.5,
	})
}

// CalculateEmbeddingSimilarityWithOptions calculates pair similarity with 2D
// Matryoshka controls. ONNX auto always resolves to mmBERT; priorities are
// validated for cross-backend API parity but do not influence model selection.
func CalculateEmbeddingSimilarityWithOptions(text1, text2 string, options SimilarityOptions) (*SimilarityOutput, error) {
	normalized, err := normalizeONNXSimilarityOptions(options)
	if err != nil {
		return nil, err
	}
	cText1, err := checkedCString(text1, "similarity text1")
	if err != nil {
		return nil, err
	}
	defer C.free(unsafe.Pointer(cText1))
	cText2, err := checkedCString(text2, "similarity text2")
	if err != nil {
		return nil, err
	}
	defer C.free(unsafe.Pointer(cText2))

	var result C.EmbeddingSimilarityResult
	status := C.calculate_embedding_similarity(
		cText1,
		cText2,
		C.int(normalized.TargetLayer),
		C.int(normalized.TargetDim),
		&result,
	)

	if status != 0 {
		return nil, embeddingStatusError("similarity calculation failed", int(status))
	}
	if result.error {
		return nil, errors.New("similarity calculation failed")
	}

	return &SimilarityOutput{
		Similarity:       float32(result.similarity),
		ModelType:        modelTypeToString(int(result.model_type)),
		ProcessingTimeMs: float32(result.processing_time_ms),
	}, nil
}

// CalculateSimilarityBatch finds top-k most similar candidates for a query
// with the legacy parameter surface.
func CalculateSimilarityBatch(query string, candidates []string, topK int, modelType string, targetDim int) (*BatchSimilarityOutput, error) {
	return CalculateSimilarityBatchWithOptions(query, candidates, topK, SimilarityOptions{
		ModelType:       modelType,
		TargetDim:       targetDim,
		QualityPriority: 0.5,
		LatencyPriority: 0.5,
	})
}

// CalculateSimilarityBatchWithOptions finds the top-k candidates with 2D
// Matryoshka controls. ONNX auto always resolves to mmBERT; priorities are
// validated for cross-backend API parity but do not influence model selection.
func CalculateSimilarityBatchWithOptions(query string, candidates []string, topK int, options SimilarityOptions) (*BatchSimilarityOutput, error) {
	normalized, err := normalizeONNXSimilarityOptions(options)
	if err != nil {
		return nil, err
	}
	if err := validateONNXNonNegativeCInt("top-k", topK); err != nil {
		return nil, err
	}
	if err := validateONNXNonNegativeCInt("candidate count", len(candidates)); err != nil {
		return nil, err
	}
	if len(candidates) == 0 {
		return nil, errors.New("no candidates provided")
	}

	cQuery, err := checkedCString(query, "similarity query")
	if err != nil {
		return nil, err
	}
	defer C.free(unsafe.Pointer(cQuery))

	// Convert candidates to C strings
	cCandidates := make([]*C.char, len(candidates))
	for i, c := range candidates {
		cCandidates[i], err = checkedCString(c, fmt.Sprintf("similarity candidates[%d]", i))
		if err != nil {
			return nil, err
		}
		defer C.free(unsafe.Pointer(cCandidates[i]))
	}

	var result C.BatchSimilarityResult
	status := C.calculate_similarity_batch(
		cQuery,
		(**C.char)(unsafe.Pointer(&cCandidates[0])),
		C.int(len(candidates)),
		C.int(topK),
		C.int(normalized.TargetLayer),
		C.int(normalized.TargetDim),
		&result,
	)

	if status != 0 {
		return nil, embeddingStatusError("batch similarity calculation failed", int(status))
	}
	if result.error {
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
func FindMostSimilar(query string, candidates []string, maxLength int) SimResult {
	if err := validateONNXNonNegativeCInt("maximum token length", maxLength); err != nil {
		return SimResult{Index: -1, Score: -1.0}
	}
	if err := validateONNXNonNegativeCInt("candidate count", len(candidates)); err != nil {
		return SimResult{Index: -1, Score: -1.0}
	}
	result, err := CalculateSimilarityBatch(query, candidates, 1, "mmbert", 0)
	if err != nil || len(result.Matches) == 0 {
		return SimResult{Index: -1, Score: -1.0}
	}
	return SimResult{
		Index: result.Matches[0].Index,
		Score: result.Matches[0].Similarity,
	}
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
func ClassifyMmBert32KPII(text string) ([]TokenEntity, error) {
	cName, err := checkedCString("pii", "PII classifier name")
	if err != nil {
		return nil, err
	}
	defer C.free(unsafe.Pointer(cName))
	cText, err := checkedCString(text, "PII classification text")
	if err != nil {
		return nil, err
	}
	defer C.free(unsafe.Pointer(cText))

	var result C.PIIResultFFI
	status := C.detect_pii(cName, cText, &result)
	defer C.free_pii_result(&result)

	if status != 0 || result.error {
		if result.error_message != nil {
			return nil, fmt.Errorf(
				"PII detection failed (status=%d, ffi_error=%t): %s",
				int(status),
				bool(result.error),
				C.GoString(result.error_message),
			)
		}
		return nil, fmt.Errorf("PII detection failed (status=%d, ffi_error=%t)", int(status), bool(result.error))
	}

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
	result, err := callClassifier(name, text)
	if err != nil {
		return ClassResult{Class: -1}, err
	}
	defer C.free_classification_result(&result)
	return ClassResult{
		Class:      int(result.class_id),
		Confidence: float32(result.confidence),
	}, nil
}

// callClassifier transfers ownership of any native result allocations to its
// caller on success. Every successful caller must call
// C.free_classification_result exactly once.
func callClassifier(name, text string) (C.ClassificationResultFFI, error) {
	cName, err := checkedCString(name, "classifier name")
	if err != nil {
		return C.ClassificationResultFFI{}, err
	}
	defer C.free(unsafe.Pointer(cName))
	cText, err := checkedCString(text, "classification text")
	if err != nil {
		return C.ClassificationResultFFI{}, err
	}
	defer C.free(unsafe.Pointer(cText))

	var result C.ClassificationResultFFI
	status := C.classify_text(cName, cText, &result)
	if status != 0 || result.error {
		C.free_classification_result(&result)
		return C.ClassificationResultFFI{}, fmt.Errorf("%s classification failed", name)
	}
	return result, nil
}

func classifyWithClassifierProbabilities(name, text string) (ClassResultWithProbs, error) {
	result, err := callClassifier(name, text)
	if err != nil {
		return ClassResultWithProbs{}, err
	}
	defer C.free_classification_result(&result)

	if result.num_classes < 0 {
		return ClassResultWithProbs{}, fmt.Errorf("%s classification returned a negative class count", name)
	}

	numClasses := int(result.num_classes)
	var nativeProbabilities []float32
	if numClasses > 0 {
		if result.probabilities == nil {
			return ClassResultWithProbs{}, fmt.Errorf("%s classification returned no probability data", name)
		}
		nativeProbabilities = unsafe.Slice((*float32)(unsafe.Pointer(result.probabilities)), numClasses)
	}

	return ownedClassResultWithProbabilities(
		int(result.class_id),
		float32(result.confidence),
		nativeProbabilities,
	), nil
}

// ownedClassResultWithProbabilities copies probability data out of native-owned
// memory before the FFI result is released.
func ownedClassResultWithProbabilities(class int, confidence float32, probabilities []float32) ClassResultWithProbs {
	owned := make([]float32, len(probabilities))
	copy(owned, probabilities)
	return ClassResultWithProbs{
		Class:         class,
		Confidence:    confidence,
		Probabilities: owned,
		NumClasses:    len(owned),
	}
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

// GetMatryoshkaConfig returns the native 2D Matryoshka capability contract.
func GetMatryoshkaConfig() (*MatryoshkaConfig, error) {
	var result C.MatryoshkaInfo
	if status := C.get_matryoshka_info(&result); status != 0 {
		return nil, fmt.Errorf("get Matryoshka configuration failed (status: %d)", int(status))
	}
	defer C.free_matryoshka_info(&result)
	if result.dimensions == nil || result.layers == nil {
		return nil, errors.New("get Matryoshka configuration returned empty fields")
	}
	return &MatryoshkaConfig{
		Dimensions: C.GoString(result.dimensions),
		Layers:     C.GoString(result.layers),
		Supports2D: bool(result.supports_2d),
	}, nil
}

// IsClassifierLoaded checks if a classifier is loaded
func IsClassifierLoaded(name string) bool {
	cName, err := checkedCString(name, "classifier name")
	if err != nil {
		return false
	}
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
	return classifyWithClassifierProbabilities("modernbert", text)
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
	entities, err := ClassifyMmBert32KPII(text)
	if err != nil {
		return TokenClassificationResult{}, err
	}
	return TokenClassificationResult{Entities: entities}, nil
}

// CalculateSimilarity calculates similarity between two texts (legacy)
func CalculateSimilarity(text1, text2 string, maxLength int) float32 {
	if err := validateONNXNonNegativeCInt("maximum token length", maxLength); err != nil {
		return -1.0
	}
	result, err := CalculateEmbeddingSimilarity(text1, text2, "mmbert", 0)
	if err != nil {
		return -1.0
	}
	return result.Similarity
}

// CalculateSimilarityDefault calculates similarity with defaults
func CalculateSimilarityDefault(text1, text2 string) float32 {
	return CalculateSimilarity(text1, text2, 0)
}

// FindMostSimilarDefault finds most similar with default settings
func FindMostSimilarDefault(query string, candidates []string) SimResult {
	return FindMostSimilar(query, candidates, 512)
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

// String returns the Candle-compatible string representation of an NLI label.
func (l NLILabel) String() string {
	switch l {
	case NLIEntailment:
		return "ENTAILMENT"
	case NLINeutral:
		return "NEUTRAL"
	case NLIContradiction:
		return "CONTRADICTION"
	default:
		return "ERROR"
	}
}

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

// NLIClassificationResult mirrors the Candle replacement-module API.
type NLIClassificationResult struct {
	Label          NLILabel `json:"label"`
	LabelStr       string   `json:"label_str"`
	Confidence     float32  `json:"confidence"`
	EntailmentProb float32  `json:"entailment_prob"`
	NeutralProb    float32  `json:"neutral_prob"`
	ContradictProb float32  `json:"contradiction_prob"`
	// ContradictionProb is retained for early ONNX consumers.
	ContradictionProb float32 `json:"-"`
}

// NLIResult is retained as a source-compatible alias for early ONNX consumers.
type NLIResult = NLIClassificationResult

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
func ClassifyNLI(premise, hypothesis string) (*NLIClassificationResult, error) {
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

// ============================================================================
// Modality Classification (stub — Candle-only)
// ============================================================================

// ModalityResult represents the output of modality routing classification.
type ModalityResult struct {
	Modality   string
	ClassID    int
	Confidence float32
}

// InitMmBert32KModalityClassifier is not supported in ONNX binding (Candle-only).
func InitMmBert32KModalityClassifier(modelPath string, useCPU bool) error {
	return errors.New("modality classifier is not supported in ONNX binding; use Candle binding or disable modality routing")
}

// ClassifyMmBert32KModality is not supported in ONNX binding (Candle-only).
func ClassifyMmBert32KModality(text string) (ModalityResult, error) {
	return ModalityResult{}, errors.New("modality classification is not supported in ONNX binding; use Candle binding or disable modality routing")
}

// ============================================================================
// MLP Selector for Model Selection (stub — Candle-only, GPU-accelerated)
// ============================================================================

// MLPDeviceType defines the device type for MLP inference.
type MLPDeviceType int

const (
	MLPDeviceCPU   MLPDeviceType = 0
	MLPDeviceCUDA  MLPDeviceType = 1
	MLPDeviceMetal MLPDeviceType = 2
)

// MLPDType defines the data type for mixed precision inference.
type MLPDType int

const (
	MLPF32  MLPDType = 0
	MLPF16  MLPDType = 1
	MLPBF16 MLPDType = 2
)

// MLPSelector is a stub for the Candle MLP implementation.
type MLPSelector struct{}

// NewMLPSelector is not supported in ONNX binding (Candle-only).
func NewMLPSelector() *MLPSelector { return &MLPSelector{} }

// NewMLPSelectorWithDevice is not supported in ONNX binding (Candle-only).
func NewMLPSelectorWithDevice(deviceType MLPDeviceType) *MLPSelector { return &MLPSelector{} }

// NewMLPSelectorWithDeviceAndDType is not supported in ONNX binding (Candle-only).
func NewMLPSelectorWithDeviceAndDType(deviceType MLPDeviceType, dtype MLPDType) *MLPSelector {
	return &MLPSelector{}
}

func (s *MLPSelector) Close()          {}
func (s *MLPSelector) IsTrained() bool { return false }
func (s *MLPSelector) ToJSON() (string, error) {
	return "", errors.New("MLP not supported in ONNX binding")
}
func (s *MLPSelector) Select(query []float64) (string, error) {
	return "", errors.New("MLP selector is not supported in ONNX binding; use Candle binding")
}

// MLPFromJSON is not supported in ONNX binding (Candle-only).
func MLPFromJSON(jsonStr string) (*MLPSelector, error) {
	return nil, errors.New("MLP selector is not supported in ONNX binding; use Candle binding")
}

// MLPFromJSONWithDevice is not supported in ONNX binding (Candle-only).
func MLPFromJSONWithDevice(jsonStr string, deviceType MLPDeviceType) (*MLPSelector, error) {
	return nil, errors.New("MLP selector is not supported in ONNX binding; use Candle binding")
}

// MLPFromJSONWithDeviceAndDType is not supported in ONNX binding (Candle-only).
func MLPFromJSONWithDeviceAndDType(jsonStr string, deviceType MLPDeviceType, dtype MLPDType) (*MLPSelector, error) {
	return nil, errors.New("MLP selector is not supported in ONNX binding; use Candle binding")
}

// ============================================================================
// Multi-Modal Embedding (ONNX Runtime — text, image, audio)
// ============================================================================

// MultiModalEmbeddingOutput represents the result of a multi-modal embedding.
type MultiModalEmbeddingOutput struct {
	Embedding        []float32
	Modality         string
	ProcessingTimeMs float32
}

func modalityToString(m int) string {
	switch m {
	case 0:
		return "text"
	case 1:
		return "image"
	case 2:
		return "audio"
	default:
		return "unknown"
	}
}

// InitMultiModalEmbeddingModel loads the multi-modal ONNX model.
// modelPath can contain encoders/tokenizer either at root or under modelPath/onnx.
func InitMultiModalEmbeddingModel(modelPath string, useCPU bool) error {
	cPath, err := checkedCString(modelPath, "multi-modal model path")
	if err != nil {
		return err
	}
	defer C.free(unsafe.Pointer(cPath))
	if !C.init_multimodal_embedding_model(cPath, C.bool(useCPU)) {
		return fmt.Errorf("failed to initialize multi-modal embedding model from %s", modelPath)
	}
	return nil
}

// MultiModalEncodeText encodes text into a shared multi-modal embedding space.
func MultiModalEncodeText(text string, targetDim int) (*MultiModalEmbeddingOutput, error) {
	if err := validateONNXNonNegativeCInt("target dimension", targetDim); err != nil {
		return nil, err
	}
	cText, err := checkedCString(text, "multi-modal text")
	if err != nil {
		return nil, err
	}
	defer C.free(unsafe.Pointer(cText))

	var result C.MultiModalEmbeddingResult
	status := C.multimodal_encode_text(cText, C.int(targetDim), &result)
	if status != 0 {
		return nil, embeddingStatusError("multi-modal text encoding failed", int(status))
	}
	if result.error {
		return nil, errors.New("multi-modal text encoding failed")
	}
	if result.data == nil || result.length <= 0 {
		return nil, errors.New("multi-modal text encoding returned empty result")
	}
	defer C.free_multimodal_embedding(result.data, result.length)

	emb := make([]float32, int(result.length))
	cData := (*[1 << 30]float32)(unsafe.Pointer(result.data))[:result.length:result.length]
	copy(emb, cData)

	return &MultiModalEmbeddingOutput{
		Embedding:        emb,
		Modality:         modalityToString(int(result.modality)),
		ProcessingTimeMs: float32(result.processing_time_ms),
	}, nil
}

// MultiModalEncodeImage encodes pre-processed pixel data (CHW, float32 [0,1]).
func MultiModalEncodeImage(pixelData []float32, height, width, targetDim int) (*MultiModalEmbeddingOutput, error) {
	if err := validateONNXNonNegativeCInt("target dimension", targetDim); err != nil {
		return nil, err
	}
	if len(pixelData) == 0 {
		return nil, errors.New("pixelData cannot be empty")
	}
	if err := validateImageGeometry(width, height); err != nil {
		return nil, fmt.Errorf("invalid pixelData geometry: %w", err)
	}
	expected := 3 * height * width
	if len(pixelData) != expected {
		return nil, fmt.Errorf("expected %d floats (3×%d×%d), got %d", expected, height, width, len(pixelData))
	}

	var result C.MultiModalEmbeddingResult
	status := C.multimodal_encode_image(
		(*C.float)(unsafe.Pointer(&pixelData[0])),
		C.int(height), C.int(width), C.int(targetDim), &result,
	)
	if status != 0 || result.error {
		return nil, errors.New("multi-modal image encoding failed")
	}
	if result.data == nil || result.length <= 0 {
		return nil, errors.New("multi-modal image encoding returned empty result")
	}
	defer C.free_multimodal_embedding(result.data, result.length)

	emb := make([]float32, int(result.length))
	cData := (*[1 << 30]float32)(unsafe.Pointer(result.data))[:result.length:result.length]
	copy(emb, cData)

	return &MultiModalEmbeddingOutput{
		Embedding:        emb,
		Modality:         modalityToString(int(result.modality)),
		ProcessingTimeMs: float32(result.processing_time_ms),
	}, nil
}

// MultiModalEncodeAudio encodes a mel spectrogram [nMels × timeFrames].
func MultiModalEncodeAudio(melData []float32, nMels, timeFrames, targetDim int) (*MultiModalEmbeddingOutput, error) {
	if err := validateMultiModalAudioInput(melData, nMels, timeFrames, targetDim); err != nil {
		return nil, err
	}

	var result C.MultiModalEmbeddingResult
	status := C.multimodal_encode_audio(
		(*C.float)(unsafe.Pointer(&melData[0])),
		C.int(nMels), C.int(timeFrames), C.int(targetDim), &result,
	)
	if status != 0 || result.error {
		return nil, errors.New("multi-modal audio encoding failed")
	}
	if result.data == nil || result.length <= 0 {
		return nil, errors.New("multi-modal audio encoding returned empty result")
	}
	defer C.free_multimodal_embedding(result.data, result.length)

	emb := make([]float32, int(result.length))
	cData := (*[1 << 30]float32)(unsafe.Pointer(result.data))[:result.length:result.length]
	copy(emb, cData)

	return &MultiModalEmbeddingOutput{
		Embedding:        emb,
		Modality:         modalityToString(int(result.modality)),
		ProcessingTimeMs: float32(result.processing_time_ms),
	}, nil
}

func validateMultiModalAudioInput(melData []float32, nMels, timeFrames, targetDim int) error {
	if err := validateONNXNonNegativeCInt("target dimension", targetDim); err != nil {
		return err
	}
	if len(melData) == 0 {
		return errors.New("melData cannot be empty")
	}
	if nMels <= 0 || timeFrames <= 0 {
		return fmt.Errorf("nMels and timeFrames must be positive, got %d×%d", nMels, timeFrames)
	}
	const maxCInt = 1<<31 - 1
	if nMels > maxCInt || timeFrames > maxCInt {
		return fmt.Errorf("nMels and timeFrames must fit C int, got %d×%d", nMels, timeFrames)
	}
	if nMels > len(melData)/timeFrames {
		return fmt.Errorf("melData length %d does not match %d×%d", len(melData), nMels, timeFrames)
	}
	expected := nMels * timeFrames
	if len(melData) != expected {
		return fmt.Errorf("melData length %d does not match %d×%d", len(melData), nMels, timeFrames)
	}
	return nil
}
