//go:build !windows && cgo && (amd64 || arm64)
// +build !windows
// +build cgo
// +build amd64 arm64

// Package onnx_binding provides Go bindings for ONNX Runtime-based semantic embedding
// with 2D Matryoshka support for AMD GPU (ROCm), NVIDIA GPU (CUDA), and CPU inference.
package onnx_binding

import (
	"fmt"
	"log"
	"sync"
	"unsafe"
)

/*
#cgo LDFLAGS: -L${SRCDIR}/target/release -lonnx_semantic_router -ldl -lm
#include <stdlib.h>
#include <stdbool.h>

// Embedding result structure
typedef struct {
    float* data;
    int length;
    bool error;
    int model_type;           // 0=mmbert, -1=error
    int sequence_length;      // Sequence length in tokens
    float processing_time_ms; // Processing time in milliseconds
} EmbeddingResult;

// Embedding similarity result structure
typedef struct {
    float similarity;         // Cosine similarity score (-1.0 to 1.0)
    int model_type;           // 0=mmbert, -1=error
    float processing_time_ms; // Processing time in milliseconds
    bool error;               // Whether an error occurred
} EmbeddingSimilarityResult;

// Batch similarity match structure
typedef struct {
    int index;        // Index of the candidate in the input array
    float similarity; // Cosine similarity score
} SimilarityMatch;

// Batch similarity result structure
typedef struct {
    SimilarityMatch* matches; // Array of top-k matches, sorted by similarity (descending)
    int num_matches;          // Number of matches returned (≤ top_k)
    int model_type;           // 0=mmbert, -1=error
    float processing_time_ms; // Processing time in milliseconds
    bool error;               // Whether an error occurred
} BatchSimilarityResult;

// Single embedding model information
typedef struct {
    char* model_name;          // "mmbert"
    bool is_loaded;            // Whether the model is loaded
    int max_sequence_length;   // Maximum sequence length
    int default_dimension;     // Default embedding dimension
    char* model_path;          // Model path (can be null if not loaded)
    bool supports_layer_exit;  // Whether layer early exit is supported
    char* available_layers;    // Available exit layers (comma-separated)
} EmbeddingModelInfo;

// Embedding models information result
typedef struct {
    EmbeddingModelInfo* models; // Array of model info
    int num_models;             // Number of models
    bool error;                 // Whether an error occurred
} EmbeddingModelsInfoResult;

// Matryoshka configuration info
typedef struct {
    char* dimensions;  // Supported dimensions (comma-separated)
    char* layers;      // Supported layers (comma-separated)
    bool supports_2d;  // Whether 2D Matryoshka is supported
} MatryoshkaInfo;

// FFI function declarations
extern bool init_mmbert_embedding_model(const char* model_path, bool use_cpu);
extern bool is_mmbert_model_initialized();
extern int get_embedding_2d_matryoshka(const char* text, int target_layer, int target_dim, EmbeddingResult* result);
extern int get_embedding(const char* text, EmbeddingResult* result);
extern int get_embedding_with_dim(const char* text, int target_dim, EmbeddingResult* result);
extern int get_embeddings_batch(const char** texts, int num_texts, int target_layer, int target_dim, EmbeddingResult* results);
extern int calculate_embedding_similarity(const char* text1, const char* text2, int target_layer, int target_dim, EmbeddingSimilarityResult* result);
extern int calculate_similarity_batch(const char* query, const char** candidates, int num_candidates, int top_k, int target_layer, int target_dim, BatchSimilarityResult* result);
extern int get_embedding_models_info(EmbeddingModelsInfoResult* result);
extern int get_matryoshka_info(MatryoshkaInfo* result);
extern void free_embedding(float* data, int length);
extern void free_batch_similarity_result(BatchSimilarityResult* result);
extern void free_embedding_models_info(EmbeddingModelsInfoResult* result);
extern void free_matryoshka_info(MatryoshkaInfo* result);
extern void free_cstring(char* s);
*/
import "C"

var (
	initOnce         sync.Once
	initErr          error
	modelInitialized bool
)

// EmbeddingOutput represents the complete embedding generation result with metadata
type EmbeddingOutput struct {
	Embedding        []float32 // The embedding vector
	ModelType        string    // Model used: "mmbert"
	SequenceLength   int       // Sequence length in tokens
	ProcessingTimeMs float32   // Processing time in milliseconds
}

// SimilarityOutput represents the result of embedding similarity calculation
type SimilarityOutput struct {
	Similarity       float32 // Cosine similarity score (-1.0 to 1.0)
	ModelType        string  // Model used: "mmbert"
	ProcessingTimeMs float32 // Processing time in milliseconds
}

// BatchSimilarityMatch represents a single match in batch similarity matching
type BatchSimilarityMatch struct {
	Index      int     // Index of the candidate in the input array
	Similarity float32 // Cosine similarity score
}

// BatchSimilarityOutput holds the result of batch similarity matching
type BatchSimilarityOutput struct {
	Matches          []BatchSimilarityMatch // Top-k matches, sorted by similarity (descending)
	ModelType        string                 // Model used: "mmbert"
	ProcessingTimeMs float32                // Processing time in milliseconds
}

// ModelInfo represents information about a single embedding model
type ModelInfo struct {
	ModelName         string // "mmbert"
	IsLoaded          bool   // Whether the model is loaded
	MaxSequenceLength int    // Maximum sequence length
	DefaultDimension  int    // Default embedding dimension
	ModelPath         string // Model path
	SupportsLayerExit bool   // Whether layer early exit is supported
	AvailableLayers   string // Available exit layers (comma-separated)
}

// ModelsInfoOutput holds information about all embedding models
type ModelsInfoOutput struct {
	Models []ModelInfo // Array of model information
}

// MatryoshkaConfig holds the 2D Matryoshka configuration
type MatryoshkaConfig struct {
	Dimensions string // Supported dimensions (comma-separated, e.g., "768,512,256,128,64")
	Layers     string // Supported layers (comma-separated, e.g., "3,6,11,22")
	Supports2D bool   // Whether 2D Matryoshka is supported
}

// InitMmBertEmbeddingModel initializes the mmBERT embedding model with 2D Matryoshka support.
//
// This model supports:
//   - 32K context length (YaRN-scaled RoPE)
//   - Multilingual (1800+ languages via Glot500)
//   - 2D Matryoshka: dimension reduction (768→64) AND layer early exit (22→3 layers)
//   - AMD GPU via ROCm, NVIDIA GPU via CUDA, or CPU
//
// After initialization, use GetEmbedding2DMatryoshka to generate embeddings.
//
// Parameters:
//   - modelPath: Path to the mmBERT model directory (must contain model.onnx and tokenizer.json)
//   - useCPU: If true, use CPU for inference; if false, use best available GPU (ROCm/CUDA)
//
// Returns:
//   - error: Non-nil if initialization fails
//
// Example:
//
//	err := InitMmBertEmbeddingModel("/path/to/mmbert-embed-32k-2d-matryoshka", false)
//	if err != nil {
//	    log.Fatal(err)
//	}
//	// Generate embedding with early exit (6 layers, 256 dimensions)
//	output, err := GetEmbedding2DMatryoshka("Hello world", 6, 256)
func InitMmBertEmbeddingModel(modelPath string, useCPU bool) error {
	// Check if already initialized
	if bool(C.is_mmbert_model_initialized()) {
		modelInitialized = true
		return nil
	}

	initOnce.Do(func() {
		if modelPath == "" {
			initErr = fmt.Errorf("modelPath cannot be empty")
			return
		}

		log.Printf("Initializing mmBERT embedding model (ONNX Runtime): %s", modelPath)

		cModelPath := C.CString(modelPath)
		defer C.free(unsafe.Pointer(cModelPath))

		success := C.init_mmbert_embedding_model(cModelPath, C.bool(useCPU))
		if !bool(success) {
			initErr = fmt.Errorf("failed to initialize mmBERT embedding model")
			return
		}

		modelInitialized = true
		log.Printf("INFO: mmBERT embedding model initialized with 2D Matryoshka support (ONNX Runtime)")
	})

	if initErr != nil {
		initOnce = sync.Once{}
		modelInitialized = false
	}

	return initErr
}

// IsModelInitialized returns whether the mmBERT model is initialized
func IsModelInitialized() bool {
	return bool(C.is_mmbert_model_initialized())
}

// cFloatArrayToGoSlice converts a C array of floats to a Go slice and frees the C memory
func cFloatArrayToGoSlice(data *C.float, length C.int) []float32 {
	if data == nil || length == 0 {
		return nil
	}

	l := int(length)
	out := make([]float32, l)

	// Create a slice that refers to the C array
	cArray := (*[1 << 30]C.float)(unsafe.Pointer(data))[:l:l]

	// Copy and convert each value
	for i := 0; i < l; i++ {
		out[i] = float32(cArray[i])
	}

	// Free the memory allocated in Rust
	C.free_embedding(data, length)
	return out
}

// GetEmbedding2DMatryoshka generates embeddings with 2D Matryoshka support.
//
// This function supports the full 2D Matryoshka API:
//   - Layer early exit: Use fewer layers (3, 6, 11, or 22) for faster inference
//   - Dimension truncation: Use smaller dimensions (64, 128, 256, 512, 768)
//
// Parameters:
//   - text: Input text to generate embedding for
//   - targetLayer: Target layer for early exit (0 for full model, recommended: 3/6/11/22)
//   - targetDim: Target embedding dimension (0 for default 768, recommended: 64/128/256/512/768)
//
// Returns:
//   - EmbeddingOutput containing the embedding vector and metadata
//   - error if embedding generation fails
//
// Example for early exit (6 layers, 256 dimensions):
//
//	output, err := GetEmbedding2DMatryoshka("Hello world", 6, 256)
//	fmt.Printf("Embedding dim: %d, took %.2fms\n", len(output.Embedding), output.ProcessingTimeMs)
//
// Quality vs Speed Tradeoffs:
//   - Layer 22, Dim 768: Best quality (100%), baseline speed
//   - Layer 11, Dim 512: Good quality (~67%), ~2x faster
//   - Layer 6, Dim 256: Moderate quality (~56%), ~3.7x faster
//   - Layer 3, Dim 64: Fastest (~55%), ~7.3x faster
func GetEmbedding2DMatryoshka(text string, targetLayer int, targetDim int) (*EmbeddingOutput, error) {
	if !modelInitialized {
		return nil, fmt.Errorf("mmBERT model not initialized. Call InitMmBertEmbeddingModel first")
	}

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	var result C.EmbeddingResult
	status := C.get_embedding_2d_matryoshka(
		cText,
		C.int(targetLayer),
		C.int(targetDim),
		&result,
	)

	// Check status code (0 = success, -1 = error)
	if status != 0 || result.error {
		return nil, fmt.Errorf("failed to generate embedding (status: %d)", status)
	}

	// Convert C array to Go slice
	embedding := cFloatArrayToGoSlice(result.data, result.length)

	return &EmbeddingOutput{
		Embedding:        embedding,
		ModelType:        "mmbert",
		SequenceLength:   int(result.sequence_length),
		ProcessingTimeMs: float32(result.processing_time_ms),
	}, nil
}

// GetEmbedding generates an embedding using the full model with default dimension (768).
//
// This is a convenience function equivalent to GetEmbedding2DMatryoshka(text, 0, 0).
//
// Parameters:
//   - text: Input text to generate embedding for
//
// Returns:
//   - EmbeddingOutput containing the 768-dimensional embedding vector and metadata
//   - error if embedding generation fails
func GetEmbedding(text string) (*EmbeddingOutput, error) {
	return GetEmbedding2DMatryoshka(text, 0, 0)
}

// GetEmbeddingWithDim generates an embedding with Matryoshka dimension truncation.
//
// This is a convenience function equivalent to GetEmbedding2DMatryoshka(text, 0, targetDim).
// Uses the full model (all layers) but truncates to the specified dimension.
//
// Parameters:
//   - text: Input text to generate embedding for
//   - targetDim: Target embedding dimension (768, 512, 256, 128, or 64; 0 for default 768)
//
// Returns:
//   - EmbeddingOutput containing the embedding vector and metadata
//   - error if embedding generation fails
func GetEmbeddingWithDim(text string, targetDim int) (*EmbeddingOutput, error) {
	return GetEmbedding2DMatryoshka(text, 0, targetDim)
}

// GetEmbeddingsBatch generates embeddings for multiple texts in batch.
//
// This is more efficient than calling GetEmbedding2DMatryoshka in a loop.
//
// Parameters:
//   - texts: Array of input texts to generate embeddings for
//   - targetLayer: Target layer for early exit (0 for full model)
//   - targetDim: Target embedding dimension (0 for default 768)
//
// Returns:
//   - []EmbeddingOutput containing the embeddings and metadata for each text
//   - error if embedding generation fails
func GetEmbeddingsBatch(texts []string, targetLayer int, targetDim int) ([]EmbeddingOutput, error) {
	if !modelInitialized {
		return nil, fmt.Errorf("mmBERT model not initialized. Call InitMmBertEmbeddingModel first")
	}

	if len(texts) == 0 {
		return nil, fmt.Errorf("texts array cannot be empty")
	}

	// Convert texts to C strings
	cTexts := make([]*C.char, len(texts))
	for i, text := range texts {
		cTexts[i] = C.CString(text)
		defer C.free(unsafe.Pointer(cTexts[i]))
	}

	// Allocate results array
	results := make([]C.EmbeddingResult, len(texts))

	status := C.get_embeddings_batch(
		(**C.char)(unsafe.Pointer(&cTexts[0])),
		C.int(len(texts)),
		C.int(targetLayer),
		C.int(targetDim),
		(*C.EmbeddingResult)(unsafe.Pointer(&results[0])),
	)

	// Check status code
	if status != 0 {
		return nil, fmt.Errorf("failed to generate batch embeddings (status: %d)", status)
	}

	// Convert results
	outputs := make([]EmbeddingOutput, len(texts))
	for i := range results {
		if results[i].error {
			return nil, fmt.Errorf("error generating embedding for text %d", i)
		}

		outputs[i] = EmbeddingOutput{
			Embedding:        cFloatArrayToGoSlice(results[i].data, results[i].length),
			ModelType:        "mmbert",
			SequenceLength:   int(results[i].sequence_length),
			ProcessingTimeMs: float32(results[i].processing_time_ms),
		}
	}

	return outputs, nil
}

// CalculateEmbeddingSimilarity calculates cosine similarity between two texts.
//
// This function:
// 1. Generates embeddings for both texts using the specified layer/dimension
// 2. Calculates cosine similarity between the embeddings
// 3. Returns similarity score along with metadata
//
// Parameters:
//   - text1, text2: The two texts to compare
//   - targetLayer: Target layer for early exit (0 for full model)
//   - targetDim: Target embedding dimension (0 for default 768)
//
// Returns:
//   - *SimilarityOutput: Contains similarity score, model used, and processing time
//   - error: If embedding generation or similarity calculation fails
//
// Example:
//
//	result, err := CalculateEmbeddingSimilarity("Hello world", "Hi there", 0, 0)
//	if err != nil {
//	    log.Fatal(err)
//	}
//	fmt.Printf("Similarity: %.4f (took: %.2fms)\n", result.Similarity, result.ProcessingTimeMs)
func CalculateEmbeddingSimilarity(text1, text2 string, targetLayer, targetDim int) (*SimilarityOutput, error) {
	if !modelInitialized {
		return nil, fmt.Errorf("mmBERT model not initialized. Call InitMmBertEmbeddingModel first")
	}

	cText1 := C.CString(text1)
	defer C.free(unsafe.Pointer(cText1))

	cText2 := C.CString(text2)
	defer C.free(unsafe.Pointer(cText2))

	var result C.EmbeddingSimilarityResult
	status := C.calculate_embedding_similarity(
		cText1,
		cText2,
		C.int(targetLayer),
		C.int(targetDim),
		&result,
	)

	// Check status code
	if status != 0 || result.error {
		return nil, fmt.Errorf("failed to calculate similarity (status: %d)", status)
	}

	return &SimilarityOutput{
		Similarity:       float32(result.similarity),
		ModelType:        "mmbert",
		ProcessingTimeMs: float32(result.processing_time_ms),
	}, nil
}

// CalculateSimilarityBatch finds top-k most similar candidates for a query.
//
// This function uses batch processing for efficiency:
// 1. Generates embeddings for query and all candidates in a single batch
// 2. Calculates cosine similarity between query and each candidate
// 3. Returns top-k matches sorted by similarity (descending)
//
// Parameters:
//   - query: The query text
//   - candidates: Array of candidate texts
//   - topK: Maximum number of matches to return (0 = return all, sorted by similarity)
//   - targetLayer: Target layer for early exit (0 for full model)
//   - targetDim: Target dimension (0 for default 768)
//
// Returns:
//   - BatchSimilarityOutput: Top-k matches sorted by similarity (descending)
//   - error: Error message if operation failed
func CalculateSimilarityBatch(query string, candidates []string, topK, targetLayer, targetDim int) (*BatchSimilarityOutput, error) {
	if !modelInitialized {
		return nil, fmt.Errorf("mmBERT model not initialized. Call InitMmBertEmbeddingModel first")
	}

	if len(candidates) == 0 {
		return nil, fmt.Errorf("candidates array cannot be empty")
	}

	// Convert query to C string
	cQuery := C.CString(query)
	defer C.free(unsafe.Pointer(cQuery))

	// Convert candidates to C string array
	cCandidates := make([]*C.char, len(candidates))
	for i, candidate := range candidates {
		cCandidates[i] = C.CString(candidate)
		defer C.free(unsafe.Pointer(cCandidates[i]))
	}

	var result C.BatchSimilarityResult
	status := C.calculate_similarity_batch(
		cQuery,
		(**C.char)(unsafe.Pointer(&cCandidates[0])),
		C.int(len(candidates)),
		C.int(topK),
		C.int(targetLayer),
		C.int(targetDim),
		&result,
	)

	// Check status code
	if status != 0 || result.error {
		return nil, fmt.Errorf("failed to calculate batch similarity (status: %d)", status)
	}

	// Convert matches to Go slice
	numMatches := int(result.num_matches)
	matches := make([]BatchSimilarityMatch, numMatches)

	if numMatches > 0 && result.matches != nil {
		matchesSlice := (*[1 << 30]C.SimilarityMatch)(unsafe.Pointer(result.matches))[:numMatches:numMatches]
		for i := 0; i < numMatches; i++ {
			matches[i] = BatchSimilarityMatch{
				Index:      int(matchesSlice[i].index),
				Similarity: float32(matchesSlice[i].similarity),
			}
		}
	}

	// Free the result
	C.free_batch_similarity_result(&result)

	return &BatchSimilarityOutput{
		Matches:          matches,
		ModelType:        "mmbert",
		ProcessingTimeMs: float32(result.processing_time_ms),
	}, nil
}

// GetEmbeddingModelsInfo retrieves information about the loaded embedding model.
//
// Returns:
//   - ModelsInfoOutput: Information about the mmBERT model
//   - error: Error message if operation failed
func GetEmbeddingModelsInfo() (*ModelsInfoOutput, error) {
	var result C.EmbeddingModelsInfoResult
	status := C.get_embedding_models_info(&result)

	// Check status code
	if status != 0 || result.error {
		return nil, fmt.Errorf("failed to get embedding models info (status: %d)", status)
	}

	// Convert models to Go slice
	numModels := int(result.num_models)
	models := make([]ModelInfo, numModels)

	if numModels > 0 && result.models != nil {
		modelsSlice := (*[1 << 30]C.EmbeddingModelInfo)(unsafe.Pointer(result.models))[:numModels:numModels]
		for i := 0; i < numModels; i++ {
			modelInfo := modelsSlice[i]
			models[i] = ModelInfo{
				ModelName:         C.GoString(modelInfo.model_name),
				IsLoaded:          bool(modelInfo.is_loaded),
				MaxSequenceLength: int(modelInfo.max_sequence_length),
				DefaultDimension:  int(modelInfo.default_dimension),
				ModelPath:         C.GoString(modelInfo.model_path),
				SupportsLayerExit: bool(modelInfo.supports_layer_exit),
				AvailableLayers:   C.GoString(modelInfo.available_layers),
			}
		}
	}

	// Free the result
	C.free_embedding_models_info(&result)

	return &ModelsInfoOutput{
		Models: models,
	}, nil
}

// GetMatryoshkaConfig retrieves the 2D Matryoshka configuration.
//
// Returns:
//   - MatryoshkaConfig: The supported dimensions and layers
//   - error: Error message if operation failed
func GetMatryoshkaConfig() (*MatryoshkaConfig, error) {
	var result C.MatryoshkaInfo
	status := C.get_matryoshka_info(&result)

	if status != 0 {
		return nil, fmt.Errorf("failed to get Matryoshka config")
	}

	config := &MatryoshkaConfig{
		Dimensions: C.GoString(result.dimensions),
		Layers:     C.GoString(result.layers),
		Supports2D: bool(result.supports_2d),
	}

	// Free the result
	C.free_matryoshka_info(&result)

	return config, nil
}
