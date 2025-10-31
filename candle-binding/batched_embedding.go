//go:build !windows && cgo
// +build !windows,cgo

package candle_binding

/*
#cgo LDFLAGS: -L${SRCDIR}/target/release -lcandle_semantic_router -ldl -lm
#include <stdlib.h>
#include <stdbool.h>

typedef struct {
    float* data;
    int length;
    bool error;
    int model_type;
    int sequence_length;
    float processing_time_ms;
} EmbeddingResult;

extern int get_embedding_batched(const char* text, const char* model_type, int target_dim, EmbeddingResult* result);
extern void free_embedding(float* data, int length);
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// GetEmbeddingBatched generates an embedding using the continuous batching model
//
// This function should be used after calling InitEmbeddingModelsBatched.
// It automatically benefits from continuous batching for concurrent requests (2-5x throughput).
//
// Parameters:
//   - text: Input text to generate embedding for
//   - modelType: "qwen3" (currently only Qwen3 supports batching)
//   - targetDim: Target dimension (0 for default, or 768, 512, 256, 128)
//
// Returns:
//   - *EmbeddingOutput: Embedding output with metadata
//   - error: Non-nil if embedding generation fails
func GetEmbeddingBatched(text string, modelType string, targetDim int) (*EmbeddingOutput, error) {
	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	cModelType := C.CString(modelType)
	defer C.free(unsafe.Pointer(cModelType))

	var result C.EmbeddingResult
	status := C.get_embedding_batched(
		cText,
		cModelType,
		C.int(targetDim),
		&result,
	)

	// Check status code (0 = success, -1 = error)
	if status != 0 || result.error {
		return nil, fmt.Errorf("failed to generate batched embedding (status: %d)", status)
	}

	// Convert C array to Go slice
	length := int(result.length)
	embedding := make([]float32, length)
	cArray := (*[1 << 30]C.float)(unsafe.Pointer(result.data))[:length:length]
	for i := 0; i < length; i++ {
		embedding[i] = float32(cArray[i])
	}

	// Free the C memory
	C.free_embedding(result.data, result.length)

	return &EmbeddingOutput{
		Embedding:        embedding,
		ModelType:        modelType,
		SequenceLength:   int(result.sequence_length),
		ProcessingTimeMs: float32(result.processing_time_ms),
	}, nil
}
