package native

import (
	"testing"
	"unsafe"
)

// Define Go equivalents of the FFI structs to ensure sizes match the Rust structs.
// These structs must match exactly what Rust's repr(C) produces.

type ffiClassificationResult struct {
	predictedClass int32
	confidence     float32
	label          *byte
}

type ffiEmbeddingResult struct {
	data             *float32
	length           int32
	errorFlag        bool
	pad              [3]byte
	modelType        int32
	sequenceLength   int32
	processingTimeMs float32
}

func TestClassificationResultLayout(t *testing.T) {
	var s ffiClassificationResult

	// predictedClass (4) + confidence (4) + label (8) = 16 bytes
	if sz := unsafe.Sizeof(s); sz != 16 {
		t.Errorf("expected ClassificationResult size 16, got %d", sz)
	}
}
