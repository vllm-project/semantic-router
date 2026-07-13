package onnx_binding

import (
	"errors"
	"testing"
)

func TestEmbeddingStatusErrorClassifiesOnlyExplicitInputLimit(t *testing.T) {
	t.Parallel()

	limitError := embeddingStatusError("generate embedding", embeddingInputTooLongStatus)
	if !errors.Is(limitError, ErrEmbeddingInputTooLong) {
		t.Fatalf("expected input limit sentinel, got %v", limitError)
	}

	internalError := embeddingStatusError("generate embedding", -1)
	if errors.Is(internalError, ErrEmbeddingInputTooLong) {
		t.Fatalf("internal status must not wrap input limit sentinel: %v", internalError)
	}
}
