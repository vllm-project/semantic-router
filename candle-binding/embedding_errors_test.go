package candle_binding

import (
	"errors"
	"strings"
	"testing"
)

func TestEmbeddingStatusErrorPreservesTypedInputLimit(t *testing.T) {
	t.Parallel()

	limitError := embeddingStatusError("generate embedding", embeddingInputTooLongStatus)
	if !errors.Is(limitError, ErrEmbeddingInputTooLong) {
		t.Fatalf("status -3 did not preserve input-too-long sentinel: %v", limitError)
	}
	internalError := embeddingStatusError("generate embedding", -1)
	if errors.Is(internalError, ErrEmbeddingInputTooLong) {
		t.Fatalf("internal status was misclassified: %v", internalError)
	}
	if !strings.Contains(internalError.Error(), "status: -1") {
		t.Fatalf("internal status missing from diagnostic: %v", internalError)
	}
}
