//go:build !windows && cgo

package classification

import (
	"errors"
	"os"
	"path/filepath"
	"testing"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
)

func TestGetMultiModalImageEmbeddingRejectsOversizedLocalCandidateBeforeRead(t *testing.T) {
	path := filepath.Join(t.TempDir(), "oversized.png")
	file, err := os.Create(path)
	if err != nil {
		t.Fatalf("create sparse image fixture: %v", err)
	}
	truncateErr := file.Truncate(candle_binding.MaxMultiModalImageEncodedBytes + 1)
	if truncateErr != nil {
		_ = file.Close()
		t.Fatalf("size sparse image fixture: %v", truncateErr)
	}
	closeErr := file.Close()
	if closeErr != nil {
		t.Fatalf("close sparse image fixture: %v", closeErr)
	}

	_, err = getMultiModalImageEmbedding(path, 0)
	if !errors.Is(err, candle_binding.ErrInvalidImageInput) {
		t.Fatalf("expected ErrInvalidImageInput for oversized candidate, got %v", err)
	}
}
