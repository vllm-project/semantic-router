//go:build !windows && cgo && (amd64 || arm64)

package onnx_binding

import (
	"errors"
	"testing"
)

func TestNativeEmbeddingCapabilitiesCanonicalizeMmBert(t *testing.T) {
	got, err := EmbeddingCapabilitiesFor("  MMBERT  ")
	if err != nil {
		t.Fatalf("EmbeddingCapabilitiesFor(mmbert) error = %v", err)
	}
	if got.ModelType != ModelTypeMmBert || got.Backend != BackendONNX || got.SupportsBatching {
		t.Fatalf("EmbeddingCapabilitiesFor(mmbert) = %#v, want canonical non-batched ONNX mmbert", got)
	}
}

func TestGetEmbeddingWithModelTypeRejectsUnsupportedTypes(t *testing.T) {
	for _, modelType := range []string{"", "qwen3", "gemma", "unknown"} {
		t.Run(modelType, func(t *testing.T) {
			_, err := GetEmbeddingWithModelType("test", modelType, 0)
			if !errors.Is(err, ErrUnsupportedModelType) {
				t.Fatalf("GetEmbeddingWithModelType(%q) error = %v, want ErrUnsupportedModelType", modelType, err)
			}
		})
	}
}
