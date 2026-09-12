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
	if got.DimensionState != DimensionStateNotLoaded && got.DimensionState != DimensionStateAvailable {
		t.Fatalf("unexpected dimension state: %q", got.DimensionState)
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

func assertCapabilityDimensions(t *testing.T, got EmbeddingCapabilities) {
	t.Helper()
	switch got.DimensionState {
	case DimensionStateNotLoaded:
		if got.NativeDimension != 0 || len(got.SupportedDimensions) != 0 {
			t.Fatalf("unloaded model reports dimension data: %#v", got)
		}
	case DimensionStateAvailable:
		if err := validateObservedDimensions(got.NativeDimension, got.SupportedDimensions); err != nil {
			t.Fatal(err)
		}
	default:
		t.Fatalf("unknown dimension state: %q", got.DimensionState)
	}
}

func assertLoadedMmBertCapabilities(t *testing.T) {
	t.Helper()
	got, err := EmbeddingCapabilitiesFor("mmbert")
	if err != nil {
		t.Fatal(err)
	}
	assertCapabilityDimensions(t, got)
	if got.DimensionState != DimensionStateAvailable {
		t.Fatalf("loaded model has dimension state %q", got.DimensionState)
	}
	output, err := GetEmbeddingWithModelType("model dimension contract", "mmbert", 0)
	if err != nil {
		t.Fatal(err)
	}
	if got.NativeDimension != len(output.Embedding) {
		t.Fatalf("native width %d does not match produced width %d", got.NativeDimension, len(output.Embedding))
	}
}
