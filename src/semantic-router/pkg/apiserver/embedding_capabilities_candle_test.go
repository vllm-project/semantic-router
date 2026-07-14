//go:build !windows && cgo && !onnx

package apiserver

import (
	"strings"
	"testing"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
)

func TestCandleEmbeddingCapabilitiesRejectUnsupportedModelDimensions(t *testing.T) {
	tests := []struct {
		name      string
		model     string
		dimension int
		wantOK    bool
	}{
		{name: "qwen 64", model: "qwen3", dimension: 64, wantOK: true},
		{name: "qwen 1024", model: "qwen3", dimension: 1024, wantOK: true},
		{name: "gemma 128", model: "gemma", dimension: 128, wantOK: true},
		{name: "gemma rejects 64", model: "gemma", dimension: 64},
		{name: "gemma rejects 1024", model: "gemma", dimension: 1024},
		{name: "mmbert rejects 1024", model: "mmbert", dimension: 1024},
		{name: "auto common dimension", model: "auto", dimension: 768, wantOK: true},
		{name: "auto rejects gemma-incompatible 64", model: "auto", dimension: 64},
		{name: "auto rejects gemma-incompatible 1024", model: "auto", dimension: 1024},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			code, message, ok := validateSimilarityOptions(test.model, test.dimension, 0, 0, 0, nil)
			if ok != test.wantOK {
				t.Fatalf("ok=%v code=%q message=%q, want ok=%v", ok, code, message, test.wantOK)
			}
			if !test.wantOK && code != "INVALID_DIMENSION" {
				t.Fatalf("rejection code=%q message=%q, want INVALID_DIMENSION", code, message)
			}
		})
	}
}

func TestCandleImageEmbeddingCapabilityUses384Default(t *testing.T) {
	if code, message, ok := validateImageEmbeddingDimension(defaultImageEmbeddingDimension); !ok {
		t.Fatalf("384-dimensional image rejected: code=%q message=%q", code, message)
	}
	code, message, ok := validateImageEmbeddingDimension(defaultEmbeddingDimension)
	if ok || code != "INVALID_DIMENSION" || !strings.Contains(message, "default: 384") {
		t.Fatalf("768-dimensional image validation = ok=%v code=%q message=%q", ok, code, message)
	}
}

func TestCandleModelInfoUsesLoadedOnlyBackendCapabilityContract(t *testing.T) {
	capabilities := nativeEmbeddingBackendCapabilities()
	if capabilities.name != "candle" {
		t.Fatalf("backend name = %q, want candle", capabilities.name)
	}

	if _, ok := loadedEmbeddingModelInfo(candle_binding.ModelInfo{
		ModelName: "qwen3",
		IsLoaded:  false,
	}, capabilities); ok {
		t.Fatal("unloaded Candle inventory placeholder was exposed as a model")
	}

	model, ok := loadedEmbeddingModelInfo(candle_binding.ModelInfo{
		ModelName:         "qwen3",
		IsLoaded:          true,
		MaxSequenceLength: 32768,
		DefaultDimension:  1024,
		ModelPath:         "models/qwen3-embedding-0.6b",
	}, capabilities)
	if !ok {
		t.Fatal("loaded Candle model was omitted")
	}
	if model.Metadata["backend"] != "candle" ||
		model.Metadata["supported_dimensions"] != "64, 128, 256, 512, 768, 1024" ||
		model.Metadata["target_layer_supported"] != "false" {
		t.Fatalf("Candle inventory metadata = %#v", model.Metadata)
	}
}
