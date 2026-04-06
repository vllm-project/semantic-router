//go:build onnx && !windows && cgo

package modelruntime

import (
	"errors"
	"testing"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
)

func TestValidateEmbeddingBackendContractRejectsUnsupportedONNXFamilies(t *testing.T) {
	contract := candle_binding.CurrentBackendContract()

	tests := []struct {
		name  string
		paths embeddingPaths
	}{
		{name: "qwen3", paths: embeddingPaths{qwen3: "models/qwen3"}},
		{name: "gemma", paths: embeddingPaths{gemma: "models/gemma"}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateEmbeddingBackendContract(contract, tt.paths, false)
			if err == nil {
				t.Fatal("expected unsupported embedding family to fail on onnx backend")
			}

			var unsupported candle_binding.UnsupportedEmbeddingFamilyError
			if !errors.As(err, &unsupported) {
				t.Fatalf("expected UnsupportedEmbeddingFamilyError, got %T", err)
			}
		})
	}
}

func TestValidateEmbeddingBackendContractAllowsMMBertOnONNX(t *testing.T) {
	contract := candle_binding.CurrentBackendContract()

	if err := validateEmbeddingBackendContract(contract, embeddingPaths{mmBert: "models/mmbert"}, false); err != nil {
		t.Fatalf("expected mmbert embeddings to be accepted on onnx backend: %v", err)
	}
	if err := validateEmbeddingBackendContract(contract, embeddingPaths{mmBert: "models/mmbert"}, true); err != nil {
		t.Fatalf("expected batched mmbert embeddings to be accepted on onnx backend: %v", err)
	}
}
