package onnx_binding

import (
	"errors"
	"testing"
)

func TestCurrentBackendContract_ONNXCapabilities(t *testing.T) {
	contract := CurrentBackendContract()

	if contract.Backend != NativeBackendONNX {
		t.Fatalf("expected onnx backend, got %q", contract.Backend)
	}
	if !contract.SupportsFeature(FeatureEmbeddings) {
		t.Fatal("expected onnx backend to support embeddings")
	}
	if contract.SupportsFeature(FeatureUnifiedClassification) {
		t.Fatal("expected onnx backend unified classification to remain unsupported")
	}
	if contract.SupportsFeature(FeatureLoRABatchInference) {
		t.Fatal("expected onnx backend LoRA batch inference to remain unsupported")
	}
	if !contract.SupportsFeature(FeatureMultimodalEmbeddings) {
		t.Fatal("expected onnx backend to support multimodal embeddings")
	}
	if contract.SupportsFeature(FeatureExplicitRuntimeReset) {
		t.Fatal("expected onnx backend explicit reset to remain unsupported")
	}
}

func TestCurrentBackendContract_ONNXEmbeddingFamilies(t *testing.T) {
	contract := CurrentBackendContract()

	if err := contract.RequireEmbeddingFamily(EmbeddingFamilyMMBert); err != nil {
		t.Fatalf("expected mmbert embeddings to be supported: %v", err)
	}
	if err := contract.RequireBatchedEmbeddingFamily(EmbeddingFamilyMMBert); err != nil {
		t.Fatalf("expected batched mmbert embeddings to be supported: %v", err)
	}

	for _, family := range []EmbeddingFamily{EmbeddingFamilyQwen3, EmbeddingFamilyGemma} {
		err := contract.RequireEmbeddingFamily(family)
		if err == nil {
			t.Fatalf("expected family %q to remain unsupported", family)
		}

		var unsupported UnsupportedEmbeddingFamilyError
		if !errors.As(err, &unsupported) {
			t.Fatalf("expected UnsupportedEmbeddingFamilyError, got %T", err)
		}
	}
}
