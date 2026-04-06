package candle_binding

import (
	"errors"
	"testing"
)

func TestCurrentBackendContract_CandleCapabilities(t *testing.T) {
	contract := CurrentBackendContract()

	if contract.Backend != NativeBackendCandle {
		t.Fatalf("expected candle backend, got %q", contract.Backend)
	}
	if !contract.SupportsFeature(FeatureEmbeddings) {
		t.Fatal("expected candle backend to support embeddings")
	}
	if !contract.SupportsFeature(FeatureUnifiedClassification) {
		t.Fatal("expected candle backend to support unified classification")
	}
	if !contract.SupportsFeature(FeatureLoRABatchInference) {
		t.Fatal("expected candle backend to support LoRA batch inference")
	}
	if !contract.SupportsFeature(FeatureMultimodalEmbeddings) {
		t.Fatal("expected candle backend to support multimodal embeddings")
	}
	if contract.SupportsFeature(FeatureExplicitRuntimeReset) {
		t.Fatal("expected candle backend explicit reset to remain unsupported")
	}
}

func TestCurrentBackendContract_CandleEmbeddingFamilies(t *testing.T) {
	contract := CurrentBackendContract()

	for _, family := range []EmbeddingFamily{EmbeddingFamilyQwen3, EmbeddingFamilyGemma, EmbeddingFamilyMMBert} {
		if err := contract.RequireEmbeddingFamily(family); err != nil {
			t.Fatalf("expected family %q to be supported: %v", family, err)
		}
	}
	if err := contract.RequireBatchedEmbeddingFamily(EmbeddingFamilyQwen3); err != nil {
		t.Fatalf("expected qwen3 batched embeddings to be supported: %v", err)
	}

	err := contract.RequireBatchedEmbeddingFamily(EmbeddingFamilyGemma)
	if err == nil {
		t.Fatal("expected gemma batched embeddings to remain unsupported")
	}

	var unsupported UnsupportedEmbeddingFamilyError
	if !errors.As(err, &unsupported) {
		t.Fatalf("expected UnsupportedEmbeddingFamilyError, got %T", err)
	}
	if !unsupported.Batched {
		t.Fatalf("expected batched family error, got %+v", unsupported)
	}
}
