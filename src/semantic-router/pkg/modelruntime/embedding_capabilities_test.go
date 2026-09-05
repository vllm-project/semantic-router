package modelruntime

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestBatchedEmbeddingNeedsSkipsSeparateBERTPath(t *testing.T) {
	cfg := &config.RouterConfig{
		SemanticCache: config.SemanticCache{
			Enabled:        true,
			EmbeddingModel: "bert",
		},
	}
	semanticCache, mlSelection, err := batchedEmbeddingNeeds(cfg, embeddingPaths{
		qwen3: "models/qwen3-embedding",
		bert:  "sentence-transformers/all-MiniLM-L6-v2",
	})
	if err != nil {
		t.Fatalf("batchedEmbeddingNeeds() queried BERT capabilities: %v", err)
	}
	if semanticCache || mlSelection {
		t.Fatalf("batchedEmbeddingNeeds() = %v/%v, want false/false for separate BERT path", semanticCache, mlSelection)
	}
}

func TestUnifiedEmbeddingModelConfigured(t *testing.T) {
	paths := embeddingPaths{
		qwen3:  "models/qwen3",
		gemma:  "models/gemma",
		mmBert: "models/mmbert",
		bert:   "models/bert",
	}
	for _, modelType := range []string{"qwen3", " GEMMA ", "MmBert"} {
		if !unifiedEmbeddingModelConfigured(paths, modelType) {
			t.Errorf("unifiedEmbeddingModelConfigured(%q) = false, want true", modelType)
		}
	}
	for _, modelType := range []string{"bert", "multimodal", "unknown", ""} {
		if unifiedEmbeddingModelConfigured(paths, modelType) {
			t.Errorf("unifiedEmbeddingModelConfigured(%q) = true, want false", modelType)
		}
	}
}
