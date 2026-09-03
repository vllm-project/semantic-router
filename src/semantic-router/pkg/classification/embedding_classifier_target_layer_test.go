package classification

import (
	"testing"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestEmbeddingClassifierPassesTargetLayerToBackend(t *testing.T) {
	capturedLayer := 0
	originalFunc := getEmbedding2DMatryoshka
	getEmbedding2DMatryoshka = func(text string, modelType string, targetLayer int, targetDim int) (*candle_binding.EmbeddingOutput, error) {
		capturedLayer = targetLayer
		return &candle_binding.EmbeddingOutput{Embedding: makeEmbedding(1.0, 0.0, 0.0)}, nil
	}
	t.Cleanup(func() {
		getEmbedding2DMatryoshka = originalFunc
	})

	classifier, err := NewEmbeddingClassifier(nil, config.HNSWConfig{
		ModelType:       "mmbert",
		TargetLayer:     6,
		TargetDimension: 256,
	})
	if err != nil {
		t.Fatalf("NewEmbeddingClassifier failed: %v", err)
	}

	if _, err := classifier.computeEmbedding("query", "mmbert"); err != nil {
		t.Fatalf("computeEmbedding failed: %v", err)
	}
	if capturedLayer != 6 {
		t.Fatalf("backend received target layer %d, want 6", capturedLayer)
	}
}
