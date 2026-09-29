package classification

import (
	"context"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

func TestEmbeddingClassifierPassesTargetLayerToBackend(t *testing.T) {
	capturedLayer := 0
	provider := &testEmbeddingProvider{text: func(_ context.Context, text string, options embedding.Options) ([]float32, error) {
		capturedLayer = options.Layer
		return makeEmbedding(1.0, 0.0, 0.0), nil
	}}

	classifier, err := NewEmbeddingClassifierWithProvider(nil, config.HNSWConfig{
		ModelType:       "mmbert",
		TargetLayer:     6,
		TargetDimension: 256,
	}, provider)
	if err != nil {
		t.Fatalf("NewEmbeddingClassifier failed: %v", err)
	}

	if _, err := classifier.computeEmbedding(context.Background(), "query", "mmbert"); err != nil {
		t.Fatalf("computeEmbedding failed: %v", err)
	}
	if capturedLayer != 6 {
		t.Fatalf("backend received target layer %d, want 6", capturedLayer)
	}
}
