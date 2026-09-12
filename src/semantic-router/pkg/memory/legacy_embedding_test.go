package memory

import (
	"context"

	candle "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

// Existing BERT fixture initialization belongs to these integration tests.
// Runtime memory stores must receive their generation-owned provider explicitly.
func memoryTestEmbeddingProvider() embedding.Provider {
	provider, _ := embedding.NewFuncProvider("test-candle-bert", 384, func(_ context.Context, text string) ([]float32, error) {
		return candle.GetEmbedding(text, 0)
	})
	return provider
}
