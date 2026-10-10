package memory

import (
	"context"
	"math"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

// Test-owned vectors exercise retrieval thresholds without loading a language model.
func memoryTestEmbeddingProvider() embedding.Provider {
	fixture := storageMemoryVectors()
	provider, _ := embedding.NewFuncProvider("memory-fixture", 384, func(ctx context.Context, text string) ([]float32, error) {
		var score float32
		switch text {
		case "What is my budget for Hawaii?", "What is my budget?":
			score = 1
		case "User's budget for Hawaii vacation is $10,000":
			score = 0.6
		case "User prefers direct flights", "User prefers direct flights to Hawaii":
			score = 0.4
		case "The weather in Hawaii is sunny":
			score = -0.5
		default:
			return fixture.Embed(ctx, text)
		}
		vector := make([]float32, 384)
		vector[0], vector[1] = score, float32(math.Sqrt(1-float64(score)*float64(score)))
		return vector, nil
	})
	return provider
}
