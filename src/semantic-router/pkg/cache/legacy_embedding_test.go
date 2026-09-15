package cache

import (
	"context"

	candle "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

// Existing model integration tests initialize their BERT fixture explicitly.
// Keep that fixture behind test-only injection; production has no native global fallback.
type legacyBERTTestProvider struct{}

func (legacyBERTTestProvider) Embed(_ context.Context, text string) ([]float32, error) {
	return candle.GetEmbedding(text, 0)
}

func (p legacyBERTTestProvider) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	result := make([][]float32, len(texts))
	for i, text := range texts {
		vector, err := p.Embed(ctx, text)
		if err != nil {
			return nil, err
		}
		result[i] = vector
	}
	return result, nil
}
func (legacyBERTTestProvider) Dimension() int  { return 384 }
func (legacyBERTTestProvider) Backend() string { return "test-candle-bert" }
func (legacyBERTTestProvider) Windows(_ context.Context, text string, limit int) ([]embedding.Window, error) {
	if limit == 0 {
		limit = 512
	}
	ranges, err := candle.TextWindows(text, limit)
	windows := make([]embedding.Window, len(ranges))
	for i, r := range ranges {
		windows[i] = embedding.Window{Start: r.Start, End: r.End}
	}
	return windows, err
}

func cacheTestEmbeddingProvider() embedding.Provider { return legacyBERTTestProvider{} }
