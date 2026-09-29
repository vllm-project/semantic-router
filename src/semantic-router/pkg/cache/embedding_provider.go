package cache

import (
	"context"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

// The generation owns the prepared provider. Cache shutdown closes its store,
// while the generation releases embedding resources after all users drain.
func computeCacheEmbedding(ctx context.Context, provider embedding.Provider, text string) ([]float32, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return invokeCacheEmbedding(ctx, provider, text)
}

// invokeCacheEmbedding is used after a cache has already checked cancellation,
// including the check before consulting the embedding memo.
func invokeCacheEmbedding(ctx context.Context, provider embedding.Provider, text string) ([]float32, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if provider == nil {
		return nil, fmt.Errorf("cache embedding provider was not prepared")
	}
	return provider.Embed(ctx, text)
}

// cacheEmbeddingOptions retains each cache backend's established model output.
// The cache selects these options; the injected provider only owns execution.
func cacheEmbeddingOptions(model string, dimension, layer int) embedding.Options {
	switch normalizeEmbeddingModel(model) {
	case "bert":
		return embedding.Options{}
	case "mmbert":
		return embedding.Options{Dimension: dimension, Layer: layer}
	case "qwen3", "gemma", "multimodal":
		return embedding.Options{Dimension: dimension}
	default:
		return embedding.Options{}
	}
}

func inMemoryEmbeddingOptions(model string) embedding.Options {
	switch normalizeEmbeddingModel(model) {
	case "mmbert":
		return embedding.Options{Dimension: 256, Layer: 6}
	default:
		return embedding.Options{}
	}
}

// No model aliases appear here: semantic storage follows its prepared provider.
// A missing provider is permitted for exact-only stores with an explicit width.
func semanticCacheEmbeddingDimension(configured int, provider embedding.Provider) int {
	if configured != 0 {
		return configured
	}
	if provider != nil {
		return provider.Dimension()
	}
	return 0
}

func resolveCacheDimension(configured int, provider embedding.Provider) (int, error) {
	if provider != nil {
		return embedding.ResolveDimension(provider, configured)
	}
	if configured > 0 {
		return configured, nil
	}
	return 0, fmt.Errorf("cache vector storage needs an explicit dimension without an embedding provider")
}
