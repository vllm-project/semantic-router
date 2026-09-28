//go:build !windows && cgo

package cache

import (
	"context"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

// ValidateBackendEmbedding checks the exact view selected after the backend
// loaded its configuration. Its owner closes a rejected backend; providers are
// borrowed from the generation and remain owned by that generation.
func ValidateBackendEmbedding(ctx context.Context, backend LegacyCacheBackend) error {
	if !backend.IsEnabled() {
		return nil
	}
	var provider embedding.Provider
	var dimension int
	switch c := backend.(type) {
	case *InMemoryCache:
		provider = c.embeddingProvider
		dimension = semanticCacheEmbeddingDimension(inMemoryEmbeddingOptions(c.embeddingModel).Dimension, provider)
	case *RedisCache:
		provider = c.embeddingProvider
		dimension = c.embeddingDimension()
	case *ValkeyCache:
		provider = c.embeddingProvider
		dimension = c.embeddingDimension()
	case *MilvusCache:
		provider = c.embeddingProvider
		dimension = c.embeddingDimension()
	case *QdrantCache:
		provider = c.embeddingProvider
		dimension = c.embeddingDimension()
	case *HybridCache:
		return ValidateBackendEmbedding(ctx, c.milvusCache)
	default:
		return fmt.Errorf("cache backend %T does not expose prepared embedding semantics", backend)
	}
	vector, err := computeCacheEmbedding(ctx, provider, "semantic router cache preparation")
	if err != nil {
		return fmt.Errorf("prepare cache embedding: %w", err)
	}
	if len(vector) == 0 || (dimension > 0 && len(vector) != dimension) {
		return fmt.Errorf("cache embedding dimension %d differs from required %d", len(vector), dimension)
	}
	windows, ok := provider.(embedding.WindowProvider)
	if !ok {
		return fmt.Errorf("cache embedding requires tokenizer windows")
	}
	if _, err := windows.Windows(ctx, "semantic router cache preparation", 0); err != nil {
		return fmt.Errorf("prepare cache tokenizer: %w", err)
	}
	return nil
}
