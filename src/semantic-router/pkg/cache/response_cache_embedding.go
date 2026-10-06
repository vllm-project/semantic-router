package cache

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

// PreparedEmbedding borrows the actual cache consumer for diagnostics. The
// caller must hold the service's generation lease until its last provider call.
// It never resolves another recipe, loads a model, or transfers ownership.
func (s *ResponseCacheService) PreparedEmbedding(model string) (embedding.Provider, error) {
	if s == nil {
		return nil, fmt.Errorf("response cache is unavailable")
	}
	adapter, ok := s.store.(*LegacyBackendAdapter)
	if !ok || adapter.embeddingProvider == nil {
		return nil, fmt.Errorf("response cache has no prepared embedding consumer")
	}
	if normalizeEmbeddingModel(model) != normalizeEmbeddingModel(adapter.embeddingModel) {
		return nil, fmt.Errorf("candidate embedding model does not match the prepared response cache consumer")
	}
	return adapter.embeddingProvider, nil
}
