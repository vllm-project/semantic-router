package classification

import (
	"crypto/sha256"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

// The request-local cache shares only identical representation requests. Each
// provider owns dimension validation and pooling; a cache must never synthesize
// an arbitrary vector prefix for a model that has no such representation.
type requestMediaEmbeddingCache struct {
	mu      sync.Mutex
	entries map[mediaEmbeddingKey]*mediaEmbeddingEntry
}
type mediaEmbeddingKey struct {
	provider  string
	modality  config.QueryModality
	payload   [32]byte
	dimension int
}
type mediaEmbeddingEntry struct {
	once      sync.Once
	embedding []float32
	err       error
}

func newRequestMediaEmbeddingCache() *requestMediaEmbeddingCache {
	return &requestMediaEmbeddingCache{entries: make(map[mediaEmbeddingKey]*mediaEmbeddingEntry)}
}

func (c *requestMediaEmbeddingCache) resolveFor(provider embedding.Provider, modality config.QueryModality, payload string, dimension int, compute func() ([]float32, error)) ([]float32, error) {
	if c == nil {
		return compute()
	}
	if provider != nil && dimension == provider.Dimension() {
		dimension = 0
	}
	key := mediaEmbeddingKey{provider: embedding.Identity(provider), modality: modality, payload: sha256.Sum256([]byte(payload)), dimension: dimension}
	c.mu.Lock()
	entry, ok := c.entries[key]
	if !ok {
		entry = &mediaEmbeddingEntry{}
		c.entries[key] = entry
	}
	c.mu.Unlock()
	entry.once.Do(func() { entry.embedding, entry.err = compute() })
	return entry.embedding, entry.err
}
