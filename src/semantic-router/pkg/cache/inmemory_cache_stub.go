//go:build windows || !cgo

package cache

import "context"

// InMemoryCache provides high-performance in-memory semantic caching
type InMemoryCache struct {
	enabled bool
}

// InMemoryCacheOptions contains configuration for the in-memory cache
type InMemoryCacheOptions struct {
	Enabled             bool
	SimilarityThreshold float32
	MaxEntries          int
	TTLSeconds          int
	EvictionPolicy      EvictionPolicyType
	UseHNSW             bool
	HNSWM               int
	HNSWEfConstruction  int
	EmbeddingModel      string
}

// NewInMemoryCache creates a new in-memory cache instance
func NewInMemoryCache(options InMemoryCacheOptions) *InMemoryCache {
	return &InMemoryCache{
		enabled: options.Enabled,
	}
}

// IsEnabled returns whether the cache is active
func (c *InMemoryCache) IsEnabled() bool {
	return c.enabled
}

// AddPendingRequest stores a request awaiting its response
func (c *InMemoryCache) AddPendingRequest(_ context.Context, requestID string, model string, query string, requestBody []byte) error {
	return nil
}

// UpdateWithResponse completes a pending request with its response
func (c *InMemoryCache) UpdateWithResponse(_ context.Context, requestID string, responseBody []byte) error {
	return nil
}

// AddEntry stores a complete request-response pair
func (c *InMemoryCache) AddEntry(_ context.Context, requestID string, model string, query string, requestBody, responseBody []byte) error {
	return nil
}

// FindSimilar searches for semantically similar cached requests
func (c *InMemoryCache) FindSimilar(_ context.Context, model string, query string) (LookupResult, error) {
	return LookupResult{}, nil
}

// FindSimilarWithThreshold searches for semantically similar cached requests using a specific threshold
func (c *InMemoryCache) FindSimilarWithThreshold(_ context.Context, model string, query string, threshold float32) (LookupResult, error) {
	return LookupResult{}, nil
}

// Close releases all resources
func (c *InMemoryCache) Close() error {
	return nil
}

// GetStats returns cache statistics
func (c *InMemoryCache) GetStats() CacheStats {
	return CacheStats{}
}

// CheckConnection checks if the cache backend is reachable
func (c *InMemoryCache) CheckConnection(_ context.Context) error {
	return nil
}
