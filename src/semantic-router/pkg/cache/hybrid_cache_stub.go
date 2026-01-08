//go:build windows || !cgo

package cache

import (
	"context"
)

// HybridCache combines in-memory HNSW index with external Milvus storage
type HybridCache struct {
	enabled bool
}

// HybridCacheOptions contains configuration for the hybrid cache
type HybridCacheOptions struct {
	Enabled                 bool
	SimilarityThreshold     float32
	TTLSeconds              int
	MaxMemoryEntries        int
	HNSWM                   int
	HNSWEfConstruction      int
	MilvusConfigPath        string
	DisableRebuildOnStartup bool
}

// NewHybridCache creates a new hybrid cache instance
func NewHybridCache(options HybridCacheOptions) (*HybridCache, error) {
	return &HybridCache{
		enabled: options.Enabled,
	}, nil
}

// IsEnabled returns whether the cache is active
func (h *HybridCache) IsEnabled() bool {
	return h.enabled
}

// AddPendingRequest stores a request awaiting its response
func (h *HybridCache) AddPendingRequest(requestID string, model string, query string, requestBody []byte) error {
	return nil
}

// UpdateWithResponse completes a pending request with its response
func (h *HybridCache) UpdateWithResponse(requestID string, responseBody []byte) error {
	return nil
}

// AddEntry stores a complete request-response pair
func (h *HybridCache) AddEntry(requestID string, model string, query string, requestBody, responseBody []byte) error {
	return nil
}

// AddEntriesBatch stores multiple request-response pairs efficiently
func (h *HybridCache) AddEntriesBatch(entries []CacheEntry) error {
	return nil
}

// FindSimilar searches for semantically similar cached requests
func (h *HybridCache) FindSimilar(model string, query string) ([]byte, bool, error) {
	return nil, false, nil
}

// FindSimilarWithThreshold searches for semantically similar cached requests with custom threshold
func (h *HybridCache) FindSimilarWithThreshold(model string, query string, threshold float32) ([]byte, bool, error) {
	return nil, false, nil
}

// RebuildFromMilvus rebuilds the in-memory HNSW index
func (h *HybridCache) RebuildFromMilvus(ctx context.Context) error {
	return nil
}

// Flush forces persistence
func (h *HybridCache) Flush() error {
	return nil
}

// Close releases all resources
func (h *HybridCache) Close() error {
	return nil
}

// GetStats returns cache statistics
func (h *HybridCache) GetStats() CacheStats {
	return CacheStats{}
}

// CheckConnection checks if the cache backend is reachable
func (h *HybridCache) CheckConnection() error {
	return nil
}
