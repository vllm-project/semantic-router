//go:build windows || !cgo

package cache

import (
	"context"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
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
	Milvus                  *config.MilvusConfig
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
func (h *HybridCache) AddPendingRequest(requestID string, model string, query string, requestBody []byte, ttlSeconds int) error {
	return nil
}

// UpdateWithResponse completes a pending request with its response
func (h *HybridCache) UpdateWithResponse(requestID string, responseBody []byte, ttlSeconds int) error {
	return nil
}

// AddEntry stores a complete request-response pair
func (h *HybridCache) AddEntry(requestID string, model string, query string, requestBody, responseBody []byte, ttlSeconds int) error {
	return nil
}

// AddDecisionEntry stores a complete request-decision pair
func (h *HybridCache) AddDecisionEntry(requestID string, model string, query string, decision *DecisionEntry, ttlSeconds int) error {
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

// FindSimilarDecision searches for semantically similar cached decisions
func (h *HybridCache) FindSimilarDecision(model string, query string) (*DecisionEntry, bool, error) {
	return nil, false, nil
}

// FindSimilarDecisionWithThreshold searches for semantically similar cached decisions with custom threshold
func (h *HybridCache) FindSimilarDecisionWithThreshold(model string, query string, threshold float32) (*DecisionEntry, bool, error) {
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
