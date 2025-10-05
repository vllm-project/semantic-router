//go:build !windows && cgo
// +build !windows,cgo

package cache

import (
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability"
)

// InMemoryCache provides a high-performance semantic cache using BERT embeddings in memory
type InMemoryCache struct {
	entries             []CacheEntry
	mu                  sync.RWMutex
	similarityThreshold float32
	maxEntries          int
	ttlSeconds          int
	enabled             bool
	hitCount            int64
	missCount           int64
	lastCleanupTime     *time.Time
	evictionPolicy      EvictionPolicy
}

// InMemoryCacheOptions contains configuration parameters for the in-memory cache
type InMemoryCacheOptions struct {
	SimilarityThreshold float32
	MaxEntries          int
	TTLSeconds          int
	Enabled             bool
	EvictionPolicy      EvictionPolicyType
}

// NewInMemoryCache initializes a new in-memory semantic cache instance
func NewInMemoryCache(options InMemoryCacheOptions) *InMemoryCache {
	observability.Debugf("Initializing in-memory cache: enabled=%t, maxEntries=%d, ttlSeconds=%d, threshold=%.3f, eviction_policy=%s",
		options.Enabled, options.MaxEntries, options.TTLSeconds, options.SimilarityThreshold, options.EvictionPolicy)

	var evictionPolicy EvictionPolicy
	switch options.EvictionPolicy {
	case LRUEvictionPolicyType:
		evictionPolicy = &LRUPolicy{}
	case LFUEvictionPolicyType:
		evictionPolicy = &LFUPolicy{}
	default: // FIFOEvictionPolicyType
		evictionPolicy = &FIFOPolicy{}
	}

	return &InMemoryCache{
		entries:             []CacheEntry{},
		similarityThreshold: options.SimilarityThreshold,
		maxEntries:          options.MaxEntries,
		ttlSeconds:          options.TTLSeconds,
		enabled:             options.Enabled,
		evictionPolicy:      evictionPolicy,
	}
}

// IsEnabled returns the current cache activation status
func (c *InMemoryCache) IsEnabled() bool {
	return c.enabled
}

// AddPendingRequest stores a request that is awaiting its response
func (c *InMemoryCache) AddPendingRequest(requestID string, model string, query string, requestBody []byte) error {
	start := time.Now()

	if !c.enabled {
		return nil
	}

	// Generate semantic embedding for the query
	embedding, err := candle_binding.GetEmbedding(query, 0) // Auto-detect dimension
	if err != nil {
		metrics.RecordCacheOperation("memory", "add_pending", "error", time.Since(start).Seconds())
		return fmt.Errorf("failed to generate embedding: %w", err)
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	// Remove expired entries to maintain cache hygiene
	c.cleanupExpiredEntries()

	// Check if eviction is needed before adding the new entry
	if c.maxEntries > 0 && len(c.entries) >= c.maxEntries {
		c.evictOne()
	}

	// Create cache entry for the pending request
	now := time.Now()
	entry := CacheEntry{
		RequestID:    requestID,
		RequestBody:  requestBody,
		Model:        model,
		Query:        query,
		Embedding:    embedding,
		Timestamp:    now,
		LastAccessAt: now,
		HitCount:     0,
	}

	c.entries = append(c.entries, entry)
	observability.Debugf("InMemoryCache.AddPendingRequest: added pending entry (total entries: %d, embedding_dim: %d)",
		len(c.entries), len(embedding))

	// Record metrics
	metrics.RecordCacheOperation("memory", "add_pending", "success", time.Since(start).Seconds())
	metrics.UpdateCacheEntries("memory", len(c.entries))

	return nil
}

// UpdateWithResponse completes a pending request by adding the response
func (c *InMemoryCache) UpdateWithResponse(requestID string, responseBody []byte) error {
	start := time.Now()

	if !c.enabled {
		return nil
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	// Clean up expired entries during the update
	c.cleanupExpiredEntries()

	// Locate the pending request and complete it
	for i, entry := range c.entries {
		if entry.RequestID == requestID && entry.ResponseBody == nil {
			// Complete the cache entry with the response
			c.entries[i].ResponseBody = responseBody
			c.entries[i].Timestamp = time.Now()
			c.entries[i].LastAccessAt = time.Now()
			observability.Debugf("InMemoryCache.UpdateWithResponse: updated entry with response (response_size: %d bytes)",
				len(responseBody))

			// Record successful completion
			metrics.RecordCacheOperation("memory", "update_response", "success", time.Since(start).Seconds())
			return nil
		}
	}

	// No matching pending request found
	metrics.RecordCacheOperation("memory", "update_response", "error", time.Since(start).Seconds())
	return fmt.Errorf("no pending request found for request ID: %s", requestID)
}

// AddEntry stores a complete request-response pair in the cache
func (c *InMemoryCache) AddEntry(requestID string, model string, query string, requestBody, responseBody []byte) error {
	start := time.Now()

	if !c.enabled {
		return nil
	}

	// Generate semantic embedding for the query
	embedding, err := candle_binding.GetEmbedding(query, 0) // Auto-detect dimension
	if err != nil {
		metrics.RecordCacheOperation("memory", "add_entry", "error", time.Since(start).Seconds())
		return fmt.Errorf("failed to generate embedding: %w", err)
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	// Clean up expired entries before adding new one
	c.cleanupExpiredEntries()

	// Check if eviction is needed before adding the new entry
	if c.maxEntries > 0 && len(c.entries) >= c.maxEntries {
		c.evictOne()
	}

	now := time.Now()
	entry := CacheEntry{
		RequestID:    requestID,
		RequestBody:  requestBody,
		ResponseBody: responseBody,
		Model:        model,
		Query:        query,
		Embedding:    embedding,
		Timestamp:    now,
		LastAccessAt: now,
		HitCount:     0,
	}

	c.entries = append(c.entries, entry)
	observability.Debugf("InMemoryCache.AddEntry: added complete entry (total entries: %d, request_size: %d, response_size: %d)",
		len(c.entries), len(requestBody), len(responseBody))
	observability.LogEvent("cache_entry_added", map[string]interface{}{
		"backend": "memory",
		"query":   query,
		"model":   model,
	})

	// Record success metrics
	metrics.RecordCacheOperation("memory", "add_entry", "success", time.Since(start).Seconds())
	metrics.UpdateCacheEntries("memory", len(c.entries))

	return nil
}

// FindSimilar searches for semantically similar cached requests
func (c *InMemoryCache) FindSimilar(model string, query string) ([]byte, bool, error) {
	start := time.Now()

	if !c.enabled {
		observability.Debugf("InMemoryCache.FindSimilar: cache disabled")
		return nil, false, nil
	}
	queryPreview := query
	if len(query) > 50 {
		queryPreview = query[:50] + "..."
	}
	observability.Debugf("InMemoryCache.FindSimilar: searching for model='%s', query='%s' (len=%d chars)",
		model, queryPreview, len(query))

	// Generate semantic embedding for similarity comparison
	queryEmbedding, err := candle_binding.GetEmbedding(query, 0) // Auto-detect dimension
	if err != nil {
		metrics.RecordCacheOperation("memory", "find_similar", "error", time.Since(start).Seconds())
		return nil, false, fmt.Errorf("failed to generate embedding: %w", err)
	}

	c.mu.RLock()

	// Check for expired entries during search
	c.cleanupExpiredEntriesReadOnly()

	var (
		bestIndex      = -1
		bestEntry      CacheEntry
		bestSimilarity float32
		entriesChecked int
	)

	// Compare with completed entries for the same model, tracking only the best match
	for entryIndex, entry := range c.entries {
		if entry.ResponseBody == nil {
			continue // Skip incomplete entries
		}

		// Only consider entries for the same model
		if entry.Model != model {
			continue
		}

		// Compute semantic similarity using dot product
		var dotProduct float32
		for i := 0; i < len(queryEmbedding) && i < len(entry.Embedding); i++ {
			dotProduct += queryEmbedding[i] * entry.Embedding[i]
		}

		entriesChecked++
		if bestIndex == -1 || dotProduct > bestSimilarity {
			bestSimilarity = dotProduct
			bestIndex = entryIndex
		}
	}
	// Snapshot the best entry before releasing the read lock
	if bestIndex >= 0 {
		bestEntry = c.entries[bestIndex]
	}

	// Unlock the read lock since we need the write lock to update the access info
	c.mu.RUnlock()

	// Handle case where no suitable entries exist
	if bestIndex < 0 {
		atomic.AddInt64(&c.missCount, 1)
		observability.Debugf("InMemoryCache.FindSimilar: no entries found with responses")
		metrics.RecordCacheOperation("memory", "find_similar", "miss", time.Since(start).Seconds())
		metrics.RecordCacheMiss()
		return nil, false, nil
	}

	// Check if the best match meets the similarity threshold
	if bestSimilarity >= c.similarityThreshold {
		atomic.AddInt64(&c.hitCount, 1)

		c.mu.Lock()
		c.updateAccessInfo(bestIndex, bestEntry)
		c.mu.Unlock()

		observability.Debugf("InMemoryCache.FindSimilar: CACHE HIT - similarity=%.4f >= threshold=%.4f, response_size=%d bytes",
			bestSimilarity, c.similarityThreshold, len(bestEntry.ResponseBody))
		observability.LogEvent("cache_hit", map[string]interface{}{
			"backend":    "memory",
			"similarity": bestSimilarity,
			"threshold":  c.similarityThreshold,
			"model":      model,
		})
		metrics.RecordCacheOperation("memory", "find_similar", "hit", time.Since(start).Seconds())
		metrics.RecordCacheHit()
		return bestEntry.ResponseBody, true, nil
	}

	atomic.AddInt64(&c.missCount, 1)
	observability.Debugf("InMemoryCache.FindSimilar: CACHE MISS - best_similarity=%.4f < threshold=%.4f (checked %d entries)",
		bestSimilarity, c.similarityThreshold, entriesChecked)
	observability.LogEvent("cache_miss", map[string]interface{}{
		"backend":         "memory",
		"best_similarity": bestSimilarity,
		"threshold":       c.similarityThreshold,
		"model":           model,
		"entries_checked": entriesChecked,
	})
	metrics.RecordCacheOperation("memory", "find_similar", "miss", time.Since(start).Seconds())
	metrics.RecordCacheMiss()
	return nil, false, nil
}

// Close releases all resources held by the cache
func (c *InMemoryCache) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Clear all entries to free memory
	c.entries = nil
	return nil
}

// GetStats provides current cache performance metrics
func (c *InMemoryCache) GetStats() CacheStats {
	c.mu.RLock()
	defer c.mu.RUnlock()

	hits := atomic.LoadInt64(&c.hitCount)
	misses := atomic.LoadInt64(&c.missCount)
	total := hits + misses

	var hitRatio float64
	if total > 0 {
		hitRatio = float64(hits) / float64(total)
	}

	stats := CacheStats{
		TotalEntries: len(c.entries),
		HitCount:     hits,
		MissCount:    misses,
		HitRatio:     hitRatio,
	}

	if c.lastCleanupTime != nil {
		stats.LastCleanupTime = c.lastCleanupTime
	}

	return stats
}

// cleanupExpiredEntries removes entries that have exceeded their TTL
// Caller must hold a write lock
func (c *InMemoryCache) cleanupExpiredEntries() {
	if c.ttlSeconds <= 0 {
		return
	}

	now := time.Now()
	validEntries := make([]CacheEntry, 0, len(c.entries))

	for _, entry := range c.entries {
		// Retain entries that are still within their TTL based on last access
		if now.Sub(entry.LastAccessAt).Seconds() < float64(c.ttlSeconds) {
			validEntries = append(validEntries, entry)
		}
	}

	if len(validEntries) < len(c.entries) {
		expiredCount := len(c.entries) - len(validEntries)
		observability.Debugf("InMemoryCache: TTL cleanup removed %d expired entries (remaining: %d)",
			expiredCount, len(validEntries))
		observability.LogEvent("cache_cleanup", map[string]interface{}{
			"backend":         "memory",
			"expired_count":   expiredCount,
			"remaining_count": len(validEntries),
			"ttl_seconds":     c.ttlSeconds,
		})
		c.entries = validEntries
		cleanupTime := time.Now()
		c.lastCleanupTime = &cleanupTime
	}
}

// cleanupExpiredEntriesReadOnly identifies expired entries without modifying the cache
// Used during read operations with only a read lock held
func (c *InMemoryCache) cleanupExpiredEntriesReadOnly() {
	if c.ttlSeconds <= 0 {
		return
	}

	now := time.Now()
	expiredCount := 0

	for _, entry := range c.entries {
		if now.Sub(entry.LastAccessAt).Seconds() >= float64(c.ttlSeconds) {
			expiredCount++
		}
	}

	if expiredCount > 0 {
		observability.Debugf("InMemoryCache: found %d expired entries during read (TTL: %ds)",
			expiredCount, c.ttlSeconds)
		observability.LogEvent("cache_expired_entries_found", map[string]interface{}{
			"backend":       "memory",
			"expired_count": expiredCount,
			"ttl_seconds":   c.ttlSeconds,
		})
	}
}

// updateAccessInfo updates the access information for the given entry index
func (c *InMemoryCache) updateAccessInfo(entryIndex int, target CacheEntry) {
	// fast path
	if entryIndex < len(c.entries) && c.entries[entryIndex].RequestID == target.RequestID {
		c.entries[entryIndex].LastAccessAt = time.Now()
		c.entries[entryIndex].HitCount++
		return
	}

	// fallback to linear search
	for i := range c.entries {
		if c.entries[i].RequestID == target.RequestID {
			c.entries[i].LastAccessAt = time.Now()
			c.entries[i].HitCount++
			break
		}
	}
}

// evictOne removes one entry based on the configured eviction policy
func (c *InMemoryCache) evictOne() {
	if len(c.entries) == 0 {
		return
	}

	victimIdx := c.evictionPolicy.SelectVictim(c.entries)
	if victimIdx < 0 || victimIdx >= len(c.entries) {
		return
	}

	evictedRequestID := c.entries[victimIdx].RequestID

	c.entries[victimIdx] = c.entries[len(c.entries)-1]
	c.entries = c.entries[:len(c.entries)-1]

	observability.LogEvent("cache_evicted", map[string]any{
		"backend":     "memory",
		"request_id":  evictedRequestID,
		"max_entries": c.maxEntries,
	})
}
