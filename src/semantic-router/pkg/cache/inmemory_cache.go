//go:build !windows && cgo
// +build !windows,cgo

package cache

import (
	"fmt"
	"sort"
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
}

// InMemoryCacheOptions contains configuration parameters for the in-memory cache
type InMemoryCacheOptions struct {
	SimilarityThreshold float32
	MaxEntries          int
	TTLSeconds          int
	Enabled             bool
}

// NewInMemoryCache initializes a new in-memory semantic cache instance
func NewInMemoryCache(options InMemoryCacheOptions) *InMemoryCache {
	observability.Debugf("Initializing in-memory cache: enabled=%t, maxEntries=%d, ttlSeconds=%d, threshold=%.3f",
		options.Enabled, options.MaxEntries, options.TTLSeconds, options.SimilarityThreshold)
	return &InMemoryCache{
		entries:             []CacheEntry{},
		similarityThreshold: options.SimilarityThreshold,
		maxEntries:          options.MaxEntries,
		ttlSeconds:          options.TTLSeconds,
		enabled:             options.Enabled,
	}
}

// IsEnabled returns the current cache activation status
func (c *InMemoryCache) IsEnabled() bool {
	return c.enabled
}

// AddPendingRequest stores a request that is awaiting its response
func (c *InMemoryCache) AddPendingRequest(model string, query string, requestBody []byte) (string, error) {
	start := time.Now()

	if !c.enabled {
		return query, nil
	}

	// Generate semantic embedding for the query
	embedding, err := candle_binding.GetEmbedding(query, 0) // Auto-detect dimension
	if err != nil {
		metrics.RecordCacheOperation("memory", "add_pending", "error", time.Since(start).Seconds())
		return "", fmt.Errorf("failed to generate embedding: %w", err)
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	// Remove expired entries to maintain cache hygiene
	c.cleanupExpiredEntries()

	// Create cache entry for the pending request
	entry := CacheEntry{
		RequestBody: requestBody,
		Model:       model,
		Query:       query,
		Embedding:   embedding,
		Timestamp:   time.Now(),
	}

	c.entries = append(c.entries, entry)
	observability.Debugf("InMemoryCache.AddPendingRequest: added pending entry (total entries: %d, embedding_dim: %d)",
		len(c.entries), len(embedding))

	// Apply entry limit to prevent unbounded memory growth
	if c.maxEntries > 0 && len(c.entries) > c.maxEntries {
		// Sort entries by timestamp to identify oldest
		sort.Slice(c.entries, func(i, j int) bool {
			return c.entries[i].Timestamp.Before(c.entries[j].Timestamp)
		})
		// Keep only the most recent entries
		removedCount := len(c.entries) - c.maxEntries
		c.entries = c.entries[len(c.entries)-c.maxEntries:]
		observability.Debugf("InMemoryCache: size limit exceeded, removed %d oldest entries (limit: %d)",
			removedCount, c.maxEntries)
		observability.LogEvent("cache_trimmed", map[string]interface{}{
			"backend":       "memory",
			"removed_count": removedCount,
			"max_entries":   c.maxEntries,
		})
	}

	// Record metrics
	metrics.RecordCacheOperation("memory", "add_pending", "success", time.Since(start).Seconds())
	metrics.UpdateCacheEntries("memory", len(c.entries))

	return query, nil
}

// UpdateWithResponse completes a pending request by adding the response
func (c *InMemoryCache) UpdateWithResponse(query string, responseBody []byte) error {
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
		if entry.Query == query && entry.ResponseBody == nil {
			// Complete the cache entry with the response
			c.entries[i].ResponseBody = responseBody
			c.entries[i].Timestamp = time.Now()
			observability.Debugf("InMemoryCache.UpdateWithResponse: updated entry with response (response_size: %d bytes)",
				len(responseBody))

			// Record successful completion
			metrics.RecordCacheOperation("memory", "update_response", "success", time.Since(start).Seconds())
			return nil
		}
	}

	// No matching pending request found
	metrics.RecordCacheOperation("memory", "update_response", "error", time.Since(start).Seconds())
	return fmt.Errorf("no pending request found for query: %s", query)
}

// AddEntry stores a complete request-response pair in the cache
func (c *InMemoryCache) AddEntry(model string, query string, requestBody, responseBody []byte) error {
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

	entry := CacheEntry{
		RequestBody:  requestBody,
		ResponseBody: responseBody,
		Model:        model,
		Query:        query,
		Embedding:    embedding,
		Timestamp:    time.Now(),
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	// Clean up expired entries before adding new one
	c.cleanupExpiredEntries()

	c.entries = append(c.entries, entry)
	observability.Debugf("InMemoryCache.AddEntry: added complete entry (total entries: %d, request_size: %d, response_size: %d)",
		len(c.entries), len(requestBody), len(responseBody))
	observability.LogEvent("cache_entry_added", map[string]interface{}{
		"backend": "memory",
		"query":   query,
		"model":   model,
	})

	// Apply entry limit if configured
	if c.maxEntries > 0 && len(c.entries) > c.maxEntries {
		// Sort by timestamp to identify oldest entries
		sort.Slice(c.entries, func(i, j int) bool {
			return c.entries[i].Timestamp.Before(c.entries[j].Timestamp)
		})
		// Keep only the most recent entries
		c.entries = c.entries[len(c.entries)-c.maxEntries:]
	}

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
	defer c.mu.RUnlock()

	// Check for expired entries during search
	c.cleanupExpiredEntriesReadOnly()

	type SimilarityResult struct {
		Entry      CacheEntry
		Similarity float32
	}

	// Compare with completed entries for the same model
	results := make([]SimilarityResult, 0, len(c.entries))
	for _, entry := range c.entries {
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

		results = append(results, SimilarityResult{
			Entry:      entry,
			Similarity: dotProduct,
		})
	}

	// Handle case where no suitable entries exist
	if len(results) == 0 {
		atomic.AddInt64(&c.missCount, 1)
		observability.Debugf("InMemoryCache.FindSimilar: no entries found with responses (total entries: %d)", len(c.entries))
		metrics.RecordCacheOperation("memory", "find_similar", "miss", time.Since(start).Seconds())
		metrics.RecordCacheMiss()
		return nil, false, nil
	}

	// Sort results by similarity score (highest first)
	sort.Slice(results, func(i, j int) bool {
		return results[i].Similarity > results[j].Similarity
	})

	// Check if the best match meets the similarity threshold
	if results[0].Similarity >= c.similarityThreshold {
		atomic.AddInt64(&c.hitCount, 1)
		observability.Debugf("InMemoryCache.FindSimilar: CACHE HIT - similarity=%.4f >= threshold=%.4f, response_size=%d bytes",
			results[0].Similarity, c.similarityThreshold, len(results[0].Entry.ResponseBody))
		observability.LogEvent("cache_hit", map[string]interface{}{
			"backend":    "memory",
			"similarity": results[0].Similarity,
			"threshold":  c.similarityThreshold,
			"model":      model,
		})
		metrics.RecordCacheOperation("memory", "find_similar", "hit", time.Since(start).Seconds())
		metrics.RecordCacheHit()
		return results[0].Entry.ResponseBody, true, nil
	}

	atomic.AddInt64(&c.missCount, 1)
	observability.Debugf("InMemoryCache.FindSimilar: CACHE MISS - best_similarity=%.4f < threshold=%.4f (checked %d entries)",
		results[0].Similarity, c.similarityThreshold, len(results))
	observability.LogEvent("cache_miss", map[string]interface{}{
		"backend":         "memory",
		"best_similarity": results[0].Similarity,
		"threshold":       c.similarityThreshold,
		"model":           model,
		"entries_checked": len(results),
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
		// Retain entries that are still within their TTL
		if now.Sub(entry.Timestamp).Seconds() < float64(c.ttlSeconds) {
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
		if now.Sub(entry.Timestamp).Seconds() >= float64(c.ttlSeconds) {
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
