//go:build !windows && cgo

package cache

import (
	"fmt"
	"math"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// HNSWNode represents a node in the HNSW graph
type HNSWNode struct {
	entryIndex int           // Index into InMemoryCache.entries
	neighbors  map[int][]int // Layer -> neighbor indices
	maxLayer   int           // Highest layer this node appears in
}

// HNSWIndex implements Hierarchical Navigable Small World graph for fast ANN search
type HNSWIndex struct {
	nodes          []*HNSWNode
	nodeIndex      map[int]*HNSWNode // entryIndex → node for O(1) lookup (critical for performance!)
	entryPoint     int               // Index of the top-level entry point
	maxLayer       int               // Maximum layer in the graph
	efConstruction int               // Size of dynamic candidate list during construction
	M              int               // Number of bi-directional links per node
	Mmax           int               // Maximum number of connections per node (=M)
	Mmax0          int               // Maximum number of connections for layer 0 (=M*2)
	ml             float64           // Normalization factor for level assignment
}

// InMemoryCache provides a high-performance semantic cache using BERT embeddings in memory
type InMemoryCache struct {
	entries             []CacheEntry
	entryMap            map[string]int // requestID -> index for O(1) lookup
	mu                  sync.RWMutex
	similarityThreshold float32
	maxEntries          int
	ttlSeconds          int
	enabled             bool
	hitCount            int64
	missCount           int64
	lastCleanupTime     *time.Time
	evictionPolicy      EvictionPolicy
	evictionPolicyType  EvictionPolicyType // Track the policy type for optimized eviction

	// O(1) eviction tracking
	optimizedLRU   *LRUPolicy
	optimizedLFU   *LFUPolicy
	optimizedFIFO  *FIFOPolicy
	expirationHeap *ExpirationHeap

	hnswIndex        *HNSWIndex
	useHNSW          bool
	hnswNeedsRebuild bool   // true while the HNSW graph is stale relative to entries
	hnswEfSearch     int    // Search-time ef parameter
	embeddingModel   string // "bert", "qwen3", or "gemma"

	// Background cleanup
	cleanupTicker *time.Ticker
	stopCleanup   chan struct{}
	closeOnce     sync.Once
}

// InMemoryCacheOptions contains configuration parameters for the in-memory cache
type InMemoryCacheOptions struct {
	SimilarityThreshold float32
	MaxEntries          int
	TTLSeconds          int
	Enabled             bool
	EvictionPolicy      EvictionPolicyType
	UseHNSW             bool   // Enable HNSW index for faster search
	HNSWM               int    // Number of bi-directional links (default: 16)
	HNSWEfConstruction  int    // Size of dynamic candidate list during construction (default: 200)
	HNSWEfSearch        int    // Size of dynamic candidate list during search (default: 50)
	EmbeddingModel      string // "bert", "qwen3", or "gemma"
}

// NewInMemoryCache initializes a new in-memory semantic cache instance
func NewInMemoryCache(options InMemoryCacheOptions) *InMemoryCache {
	logging.Debugf("Initializing in-memory cache: enabled=%t, maxEntries=%d, ttlSeconds=%d, threshold=%.3f, eviction_policy=%s, useHNSW=%t",
		options.Enabled, options.MaxEntries, options.TTLSeconds, options.SimilarityThreshold, options.EvictionPolicy, options.UseHNSW)

	// Set HNSW search ef parameter
	efSearch := options.HNSWEfSearch
	if efSearch <= 0 {
		efSearch = 50 // Default value
	}

	// Set default embedding model if not specified, normalize to lowercase
	embeddingModel := strings.ToLower(strings.TrimSpace(options.EmbeddingModel))
	if embeddingModel == "" {
		embeddingModel = "bert" // Default: BERT (fastest, lowest memory)
	}

	logging.Debugf("Semantic cache embedding model: %s", embeddingModel)

	cache := &InMemoryCache{
		entries:             []CacheEntry{},
		entryMap:            make(map[string]int),
		similarityThreshold: options.SimilarityThreshold,
		maxEntries:          options.MaxEntries,
		ttlSeconds:          options.TTLSeconds,
		enabled:             options.Enabled,
		evictionPolicyType:  options.EvictionPolicy,
		expirationHeap:      NewExpirationHeap(),
		useHNSW:             options.UseHNSW,
		hnswEfSearch:        efSearch,
		embeddingModel:      embeddingModel,
	}

	// Initialize O(1) eviction policy
	switch options.EvictionPolicy {
	case LRUEvictionPolicyType:
		cache.optimizedLRU = NewLRUPolicy()
		cache.evictionPolicy = cache.optimizedLRU
		logging.Debugf("LRU policy initialized for O(1) eviction")
	case LFUEvictionPolicyType:
		cache.optimizedLFU = NewLFUPolicy()
		cache.evictionPolicy = cache.optimizedLFU
		logging.Debugf("LFU policy initialized for O(1) eviction")
	default: // FIFO
		cache.optimizedFIFO = NewFIFOPolicy()
		cache.evictionPolicy = cache.optimizedFIFO
		logging.Debugf("FIFO policy initialized for O(1) eviction")
	}

	// Initialize HNSW index if enabled
	if options.UseHNSW {
		M := options.HNSWM
		if M <= 0 {
			M = 16 // Default value
		}
		efConstruction := options.HNSWEfConstruction
		if efConstruction <= 0 {
			efConstruction = 200 // Default value
		}
		cache.hnswIndex = newHNSWIndex(M, efConstruction)
		logging.Debugf("HNSW index initialized: M=%d, efConstruction=%d", M, efConstruction)
	}

	// Start background cleanup goroutine if TTL is enabled
	if options.Enabled && options.TTLSeconds > 0 {
		cache.stopCleanup = make(chan struct{})
		// Run cleanup every TTL/2 seconds (e.g., every 30s for 60s TTL)
		cleanupInterval := time.Duration(options.TTLSeconds/2) * time.Second
		if cleanupInterval < 10*time.Second {
			cleanupInterval = 10 * time.Second // Minimum 10 seconds
		}
		cache.cleanupTicker = time.NewTicker(cleanupInterval)
		go cache.backgroundCleanup()
		logging.Debugf("Background cleanup started: interval=%v", cleanupInterval)
	}

	return cache
}

// IsEnabled returns the current cache activation status
func (c *InMemoryCache) IsEnabled() bool {
	return c.enabled
}

// CheckConnection verifies the cache connection is healthy
// For in-memory cache, this is always healthy (no external connection)
func (c *InMemoryCache) CheckConnection() error {
	// In-memory cache has no external connection to check
	return nil
}

// generateEmbedding generates an embedding using the configured model
func (c *InMemoryCache) generateEmbedding(text string) ([]float32, error) {
	// Normalize to lowercase for case-insensitive comparison
	modelName := strings.ToLower(strings.TrimSpace(c.embeddingModel))

	switch modelName {
	case "qwen3":
		// Use GetEmbeddingBatched for Qwen3 with TRUE continuous batching
		// Now properly fixed to avoid CUDA context issues!
		output, err := candle_binding.GetEmbeddingBatched(text, modelName, 0)
		if err != nil {
			return nil, err
		}
		return output.Embedding, nil
	case "gemma":
		// Use GetEmbeddingWithModelType for Gemma (standard version)
		output, err := candle_binding.GetEmbeddingWithModelType(text, modelName, 0)
		if err != nil {
			return nil, err
		}
		return output.Embedding, nil
	case "bert", "":
		// Use traditional GetEmbedding for BERT (default)
		return candle_binding.GetEmbedding(text, 0)
	default:
		return nil, fmt.Errorf("unsupported embedding model: %s (must be 'bert', 'qwen3', or 'gemma')", c.embeddingModel)
	}
}

// AddPendingRequest stores a request that is awaiting its response
func (c *InMemoryCache) AddPendingRequest(
	requestID string,
	model string,
	query string,
	requestBody []byte,
	ttlSeconds int,
) error {
	start := time.Now()

	if !c.enabled {
		return nil
	}

	// Handle TTL=0: skip caching entirely
	if ttlSeconds == 0 {
		logging.Debugf("InMemoryCache.AddPendingRequest: skipping cache (ttl_seconds=0)")
		return nil
	}

	// Determine effective TTL: use provided value or fall back to cache default
	effectiveTTL := ttlSeconds
	if ttlSeconds == -1 {
		effectiveTTL = c.ttlSeconds
	}

	// Generate semantic embedding using the configured model
	embedding, err := c.generateEmbedding(query)
	if err != nil {
		metrics.RecordCacheOperation("memory", "add_pending", "error", time.Since(start).Seconds())
		return fmt.Errorf("failed to generate embedding: %w", err)
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	// Remove expired entries to maintain cache hygiene, but defer the HNSW rebuild to the insertion below if HNSW is enabled.
	c.cleanupExpiredEntriesDeferred()

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
		TTLSeconds:   ttlSeconds,
	}

	// Calculate expiration time if TTL is set
	if effectiveTTL > 0 {
		entry.ExpiresAt = now.Add(time.Duration(effectiveTTL) * time.Second)
	}

	c.entries = append(c.entries, entry)
	entryIndex := len(c.entries) - 1
	c.entryMap[requestID] = entryIndex

	// Register with optimized eviction policy for O(1) eviction
	c.registerEntryWithEvictionPolicy(entryIndex, requestID)

	// Register with expiration heap for efficient TTL cleanup
	if effectiveTTL > 0 {
		c.expirationHeap.Add(requestID, entryIndex, entry.ExpiresAt)
	}

	// Add to HNSW index if enabled. Do not call c.hnswIndex.addNode directly to keep in sync with entries slice when evictions/cleanups occurred.
	c.addEntryToHNSWIndex(entryIndex, embedding)

	logging.Debugf("InMemoryCache.AddPendingRequest: added pending entry (total entries: %d, embedding_dim: %d, useHNSW: %t, ttl=%d)",
		len(c.entries), len(embedding), c.useHNSW, effectiveTTL)

	// Record metrics
	metrics.RecordCacheOperation("memory", "add_pending", "success", time.Since(start).Seconds())
	metrics.UpdateCacheEntries("memory", len(c.entries))

	return nil
}

// UpdateWithResponse completes a pending request by adding the response
func (c *InMemoryCache) UpdateWithResponse(requestID string, responseBody []byte, ttlSeconds int) error {
	start := time.Now()

	if !c.enabled {
		return nil
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	// Clean up expired entries during the update
	c.cleanupExpiredEntries()

	// Locate the pending request and complete it
	now := time.Now()
	for i, entry := range c.entries {
		if entry.RequestID == requestID && entry.ResponseBody == nil {
			// Complete the cache entry with the response
			c.entries[i].ResponseBody = responseBody
			c.entries[i].Timestamp = now
			c.entries[i].LastAccessAt = now

			// Update TTL if provided (ttlSeconds != -1)
			// If ttlSeconds == 0, this means we shouldn't cache - but entry already exists, so just mark as complete
			if ttlSeconds != -1 {
				c.entries[i].TTLSeconds = ttlSeconds
				// Recalculate expiration time
				effectiveTTL := ttlSeconds
				if ttlSeconds == -1 {
					effectiveTTL = c.ttlSeconds
				}
				if effectiveTTL > 0 {
					c.entries[i].ExpiresAt = now.Add(time.Duration(effectiveTTL) * time.Second)
					// Update expiration heap with new expiration time
					c.expirationHeap.UpdateExpiration(requestID, c.entries[i].ExpiresAt)
				}
			}

			logging.Debugf("InMemoryCache.UpdateWithResponse: updated entry with response (response_size: %d bytes, ttl=%d)",
				len(responseBody), c.entries[i].TTLSeconds)

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
func (c *InMemoryCache) AddEntry(
	requestID string,
	model string,
	query string,
	requestBody []byte,
	responseBody []byte,
	ttlSeconds int,
) error {
	start := time.Now()

	if !c.enabled {
		return nil
	}

	if ttlSeconds == 0 {
		logging.Debugf("InMemoryCache.AddEntry: skipping cache (ttl_seconds=0)")
		return nil
	}

	effectiveTTL := ttlSeconds
	if ttlSeconds == -1 {
		effectiveTTL = c.ttlSeconds
	}

	// Generate semantic embedding using the configured model
	embedding, err := c.generateEmbedding(query)
	if err != nil {
		metrics.RecordCacheOperation("memory", "add_entry", "error", time.Since(start).Seconds())
		return fmt.Errorf("failed to generate embedding: %w", err)
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	// Remove expired entries to maintain cache hygiene, but defer the HNSW rebuild to the insertion below if HNSW is enabled.
	c.cleanupExpiredEntriesDeferred()

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
		TTLSeconds:   ttlSeconds,
	}

	// Calculate expiration time if TTL is set
	if effectiveTTL > 0 {
		entry.ExpiresAt = now.Add(time.Duration(effectiveTTL) * time.Second)
	}

	c.entries = append(c.entries, entry)
	entryIndex := len(c.entries) - 1
	c.entryMap[requestID] = entryIndex

	// Register with optimized eviction policy for O(1) eviction
	c.registerEntryWithEvictionPolicy(entryIndex, requestID)

	// Register with expiration heap for efficient TTL cleanup
	if effectiveTTL > 0 {
		c.expirationHeap.Add(requestID, entryIndex, entry.ExpiresAt)
	}

	// Add to HNSW index if enabled. Do not call c.hnswIndex.addNode directly to keep in sync with entries slice when evictions/cleanups occurred.
	c.addEntryToHNSWIndex(entryIndex, embedding)

	logging.Debugf("InMemoryCache.AddEntry: added complete entry (total entries: %d, request_size: %d, response_size: %d, useHNSW: %t, ttl=%d)",
		len(c.entries), len(requestBody), len(responseBody), c.useHNSW, effectiveTTL)
	logging.LogEvent("cache_entry_added", map[string]interface{}{
		"backend": "memory",
		"query":   query,
		"model":   model,
		"useHNSW": c.useHNSW,
	})

	// Record success metrics
	metrics.RecordCacheOperation("memory", "add_entry", "success", time.Since(start).Seconds())
	metrics.UpdateCacheEntries("memory", len(c.entries))

	return nil
}

// FindSimilar searches for semantically similar cached requests using the default threshold
func (c *InMemoryCache) FindSimilar(model string, query string) ([]byte, bool, error) {
	return c.FindSimilarWithThreshold(model, query, c.similarityThreshold)
}

// FindSimilarWithThreshold searches for semantically similar cached requests using a specific threshold
func (c *InMemoryCache) FindSimilarWithThreshold(model string, query string, threshold float32) ([]byte, bool, error) {
	start := time.Now()

	if !c.enabled {
		logging.Debugf("InMemoryCache.FindSimilarWithThreshold: cache disabled")
		return nil, false, nil
	}
	queryPreview := query
	if len(query) > 50 {
		queryPreview = query[:50] + "..."
	}
	logging.Debugf("InMemoryCache.FindSimilarWithThreshold: searching for model='%s', query='%s' (len=%d chars), threshold=%.4f",
		model, queryPreview, len(query), threshold)

	// Generate semantic embedding using the configured model
	queryEmbedding, err := c.generateEmbedding(query)
	if err != nil {
		metrics.RecordCacheOperation("memory", "find_similar", "error", time.Since(start).Seconds())
		return nil, false, fmt.Errorf("failed to generate embedding: %w", err)
	}

	c.mu.RLock()
	var (
		bestIndex      = -1
		bestEntry      CacheEntry
		bestSimilarity float32
		entriesChecked int
		expiredCount   int
	)
	// Capture the lookup time after acquiring the read lock so TTL checks aren't skewed by embedding work or lock wait
	now := time.Now()

	// Use HNSW index for fast search if enabled
	if c.useHNSW && c.hnswIndex != nil {
		// Defensive check and rebuild HNSW index if marked as needing rebuild.
		if c.hnswNeedsRebuild {
			logging.Debugf("InMemoryCache.FindSimilar: HNSW index marked as needing rebuild, rebuilding now")
			// Usually this is not reachable since we rebuild during insertions.
			c.mu.RUnlock()
			c.mu.Lock()
			if c.hnswNeedsRebuild { // Double-check under write lock
				c.rebuildHNSWIndex()
			}
			c.mu.Unlock()
			c.mu.RLock()
		}

		// Search using HNSW index with configured ef parameter
		candidateIndices := c.hnswIndex.searchKNN(queryEmbedding, 10, c.hnswEfSearch, c.entries)

		// Filter candidates by expiration, then find best match
		for _, entryIndex := range candidateIndices {
			if entryIndex < 0 || entryIndex >= len(c.entries) {
				continue
			}

			entry := c.entries[entryIndex]

			// Skip incomplete entries
			if entry.ResponseBody == nil {
				continue
			}

			// Skip entries that have expired before considering them
			if c.isExpired(entry, now) {
				expiredCount++
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

		logging.Debugf("InMemoryCache.FindSimilar: HNSW search checked %d candidates", len(candidateIndices))
	} else {
		// Fallback to linear search
		for entryIndex, entry := range c.entries {
			// Skip incomplete entries
			if entry.ResponseBody == nil {
				continue
			}

			// Skip entries that have expired before considering them
			if c.isExpired(entry, now) {
				expiredCount++
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

		if !c.useHNSW {
			logging.Debugf("InMemoryCache.FindSimilar: Linear search used (HNSW disabled)")
		}
	}

	// Snapshot the best entry before releasing the read lock
	if bestIndex >= 0 {
		bestEntry = c.entries[bestIndex]
	}

	// Unlock the read lock since we need the write lock to update the access info
	c.mu.RUnlock()

	// Log if any expired entries were skipped
	if expiredCount > 0 {
		logging.Debugf("InMemoryCache: excluded %d expired entries during search (TTL: %ds)",
			expiredCount, c.ttlSeconds)
		logging.LogEvent("cache_expired_entries_found", map[string]interface{}{
			"backend":       "memory",
			"expired_count": expiredCount,
			"ttl_seconds":   c.ttlSeconds,
		})
	}

	// Handle case where no suitable entries exist
	if bestIndex < 0 {
		atomic.AddInt64(&c.missCount, 1)
		logging.Debugf("InMemoryCache.FindSimilarWithThreshold: no entries found with responses")
		metrics.RecordCacheOperation("memory", "find_similar", "miss", time.Since(start).Seconds())
		return nil, false, nil
	}

	// Check if the best match meets the similarity threshold
	if bestSimilarity >= threshold {
		atomic.AddInt64(&c.hitCount, 1)

		c.mu.Lock()
		c.updateAccessInfo(bestIndex, bestEntry)
		c.mu.Unlock()

		logging.Debugf("InMemoryCache.FindSimilarWithThreshold: CACHE HIT - similarity=%.4f >= threshold=%.4f, response_size=%d bytes",
			bestSimilarity, threshold, len(bestEntry.ResponseBody))
		logging.LogEvent("cache_hit", map[string]interface{}{
			"backend":    "memory",
			"similarity": bestSimilarity,
			"threshold":  threshold,
			"model":      model,
		})
		metrics.RecordCacheOperation("memory", "find_similar", "hit", time.Since(start).Seconds())
		return bestEntry.ResponseBody, true, nil
	}

	atomic.AddInt64(&c.missCount, 1)
	logging.Debugf("InMemoryCache.FindSimilarWithThreshold: CACHE MISS - best_similarity=%.4f < threshold=%.4f (checked %d entries)",
		bestSimilarity, threshold, entriesChecked)
	logging.LogEvent("cache_miss", map[string]interface{}{
		"backend":         "memory",
		"best_similarity": bestSimilarity,
		"threshold":       threshold,
		"model":           model,
		"entries_checked": entriesChecked,
	})
	metrics.RecordCacheOperation("memory", "find_similar", "miss", time.Since(start).Seconds())
	return nil, false, nil
}

// Close releases all resources held by the cache
func (c *InMemoryCache) Close() error {
	// Use sync.Once to ensure cleanup happens only once
	c.closeOnce.Do(func() {
		// Stop background cleanup goroutine
		if c.stopCleanup != nil {
			close(c.stopCleanup)
		}
		if c.cleanupTicker != nil {
			c.cleanupTicker.Stop()
		}

		c.mu.Lock()
		defer c.mu.Unlock()

		// Clear all entries to free memory
		c.entries = nil

		// Zero cache entries metrics
		metrics.UpdateCacheEntries("memory", 0)
	})

	return nil
}

// backgroundCleanup runs periodic cleanup of expired entries
func (c *InMemoryCache) backgroundCleanup() {
	for {
		select {
		case <-c.cleanupTicker.C:
			c.mu.Lock()
			c.cleanupExpiredEntries()
			c.mu.Unlock()
		case <-c.stopCleanup:
			return
		}
	}
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

// cleanupExpiredEntries removes entries that have exceeded their TTL and immediately rebuilds HNSW if HNSW is enabled and cleanup occurs.
//
// Caller must hold a write lock.
func (c *InMemoryCache) cleanupExpiredEntries() {
	c.cleanupExpiredEntriesInternal(false)
}

// cleanupExpiredEntriesDeferred removes expired entries.
//
// If HNSW is enabled and cleanup occurs, it marks HNSW as needing rebuild but defers the rebuild until next call to addEntryToHNSWIndex or rebuildHNSWIndex.
// This is used in write paths that already plan to mutate the slice again (evictions, appends) so we only rebuild once per batch.
//
// Caller must hold a write lock.
func (c *InMemoryCache) cleanupExpiredEntriesDeferred() {
	c.cleanupExpiredEntriesInternal(true)
}

// cleanupExpiredEntriesInternal optionally postpones HNSW rebuild until the caller finishes batching updates.
// Uses expiration heap for O(k) cleanup where k = number of expired entries.
func (c *InMemoryCache) cleanupExpiredEntriesInternal(deferRebuild bool) {
	if c.ttlSeconds <= 0 {
		return
	}

	now := time.Now()

	// Use expiration heap for efficient O(k) cleanup where k = expired entries
	expiredRequestIDs := c.expirationHeap.PopExpired(now)

	if len(expiredRequestIDs) == 0 {
		return
	}

	// Remove expired entries from the entries slice
	// Build a set of expired request IDs for O(1) lookup
	expiredSet := make(map[string]bool, len(expiredRequestIDs))
	for _, id := range expiredRequestIDs {
		expiredSet[id] = true
	}

	// Compact the entries slice, keeping non-expired entries
	writeIdx := 0
	for readIdx := 0; readIdx < len(c.entries); readIdx++ {
		entry := c.entries[readIdx]
		if !expiredSet[entry.RequestID] {
			if writeIdx != readIdx {
				c.entries[writeIdx] = entry
				// Update tracking for the moved entry
				c.updateMovedEntryIndex(entry.RequestID, readIdx, writeIdx)
			}
			writeIdx++
		} else {
			// Remove from tracking structures
			c.removeEntryFromTracking(readIdx, entry.RequestID)
		}
	}
	c.entries = c.entries[:writeIdx]

	expiredCount := len(expiredRequestIDs)
	logging.Debugf("InMemoryCache: TTL cleanup removed %d expired entries (remaining: %d)",
		expiredCount, len(c.entries))
	logging.LogEvent("cache_cleanup", map[string]interface{}{
		"backend":         "memory",
		"expired_count":   expiredCount,
		"remaining_count": len(c.entries),
		"ttl_seconds":     c.ttlSeconds,
	})
	cleanupTime := time.Now()
	c.lastCleanupTime = &cleanupTime

	// Record cleanup operation metric
	if expiredCount > 0 {
		metrics.RecordCacheOperation("memory", "cleanup_expired", "success", time.Since(now).Seconds())
	}

	// Rebuild HNSW index if entries were removed and deferRebuild is false
	if expiredCount > 0 && c.useHNSW && c.hnswIndex != nil {
		logging.Debugf("InMemoryCache: TTL cleanup removed entries, marking HNSW index as needing rebuild")
		c.hnswNeedsRebuild = true
		c.hnswIndex.markStale()
		if !deferRebuild {
			c.rebuildHNSWIndex()
		}
	}

	// Update metrics after cleanup
	metrics.UpdateCacheEntries("memory", len(c.entries))
}

// isExpired checks if a cache entry has expired based on its last access time
func (c *InMemoryCache) isExpired(entry CacheEntry, now time.Time) bool {
	// Check per-entry expiration first
	if !entry.ExpiresAt.IsZero() {
		return now.After(entry.ExpiresAt)
	}

	// Fall back to global TTL for backward compatibility
	if c.ttlSeconds <= 0 {
		return false
	}

	return now.Sub(entry.LastAccessAt) >= time.Duration(c.ttlSeconds)*time.Second
}

// updateAccessInfo updates the access information for the given entry index
func (c *InMemoryCache) updateAccessInfo(entryIndex int, target CacheEntry) {
	now := time.Now()

	// fast path
	if entryIndex < len(c.entries) && c.entries[entryIndex].RequestID == target.RequestID {
		c.entries[entryIndex].LastAccessAt = now
		c.entries[entryIndex].HitCount++

		// Update optimized eviction policy tracking
		c.notifyAccessToEvictionPolicy(entryIndex, target.RequestID)

		// Extend TTL in expiration heap (sliding window TTL)
		// Use per-entry TTL if set, otherwise use global TTL
		effectiveTTL := c.ttlSeconds
		if c.entries[entryIndex].TTLSeconds > 0 {
			effectiveTTL = c.entries[entryIndex].TTLSeconds
		}
		if effectiveTTL > 0 {
			newExpiresAt := now.Add(time.Duration(effectiveTTL) * time.Second)
			c.entries[entryIndex].ExpiresAt = newExpiresAt
			c.expirationHeap.UpdateExpiration(target.RequestID, newExpiresAt)
		}
		return
	}

	// fallback to linear search
	for i := range c.entries {
		if c.entries[i].RequestID == target.RequestID {
			c.entries[i].LastAccessAt = now
			c.entries[i].HitCount++

			// Update optimized eviction policy tracking
			c.notifyAccessToEvictionPolicy(i, target.RequestID)

			// Extend TTL in expiration heap (sliding window TTL)
			// Use per-entry TTL if set, otherwise use global TTL
			effectiveTTL := c.ttlSeconds
			if c.entries[i].TTLSeconds > 0 {
				effectiveTTL = c.entries[i].TTLSeconds
			}
			if effectiveTTL > 0 {
				newExpiresAt := now.Add(time.Duration(effectiveTTL) * time.Second)
				c.entries[i].ExpiresAt = newExpiresAt
				c.expirationHeap.UpdateExpiration(target.RequestID, newExpiresAt)
			}
			break
		}
	}
}

// addEntryToHNSWIndex adds a new entry to the HNSW index, rebuilding if hnswNeedsRebuild is true.
// If HNSW is disabled, this is a no-op.
//
// Caller must hold a write lock.
func (c *InMemoryCache) addEntryToHNSWIndex(entryIndex int, embedding []float32) {
	if !c.useHNSW || c.hnswIndex == nil {
		return
	}

	if c.hnswNeedsRebuild {
		logging.Debugf("InMemoryCache.addEntryToHNSWIndex: HNSW index marked as needing rebuild, rebuilding now")
		c.rebuildHNSWIndex() // Rebuild HNSW index if stale
	} else {
		logging.Debugf("InMemoryCache.addEntryToHNSWIndex: adding new node to HNSW index for entryIndex=%d", entryIndex)
		c.hnswIndex.addNode(entryIndex, embedding, c.entries) // Not stale, just add the new node
	}
}

// evictOne removes one entry based on the configured eviction policy.
// It marks HNSW as needing rebuild if HNSW is enabled and an eviction occurs. HNSW will be rebuilt on next call to addEntryToHNSWIndex or rebuildHNSWIndex.
//
// Caller must hold a write lock.
func (c *InMemoryCache) evictOne() {
	if len(c.entries) == 0 {
		return
	}

	// Use optimized O(1) eviction
	victimIdx := c.evictUsingOptimizedPolicy()
	if victimIdx < 0 || victimIdx >= len(c.entries) {
		// Fallback to legacy O(n) eviction if optimized policy is not available
		victimIdx = c.evictionPolicy.SelectVictim(c.entries)
		if victimIdx < 0 || victimIdx >= len(c.entries) {
			return
		}
	}

	evictedRequestID := c.entries[victimIdx].RequestID

	// If using HNSW, we need to rebuild the index after eviction
	// For simplicity, we'll mark that a rebuild is needed
	if c.useHNSW && c.hnswIndex != nil {
		logging.Debugf("InMemoryCache.evictOne: HNSW index marked as needing rebuild due to eviction")
		// Note: HNSW doesn't support efficient deletion, leave the rebuild for the next insertion so we only rebuild once for eviction + append.
		c.hnswNeedsRebuild = true
		c.hnswIndex.markStale()
	}

	// Remove from optimized tracking structures
	c.removeEntryFromTracking(victimIdx, evictedRequestID)

	// Swap with last entry and shrink slice
	lastIdx := len(c.entries) - 1
	if victimIdx != lastIdx {
		movedEntry := c.entries[lastIdx]
		c.entries[victimIdx] = movedEntry
		// Update tracking for the moved entry
		c.updateMovedEntryIndex(movedEntry.RequestID, lastIdx, victimIdx)
	}
	c.entries = c.entries[:lastIdx]

	logging.LogEvent("cache_evicted", map[string]any{
		"backend":     "memory",
		"request_id":  evictedRequestID,
		"max_entries": c.maxEntries,
	})

	// Record eviction metric
	metrics.RecordCacheOperation("memory", "evict", "success", 0)

	// Update cache entries count after eviction
	metrics.UpdateCacheEntries("memory", len(c.entries))
}

// ===== Optimized Eviction Policy Helpers =====

// registerEntryWithEvictionPolicy registers a new entry with the appropriate optimized eviction policy.
// Caller must hold a write lock.
func (c *InMemoryCache) registerEntryWithEvictionPolicy(entryIndex int, requestID string) {
	switch c.evictionPolicyType {
	case LRUEvictionPolicyType:
		if c.optimizedLRU != nil {
			c.optimizedLRU.OnInsert(entryIndex, requestID)
		}
	case LFUEvictionPolicyType:
		if c.optimizedLFU != nil {
			c.optimizedLFU.OnInsert(entryIndex, requestID)
		}
	default: // FIFO
		if c.optimizedFIFO != nil {
			c.optimizedFIFO.OnInsert(entryIndex, requestID)
		}
	}
}

// evictUsingOptimizedPolicy uses the optimized O(1) eviction policy to select and evict a victim.
// Returns the victim index, or -1 if no victim was evicted.
// Caller must hold a write lock.
func (c *InMemoryCache) evictUsingOptimizedPolicy() int {
	switch c.evictionPolicyType {
	case LRUEvictionPolicyType:
		if c.optimizedLRU != nil {
			return c.optimizedLRU.Evict()
		}
	case LFUEvictionPolicyType:
		if c.optimizedLFU != nil {
			return c.optimizedLFU.Evict()
		}
	default: // FIFO
		if c.optimizedFIFO != nil {
			return c.optimizedFIFO.Evict()
		}
	}
	return -1
}

// removeEntryFromTracking removes an entry from all tracking structures.
// Caller must hold a write lock.
func (c *InMemoryCache) removeEntryFromTracking(entryIndex int, requestID string) {
	// Remove from entryMap
	delete(c.entryMap, requestID)

	// Remove from expiration heap
	c.expirationHeap.Remove(requestID)

	// Remove from optimized eviction policy
	// Note: This is idempotent - safe to call even if already removed by Evict()
	switch c.evictionPolicyType {
	case LRUEvictionPolicyType:
		if c.optimizedLRU != nil {
			c.optimizedLRU.OnRemove(entryIndex, requestID)
		}
	case LFUEvictionPolicyType:
		if c.optimizedLFU != nil {
			c.optimizedLFU.OnRemove(entryIndex, requestID)
		}
	default: // FIFO
		if c.optimizedFIFO != nil {
			c.optimizedFIFO.OnRemove(entryIndex, requestID)
		}
	}
}

// updateMovedEntryIndex updates tracking structures when an entry is moved to a new index.
// This is called after swap operations during eviction or cleanup.
// Caller must hold a write lock.
func (c *InMemoryCache) updateMovedEntryIndex(requestID string, oldIdx, newIdx int) {
	// Update entryMap
	c.entryMap[requestID] = newIdx

	// Update expiration heap
	c.expirationHeap.UpdateIndex(requestID, newIdx)

	// Update optimized eviction policy
	switch c.evictionPolicyType {
	case LRUEvictionPolicyType:
		if c.optimizedLRU != nil {
			c.optimizedLRU.UpdateIndex(requestID, oldIdx, newIdx)
		}
	case LFUEvictionPolicyType:
		if c.optimizedLFU != nil {
			c.optimizedLFU.UpdateIndex(requestID, oldIdx, newIdx)
		}
	default: // FIFO
		if c.optimizedFIFO != nil {
			c.optimizedFIFO.UpdateIndex(requestID, oldIdx, newIdx)
		}
	}
}

// notifyAccessToEvictionPolicy notifies the eviction policy of an access to update recency/frequency.
// Caller must hold a write lock.
func (c *InMemoryCache) notifyAccessToEvictionPolicy(entryIndex int, requestID string) {
	switch c.evictionPolicyType {
	case LRUEvictionPolicyType:
		if c.optimizedLRU != nil {
			c.optimizedLRU.OnAccess(entryIndex, requestID)
		}
	case LFUEvictionPolicyType:
		if c.optimizedLFU != nil {
			c.optimizedLFU.OnAccess(entryIndex, requestID)
		}
		// FIFO doesn't need access tracking
	}
}

// ===== HNSW Index Implementation =====

// rebuildHNSWIndex rebuilds the HNSW index from scratch.
// Caller must hold a write lock.
func (c *InMemoryCache) rebuildHNSWIndex() {
	if !c.useHNSW || c.hnswIndex == nil {
		c.hnswNeedsRebuild = false
		return
	}

	logging.Debugf("InMemoryCache: Rebuilding HNSW index with %d entries", len(c.entries))

	// Clear the existing index
	c.hnswIndex.nodes = []*HNSWNode{}
	c.hnswIndex.nodeIndex = make(map[int]*HNSWNode) // Clear O(1) lookup map
	c.hnswIndex.entryPoint = -1
	c.hnswIndex.maxLayer = -1

	// Rebuild by adding all entries
	for i, entry := range c.entries {
		if len(entry.Embedding) > 0 {
			c.hnswIndex.addNode(i, entry.Embedding, c.entries)
		}
	}

	logging.Debugf("InMemoryCache: HNSW index rebuilt with %d nodes", len(c.hnswIndex.nodes))
	c.hnswNeedsRebuild = false
}

// newHNSWIndex creates a new HNSW index
func newHNSWIndex(m, efConstruction int) *HNSWIndex {
	return &HNSWIndex{
		nodes:          []*HNSWNode{},
		nodeIndex:      make(map[int]*HNSWNode), // Initialize O(1) lookup map
		entryPoint:     -1,
		maxLayer:       -1,
		efConstruction: efConstruction,
		M:              m,
		Mmax:           m,
		Mmax0:          m * 2,
		ml:             1.0 / math.Log(float64(m)),
	}
}

// markStale marks the index as needing a rebuild
func (h *HNSWIndex) markStale() {
	// Simple approach: clear the index
	h.nodes = []*HNSWNode{}
	h.nodeIndex = make(map[int]*HNSWNode) // Clear O(1) lookup map
	h.entryPoint = -1
	h.maxLayer = -1
}

// selectLevel randomly selects a level for a new node
func (h *HNSWIndex) selectLevel() int {
	// Use exponential decay probability
	r := -math.Log(math.Max(1e-9, 1.0-float64(time.Now().UnixNano()%1000000)/1000000.0))
	return int(r * h.ml)
}

// addNode adds a new node to the HNSW index.
//
// For InMemoryCache, it is called via addEntryToHNSWIndex to keep in sync with entries slice.
func (h *HNSWIndex) addNode(entryIndex int, embedding []float32, entries []CacheEntry) {
	level := h.selectLevel()

	node := &HNSWNode{
		entryIndex: entryIndex,
		neighbors:  make(map[int][]int),
		maxLayer:   level,
	}

	// If this is the first node, make it the entry point
	if h.entryPoint == -1 {
		h.entryPoint = entryIndex
		h.maxLayer = level
		h.nodes = append(h.nodes, node)
		h.nodeIndex[entryIndex] = node // Add to O(1) lookup map
		return
	}

	// Find nearest neighbors and connect
	for lc := min(level, h.maxLayer); lc >= 0; lc-- {
		candidates := h.searchLayer(embedding, h.entryPoint, h.efConstruction, lc, entries)

		// Select M nearest neighbors
		M := h.Mmax
		if lc == 0 {
			M = h.Mmax0
		}
		neighbors := h.selectNeighbors(candidates, M, embedding, entries)

		// Add bidirectional links
		node.neighbors[lc] = neighbors
		for _, neighborIdx := range neighbors {
			// Fast O(1) lookup using nodeIndex map
			if n := h.nodeIndex[neighborIdx]; n != nil {
				if n.neighbors[lc] == nil {
					n.neighbors[lc] = []int{}
				}
				n.neighbors[lc] = append(n.neighbors[lc], entryIndex)

				// Prune neighbors if needed
				if len(n.neighbors[lc]) > M {
					// Use neighbor's own embedding as query for pruning
					n.neighbors[lc] = h.selectNeighbors(n.neighbors[lc], M, entries[neighborIdx].Embedding, entries)
				}
			}
		}
	}

	// Update entry point if this node has a higher level
	if level > h.maxLayer {
		h.maxLayer = level
		h.entryPoint = entryIndex
	}

	h.nodes = append(h.nodes, node)
	h.nodeIndex[entryIndex] = node // Add to O(1) lookup map
}

// searchKNN performs k-nearest neighbor search
func (h *HNSWIndex) searchKNN(queryEmbedding []float32, k, ef int, entries []CacheEntry) []int {
	if h.entryPoint == -1 || len(h.nodes) == 0 {
		return []int{}
	}

	// Search from top layer to layer 1
	currentNearest := h.entryPoint
	for lc := h.maxLayer; lc > 0; lc-- {
		nearest := h.searchLayer(queryEmbedding, currentNearest, 1, lc, entries)
		if len(nearest) > 0 {
			currentNearest = nearest[0]
		}
	}

	// Search at layer 0 with ef
	return h.searchLayer(queryEmbedding, currentNearest, ef, 0, entries)
}

// searchLayer searches for nearest neighbors at a specific layer
func (h *HNSWIndex) searchLayer(queryEmbedding []float32, entryPoint, ef, layer int, entries []CacheEntry) []int {
	visited := make(map[int]bool)
	candidates := newMinHeap() // set of candidates, explore closest candidate first
	results := newMaxHeap()    // dynamic list of found nearest neighbors, track current frontier, worst distance on top

	// Calculate distance to entry point
	if entryPoint >= 0 && entryPoint < len(entries) {
		dist := h.distance(queryEmbedding, entries[entryPoint].Embedding)
		candidates.push(entryPoint, dist)
		results.push(entryPoint, dist)
		visited[entryPoint] = true
	}

	for candidates.len() > 0 {
		currentIdx, currentDist := candidates.pop()

		// If we have enough results and the current distance is worse than the worst in results, we can stop
		if results.len() > 0 && currentDist > results.peekDist() {
			break
		}

		// Fast O(1) lookup using nodeIndex map
		currentNode := h.nodeIndex[currentIdx]
		if currentNode == nil || currentNode.neighbors[layer] == nil {
			continue
		}

		// Check neighbors
		for _, neighborIdx := range currentNode.neighbors[layer] {
			if visited[neighborIdx] {
				continue
			}
			visited[neighborIdx] = true

			if neighborIdx >= 0 && neighborIdx < len(entries) {
				dist := h.distance(queryEmbedding, entries[neighborIdx].Embedding)

				if results.len() < ef {
					candidates.push(neighborIdx, dist)
					results.push(neighborIdx, dist)
				} else if dist < results.peekDist() {
					candidates.push(neighborIdx, dist)
					results.push(neighborIdx, dist)
					if results.len() > ef {
						results.pop()
					}
				}
			}
		}
	}

	return results.items()
}

// selectNeighbors selects the best neighbors by sorting by distance
// This is CRITICAL for HNSW graph quality - must select NEAREST neighbors, not arbitrary ones!
func (h *HNSWIndex) selectNeighbors(candidates []int, m int, queryEmb []float32, entries []CacheEntry) []int {
	// Validate queryEmb: must not be nil or empty to ensure correct distance calculations
	if len(queryEmb) == 0 {
		logging.Errorf("selectNeighbors: queryEmb is empty - cannot compute distances")
		return []int{}
	}

	if len(candidates) <= m {
		return candidates
	}

	// Create a temporary slice with distances for sorting
	type neighborDist struct {
		idx  int
		dist float32
	}

	neighbors := make([]neighborDist, len(candidates))

	// Compute distance from query to each candidate (using SIMD!)
	for i, idx := range candidates {
		if idx >= 0 && idx < len(entries) {
			// Validate dimension match to prevent silent data corruption in HNSW graph
			if len(entries[idx].Embedding) != len(queryEmb) {
				logging.Errorf("selectNeighbors: dimension mismatch - query has %d dims, candidate %d has %d dims",
					len(queryEmb), idx, len(entries[idx].Embedding))
				// Skip this candidate rather than corrupting the graph
				continue
			}
			neighbors[i] = neighborDist{
				idx:  idx,
				dist: h.distance(queryEmb, entries[idx].Embedding),
			}
		}
	}

	// Sort by distance (ascending - smallest distance first)
	// Use a simple selection sort since m is typically small (16-32)
	for i := 0; i < m && i < len(neighbors); i++ {
		minIdx := i
		for j := i + 1; j < len(neighbors); j++ {
			if neighbors[j].dist < neighbors[minIdx].dist {
				minIdx = j
			}
		}
		if minIdx != i {
			neighbors[i], neighbors[minIdx] = neighbors[minIdx], neighbors[i]
		}
	}

	// Return the m nearest neighbors
	result := make([]int, m)
	for i := 0; i < m; i++ {
		result[i] = neighbors[i].idx
	}
	return result
}

// distance calculates cosine similarity (as dot product since embeddings are normalized)
func (h *HNSWIndex) distance(a, b []float32) float32 {
	// We use negative dot product so that larger similarity = smaller distance
	// Use SIMD-optimized dot product (AVX2/AVX512)
	dotProduct := dotProductSIMD(a, b)
	return -dotProduct // Negate so higher similarity = lower distance
}

// Helper priority queue implementations for HNSW

type heapItem struct {
	index int
	dist  float32
}

type minHeap struct {
	data []heapItem
}

func newMinHeap() *minHeap {
	return &minHeap{data: []heapItem{}}
}

func (h *minHeap) push(index int, dist float32) {
	h.data = append(h.data, heapItem{index, dist})
	h.bubbleUp(len(h.data) - 1)
}

func (h *minHeap) pop() (int, float32) {
	if len(h.data) == 0 {
		return -1, 0
	}
	result := h.data[0]
	h.data[0] = h.data[len(h.data)-1]
	h.data = h.data[:len(h.data)-1]
	if len(h.data) > 0 {
		h.bubbleDown(0)
	}
	return result.index, result.dist
}

func (h *minHeap) len() int {
	return len(h.data)
}

func (h *minHeap) bubbleUp(i int) {
	for i > 0 {
		parent := (i - 1) / 2
		if h.data[i].dist >= h.data[parent].dist {
			break
		}
		h.data[i], h.data[parent] = h.data[parent], h.data[i]
		i = parent
	}
}

func (h *minHeap) bubbleDown(i int) {
	for {
		left := 2*i + 1
		right := 2*i + 2
		smallest := i

		if left < len(h.data) && h.data[left].dist < h.data[smallest].dist {
			smallest = left
		}
		if right < len(h.data) && h.data[right].dist < h.data[smallest].dist {
			smallest = right
		}
		if smallest == i {
			break
		}
		h.data[i], h.data[smallest] = h.data[smallest], h.data[i]
		i = smallest
	}
}

type maxHeap struct {
	data []heapItem
}

func newMaxHeap() *maxHeap {
	return &maxHeap{data: []heapItem{}}
}

func (h *maxHeap) push(index int, dist float32) {
	h.data = append(h.data, heapItem{index, dist})
	h.bubbleUp(len(h.data) - 1)
}

func (h *maxHeap) pop() (int, float32) {
	if len(h.data) == 0 {
		return -1, 0
	}
	result := h.data[0]
	h.data[0] = h.data[len(h.data)-1]
	h.data = h.data[:len(h.data)-1]
	if len(h.data) > 0 {
		h.bubbleDown(0)
	}
	return result.index, result.dist
}

func (h *maxHeap) len() int {
	return len(h.data)
}

func (h *maxHeap) peekDist() float32 {
	if len(h.data) == 0 {
		return math.MaxFloat32
	}
	return h.data[0].dist
}

func (h *maxHeap) items() []int {
	result := make([]int, len(h.data))
	for i, item := range h.data {
		result[i] = item.index
	}
	return result
}

func (h *maxHeap) bubbleUp(i int) {
	for i > 0 {
		parent := (i - 1) / 2
		if h.data[i].dist <= h.data[parent].dist {
			break
		}
		h.data[i], h.data[parent] = h.data[parent], h.data[i]
		i = parent
	}
}

func (h *maxHeap) bubbleDown(i int) {
	for {
		left := 2*i + 1
		right := 2*i + 2
		largest := i

		if left < len(h.data) && h.data[left].dist > h.data[largest].dist {
			largest = left
		}
		if right < len(h.data) && h.data[right].dist > h.data[largest].dist {
			largest = right
		}
		if largest == i {
			break
		}
		h.data[i], h.data[largest] = h.data[largest], h.data[i]
		i = largest
	}
}
