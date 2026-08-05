//go:build !windows && cgo

package cache

import (
	"fmt"
	"sync"
	"time"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// InMemoryCache provides a high-performance semantic cache using BERT embeddings in memory
type InMemoryCache struct {
	SimilarityTracker   // embedded — provides LastSimilarity()
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
	embeddingModel   string // "bert", "qwen3", "gemma", "mmbert", or "multimodal"

	// embMemo deduplicates query-embedding inference: a cache-miss request
	// otherwise embeds the same query twice (lookup + pending write), so the
	// memo turns the second compute into a memory hit.
	embMemo *embeddingMemo

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
	EmbeddingModel      string // "bert", "qwen3", "gemma", "mmbert", or "multimodal"
}

func attachInMemoryEvictionPolicy(cache *InMemoryCache, policy EvictionPolicyType) {
	switch policy {
	case LRUEvictionPolicyType:
		cache.optimizedLRU = NewLRUPolicy()
		cache.evictionPolicy = cache.optimizedLRU
		logging.ComponentDebugEvent("cache", "inmemory_cache_eviction_policy_initialized", map[string]interface{}{
			"policy": "lru",
		})
	case LFUEvictionPolicyType:
		cache.optimizedLFU = NewLFUPolicy()
		cache.evictionPolicy = cache.optimizedLFU
		logging.ComponentDebugEvent("cache", "inmemory_cache_eviction_policy_initialized", map[string]interface{}{
			"policy": "lfu",
		})
	default: // FIFO
		cache.optimizedFIFO = NewFIFOPolicy()
		cache.evictionPolicy = cache.optimizedFIFO
		logging.ComponentDebugEvent("cache", "inmemory_cache_eviction_policy_initialized", map[string]interface{}{
			"policy": "fifo",
		})
	}
}

func attachInMemoryHNSW(cache *InMemoryCache, options InMemoryCacheOptions, efSearch int) {
	if !options.UseHNSW {
		return
	}
	M := options.HNSWM
	if M <= 0 {
		M = 16 // Default value
	}
	efConstruction := options.HNSWEfConstruction
	if efConstruction <= 0 {
		efConstruction = 200 // Default value
	}
	cache.hnswIndex = newHNSWIndex(M, efConstruction)
	logging.ComponentDebugEvent("cache", "inmemory_cache_hnsw_initialized", map[string]interface{}{
		"hnsw_m":               M,
		"hnsw_ef_construction": efConstruction,
		"hnsw_ef_search":       efSearch,
	})
}

func startInMemoryTTLCleanup(cache *InMemoryCache, options InMemoryCacheOptions) {
	if !options.Enabled || options.TTLSeconds <= 0 {
		return
	}
	cache.stopCleanup = make(chan struct{})
	cleanupInterval := time.Duration(options.TTLSeconds/2) * time.Second
	if cleanupInterval < 10*time.Second {
		cleanupInterval = 10 * time.Second // Minimum 10 seconds
	}
	cache.cleanupTicker = time.NewTicker(cleanupInterval)
	go cache.backgroundCleanup()
	logging.ComponentDebugEvent("cache", "inmemory_cache_cleanup_started", map[string]interface{}{
		"interval_seconds": cleanupInterval.Seconds(),
	})
}

// NewInMemoryCache initializes a new in-memory semantic cache instance
func NewInMemoryCache(options InMemoryCacheOptions) *InMemoryCache {
	// Set HNSW search ef parameter
	efSearch := options.HNSWEfSearch
	if efSearch <= 0 {
		efSearch = 50 // Default value
	}

	// Store the canonical model name used by embedding dispatch.
	embeddingModel := normalizeEmbeddingModel(options.EmbeddingModel)

	logging.ComponentEvent("cache", "inmemory_cache_init_started", map[string]interface{}{
		"enabled":              options.Enabled,
		"max_entries":          options.MaxEntries,
		"ttl_seconds":          options.TTLSeconds,
		"similarity_threshold": options.SimilarityThreshold,
		"eviction_policy":      options.EvictionPolicy,
		"use_hnsw":             options.UseHNSW,
		"embedding_model":      embeddingModel,
	})

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
		embMemo:             newEmbeddingMemo(defaultEmbeddingMemoSize),
	}

	logging.ComponentEvent("cache", "inmemory_cache_initialized", map[string]interface{}{
		"enabled":              options.Enabled,
		"max_entries":          options.MaxEntries,
		"ttl_seconds":          options.TTLSeconds,
		"similarity_threshold": options.SimilarityThreshold,
		"eviction_policy":      options.EvictionPolicy,
		"use_hnsw":             options.UseHNSW,
		"hnsw_ef_search":       efSearch,
		"embedding_model":      embeddingModel,
	})

	attachInMemoryEvictionPolicy(cache, options.EvictionPolicy)
	attachInMemoryHNSW(cache, options, efSearch)
	startInMemoryTTLCleanup(cache, options)

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

// generateEmbedding returns an embedding for text using the configured model,
// served from the embedding memo when the same text was embedded recently
// (e.g. the lookup + pending-write pair of a single cache-miss request).
func (c *InMemoryCache) generateEmbedding(text string) ([]float32, error) {
	if c.embMemo != nil {
		return c.embMemo.getOrCompute(text, c.computeEmbedding)
	}
	return c.computeEmbedding(text)
}

// computeEmbedding runs the configured embedding model for text (no caching).
func (c *InMemoryCache) computeEmbedding(text string) ([]float32, error) {
	modelName := c.embeddingModel

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
	case "mmbert":
		// Use GetEmbedding2DMatryoshka for mmBERT with 2D Matryoshka support
		// Default to layer 6 (~3.6x speedup) and dimension 256 for good balance
		output, err := candle_binding.GetEmbedding2DMatryoshka(text, modelName, 6, 256)
		if err != nil {
			return nil, err
		}
		return output.Embedding, nil
	case "multimodal":
		// Use multimodal text encoder branch (384-dim default)
		output, err := candle_binding.GetEmbeddingWithModelType(text, modelName, 384)
		if err != nil {
			return nil, err
		}
		return output.Embedding, nil
	case "bert":
		// Use traditional GetEmbedding for BERT (default)
		return candle_binding.GetEmbedding(text, 0)
	default:
		return nil, fmt.Errorf("unsupported embedding model: %s (must be 'bert', 'qwen3', 'gemma', 'mmbert', or 'multimodal')", c.embeddingModel)
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
	// Fast path: use entryMap for O(1) lookup
	targetIdx := -1
	if idx, ok := c.entryMap[requestID]; ok && idx >= 0 && idx < len(c.entries) &&
		c.entries[idx].RequestID == requestID && c.entries[idx].ResponseBody == nil {
		targetIdx = idx
	}
	// Fallback to linear search if not found
	if targetIdx == -1 {
		for i, entry := range c.entries {
			if entry.RequestID == requestID && entry.ResponseBody == nil {
				targetIdx = i
				c.entryMap[requestID] = i
				break
			}
		}
	}
	// No matching pending request found
	if targetIdx == -1 {
		metrics.RecordCacheOperation("memory", "update_response", "error", time.Since(start).Seconds())
		return fmt.Errorf("no pending request found for request ID: %s", requestID)
	}

	// Complete the cache entry with the response
	c.entries[targetIdx].ResponseBody = responseBody
	c.entries[targetIdx].Timestamp = now
	c.entries[targetIdx].LastAccessAt = now
	// Update TTL if provided (ttlSeconds != -1)
	// If ttlSeconds == 0, this means we shouldn't cache - but entry already exists, so just mark as complete
	if ttlSeconds != -1 {
		c.entries[targetIdx].TTLSeconds = ttlSeconds
		if ttlSeconds > 0 {
			c.entries[targetIdx].ExpiresAt = now.Add(time.Duration(ttlSeconds) * time.Second)
			// Update expiration heap with new expiration time
			c.expirationHeap.UpdateExpiration(requestID, c.entries[targetIdx].ExpiresAt)
		}
	}

	// Record successful completion
	logging.Debugf("InMemoryCache.UpdateWithResponse: updated entry with response (response_size: %d bytes, ttl=%d)",
		len(responseBody), c.entries[targetIdx].TTLSeconds)
	metrics.RecordCacheOperation("memory", "update_response", "success", time.Since(start).Seconds())
	return nil
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
