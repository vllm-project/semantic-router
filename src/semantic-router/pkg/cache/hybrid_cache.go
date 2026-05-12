//go:build !windows && cgo

package cache

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

const (
	// Buffer pool limits to prevent memory bloat
	maxVisitedMapSize     = 1000 // Maximum size for visited map before discarding buffer
	maxCandidatesCapacity = 200  // Maximum capacity for candidates heap before discarding buffer
	maxResultsCapacity    = 200  // Maximum capacity for results heap before discarding buffer
	maxHNSWLayers         = 16   // Maximum number of layers in HNSW index
)

// searchBuffers holds reusable buffers for HNSW search to reduce GC pressure
type searchBuffers struct {
	visited    map[int]bool
	candidates *minHeap
	results    *maxHeap
}

// Global pool for search buffers (reduces allocations)
var searchBufferPool = sync.Pool{
	New: func() interface{} {
		return &searchBuffers{
			visited:    make(map[int]bool, 100),
			candidates: newMinHeap(),
			results:    newMaxHeap(),
		}
	},
}

// getSearchBuffers gets reusable buffers from pool
func getSearchBuffers() *searchBuffers {
	buf := searchBufferPool.Get().(*searchBuffers)
	// Clear maps and heaps for reuse
	clear(buf.visited)
	buf.candidates.data = buf.candidates.data[:0]
	buf.results.data = buf.results.data[:0]
	return buf
}

// putSearchBuffers returns buffers to pool
func putSearchBuffers(buf *searchBuffers) {
	// Don't return to pool if buffers grew too large (avoid memory bloat)
	if len(buf.visited) > maxVisitedMapSize || cap(buf.candidates.data) > maxCandidatesCapacity || cap(buf.results.data) > maxResultsCapacity {
		return
	}
	searchBufferPool.Put(buf)
}

// HybridCache combines in-memory HNSW index with external Milvus storage
// Architecture:
//   - In-memory: HNSW index with ALL embeddings (for fast O(log n) search)
//   - Milvus: ALL documents (fetched by ID after search)
//
// This provides fast search while supporting millions of entries without storing docs in memory
type HybridCache struct {
	SimilarityTracker // embedded — provides LastSimilarity()
	// In-memory components (search only)
	hnswIndex  *HNSWIndex
	embeddings [][]float32
	idMap      map[int]string // Entry index → Milvus ID

	// External storage (all documents)
	milvusCache *MilvusCache

	// Configuration
	similarityThreshold float32
	maxMemoryEntries    int // Max entries in HNSW index
	ttlSeconds          int
	enabled             bool

	// Statistics
	hitCount   int64
	missCount  int64
	evictCount int64

	// Concurrency control
	mu sync.RWMutex
}

// HybridCacheOptions contains configuration for the hybrid cache
type HybridCacheOptions struct {
	// Core settings
	Enabled             bool
	SimilarityThreshold float32
	TTLSeconds          int

	// HNSW settings
	MaxMemoryEntries   int // Max entries in HNSW (default: 100,000)
	HNSWM              int // HNSW M parameter
	HNSWEfConstruction int // HNSW efConstruction parameter

	// Milvus settings
	Milvus *config.MilvusConfig

	// Embedding settings
	EmbeddingModel string // "bert", "qwen3", "gemma", "mmbert", or "multimodal"

	// (Deprecated) Milvus settings configuration path
	MilvusConfigPath string

	// Startup settings
	DisableRebuildOnStartup bool // Skip rebuilding HNSW index from Milvus on startup (default: false, meaning rebuild IS enabled)
}

// NewHybridCache creates a new hybrid cache instance
func NewHybridCache(options HybridCacheOptions) (*HybridCache, error) {
	logging.ComponentEvent("cache", "hybrid_cache_init_started", map[string]interface{}{
		"enabled":              options.Enabled,
		"max_memory_entries":   options.MaxMemoryEntries,
		"similarity_threshold": options.SimilarityThreshold,
		"ttl_seconds":          options.TTLSeconds,
		"rebuild_on_startup":   !options.DisableRebuildOnStartup,
	})

	if !options.Enabled {
		logging.ComponentDebugEvent("cache", "hybrid_cache_init_skipped", map[string]interface{}{
			"reason":  "disabled",
			"enabled": false,
		})
		return &HybridCache{
			enabled: false,
		}, nil
	}

	// Initialize Milvus backend
	milvusCache, err := NewMilvusCache(milvusCacheOptionsFromHybridOptions(options))
	if err != nil {
		return nil, fmt.Errorf("failed to initialize Milvus backend: %w", err)
	}

	// Set defaults
	if options.MaxMemoryEntries <= 0 {
		options.MaxMemoryEntries = 100000 // Default: 100K entries in memory
	}
	if options.HNSWM <= 0 {
		options.HNSWM = 16
	}
	if options.HNSWEfConstruction <= 0 {
		options.HNSWEfConstruction = 200
	}

	// Initialize HNSW index
	hnswIndex := newHNSWIndex(options.HNSWM, options.HNSWEfConstruction)

	cache := &HybridCache{
		hnswIndex:           hnswIndex,
		embeddings:          make([][]float32, 0, options.MaxMemoryEntries),
		idMap:               make(map[int]string),
		milvusCache:         milvusCache,
		similarityThreshold: options.SimilarityThreshold,
		maxMemoryEntries:    options.MaxMemoryEntries,
		ttlSeconds:          options.TTLSeconds,
		enabled:             true,
	}

	logging.ComponentEvent("cache", "hybrid_cache_initialized", map[string]interface{}{
		"hnsw_m":               options.HNSWM,
		"hnsw_ef_construction": options.HNSWEfConstruction,
		"max_memory_entries":   options.MaxMemoryEntries,
		"similarity_threshold": options.SimilarityThreshold,
		"ttl_seconds":          options.TTLSeconds,
		"rebuild_on_startup":   !options.DisableRebuildOnStartup,
	})

	// Rebuild HNSW index from Milvus on startup (enabled by default)
	// This ensures the in-memory index is populated after a restart
	// Set DisableRebuildOnStartup=true to skip this step (not recommended for production)
	if !options.DisableRebuildOnStartup {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
		defer cancel()

		if err := cache.RebuildFromMilvus(ctx); err != nil {
			logging.ComponentWarnEvent("cache", "hybrid_cache_rebuild_failed", map[string]interface{}{
				"source":      "startup",
				"error":       err.Error(),
				"empty_index": true,
			})
			// Don't fail initialization, just log warning and continue with empty index
		}
	} else {
		logging.ComponentWarnEvent("cache", "hybrid_cache_rebuild_skipped", map[string]interface{}{
			"source":      "startup",
			"reason":      "disabled_on_startup",
			"empty_index": true,
		})
	}

	return cache, nil
}

func milvusCacheOptionsFromHybridOptions(options HybridCacheOptions) MilvusCacheOptions {
	milvusOptions := MilvusCacheOptions{
		Enabled:             true,
		SimilarityThreshold: options.SimilarityThreshold,
		TTLSeconds:          options.TTLSeconds,
		EmbeddingModel:      options.EmbeddingModel,
	}

	if options.Milvus != nil {
		milvusOptions.Config = options.Milvus
		return milvusOptions
	}

	milvusOptions.ConfigPath = options.MilvusConfigPath
	return milvusOptions
}

func (h *HybridCache) generateEmbedding(text string) ([]float32, error) {
	if h.milvusCache == nil {
		return nil, fmt.Errorf("milvus cache is not initialized")
	}

	return h.milvusCache.getEmbedding(text)
}

// IsEnabled returns whether the cache is active
func (h *HybridCache) IsEnabled() bool {
	return h.enabled
}

// CheckConnection verifies the cache backend connection is healthy
// For hybrid cache, this checks the Milvus connection
func (h *HybridCache) CheckConnection() error {
	if !h.enabled {
		return nil
	}

	if h.milvusCache == nil {
		return fmt.Errorf("milvus cache is not initialized")
	}

	// Delegate to Milvus cache connection check
	return h.milvusCache.CheckConnection()
}

// RebuildFromMilvus rebuilds the in-memory HNSW index from persistent Milvus storage
// This is called on startup to recover the index after a restart
func (h *HybridCache) RebuildFromMilvus(ctx context.Context) error {
	if !h.enabled {
		return nil
	}

	start := time.Now()
	logging.ComponentEvent("cache", "hybrid_cache_rebuild_started", map[string]interface{}{
		"max_memory_entries": h.maxMemoryEntries,
	})

	// Query all entries from Milvus
	requestIDs, embeddings, err := h.milvusCache.GetAllEntries(ctx)
	if err != nil {
		return fmt.Errorf("failed to get entries from Milvus: %w", err)
	}

	if len(requestIDs) == 0 {
		logging.ComponentEvent("cache", "hybrid_cache_rebuild_completed", map[string]interface{}{
			"entries_loaded":     0,
			"entries_available":  0,
			"duration_seconds":   time.Since(start).Seconds(),
			"entries_per_second": 0.0,
			"empty":              true,
			"truncated":          false,
		})
		return nil
	}

	// Lock for the entire rebuild process
	h.mu.Lock()
	defer h.mu.Unlock()

	// Clear existing index
	loadLimit := min(len(embeddings), h.maxMemoryEntries)
	h.embeddings = make([][]float32, 0, loadLimit)
	h.idMap = make(map[int]string, loadLimit)
	h.hnswIndex = newHNSWIndex(h.hnswIndex.M, h.hnswIndex.efConstruction)

	// Rebuild HNSW index with progress logging
	batchSize := 1000
	for i, embedding := range embeddings {
		// Check memory limits
		if len(h.embeddings) >= h.maxMemoryEntries {
			logging.ComponentWarnEvent("cache", "hybrid_cache_rebuild_truncated", map[string]interface{}{
				"max_memory_entries": h.maxMemoryEntries,
				"entries_loaded":     i,
				"entries_available":  len(embeddings),
			})
			break
		}

		// Add to HNSW
		entryIndex := len(h.embeddings)
		h.embeddings = append(h.embeddings, embedding)
		h.idMap[entryIndex] = requestIDs[i]
		h.addNodeHybrid(entryIndex, embedding)

		// Progress logging for large datasets
		if (i+1)%batchSize == 0 {
			elapsed := time.Since(start)
			rate := 0.0
			if elapsed.Seconds() > 0 {
				rate = float64(i+1) / elapsed.Seconds()
			}
			remaining := len(embeddings) - (i + 1)
			etaSeconds := 0.0
			if rate > 0 {
				etaSeconds = float64(remaining) / rate
			}
			logging.ComponentDebugEvent("cache", "hybrid_cache_rebuild_progress", map[string]interface{}{
				"entries_loaded":     i + 1,
				"entries_available":  len(embeddings),
				"progress_percent":   float64(i+1) / float64(len(embeddings)) * 100,
				"entries_per_second": rate,
				"eta_seconds":        etaSeconds,
			})
		}
	}

	elapsed := time.Since(start)
	rate := 0.0
	if elapsed.Seconds() > 0 {
		rate = float64(len(h.embeddings)) / elapsed.Seconds()
	}
	logging.ComponentEvent("cache", "hybrid_cache_rebuild_completed", map[string]interface{}{
		"entries_loaded":     len(h.embeddings),
		"entries_available":  len(embeddings),
		"duration_seconds":   elapsed.Seconds(),
		"entries_per_second": rate,
		"empty":              len(h.embeddings) == 0,
		"truncated":          len(h.embeddings) < len(embeddings),
	})

	metrics.UpdateCacheEntries("hybrid", len(h.embeddings))

	return nil
}

// AddPendingRequest stores a request awaiting its response
func (h *HybridCache) AddPendingRequest(requestID string, model string, query string, requestBody []byte, ttlSeconds int) error {
	start := time.Now()

	if !h.enabled {
		return nil
	}

	// Handle TTL=0: skip caching entirely
	if ttlSeconds == 0 {
		logging.Debugf("HybridCache.AddPendingRequest: skipping cache (ttl_seconds=0)")
		return nil
	}

	// Generate embedding
	embedding, err := h.generateEmbedding(query)
	if err != nil {
		metrics.RecordCacheOperation("hybrid", "add_pending", "error", time.Since(start).Seconds())
		return fmt.Errorf("failed to generate embedding: %w", err)
	}

	// Store in Milvus (write-through)
	if err := h.milvusCache.AddPendingRequest(requestID, model, query, requestBody, ttlSeconds); err != nil {
		metrics.RecordCacheOperation("hybrid", "add_pending", "error", time.Since(start).Seconds())
		return fmt.Errorf("milvus add pending failed: %w", err)
	}

	// Add to in-memory HNSW index
	h.mu.Lock()
	defer h.mu.Unlock()

	// Check if we need to evict
	if len(h.embeddings) >= h.maxMemoryEntries {
		h.evictOneUnsafe()
	}

	// Add to HNSW
	entryIndex := len(h.embeddings)
	h.embeddings = append(h.embeddings, embedding)
	h.idMap[entryIndex] = requestID
	h.addNodeHybrid(entryIndex, embedding)

	logging.Debugf("HybridCache.AddPendingRequest: added to HNSW index=%d, milvusID=%s, ttl=%d",
		entryIndex, requestID, ttlSeconds)

	metrics.RecordCacheOperation("hybrid", "add_pending", "success", time.Since(start).Seconds())
	metrics.UpdateCacheEntries("hybrid", len(h.embeddings))

	return nil
}

// UpdateWithResponse completes a pending request with its response
func (h *HybridCache) UpdateWithResponse(requestID string, responseBody []byte, ttlSeconds int) error {
	start := time.Now()

	if !h.enabled {
		return nil
	}

	// Update in Milvus
	if err := h.milvusCache.UpdateWithResponse(requestID, responseBody, ttlSeconds); err != nil {
		metrics.RecordCacheOperation("hybrid", "update_response", "error", time.Since(start).Seconds())
		return fmt.Errorf("milvus update failed: %w", err)
	}

	// HNSW index already has the embedding, no update needed there

	logging.Debugf("HybridCache.UpdateWithResponse: updated milvusID=%s, ttl=%d", requestID, ttlSeconds)
	metrics.RecordCacheOperation("hybrid", "update_response", "success", time.Since(start).Seconds())

	return nil
}

// AddEntry stores a complete request-response pair
func (h *HybridCache) AddEntry(requestID string, model string, query string, requestBody, responseBody []byte, ttlSeconds int) error {
	start := time.Now()

	if !h.enabled {
		return nil
	}

	// Handle TTL=0: skip caching entirely
	if ttlSeconds == 0 {
		logging.Debugf("HybridCache.AddEntry: skipping cache (ttl_seconds=0)")
		return nil
	}

	// Generate embedding
	embedding, err := h.generateEmbedding(query)
	if err != nil {
		metrics.RecordCacheOperation("hybrid", "add_entry", "error", time.Since(start).Seconds())
		return fmt.Errorf("failed to generate embedding: %w", err)
	}

	// Store in Milvus (write-through)
	if err := h.milvusCache.AddEntry(requestID, model, query, requestBody, responseBody, ttlSeconds); err != nil {
		metrics.RecordCacheOperation("hybrid", "add_entry", "error", time.Since(start).Seconds())
		return fmt.Errorf("milvus add entry failed: %w", err)
	}

	// Add to in-memory HNSW index
	h.mu.Lock()
	defer h.mu.Unlock()

	// Check if we need to evict
	if len(h.embeddings) >= h.maxMemoryEntries {
		h.evictOneUnsafe()
	}

	// Add to HNSW
	entryIndex := len(h.embeddings)
	h.embeddings = append(h.embeddings, embedding)
	h.idMap[entryIndex] = requestID
	h.addNodeHybrid(entryIndex, embedding)

	logging.Debugf("HybridCache.AddEntry: added to HNSW index=%d, milvusID=%s, ttl=%d",
		entryIndex, requestID, ttlSeconds)
	logging.LogEvent("hybrid_cache_entry_added", map[string]interface{}{
		"backend": "hybrid",
		"query":   query,
		"model":   model,
		"in_hnsw": true,
	})

	metrics.RecordCacheOperation("hybrid", "add_entry", "success", time.Since(start).Seconds())
	metrics.UpdateCacheEntries("hybrid", len(h.embeddings))

	return nil
}

// AddEntriesBatch stores multiple request-response pairs efficiently
func (h *HybridCache) AddEntriesBatch(entries []CacheEntry) error {
	start := time.Now()

	if !h.enabled {
		return nil
	}

	if len(entries) == 0 {
		return nil
	}

	logging.Debugf("HybridCache.AddEntriesBatch: adding %d entries in batch", len(entries))

	// Generate all embeddings first
	embeddings := make([][]float32, len(entries))
	for i, entry := range entries {
		embedding, err := h.generateEmbedding(entry.Query)
		if err != nil {
			metrics.RecordCacheOperation("hybrid", "add_entries_batch", "error", time.Since(start).Seconds())
			return fmt.Errorf("failed to generate embedding for entry %d: %w", i, err)
		}
		embeddings[i] = embedding
	}

	// Store all in Milvus at once (write-through)
	if err := h.milvusCache.AddEntriesBatch(entries); err != nil {
		metrics.RecordCacheOperation("hybrid", "add_entries_batch", "error", time.Since(start).Seconds())
		return fmt.Errorf("milvus batch add failed: %w", err)
	}

	// Add all to in-memory HNSW index
	h.mu.Lock()
	defer h.mu.Unlock()

	for i, entry := range entries {
		// Check if we need to evict
		if len(h.embeddings) >= h.maxMemoryEntries {
			h.evictOneUnsafe()
		}

		// Add to HNSW
		entryIndex := len(h.embeddings)
		h.embeddings = append(h.embeddings, embeddings[i])
		h.idMap[entryIndex] = entry.RequestID
		h.addNodeHybrid(entryIndex, embeddings[i])
	}

	elapsed := time.Since(start)
	logging.Debugf("HybridCache.AddEntriesBatch: added %d entries in %v (%.0f entries/sec)",
		len(entries), elapsed, float64(len(entries))/elapsed.Seconds())
	logging.LogEvent("hybrid_cache_entries_added", map[string]interface{}{
		"backend": "hybrid",
		"count":   len(entries),
		"in_hnsw": true,
	})

	metrics.RecordCacheOperation("hybrid", "add_entries_batch", "success", elapsed.Seconds())
	metrics.UpdateCacheEntries("hybrid", len(h.embeddings))

	return nil
}

// Flush forces Milvus to persist all buffered data to disk
func (h *HybridCache) Flush() error {
	if !h.enabled {
		return nil
	}

	return h.milvusCache.Flush()
}

// Close releases all resources
func (h *HybridCache) Close() error {
	if !h.enabled {
		return nil
	}

	h.mu.Lock()
	defer h.mu.Unlock()

	// Close Milvus connection
	if h.milvusCache != nil {
		if err := h.milvusCache.Close(); err != nil {
			logging.Debugf("HybridCache.Close: Milvus close error: %v", err)
		}
	}

	// Clear in-memory structures
	h.embeddings = nil
	h.idMap = nil
	h.hnswIndex = nil

	metrics.UpdateCacheEntries("hybrid", 0)

	return nil
}

// GetStats returns cache statistics
func (h *HybridCache) GetStats() CacheStats {
	h.mu.RLock()
	defer h.mu.RUnlock()

	hits := atomic.LoadInt64(&h.hitCount)
	misses := atomic.LoadInt64(&h.missCount)
	total := hits + misses

	var hitRatio float64
	if total > 0 {
		hitRatio = float64(hits) / float64(total)
	}

	return CacheStats{
		TotalEntries: len(h.embeddings),
		HitCount:     hits,
		MissCount:    misses,
		HitRatio:     hitRatio,
	}
}

// Helper methods

// evictOneUnsafe removes one entry from HNSW index (must hold write lock)
func (h *HybridCache) evictOneUnsafe() {
	if len(h.embeddings) == 0 {
		return
	}

	// Simple FIFO eviction: remove oldest entry (index 0)
	victimIdx := 0

	// Get milvusID before removing from map (for logging)
	milvusID := h.idMap[victimIdx]

	// Remove the embedding from the slice
	h.embeddings = h.embeddings[1:]

	// Rebuild idMap with adjusted indices (all indices shift down by 1)
	newIDMap := make(map[int]string, len(h.idMap)-1)
	for idx, id := range h.idMap {
		if idx > victimIdx {
			newIDMap[idx-1] = id // Shift index down
		}
		// Skip victimIdx (it's being evicted)
	}
	h.idMap = newIDMap

	// Mark HNSW index as stale (needs rebuild with new indices)
	h.hnswIndex.markStale()

	atomic.AddInt64(&h.evictCount, 1)

	logging.Debugf("HybridCache.evictOne: evicted entry at index %d (milvus_id=%s), new size=%d",
		victimIdx, milvusID, len(h.embeddings))
	logging.LogEvent("hybrid_cache_evicted", map[string]interface{}{
		"backend":     "hybrid",
		"milvus_id":   milvusID,
		"hnsw_index":  victimIdx,
		"new_size":    len(h.embeddings),
		"max_entries": h.maxMemoryEntries,
	})
}
