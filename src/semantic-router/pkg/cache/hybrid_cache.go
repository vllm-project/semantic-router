//go:build !windows && cgo

package cache

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
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
	for k := range buf.visited {
		delete(buf.visited, k)
	}
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

	// (Deprecated) Milvus settings configuration path
	MilvusConfigPath string

	// Startup settings
	DisableRebuildOnStartup bool // Skip rebuilding HNSW index from Milvus on startup (default: false, meaning rebuild IS enabled)
}

// NewHybridCache creates a new hybrid cache instance
func NewHybridCache(options HybridCacheOptions) (*HybridCache, error) {
	logging.Infof("Initializing hybrid cache: enabled=%t, maxMemoryEntries=%d, threshold=%.3f",
		options.Enabled, options.MaxMemoryEntries, options.SimilarityThreshold)

	if !options.Enabled {
		logging.Debugf("Hybrid cache disabled, returning inactive instance")
		return &HybridCache{
			enabled: false,
		}, nil
	}

	// Initialize Milvus backend
	var milvusOptions MilvusCacheOptions
	if options.Milvus != nil {
		milvusOptions = MilvusCacheOptions{
			Enabled:             true,
			SimilarityThreshold: options.SimilarityThreshold,
			TTLSeconds:          options.TTLSeconds,
			Config:              options.Milvus,
		}
	} else {
		milvusOptions = MilvusCacheOptions{
			Enabled:             true,
			SimilarityThreshold: options.SimilarityThreshold,
			TTLSeconds:          options.TTLSeconds,
			ConfigPath:          options.MilvusConfigPath,
		}
	}

	milvusCache, err := NewMilvusCache(milvusOptions)
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

	logging.Infof("Hybrid cache initialized: HNSW(M=%d, ef=%d), maxMemory=%d",
		options.HNSWM, options.HNSWEfConstruction, options.MaxMemoryEntries)

	// Rebuild HNSW index from Milvus on startup (enabled by default)
	// This ensures the in-memory index is populated after a restart
	// Set DisableRebuildOnStartup=true to skip this step (not recommended for production)
	if !options.DisableRebuildOnStartup {
		logging.Infof("Hybrid cache: rebuilding HNSW index from Milvus...")
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
		defer cancel()

		if err := cache.RebuildFromMilvus(ctx); err != nil {
			logging.Warnf("Hybrid cache: failed to rebuild HNSW index from Milvus: %v", err)
			logging.Warnf("Hybrid cache: continuing with empty HNSW index")
			// Don't fail initialization, just log warning and continue with empty index
		} else {
			logging.Infof("Hybrid cache: HNSW index rebuild complete")
		}
	} else {
		logging.Warnf("Hybrid cache: skipping HNSW index rebuild (DisableRebuildOnStartup=true)")
		logging.Warnf("Hybrid cache: index will be empty until entries are added")
	}

	return cache, nil
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
	logging.Infof("HybridCache.RebuildFromMilvus: starting HNSW index rebuild from Milvus")

	// Query all entries from Milvus
	requestIDs, embeddings, err := h.milvusCache.GetAllEntries(ctx)
	if err != nil {
		return fmt.Errorf("failed to get entries from Milvus: %w", err)
	}

	if len(requestIDs) == 0 {
		logging.Infof("HybridCache.RebuildFromMilvus: no entries to rebuild, starting with empty index")
		return nil
	}

	logging.Infof("HybridCache.RebuildFromMilvus: rebuilding HNSW index with %d entries", len(requestIDs))

	// Lock for the entire rebuild process
	h.mu.Lock()
	defer h.mu.Unlock()

	// Clear existing index
	h.embeddings = make([][]float32, 0, len(embeddings))
	h.idMap = make(map[int]string)
	h.hnswIndex = newHNSWIndex(h.hnswIndex.M, h.hnswIndex.efConstruction)

	// Rebuild HNSW index with progress logging
	batchSize := 1000
	for i, embedding := range embeddings {
		// Check memory limits
		if len(h.embeddings) >= h.maxMemoryEntries {
			logging.Warnf("HybridCache.RebuildFromMilvus: reached max memory entries (%d), stopping rebuild at %d/%d",
				h.maxMemoryEntries, i, len(embeddings))
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
			rate := float64(i+1) / elapsed.Seconds()
			remaining := len(embeddings) - (i + 1)
			eta := time.Duration(float64(remaining)/rate) * time.Second
			logging.Infof("HybridCache.RebuildFromMilvus: progress %d/%d (%.1f%%, %.0f entries/sec, ETA: %v)",
				i+1, len(embeddings), float64(i+1)/float64(len(embeddings))*100, rate, eta)
		}
	}

	elapsed := time.Since(start)
	rate := float64(len(h.embeddings)) / elapsed.Seconds()
	logging.Infof("HybridCache.RebuildFromMilvus: rebuild complete - %d entries in %v (%.0f entries/sec)",
		len(h.embeddings), elapsed, rate)

	logging.LogEvent("hybrid_cache_rebuilt", map[string]interface{}{
		"backend":           "hybrid",
		"entries_loaded":    len(h.embeddings),
		"entries_in_milvus": len(embeddings),
		"duration_seconds":  elapsed.Seconds(),
		"entries_per_sec":   rate,
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
	embedding, err := candle_binding.GetEmbedding(query, 0)
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
	embedding, err := candle_binding.GetEmbedding(query, 0)
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
		embedding, err := candle_binding.GetEmbedding(entry.Query, 0)
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

// FindSimilar searches for semantically similar cached requests
func (h *HybridCache) FindSimilar(model string, query string) ([]byte, bool, error) {
	start := time.Now()

	if !h.enabled {
		return nil, false, nil
	}

	queryPreview := query
	if len(query) > 50 {
		queryPreview = query[:50] + "..."
	}
	logging.Debugf("HybridCache.FindSimilar: searching for model='%s', query='%s'",
		model, queryPreview)

	// Generate query embedding
	queryEmbedding, err := candle_binding.GetEmbedding(query, 0)
	if err != nil {
		metrics.RecordCacheOperation("hybrid", "find_similar", "error", time.Since(start).Seconds())
		return nil, false, fmt.Errorf("failed to generate embedding: %w", err)
	}

	// Search HNSW index for candidates above similarity threshold
	// For semantic cache, we only need the first match, so search with k=1
	// and stop early when finding a match above threshold
	h.mu.RLock()
	candidates := h.searchKNNHybridWithThreshold(queryEmbedding, 1, 20, h.similarityThreshold)
	threshold := h.similarityThreshold
	h.mu.RUnlock()

	// Filter by similarity threshold before fetching from Milvus
	var qualifiedCandidates []searchResult
	for _, candidate := range candidates {
		if candidate.similarity >= threshold {
			qualifiedCandidates = append(qualifiedCandidates, candidate)
		}
	}

	// Map qualified candidates to Milvus IDs (need lock for idMap access)
	type candidateWithID struct {
		milvusID   string
		similarity float32
		index      int
	}

	h.mu.RLock()
	candidatesWithIDs := make([]candidateWithID, 0, len(qualifiedCandidates))
	for _, candidate := range qualifiedCandidates {
		if milvusID, ok := h.idMap[candidate.index]; ok {
			candidatesWithIDs = append(candidatesWithIDs, candidateWithID{
				milvusID:   milvusID,
				similarity: candidate.similarity,
				index:      candidate.index,
			})
		}
	}
	h.mu.RUnlock()

	if len(candidatesWithIDs) == 0 {
		atomic.AddInt64(&h.missCount, 1)
		if len(candidates) > 0 {
			logging.Debugf("HybridCache.FindSimilar: %d candidates found but none above threshold %.3f",
				len(candidates), h.similarityThreshold)
		} else {
			logging.Debugf("HybridCache.FindSimilar: no candidates found in HNSW")
		}
		metrics.RecordCacheOperation("hybrid", "find_similar", "miss", time.Since(start).Seconds())
		return nil, false, nil
	}

	logging.Debugf("HybridCache.FindSimilar: HNSW returned %d candidates, %d above threshold",
		len(candidates), len(candidatesWithIDs))

	// Fetch document from Milvus for qualified candidates
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// Try candidates in order (already sorted by similarity from HNSW)
	for _, candidate := range candidatesWithIDs {
		// Fetch document from Milvus by ID (direct lookup by primary key)
		fetchCtx, fetchCancel := context.WithTimeout(ctx, 2*time.Second)
		responseBody, err := h.milvusCache.GetByID(fetchCtx, candidate.milvusID)
		fetchCancel()

		if err != nil {
			logging.Debugf("HybridCache.FindSimilar: Milvus GetByID failed for %s: %v",
				candidate.milvusID, err)
			continue
		}

		if responseBody != nil {
			atomic.AddInt64(&h.hitCount, 1)
			logging.Debugf("HybridCache.FindSimilar: MILVUS HIT - similarity=%.4f (threshold=%.3f)",
				candidate.similarity, h.similarityThreshold)
			logging.LogEvent("hybrid_cache_hit", map[string]interface{}{
				"backend":    "hybrid",
				"source":     "milvus",
				"similarity": candidate.similarity,
				"threshold":  h.similarityThreshold,
				"model":      model,
				"latency_ms": time.Since(start).Milliseconds(),
			})
			metrics.RecordCacheOperation("hybrid", "find_similar", "hit_milvus", time.Since(start).Seconds())
			return responseBody, true, nil
		}
	}

	// No match found above threshold
	atomic.AddInt64(&h.missCount, 1)
	logging.Debugf("HybridCache.FindSimilar: CACHE MISS - no match above threshold")
	logging.LogEvent("hybrid_cache_miss", map[string]interface{}{
		"backend":    "hybrid",
		"threshold":  h.similarityThreshold,
		"model":      model,
		"candidates": len(candidatesWithIDs),
	})
	metrics.RecordCacheOperation("hybrid", "find_similar", "miss", time.Since(start).Seconds())

	return nil, false, nil
}

// FindSimilarWithThreshold searches for semantically similar cached requests using a specific threshold
func (h *HybridCache) FindSimilarWithThreshold(model string, query string, threshold float32) ([]byte, bool, error) {
	start := time.Now()

	if !h.enabled {
		return nil, false, nil
	}

	queryPreview := query
	if len(query) > 50 {
		queryPreview = query[:50] + "..."
	}
	logging.Debugf("HybridCache.FindSimilarWithThreshold: searching for model='%s', query='%s', threshold=%.3f",
		model, queryPreview, threshold)

	// Generate query embedding
	queryEmbedding, err := candle_binding.GetEmbedding(query, 0)
	if err != nil {
		metrics.RecordCacheOperation("hybrid", "find_similar_threshold", "error", time.Since(start).Seconds())
		return nil, false, fmt.Errorf("failed to generate embedding: %w", err)
	}

	// Search HNSW index for candidates above similarity threshold
	// For semantic cache, we only need the first match, so search with k=1
	// and stop early when finding a match above threshold
	h.mu.RLock()
	candidates := h.searchKNNHybridWithThreshold(queryEmbedding, 1, 20, threshold)
	h.mu.RUnlock()

	// Filter by similarity threshold before fetching from Milvus
	var qualifiedCandidates []searchResult
	for _, candidate := range candidates {
		if candidate.similarity >= threshold {
			qualifiedCandidates = append(qualifiedCandidates, candidate)
		}
	}

	// Map qualified candidates to Milvus IDs (need lock for idMap access)
	type candidateWithID struct {
		milvusID   string
		similarity float32
		index      int
	}

	h.mu.RLock()
	candidatesWithIDs := make([]candidateWithID, 0, len(qualifiedCandidates))
	for _, candidate := range qualifiedCandidates {
		if milvusID, ok := h.idMap[candidate.index]; ok {
			candidatesWithIDs = append(candidatesWithIDs, candidateWithID{
				milvusID:   milvusID,
				similarity: candidate.similarity,
				index:      candidate.index,
			})
		}
	}
	h.mu.RUnlock()

	if len(candidatesWithIDs) == 0 {
		atomic.AddInt64(&h.missCount, 1)
		if len(candidates) > 0 {
			logging.Debugf("HybridCache.FindSimilarWithThreshold: %d candidates found but none above threshold %.3f",
				len(candidates), threshold)
		} else {
			logging.Debugf("HybridCache.FindSimilarWithThreshold: no candidates found in HNSW")
		}
		metrics.RecordCacheOperation("hybrid", "find_similar_threshold", "miss", time.Since(start).Seconds())
		return nil, false, nil
	}

	logging.Debugf("HybridCache.FindSimilarWithThreshold: HNSW returned %d candidates, %d above threshold",
		len(candidates), len(candidatesWithIDs))

	// Fetch document from Milvus for qualified candidates
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// Try candidates in order (already sorted by similarity from HNSW)
	for _, candidate := range candidatesWithIDs {
		// Fetch document from Milvus by ID (direct lookup by primary key)
		fetchCtx, fetchCancel := context.WithTimeout(ctx, 2*time.Second)
		responseBody, err := h.milvusCache.GetByID(fetchCtx, candidate.milvusID)
		fetchCancel()

		if err != nil {
			logging.Debugf("HybridCache.FindSimilarWithThreshold: Milvus GetByID failed for %s: %v",
				candidate.milvusID, err)
			continue
		}

		if responseBody != nil {
			atomic.AddInt64(&h.hitCount, 1)
			logging.Debugf("HybridCache.FindSimilarWithThreshold: MILVUS HIT - similarity=%.4f (threshold=%.3f)",
				candidate.similarity, threshold)
			logging.LogEvent("hybrid_cache_hit", map[string]interface{}{
				"backend":    "hybrid",
				"source":     "milvus",
				"similarity": candidate.similarity,
				"threshold":  threshold,
				"model":      model,
				"latency_ms": time.Since(start).Milliseconds(),
			})
			metrics.RecordCacheOperation("hybrid", "find_similar_threshold", "hit_milvus", time.Since(start).Seconds())
			return responseBody, true, nil
		}
	}

	// No match found above threshold
	atomic.AddInt64(&h.missCount, 1)
	logging.Debugf("HybridCache.FindSimilarWithThreshold: CACHE MISS - no match above threshold")
	logging.LogEvent("hybrid_cache_miss", map[string]interface{}{
		"backend":    "hybrid",
		"threshold":  threshold,
		"model":      model,
		"candidates": len(candidatesWithIDs),
	})
	metrics.RecordCacheOperation("hybrid", "find_similar_threshold", "miss", time.Since(start).Seconds())

	return nil, false, nil
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

// searchResult holds a candidate with its similarity score
type searchResult struct {
	index      int
	similarity float32
}

// dotProduct calculates the dot product between two vectors
// Uses SIMD instructions (AVX2/AVX-512) when available for performance
// Falls back to scalar implementation on non-x86 platforms
func dotProduct(a, b []float32) float32 {
	return dotProductSIMD(a, b)
}

// hybridHNSWAdapter adapts the HNSW index to work with [][]float32 instead of []CacheEntry
type hybridHNSWAdapter struct {
	embeddings [][]float32
}

func (h *hybridHNSWAdapter) getEmbedding(idx int) []float32 {
	if idx < 0 || idx >= len(h.embeddings) {
		return nil
	}
	return h.embeddings[idx]
}

func (h *hybridHNSWAdapter) distance(idx1, idx2 int) float32 {
	emb1 := h.getEmbedding(idx1)
	emb2 := h.getEmbedding(idx2)
	if emb1 == nil || emb2 == nil {
		return 0
	}
	return dotProduct(emb1, emb2)
}

// addNodeHybrid adds a node to the HNSW index (hybrid version)
func (h *HybridCache) addNodeHybrid(entryIndex int, embedding []float32) {
	// Lock is already held by caller (mu.Lock())

	level := h.selectLevelHybrid()
	node := &HNSWNode{
		entryIndex: entryIndex,
		neighbors:  make(map[int][]int),
		maxLayer:   level,
	}

	for i := 0; i <= level; i++ {
		node.neighbors[i] = make([]int, 0)
	}

	h.hnswIndex.nodes = append(h.hnswIndex.nodes, node)
	h.hnswIndex.nodeIndex[entryIndex] = node // Add to O(1) lookup map

	if h.hnswIndex.entryPoint == -1 {
		h.hnswIndex.entryPoint = entryIndex
		h.hnswIndex.maxLayer = level
		return
	}

	// Find nearest neighbors at each layer
	adapter := &hybridHNSWAdapter{embeddings: h.embeddings}

	// Start from top layer
	currNearest := h.hnswIndex.entryPoint
	for lc := h.hnswIndex.maxLayer; lc > level; lc-- {
		// Search for nearest at this layer - Fast O(1) lookup
		candidates := []int{currNearest}
		if hn := h.hnswIndex.nodeIndex[currNearest]; hn != nil && hn.neighbors[lc] != nil {
			for _, neighbor := range hn.neighbors[lc] {
				if neighbor >= 0 && neighbor < len(h.embeddings) {
					candidates = append(candidates, neighbor)
				}
			}
		}

		// Find closest
		bestDist := adapter.distance(entryIndex, currNearest)
		for _, candidate := range candidates {
			dist := adapter.distance(entryIndex, candidate)
			if dist > bestDist {
				bestDist = dist
				currNearest = candidate
			}
		}
	}

	// Insert at appropriate layers
	for lc := level; lc >= 0; lc-- {
		// Find neighbors at this layer
		neighbors := h.searchLayerHybrid(embedding, h.hnswIndex.efConstruction, lc, []int{currNearest})

		m := h.hnswIndex.M
		if lc == 0 {
			m = h.hnswIndex.Mmax0
		}

		selectedNeighbors := h.selectNeighborsHybrid(neighbors, m)

		// Add bidirectional links
		for _, neighborID := range selectedNeighbors {
			node.neighbors[lc] = append(node.neighbors[lc], neighborID)

			// Add reverse link - Fast O(1) lookup
			if neighborNode := h.hnswIndex.nodeIndex[neighborID]; neighborNode != nil {
				if neighborNode.neighbors[lc] == nil {
					neighborNode.neighbors[lc] = make([]int, 0)
				}
				neighborNode.neighbors[lc] = append(neighborNode.neighbors[lc], entryIndex)
			}
		}
	}

	if level > h.hnswIndex.maxLayer {
		h.hnswIndex.maxLayer = level
		h.hnswIndex.entryPoint = entryIndex
	}
}

// selectLevelHybrid randomly selects a level for a new node
func (h *HybridCache) selectLevelHybrid() int {
	// Use exponential decay to select level
	// Most nodes at layer 0, fewer at higher layers
	level := 0
	for level < maxHNSWLayers {
		if randFloat() > h.hnswIndex.ml {
			break
		}
		level++
	}
	return level
}

// randFloat returns a random float between 0 and 1
func randFloat() float64 {
	// Simple random using time-based seed
	return float64(time.Now().UnixNano()%1000) / 1000.0
}

// searchLayerHybrid searches for nearest neighbors at a specific layer
func (h *HybridCache) searchLayerHybrid(query []float32, ef int, layer int, entryPoints []int) []int {
	// Reuse buffers from pool to reduce allocations
	buf := getSearchBuffers()
	defer putSearchBuffers(buf)

	visited := buf.visited
	candidates := buf.candidates
	results := buf.results

	for _, ep := range entryPoints {
		if ep < 0 || ep >= len(h.embeddings) {
			continue
		}
		dist := -dotProduct(query, h.embeddings[ep]) // Negative product so that higher similarity = lower distance
		candidates.push(ep, dist)
		results.push(ep, dist)
		visited[ep] = true
	}

	for len(candidates.data) > 0 {
		currentIdx, currentDist := candidates.pop()
		if len(results.data) > 0 && currentDist > results.data[0].dist {
			break
		}

		// Fast O(1) lookup using nodeIndex map
		currentNode := h.hnswIndex.nodeIndex[currentIdx]
		if currentNode == nil || currentNode.neighbors[layer] == nil {
			continue
		}

		for _, neighborID := range currentNode.neighbors[layer] {
			if visited[neighborID] || neighborID < 0 || neighborID >= len(h.embeddings) {
				continue
			}
			visited[neighborID] = true

			dist := -dotProduct(query, h.embeddings[neighborID])

			if len(results.data) < ef || dist < results.data[0].dist {
				candidates.push(neighborID, dist)
				results.push(neighborID, dist)

				if len(results.data) > ef {
					results.pop()
				}
			}
		}
	}

	// Extract IDs from heap and reverse to get correct order
	resultIDs := make([]int, 0, len(results.data))
	for len(results.data) > 0 {
		idx, _ := results.pop()
		resultIDs = append(resultIDs, idx)
	}

	// Reverse in place to match similarity order
	for i, j := 0, len(resultIDs)-1; i < j; i, j = i+1, j-1 {
		resultIDs[i], resultIDs[j] = resultIDs[j], resultIDs[i]
	}

	return resultIDs
}

// selectNeighborsHybrid selects the best neighbors from candidates (hybrid version)
func (h *HybridCache) selectNeighborsHybrid(candidates []int, m int) []int {
	if len(candidates) <= m {
		return candidates
	}

	// Simple selection: take first M candidates
	return candidates[:m]
}

// searchKNNHybridWithThreshold searches for k nearest neighbors with early stopping
// Stops immediately when finding a match above the similarity threshold
// This is optimal for semantic cache where we only need the first good match
func (h *HybridCache) searchKNNHybridWithThreshold(query []float32, k int, ef int, threshold float32) []searchResult {
	// Lock is already held by caller (mu.RLock())

	if h.hnswIndex.entryPoint == -1 || len(h.embeddings) == 0 {
		return nil
	}

	// Search from top layer down to layer 1 for navigation
	currNearest := []int{h.hnswIndex.entryPoint}

	for lc := h.hnswIndex.maxLayer; lc > 0; lc-- {
		currNearest = h.searchLayerHybrid(query, 1, lc, currNearest)
	}

	// Search at layer 0 with early stopping at threshold
	candidateIndices := h.searchLayerHybridWithEarlyStop(query, ef, 0, currNearest, threshold)

	// Convert to searchResults with similarity scores
	results := make([]searchResult, 0, len(candidateIndices))
	for _, idx := range candidateIndices {
		if idx >= 0 && idx < len(h.embeddings) {
			similarity := dotProductSIMD(query, h.embeddings[idx])

			// Return immediately if we found a match above threshold
			if similarity >= threshold {
				results = append(results, searchResult{
					index:      idx,
					similarity: similarity,
				})
				return results
			}

			results = append(results, searchResult{
				index:      idx,
				similarity: similarity,
			})
		}
	}

	// Return top k (or fewer if early stopped)
	if len(results) > k {
		return results[:k]
	}
	return results
}

// searchLayerHybridWithEarlyStop searches a layer and stops when finding a match above threshold
func (h *HybridCache) searchLayerHybridWithEarlyStop(query []float32, ef int, layer int, entryPoints []int, threshold float32) []int {
	buf := getSearchBuffers()
	defer putSearchBuffers(buf)

	visited := buf.visited
	candidates := buf.candidates
	results := buf.results

	for _, ep := range entryPoints {
		if ep < 0 || ep >= len(h.embeddings) {
			continue
		}
		dist := -dotProductSIMD(query, h.embeddings[ep]) // Negative product so that higher similarity = lower distance
		candidates.push(ep, dist)
		results.push(ep, dist)
		visited[ep] = true

		// Check if this entry point already meets the threshold
		if -dist >= threshold {
			return []int{ep}
		}
	}

	for len(candidates.data) > 0 {
		currentIdx, currentDist := candidates.pop()
		if len(results.data) > 0 && currentDist > results.data[0].dist {
			break
		}

		currentNode := h.hnswIndex.nodeIndex[currentIdx]
		if currentNode == nil || currentNode.neighbors[layer] == nil {
			continue
		}

		for _, neighborID := range currentNode.neighbors[layer] {
			if visited[neighborID] || neighborID < 0 || neighborID >= len(h.embeddings) {
				continue
			}
			visited[neighborID] = true

			similarity := dotProductSIMD(query, h.embeddings[neighborID])
			dist := -similarity

			// Stop if this neighbor meets the threshold
			if similarity >= threshold {
				return []int{neighborID}
			}

			if len(results.data) < ef || dist < results.data[0].dist {
				candidates.push(neighborID, dist)
				results.push(neighborID, dist)

				if len(results.data) > ef {
					results.pop()
				}
			}
		}
	}

	// Extract IDs (sorted by similarity)
	resultIDs := make([]int, 0, len(results.data))
	for len(results.data) > 0 {
		idx, _ := results.pop()
		resultIDs = append(resultIDs, idx)
	}

	// Reverse in place
	for i, j := 0, len(resultIDs)-1; i < j; i, j = i+1, j-1 {
		resultIDs[i], resultIDs[j] = resultIDs[j], resultIDs[i]
	}

	return resultIDs
}
