package extproc

import (
	"crypto/sha256"
	"fmt"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// RAGCacheEntry represents a cached RAG retrieval result
type RAGCacheEntry struct {
	Context          string
	RetrievedAt      time.Time
	SimilarityScores []float32
}

// RAGResultCache provides in-memory caching for RAG retrieval results
type RAGResultCache struct {
	cache       map[string]*RAGCacheEntry
	accessOrder []string // Tracks insertion/access order for efficient LRU eviction
	mu          sync.RWMutex
	maxSize     int
}

var (
	ragCache     *RAGResultCache
	ragCacheOnce sync.Once
)

// getRAGCacheInstance returns the singleton RAG cache instance
func getRAGCacheInstance() *RAGResultCache {
	ragCacheOnce.Do(func() {
		ragCache = &RAGResultCache{
			cache:       make(map[string]*RAGCacheEntry),
			accessOrder: make([]string, 0, 10000), // Pre-allocate with capacity
			maxSize:     10000,                    // Maximum cache entries
		}
	})
	return ragCache
}

// getRAGCache retrieves cached RAG result if available
func (r *OpenAIRouter) getRAGCache(query string, ragConfig *config.RAGPluginConfig) (string, bool) {
	if !ragConfig.CacheResults {
		return "", false
	}

	cache := getRAGCacheInstance()
	key := r.buildRAGCacheKey(query, ragConfig)

	cache.mu.RLock()
	entry, exists := cache.cache[key]
	if !exists {
		cache.mu.RUnlock()
		return "", false
	}

	// Check TTL
	ttl := 3600 // Default 1 hour
	if ragConfig.CacheTTLSeconds != nil {
		ttl = *ragConfig.CacheTTLSeconds
	}

	expired := ttl > 0 && time.Since(entry.RetrievedAt) > time.Duration(ttl)*time.Second
	context := entry.Context
	cache.mu.RUnlock()

	if expired {
		// Expired, remove from cache (need write lock for deletion)
		cache.mu.Lock()
		// Double-check after acquiring write lock
		if entry, stillExists := cache.cache[key]; stillExists {
			if time.Since(entry.RetrievedAt) > time.Duration(ttl)*time.Second {
				delete(cache.cache, key)
				// Remove from access order slice
				r.removeFromAccessOrder(cache, key)
			}
		}
		cache.mu.Unlock()
		return "", false
	}

	// Update access order for LRU (move to end)
	// Note: This requires a write lock, but we can do it optimistically
	// For better performance, we could skip this and only update on eviction
	// For now, we'll update it to maintain proper LRU behavior
	cache.mu.Lock()
	r.moveToEnd(cache, key)
	cache.mu.Unlock()

	return context, true
}

// setRAGCache stores RAG result in cache
func (r *OpenAIRouter) setRAGCache(query string, context string, ragConfig *config.RAGPluginConfig) {
	if !ragConfig.CacheResults || context == "" {
		return
	}

	cache := getRAGCacheInstance()
	key := r.buildRAGCacheKey(query, ragConfig)

	cache.mu.Lock()
	defer cache.mu.Unlock()

	// Evict if cache is full (LRU: remove oldest)
	if len(cache.cache) >= cache.maxSize {
		r.evictOldestRAGCacheEntry(cache)
	}

	entry := &RAGCacheEntry{
		Context:     context,
		RetrievedAt: time.Now(),
	}

	// Add to cache and update access order
	_, isUpdate := cache.cache[key]
	cache.cache[key] = entry
	if !isUpdate {
		// New entry, add to end of access order
		cache.accessOrder = append(cache.accessOrder, key)
	} else {
		// Update existing entry, move to end (most recently used)
		r.moveToEnd(cache, key)
	}
}

// buildRAGCacheKey builds a cache key from query and config
func (r *OpenAIRouter) buildRAGCacheKey(query string, ragConfig *config.RAGPluginConfig) string {
	// Include backend, topK, and threshold in key for cache differentiation
	topK := 5
	if ragConfig.TopK != nil {
		topK = *ragConfig.TopK
	}
	threshold := "0.7"
	if ragConfig.SimilarityThreshold != nil {
		threshold = fmt.Sprintf("%.3f", *ragConfig.SimilarityThreshold)
	}

	// Use SHA-256 for cache key hashing (non-cryptographic use, but more modern than MD5)
	keyStr := fmt.Sprintf("%s:%s:%d:%s", ragConfig.Backend, query, topK, threshold)
	hash := sha256.Sum256([]byte(keyStr))
	return fmt.Sprintf("%x", hash)
}

// evictOldestRAGCacheEntry removes the oldest cache entry using LRU strategy
// Uses accessOrder slice for O(1) eviction of the least recently used entry
func (r *OpenAIRouter) evictOldestRAGCacheEntry(cache *RAGResultCache) {
	if len(cache.accessOrder) == 0 {
		return
	}

	// Remove the first (oldest) entry from access order
	oldestKey := cache.accessOrder[0]
	cache.accessOrder = cache.accessOrder[1:]

	// Remove from cache map
	delete(cache.cache, oldestKey)
	logging.Debugf("Evicted oldest RAG cache entry: %s", oldestKey)
}

// removeFromAccessOrder removes a key from the access order slice
// This is used when entries expire and are removed
func (r *OpenAIRouter) removeFromAccessOrder(cache *RAGResultCache, key string) {
	for i, k := range cache.accessOrder {
		if k == key {
			// Remove element by shifting slice
			cache.accessOrder = append(cache.accessOrder[:i], cache.accessOrder[i+1:]...)
			return
		}
	}
}

// moveToEnd moves a key to the end of the access order slice (marking it as most recently used)
func (r *OpenAIRouter) moveToEnd(cache *RAGResultCache, key string) {
	// Find and remove the key from its current position
	for i, k := range cache.accessOrder {
		if k == key {
			// Remove from current position
			cache.accessOrder = append(cache.accessOrder[:i], cache.accessOrder[i+1:]...)
			// Add to end (most recently used)
			cache.accessOrder = append(cache.accessOrder, key)
			return
		}
	}
	// Key not found in access order, add it (shouldn't happen, but handle gracefully)
	cache.accessOrder = append(cache.accessOrder, key)
}
