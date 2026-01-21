package extproc

import (
	"crypto/md5"
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
	cache   map[string]*RAGCacheEntry
	mu      sync.RWMutex
	maxSize int
}

var (
	ragCache     *RAGResultCache
	ragCacheOnce sync.Once
)

// getRAGCacheInstance returns the singleton RAG cache instance
func getRAGCacheInstance() *RAGResultCache {
	ragCacheOnce.Do(func() {
		ragCache = &RAGResultCache{
			cache:   make(map[string]*RAGCacheEntry),
			maxSize: 10000, // Maximum cache entries
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
	defer cache.mu.RUnlock()

	entry, exists := cache.cache[key]
	if !exists {
		return "", false
	}

	// Check TTL
	ttl := 3600 // Default 1 hour
	if ragConfig.CacheTTLSeconds != nil {
		ttl = *ragConfig.CacheTTLSeconds
	}

	if ttl > 0 && time.Since(entry.RetrievedAt) > time.Duration(ttl)*time.Second {
		// Expired, remove from cache
		delete(cache.cache, key)
		return "", false
	}

	return entry.Context, true
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

	// Evict if cache is full (simple LRU: remove oldest)
	if len(cache.cache) >= cache.maxSize {
		r.evictOldestRAGCacheEntry(cache)
	}

	entry := &RAGCacheEntry{
		Context:     context,
		RetrievedAt: time.Now(),
	}

	cache.cache[key] = entry
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

	keyStr := fmt.Sprintf("%s:%s:%d:%s", ragConfig.Backend, query, topK, threshold)
	hash := md5.Sum([]byte(keyStr))
	return fmt.Sprintf("%x", hash)
}

// evictOldestRAGCacheEntry removes the oldest cache entry
func (r *OpenAIRouter) evictOldestRAGCacheEntry(cache *RAGResultCache) {
	var oldestKey string
	var oldestTime time.Time
	first := true

	for key, entry := range cache.cache {
		if first || entry.RetrievedAt.Before(oldestTime) {
			oldestKey = key
			oldestTime = entry.RetrievedAt
			first = false
		}
	}

	if oldestKey != "" {
		delete(cache.cache, oldestKey)
		logging.Debugf("Evicted oldest RAG cache entry: %s", oldestKey)
	}
}
