package extproc

import (
	"container/list"
	"crypto/sha256"
	"encoding/json"
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

// ragLRUItem is the value stored in the LRU list; it carries the map key so
// eviction can remove the map entry in O(1) without a reverse lookup.
type ragLRUItem struct {
	key   string
	entry *RAGCacheEntry
}

// RAGResultCache provides in-memory caching for RAG retrieval results.
// Uses a doubly-linked list + map for O(1) get, set, and eviction.
// Eviction order is write/update-recency: only setRAGCache (new insert or
// re-set of an existing key) moves an entry to MRU. Reads do not promote
// entries, avoiding a write-lock on every cache hit. Reads use RLock; only
// writes and evictions use Lock.
type RAGResultCache struct {
	cache   map[string]*list.Element // key → list element
	lru     *list.List               // front = LRU, back = MRU
	mu      sync.RWMutex
	maxSize int
}

var (
	ragCache     *RAGResultCache
	ragCacheOnce sync.Once
)

func getRAGCacheInstance() *RAGResultCache {
	ragCacheOnce.Do(func() {
		ragCache = &RAGResultCache{
			cache:   make(map[string]*list.Element, 10000),
			lru:     list.New(),
			maxSize: 10000,
		}
	})
	return ragCache
}

// getRAGCache retrieves a cached RAG result if available and not expired.
// Cache hits use RLock only; the LRU position is updated lazily on the next
// write to avoid promoting a write lock on every read.
func (r *OpenAIRouter) getRAGCache(recipe config.RecipeName, query string, ragConfig *config.RAGPluginConfig) (string, bool) {
	if !ragConfig.CacheResults {
		return "", false
	}

	ttl := 3600
	if ragConfig.CacheTTLSeconds != nil {
		ttl = *ragConfig.CacheTTLSeconds
	}

	cache := getRAGCacheInstance()
	key := r.buildRAGCacheKey(recipe, query, ragConfig)
	if key == "" {
		return "", false
	}

	cache.mu.RLock()
	el, exists := cache.cache[key]
	if !exists {
		cache.mu.RUnlock()
		return "", false
	}
	item := el.Value.(*ragLRUItem)
	expired := ttl > 0 && time.Since(item.entry.RetrievedAt) > time.Duration(ttl)*time.Second
	ctx := item.entry.Context
	cache.mu.RUnlock()

	if expired {
		cache.mu.Lock()
		// Re-check under write lock to avoid double-delete.
		if el2, ok := cache.cache[key]; ok {
			it := el2.Value.(*ragLRUItem)
			if ttl > 0 && time.Since(it.entry.RetrievedAt) > time.Duration(ttl)*time.Second {
				cache.lru.Remove(el2)
				delete(cache.cache, key)
			}
		}
		cache.mu.Unlock()
		return "", false
	}

	return ctx, true
}

// setRAGCache stores a RAG result. Evicts the LRU entry when the cache is full.
func (r *OpenAIRouter) setRAGCache(recipe config.RecipeName, query string, context string, ragConfig *config.RAGPluginConfig) {
	if !ragConfig.CacheResults || context == "" {
		return
	}

	cache := getRAGCacheInstance()
	key := r.buildRAGCacheKey(recipe, query, ragConfig)
	if key == "" {
		return
	}

	entry := &RAGCacheEntry{
		Context:     context,
		RetrievedAt: time.Now(),
	}

	cache.mu.Lock()
	defer cache.mu.Unlock()

	if el, exists := cache.cache[key]; exists {
		// Update existing entry and mark as MRU.
		el.Value.(*ragLRUItem).entry = entry
		cache.lru.MoveToBack(el)
		return
	}

	if len(cache.cache) >= cache.maxSize {
		r.evictLRUEntry(cache)
	}

	el := cache.lru.PushBack(&ragLRUItem{key: key, entry: entry})
	cache.cache[key] = el
}

// evictLRUEntry removes the least-recently-used entry. Must be called with Lock held.
func (r *OpenAIRouter) evictLRUEntry(cache *RAGResultCache) {
	front := cache.lru.Front()
	if front == nil {
		return
	}
	item := cache.lru.Remove(front).(*ragLRUItem)
	delete(cache.cache, item.key)
	logging.Debugf("Evicted LRU RAG cache entry: %s", item.key)
}

// buildRAGCacheKey builds a cache key from query and config.
func (r *OpenAIRouter) buildRAGCacheKey(recipe config.RecipeName, query string, ragConfig *config.RAGPluginConfig) string {
	identity, compatible := r.ragCacheRepresentation(ragConfig)
	if !compatible {
		return ""
	}
	rerankerIdentity := ""
	if ragConfig.Rerank != nil {
		scorer := r.rerankers[recipe]
		if scorer == nil || scorer.CacheIdentity() == "" {
			return ""
		}
		rerankerIdentity = scorer.CacheIdentity()
	}
	// Include backend configuration: different vector-store IDs, file filters,
	// and endpoints must never share retrieved text just because queries match.
	key, err := json.Marshal(struct {
		Recipe            config.RecipeName       `json:"recipe"`
		RerankerIdentity  string                  `json:"reranker_identity,omitempty"`
		Query             string                  `json:"query"`
		Config            *config.RAGPluginConfig `json:"config"`
		EmbeddingIdentity string                  `json:"embedding_identity"`
	}{recipe, rerankerIdentity, query, ragConfig, identity})
	if err != nil {
		return ""
	}
	hash := sha256.Sum256(key)
	return fmt.Sprintf("%x", hash)
}

// A cached result cannot bypass model compatibility checks. An incompatible
// hybrid child disables caching, while normal retrieval can still use its
// configured fallback backend.
func (r *OpenAIRouter) ragCacheRepresentation(cfg *config.RAGPluginConfig) (string, bool) {
	switch cfg.Backend {
	case "vectorstore":
		store, err := cfg.VectorStoreBackendConfig()
		manager := r.currentVectorStoreManager()
		if err != nil || store == nil || manager == nil || manager.CheckEmbeddingCompatibility(store.VectorStoreID) != nil {
			return "", false
		}
		return manager.EmbeddingIdentity(), true
	case "hybrid":
		hybrid, err := cfg.HybridBackendConfig()
		if err != nil || hybrid == nil {
			return "", false
		}
		primary, ok := r.ragCacheRepresentation(&config.RAGPluginConfig{Backend: hybrid.Primary, BackendConfig: hybrid.PrimaryConfig})
		if !ok {
			return "", false
		}
		fallback, ok := r.ragCacheRepresentation(&config.RAGPluginConfig{Backend: hybrid.Fallback, BackendConfig: hybrid.FallbackConfig})
		return primary + ":" + fallback, ok
	default:
		return "", true
	}
}
