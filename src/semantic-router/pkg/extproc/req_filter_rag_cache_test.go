package extproc

import (
	"container/list"
	"sync"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// newTestRAGCache returns a fresh RAGResultCache bypassing the singleton, so
// each test starts with an empty, isolated cache.
func newTestRAGCache(maxSize int) *RAGResultCache {
	return &RAGResultCache{
		cache:   make(map[string]*list.Element, maxSize),
		lru:     list.New(),
		maxSize: maxSize,
	}
}

// resetRAGSingleton replaces the package-level singleton with a fresh cache of
// the given maxSize and resets the Once so getRAGCacheInstance returns it.
// Must only be called from tests; callers are responsible for restoring state.
func resetRAGSingleton(maxSize int) *RAGResultCache {
	fresh := newTestRAGCache(maxSize)
	ragCache = fresh
	ragCacheOnce = sync.Once{}
	ragCacheOnce.Do(func() { ragCache = fresh })
	return fresh
}

// setDirect inserts an entry into a standalone cache, bypassing the singleton.
// It mirrors the production logic in setRAGCache exactly.
func setDirect(r *OpenAIRouter, cache *RAGResultCache, key string, ctx string) {
	entry := &RAGCacheEntry{Context: ctx, RetrievedAt: time.Now()}
	cache.mu.Lock()
	defer cache.mu.Unlock()
	if el, exists := cache.cache[key]; exists {
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

// getDirect reads from a standalone cache, bypassing the singleton.
// It mirrors the production logic in getRAGCache exactly.
func getDirect(cache *RAGResultCache, key string, ttl int) (string, bool) {
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

// TestRAGCache_EvictionOrder_WriteLRU verifies the write/update-recency eviction
// contract (comment #2 on PR #2040): a key that is only read (never re-written)
// stays at the LRU front and is evicted before a key written more recently, even
// if the read-only key was accessed far more frequently.
func TestRAGCache_EvictionOrder_WriteLRU(t *testing.T) {
	r := &OpenAIRouter{}
	cache := newTestRAGCache(2)

	setDirect(r, cache, "keyA", "ctxA") // list: [A]      — A is LRU front
	setDirect(r, cache, "keyB", "ctxB") // list: [A, B]   — A still LRU front

	// Read keyA many times. Under write/update-recency eviction, reads must NOT
	// promote keyA toward MRU — it must remain at the LRU front.
	for i := 0; i < 5; i++ {
		v, ok := getDirect(cache, "keyA", 3600)
		if !ok || v != "ctxA" {
			t.Fatalf("expected cache hit for keyA on read %d, got ok=%v v=%q", i, ok, v)
		}
	}

	// Insert keyC: cache is full so the LRU entry (keyA, written first, never
	// re-written) must be evicted — not keyB which was written after.
	setDirect(r, cache, "keyC", "ctxC")

	if _, ok := getDirect(cache, "keyA", 3600); ok {
		t.Error("keyA should have been evicted (it is the LRU write), but it is still in cache")
	}
	if v, ok := getDirect(cache, "keyB", 3600); !ok || v != "ctxB" {
		t.Errorf("keyB should still be in cache after eviction, got ok=%v v=%q", ok, v)
	}
	if v, ok := getDirect(cache, "keyC", 3600); !ok || v != "ctxC" {
		t.Errorf("keyC should be in cache after insertion, got ok=%v v=%q", ok, v)
	}
}

// TestRAGCache_TTLExpiry_MapAndListConsistency exercises the production
// getRAGCache path (comment #3 on PR #2040) and asserts that after TTL expiry
// both cache.cache and cache.lru are cleaned up atomically, leaving no orphans
// in either structure.
func TestRAGCache_TTLExpiry_MapAndListConsistency(t *testing.T) {
	r := &OpenAIRouter{}
	ttlSec := 1
	cfg := &config.RAGPluginConfig{
		Backend:         "ttltest",
		CacheResults:    true,
		CacheTTLSeconds: &ttlSec,
	}

	// Redirect the singleton to an isolated cache for this test.
	cache := resetRAGSingleton(10)

	// Inject an entry directly with a RetrievedAt already 2 hours in the past,
	// so it is expired relative to the 1-second TTL.
	key := r.buildRAGCacheKey("expired-query", cfg)
	expiredEntry := &RAGCacheEntry{
		Context:     "stale-context",
		RetrievedAt: time.Now().Add(-2 * time.Hour),
	}
	cache.mu.Lock()
	el := cache.lru.PushBack(&ragLRUItem{key: key, entry: expiredEntry})
	cache.cache[key] = el
	cache.mu.Unlock()

	// Call the real production function — it must detect expiry, remove the
	// entry from both cache.cache and cache.lru, and return a miss.
	ctx, ok := r.getRAGCache("expired-query", cfg)
	if ok || ctx != "" {
		t.Fatalf("expected miss for expired entry, got ok=%v ctx=%q", ok, ctx)
	}

	// Both structures must be empty — no orphan in either map or list.
	cache.mu.RLock()
	mapLen := len(cache.cache)
	listLen := cache.lru.Len()
	cache.mu.RUnlock()

	if mapLen != 0 {
		t.Errorf("cache.cache must be empty after TTL expiry, got len=%d", mapLen)
	}
	if listLen != 0 {
		t.Errorf("cache.lru must be empty after TTL expiry, got len=%d", listLen)
	}
}

// TestRAGCache_UpdateExistingKey_SizeAndMRU verifies that re-setting an existing
// key via setRAGCache does not grow the cache and moves the entry to MRU.
// This covers the update path (comment #2 on PR #2040).
func TestRAGCache_UpdateExistingKey_SizeAndMRU(t *testing.T) {
	r := &OpenAIRouter{}
	cache := newTestRAGCache(10)

	setDirect(r, cache, "keyA", "v1")
	setDirect(r, cache, "keyB", "v2")
	// Re-set keyA: must update value, keep size at 2, and move keyA to MRU.
	setDirect(r, cache, "keyA", "v1-updated")

	cache.mu.RLock()
	mapLen := len(cache.cache)
	mru := cache.lru.Back().Value.(*ragLRUItem).key
	lru := cache.lru.Front().Value.(*ragLRUItem).key
	cache.mu.RUnlock()

	if mapLen != 2 {
		t.Errorf("cache size must stay 2 after update, got %d", mapLen)
	}
	if mru != "keyA" {
		t.Errorf("keyA must be MRU (back of list) after re-set, got %q", mru)
	}
	if lru != "keyB" {
		t.Errorf("keyB must be LRU (front of list) after keyA re-set, got %q", lru)
	}

	v, ok := getDirect(cache, "keyA", 3600)
	if !ok || v != "v1-updated" {
		t.Errorf("expected updated value for keyA, got ok=%v v=%q", ok, v)
	}
}

// TestRAGCache_ConcurrentReadsDontDeadlock ensures that 50 concurrent reads
// under RLock do not block or deadlock each other — validating that reads never
// acquire a write lock.
func TestRAGCache_ConcurrentReadsDontDeadlock(t *testing.T) {
	r := &OpenAIRouter{}
	cache := newTestRAGCache(10)
	setDirect(r, cache, "key", "ctx")

	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			getDirect(cache, "key", 3600)
		}()
	}
	wg.Wait()
}

// TestGetRAGCache_CacheDisabled verifies that getRAGCache returns a miss
// immediately when CacheResults is false, without touching the singleton.
func TestGetRAGCache_CacheDisabled(t *testing.T) {
	r := &OpenAIRouter{}
	cfg := &config.RAGPluginConfig{CacheResults: false}
	ctx, ok := r.getRAGCache("query", cfg)
	if ok || ctx != "" {
		t.Errorf("expected miss when CacheResults=false, got ok=%v ctx=%q", ok, ctx)
	}
}
