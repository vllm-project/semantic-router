package tokenaccounting

import (
	"container/list"
	"hash/maphash"
	"sync"
)

const (
	// DefaultCacheEntries bounds the per-text count cache.
	DefaultCacheEntries = 4096
	// cacheMinBytes keeps short texts out of the cache: counting them costs
	// less than a lookup. Long agent histories are resent every turn, and
	// their unchanged messages and tool results are what the cache saves.
	cacheMinBytes = 256
)

type cacheKey struct {
	family string
	hash   uint64
	bytes  int
}

type cacheEntry struct {
	key    cacheKey
	tokens int
}

// countCache is a bounded LRU of exact text counts. A key is the family, a
// 64-bit seeded hash and the byte length of the text, so an entry never holds
// request content. Two different texts of one length colliding in the hash
// would share a count; at 2^-64 per pair that is not a correctness concern
// for an accounting estimate's cache.
type countCache struct {
	mu      sync.Mutex
	limit   int
	seed    maphash.Seed
	order   *list.List
	entries map[cacheKey]*list.Element
	hits    uint64
	misses  uint64
}

func newCountCache(limit int) *countCache {
	if limit <= 0 {
		return nil
	}
	return &countCache{
		limit:   limit,
		seed:    maphash.MakeSeed(),
		order:   list.New(),
		entries: map[cacheKey]*list.Element{},
	}
}

func (cache *countCache) count(family *Family, text string) int {
	if cache == nil || len(text) < cacheMinBytes {
		return family.Tokenizer.CountTokens(text)
	}
	key := cacheKey{family: family.Name, hash: maphash.String(cache.seed, text), bytes: len(text)}
	cache.mu.Lock()
	if element, ok := cache.entries[key]; ok {
		cache.order.MoveToFront(element)
		cache.hits++
		tokens := element.Value.(*cacheEntry).tokens
		cache.mu.Unlock()
		return tokens
	}
	cache.misses++
	cache.mu.Unlock()

	// Count outside the lock: tokenization is the slow part, and two
	// requests counting the same text only repeat work.
	tokens := family.Tokenizer.CountTokens(text)

	cache.mu.Lock()
	defer cache.mu.Unlock()
	if element, ok := cache.entries[key]; ok {
		cache.order.MoveToFront(element)
		return tokens
	}
	cache.entries[key] = cache.order.PushFront(&cacheEntry{key: key, tokens: tokens})
	for cache.order.Len() > cache.limit {
		oldest := cache.order.Back()
		cache.order.Remove(oldest)
		delete(cache.entries, oldest.Value.(*cacheEntry).key)
	}
	return tokens
}

// CacheStats reports the count cache's size and use.
type CacheStats struct {
	Entries int
	Hits    uint64
	Misses  uint64
}

func (cache *countCache) stats() CacheStats {
	if cache == nil {
		return CacheStats{}
	}
	cache.mu.Lock()
	defer cache.mu.Unlock()
	return CacheStats{Entries: cache.order.Len(), Hits: cache.hits, Misses: cache.misses}
}
