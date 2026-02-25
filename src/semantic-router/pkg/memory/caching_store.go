package memory

import (
	"context"
	"time"
)

const cacheBackendLabel = "milvus" // backend label for cache metrics (cache sits in front of Milvus)

// CachingStore wraps a Store and adds a Redis hot cache for Retrieve.
// Retrieve: check cache first; on miss, call underlying store and populate cache.
// Store/Update/Forget/ForgetByScope: call underlying then invalidate cache for affected user(s).
type CachingStore struct {
	store Store
	cache *RedisCache
}

// NewCachingStore returns a Store that caches retrieval results in Redis.
// If cache is nil, all operations delegate to store with no caching.
func NewCachingStore(store Store, cache *RedisCache) Store {
	if cache == nil {
		return store
	}
	return &CachingStore{store: store, cache: cache}
}

// Retrieve implements Store. It checks the cache first; on miss, calls the underlying store and caches the result.
func (c *CachingStore) Retrieve(ctx context.Context, opts RetrieveOptions) ([]*RetrieveResult, error) {
	if c.cache != nil {
		start := time.Now()
		results, ok := c.cache.Get(ctx, opts)
		elapsed := time.Since(start).Seconds()
		if ok {
			RecordMemoryCacheHit(cacheBackendLabel, elapsed)
			return results, nil
		}
		RecordMemoryCacheMiss(cacheBackendLabel)
	}
	results, err := c.store.Retrieve(ctx, opts)
	if err != nil {
		return nil, err
	}
	if c.cache != nil {
		c.cache.Set(ctx, opts, results)
	}
	return results, nil
}

// Store implements Store; delegates then invalidates cache for the memory's user.
func (c *CachingStore) Store(ctx context.Context, memory *Memory) error {
	err := c.store.Store(ctx, memory)
	if err == nil && c.cache != nil && memory != nil && memory.UserID != "" {
		c.cache.InvalidateByUser(ctx, memory.UserID)
	}
	return err
}

// Get implements Store.
func (c *CachingStore) Get(ctx context.Context, id string) (*Memory, error) {
	return c.store.Get(ctx, id)
}

// Update implements Store; delegates then invalidates cache for the memory's user.
func (c *CachingStore) Update(ctx context.Context, id string, memory *Memory) error {
	err := c.store.Update(ctx, id, memory)
	if err == nil && c.cache != nil && memory != nil && memory.UserID != "" {
		c.cache.InvalidateByUser(ctx, memory.UserID)
	}
	return err
}

// List implements Store.
func (c *CachingStore) List(ctx context.Context, opts ListOptions) (*ListResult, error) {
	return c.store.List(ctx, opts)
}

// Forget implements Store; delegates then invalidates cache. We don't have userID from id alone, so invalidate is best-effort:
// we could skip invalidation here and rely on TTL, or we could Get(id) to get userID then invalidate. For simplicity we don't
// invalidate on Forget(id) unless we fetch the memory (extra round-trip). Alternatively we could invalidate all users for this
// cache - too broad. So we only invalidate on Store/Update/ForgetByScope where we have userID. On Forget(id) we skip cache invalidation
// (stale cache until TTL or next write for that user). To do it right we'd need to Get the memory to get UserID.
func (c *CachingStore) Forget(ctx context.Context, id string) error {
	if c.cache != nil {
		mem, err := c.store.Get(ctx, id)
		if err == nil && mem != nil && mem.UserID != "" {
			defer func() { c.cache.InvalidateByUser(ctx, mem.UserID) }()
		}
	}
	return c.store.Forget(ctx, id)
}

// ForgetByScope implements Store; delegates then invalidates cache for the scope's user.
func (c *CachingStore) ForgetByScope(ctx context.Context, scope MemoryScope) error {
	err := c.store.ForgetByScope(ctx, scope)
	if err == nil && c.cache != nil && scope.UserID != "" {
		c.cache.InvalidateByUser(ctx, scope.UserID)
	}
	return err
}

// IsEnabled implements Store.
func (c *CachingStore) IsEnabled() bool {
	return c.store.IsEnabled()
}

// CheckConnection implements Store.
func (c *CachingStore) CheckConnection(ctx context.Context) error {
	return c.store.CheckConnection(ctx)
}

// Close implements Store; closes the cache client if present.
func (c *CachingStore) Close() error {
	if c.cache != nil {
		_ = c.cache.Close()
	}
	return c.store.Close()
}

// ensure CachingStore implements Store
var _ Store = (*CachingStore)(nil)
