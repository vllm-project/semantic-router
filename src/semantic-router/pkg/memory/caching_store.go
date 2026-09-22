package memory

import (
	"context"
	"time"
)

const defaultCacheBackendLabel = "milvus" // fallback label when none is provided

// cacheInvalidateTimeout bounds one post-write invalidation. Invalidation
// ignores caller cancellation by design, so a retiring runner can overrun its
// drain grace by one budget per in-flight write. That costs a
// persistence_shutdown_deadline warning, never correctness, but the grace is
// configurable down to a second: keep this small enough to stay in that range.
const cacheInvalidateTimeout = time.Second

// CachingStore wraps a Store and adds a Redis hot cache for Retrieve.
// Retrieve: check cache first; on miss, call underlying store and populate cache.
// Store/Update/Forget/ForgetByScope: call underlying then invalidate cache for affected user(s).
type CachingStore struct {
	store        Store
	cache        *RedisCache
	backendLabel string
}

// NewCachingStore returns a Store that caches retrieval results in Redis.
// If cache is nil, all operations delegate to store with no caching.
// backendLabel is used for cache hit/miss metrics (e.g. "milvus", "valkey").
func NewCachingStore(store Store, cache *RedisCache, backendLabel string) Store {
	if cache == nil {
		return store
	}
	return &CachingStore{store: store, cache: cache, backendLabel: backendLabel}
}

// label returns the metrics label for this cache's backing store.
func (c *CachingStore) label() string {
	if c.backendLabel == "" {
		return defaultCacheBackendLabel
	}
	return c.backendLabel
}

// invalidate drops the user's cached retrievals after a committed write.
// The caller's context may already be done - a write that used up its whole job
// deadline, or one that raced shutdown - and on that context invalidation is
// skipped exactly when it matters, leaving stale entries readable until TTL. So
// drop the cancellation chain (keeping trace values) and use our own budget.
func (c *CachingStore) invalidate(ctx context.Context, userID string) {
	if c.cache == nil || userID == "" {
		return
	}
	ctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), cacheInvalidateTimeout)
	defer cancel()
	start := time.Now()
	if err := c.cache.InvalidateByUser(ctx, userID); err != nil {
		RecordMemoryStoreOperation(c.label(), "cache_invalidate", "failed", time.Since(start).Seconds())
		return
	}
	RecordMemoryStoreOperation(c.label(), "cache_invalidate", "success", time.Since(start).Seconds())
}

// Retrieve implements Store. It checks the cache first; on miss, calls the underlying store and caches the result.
func (c *CachingStore) Retrieve(ctx context.Context, opts RetrieveOptions) ([]*RetrieveResult, error) {
	label := c.label()
	if c.cache != nil {
		start := time.Now()
		results, ok := c.cache.Get(ctx, opts)
		elapsed := time.Since(start).Seconds()
		if ok {
			RecordMemoryCacheHit(label, elapsed)
			return results, nil
		}
		RecordMemoryCacheMiss(label)
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
	if err == nil && memory != nil {
		c.invalidate(ctx, memory.UserID)
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
	if err == nil && memory != nil {
		c.invalidate(ctx, memory.UserID)
	}
	return err
}

// List implements Store.
func (c *CachingStore) List(ctx context.Context, opts ListOptions) (*ListResult, error) {
	return c.store.List(ctx, opts)
}

// Forget implements Store. The id alone carries no userID, so pay one Get to
// learn the owner before deleting. That lookup runs on the same detached budget
// as the invalidation it feeds: on the caller's context a dead deadline would
// lose the owner, and the deleted memory would stay readable until TTL. If the
// lookup fails anyway the owner is unknown and their entries keep that fate.
func (c *CachingStore) Forget(ctx context.Context, id string) error {
	var owner string
	if c.cache != nil {
		lookup, cancel := context.WithTimeout(
			context.WithoutCancel(ctx), cacheInvalidateTimeout,
		)
		mem, err := c.store.Get(lookup, id)
		cancel()
		if err == nil && mem != nil {
			owner = mem.UserID
		}
	}
	err := c.store.Forget(ctx, id)
	if err == nil {
		c.invalidate(ctx, owner)
	}
	return err
}

// ForgetByScope implements Store; delegates then invalidates cache for the scope's user.
func (c *CachingStore) ForgetByScope(ctx context.Context, scope MemoryScope) error {
	err := c.store.ForgetByScope(ctx, scope)
	if err == nil {
		c.invalidate(ctx, scope.UserID)
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
