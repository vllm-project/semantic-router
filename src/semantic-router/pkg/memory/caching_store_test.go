package memory

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/alicebob/miniredis/v2/server"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/internal/testutil/storagetest"
)

func TestNewCachingStore_WithNilCache_ReturnsSameStore(t *testing.T) {
	underlying := newTestInMemoryStore()
	wrapped := NewCachingStore(underlying, nil, "milvus")
	// When cache is nil, NewCachingStore returns the same store (no wrapper)
	assert.Same(t, underlying, wrapped)
}

// StorageIntegration: redis
func TestCachingStore_DelegatesToUnderlying(t *testing.T) {
	underlying := newTestInMemoryStore()
	// Use a non-nil cache that we don't connect (we only test delegation)
	// NewCachingStore with nil cache returns store; with non-nil cache returns CachingStore.
	// So we need a real Redis cache to test the wrapper. Skip integration and just test that
	// the wrapper implements Store and delegates Get/List/IsEnabled/CheckConnection/Close.
	storagetest.Require(t, "redis")
	cacheCfg := &RedisCacheConfig{Address: storageRedisAddress(), TTLSeconds: 60}
	redisCache, err := NewRedisCache(context.Background(), cacheCfg)
	if err != nil {
		storagetest.Unavailable(t, "redis", fmt.Sprintf("Redis not available for CachingStore test: %v", err))
	}
	defer func() { _ = redisCache.Close() }()

	wrapped := NewCachingStore(underlying, redisCache, "milvus")
	require.NotNil(t, wrapped)
	assert.True(t, wrapped.IsEnabled())

	ctx := context.Background()
	_, err = wrapped.Get(ctx, "nonexistent")
	assert.Error(t, err)

	list, err := wrapped.List(ctx, ListOptions{UserID: "u1", Limit: 10})
	require.NoError(t, err)
	assert.NotNil(t, list)
	assert.Empty(t, list.Memories)

	assert.NoError(t, wrapped.CheckConnection(ctx))
	assert.NoError(t, wrapped.Close())
}

// StorageIntegration: redis
func TestCachingStore_Retrieve_MissThenHit(t *testing.T) {
	storagetest.Require(t, "redis")
	cacheCfg := &RedisCacheConfig{
		Address:    storageRedisAddress(),
		KeyPrefix:  fmt.Sprintf("%s:%d:", t.Name(), time.Now().UnixNano()),
		TTLSeconds: 60,
	}
	redisCache, err := NewRedisCache(context.Background(), cacheCfg)
	if err != nil {
		storagetest.Unavailable(t, "redis", fmt.Sprintf("Redis not available: %v", err))
	}
	defer func() { _ = redisCache.Close() }()

	// Declare the query/record relationship explicitly; this test exercises
	// Redis caching, while live Vela tests exercise embedding semantics.
	underlying := NewInMemoryStoreWithConfig(EmbeddingConfig{
		Provider: storagetest.Vectors{Size: 384, Aliases: map[string]string{
			"coffee": "user likes coffee",
		}},
		Model: EmbeddingModelBERT,
	})
	// Store one memory so Retrieve can return something
	mem := &Memory{ID: "m1", Type: MemoryTypeSemantic, Content: "user likes coffee", UserID: "u1"}
	require.NoError(t, underlying.Store(context.Background(), mem))

	wrapped := NewCachingStore(underlying, redisCache, "milvus")
	opts := RetrieveOptions{Query: "coffee", UserID: "u1", Limit: 5, Threshold: 0.5}

	// First call: miss, then populate cache
	r1, err := wrapped.Retrieve(context.Background(), opts)
	require.NoError(t, err)
	require.Len(t, r1, 1)
	assert.Equal(t, mem.ID, r1[0].Memory.ID)
	cached, hit := redisCache.Get(context.Background(), opts)
	require.True(t, hit, "first retrieval must populate Redis")
	require.Len(t, cached, 1)

	// Bypass the wrapper to remove the source without invalidating Redis. The
	// next retrieval can only retain the result if it actually uses the cache.
	require.NoError(t, underlying.Forget(context.Background(), mem.ID))

	// Second call: hit from cache (same opts)
	r2, err := wrapped.Retrieve(context.Background(), opts)
	require.NoError(t, err)
	require.Len(t, r2, 1)
	assert.Equal(t, r1[0].Memory.Content, r2[0].Memory.Content)
}

// TestCachingStore_Retrieve_EmptyResultsCached verifies that empty retrieval results are cached so the second call is a hit.
// StorageIntegration: redis
func TestCachingStore_Retrieve_EmptyResultsCached(t *testing.T) {
	storagetest.Require(t, "redis")
	cacheCfg := &RedisCacheConfig{Address: storageRedisAddress(), TTLSeconds: 60}
	redisCache, err := NewRedisCache(context.Background(), cacheCfg)
	if err != nil {
		storagetest.Unavailable(t, "redis", fmt.Sprintf("Redis not available: %v", err))
	}
	defer func() { _ = redisCache.Close() }()
	underlying := newTestInMemoryStore()
	wrapped := NewCachingStore(underlying, redisCache, "milvus")
	opts := RetrieveOptions{Query: "nonexistentquery123", UserID: "u_none", Limit: 5, Threshold: 0.5}
	// First call: miss, underlying returns some result (possibly empty)
	r1, err := wrapped.Retrieve(context.Background(), opts)
	require.NoError(t, err)
	// Second call: hit from cache, same result
	r2, err := wrapped.Retrieve(context.Background(), opts)
	require.NoError(t, err)
	assert.Len(t, r2, len(r1), "cached result should match first result length")
}

// TestCachingStore_Store_InvalidatesCache verifies that after Store, a subsequent Retrieve for that user misses cache.
// StorageIntegration: redis
func TestCachingStore_Store_InvalidatesCache(t *testing.T) {
	storagetest.Require(t, "redis")
	cacheCfg := &RedisCacheConfig{
		Address:    storageRedisAddress(),
		KeyPrefix:  fmt.Sprintf("%s:%d:", t.Name(), time.Now().UnixNano()),
		TTLSeconds: 60,
	}
	redisCache, err := NewRedisCache(context.Background(), cacheCfg)
	if err != nil {
		storagetest.Unavailable(t, "redis", fmt.Sprintf("Redis not available: %v", err))
	}
	defer func() { _ = redisCache.Close() }()
	underlying := NewInMemoryStoreWithConfig(EmbeddingConfig{
		Provider: storagetest.Vectors{Size: 384, Aliases: map[string]string{
			"coffee": "likes coffee",
		}},
		Model: EmbeddingModelBERT,
	})
	wrapped := NewCachingStore(underlying, redisCache, "milvus")
	opts := RetrieveOptions{Query: "coffee", UserID: "u1", Limit: 5, Threshold: 0.5}
	// Prime cache with one memory
	mem := &Memory{ID: "m1", Type: MemoryTypeSemantic, Content: "likes coffee", UserID: "u1"}
	require.NoError(t, underlying.Store(context.Background(), mem))
	r1, err := wrapped.Retrieve(context.Background(), opts)
	require.NoError(t, err)
	require.Len(t, r1, 1)
	r2, err := wrapped.Retrieve(context.Background(), opts)
	require.NoError(t, err)
	require.Len(t, r2, 1)
	_, hit := redisCache.Get(context.Background(), opts)
	require.True(t, hit, "retrieval must be cached before invalidation")
	// Store another memory for same user (invalidates cache)
	mem2 := &Memory{ID: "m2", Type: MemoryTypeSemantic, Content: "likes tea", UserID: "u1"}
	require.NoError(t, wrapped.Store(context.Background(), mem2))
	_, hit = redisCache.Get(context.Background(), opts)
	require.False(t, hit, "storing a memory must invalidate the user's cached retrieval")
	// Next Retrieve must refill Redis with the still-matching coffee memory.
	r3, err := wrapped.Retrieve(context.Background(), opts)
	require.NoError(t, err)
	require.Len(t, r3, 1)
	assert.Equal(t, mem.ID, r3[0].Memory.ID)
	_, hit = redisCache.Get(context.Background(), opts)
	assert.True(t, hit, "retrieval after invalidation must refill Redis")
}

// TestCachingStore_ForgetByScope_InvalidatesCache verifies that ForgetByScope invalidates cache for that user.
// StorageIntegration: redis
func TestCachingStore_ForgetByScope_InvalidatesCache(t *testing.T) {
	storagetest.Require(t, "redis")
	cacheCfg := &RedisCacheConfig{Address: storageRedisAddress(), TTLSeconds: 60}
	redisCache, err := NewRedisCache(context.Background(), cacheCfg)
	if err != nil {
		storagetest.Unavailable(t, "redis", fmt.Sprintf("Redis not available: %v", err))
	}
	defer func() { _ = redisCache.Close() }()
	underlying := newTestInMemoryStore()
	mem := &Memory{ID: "m1", Type: MemoryTypeSemantic, Content: "content", UserID: "u1"}
	require.NoError(t, underlying.Store(context.Background(), mem))
	wrapped := NewCachingStore(underlying, redisCache, "milvus")
	opts := RetrieveOptions{Query: "content", UserID: "u1", Limit: 5, Threshold: 0.5}
	_, err = wrapped.Retrieve(context.Background(), opts)
	require.NoError(t, err)
	err = wrapped.ForgetByScope(context.Background(), MemoryScope{UserID: "u1"})
	require.NoError(t, err)
	// Cache was invalidated; next Retrieve hits underlying and gets 0
	r, err := wrapped.Retrieve(context.Background(), opts)
	require.NoError(t, err)
	assert.Empty(t, r)
}

// TestNewCachingStore_BackendLabel verifies that the backend label is stored and
// that both "milvus" and "valkey" labels are accepted without error.
// StorageIntegration: redis
func TestNewCachingStore_BackendLabel(t *testing.T) {
	t.Parallel()

	for _, label := range []string{"milvus", "valkey", "custom"} {
		t.Run(label, func(t *testing.T) {
			t.Parallel()
			underlying := newTestInMemoryStore()
			// nil cache returns the underlying store directly (no wrapper), so use a
			// non-nil cache to exercise the CachingStore path. Skip if Redis unavailable.
			storagetest.Require(t, "redis")
			cacheCfg := &RedisCacheConfig{Address: storageRedisAddress(), TTLSeconds: 60}
			redisCache, err := NewRedisCache(context.Background(), cacheCfg)
			if err != nil {
				storagetest.Unavailable(t, "redis", fmt.Sprintf("Redis not available: %v", err))
			}
			defer func() { _ = redisCache.Close() }()

			wrapped := NewCachingStore(underlying, redisCache, label)
			require.NotNil(t, wrapped)
			cs, ok := wrapped.(*CachingStore)
			require.True(t, ok, "expected *CachingStore wrapper")
			assert.Equal(t, label, cs.backendLabel)
		})
	}
}

// ctxIgnoringStore models a backing store whose write commits even though the
// caller's context is already done - the job context can expire mid-write, or
// shutdown can cancel it. That write is durable, so the cache must not keep
// serving the pre-write answer.
type ctxIgnoringStore struct {
	writeErr error
	mem      *Memory
}

func (s *ctxIgnoringStore) Store(context.Context, *Memory) error { return s.writeErr }

func (s *ctxIgnoringStore) Update(context.Context, string, *Memory) error { return s.writeErr }

// Get, unlike the writes, honours the context: a real store's read fails on a
// dead one, and Forget has to resolve an owner through it before deleting.
func (s *ctxIgnoringStore) Get(ctx context.Context, _ string) (*Memory, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return s.mem, nil
}

func (s *ctxIgnoringStore) Forget(context.Context, string) error { return s.writeErr }

func (s *ctxIgnoringStore) ForgetByScope(context.Context, MemoryScope) error { return s.writeErr }

func (s *ctxIgnoringStore) Retrieve(context.Context, RetrieveOptions) ([]*RetrieveResult, error) {
	return nil, nil
}

func (s *ctxIgnoringStore) List(context.Context, ListOptions) (*ListResult, error) {
	return &ListResult{}, nil
}

func (s *ctxIgnoringStore) IsEnabled() bool { return true }

func (s *ctxIgnoringStore) CheckConnection(context.Context) error { return nil }

func (s *ctxIgnoringStore) Close() error { return nil }

var _ Store = (*ctxIgnoringStore)(nil)

func newInvalidationFixture(t *testing.T, backing Store) (*CachingStore, *RedisCache) {
	t.Helper()
	mr, err := miniredis.Run()
	require.NoError(t, err)
	t.Cleanup(mr.Close)

	cache, err := NewRedisCache(context.Background(), &RedisCacheConfig{
		Address: mr.Addr(), KeyPrefix: t.Name() + ":", TTLSeconds: 300,
	})
	require.NoError(t, err)
	t.Cleanup(func() { _ = cache.Close() })

	return &CachingStore{store: backing, cache: cache, backendLabel: "valkey"}, cache
}

// deadCtxs covers both ways a persistence job context dies: its own deadline,
// and the runner's shutdown cancellation.
func deadCtxs() map[string]func() context.Context {
	return map[string]func() context.Context{
		"deadline_exceeded": func() context.Context {
			ctx, cancel := context.WithDeadline(context.Background(), time.Now().Add(-time.Second))
			cancel()
			return ctx
		},
		"cancelled": func() context.Context {
			ctx, cancel := context.WithCancel(context.Background())
			cancel()
			return ctx
		},
	}
}

func TestCachingStoreInvalidatesAfterCommittedWriteOnDeadContext(t *testing.T) {
	const userID = "u-invalidate"
	opts := RetrieveOptions{Query: "coffee", UserID: userID, Limit: 5, Threshold: 0.5}
	mem := &Memory{ID: "m1", UserID: userID, Content: "likes coffee"}

	writes := map[string]func(*CachingStore, context.Context) error{
		"Store":  func(c *CachingStore, ctx context.Context) error { return c.Store(ctx, mem) },
		"Update": func(c *CachingStore, ctx context.Context) error { return c.Update(ctx, mem.ID, mem) },
		"Forget": func(c *CachingStore, ctx context.Context) error { return c.Forget(ctx, mem.ID) },
		"ForgetByScope": func(c *CachingStore, ctx context.Context) error {
			return c.ForgetByScope(ctx, MemoryScope{UserID: userID})
		},
	}

	for writeName, write := range writes {
		for ctxName, newCtx := range deadCtxs() {
			t.Run(writeName+"/"+ctxName, func(t *testing.T) {
				wrapped, cache := newInvalidationFixture(t, &ctxIgnoringStore{mem: mem})

				cache.Set(context.Background(), opts, []*RetrieveResult{{Memory: mem}})
				_, cached := cache.Get(context.Background(), opts)
				require.True(t, cached, "precondition: entry is cached before the write")

				require.NoError(t, write(wrapped, newCtx()))

				_, cached = cache.Get(context.Background(), opts)
				require.False(t, cached, "committed write must invalidate even though the caller's context was already done")
			})
		}
	}
}

func TestCachingStoreKeepsCacheWhenWriteFails(t *testing.T) {
	const userID = "u-failed-write"
	opts := RetrieveOptions{Query: "tea", UserID: userID, Limit: 5}
	mem := &Memory{ID: "m2", UserID: userID, Content: "likes tea"}
	writeErr := errors.New("backend rejected the write")

	writes := map[string]func(*CachingStore, context.Context) error{
		"Store":  func(c *CachingStore, ctx context.Context) error { return c.Store(ctx, mem) },
		"Update": func(c *CachingStore, ctx context.Context) error { return c.Update(ctx, mem.ID, mem) },
		"Forget": func(c *CachingStore, ctx context.Context) error { return c.Forget(ctx, mem.ID) },
		"ForgetByScope": func(c *CachingStore, ctx context.Context) error {
			return c.ForgetByScope(ctx, MemoryScope{UserID: userID})
		},
	}

	for name, write := range writes {
		t.Run(name, func(t *testing.T) {
			wrapped, cache := newInvalidationFixture(
				t, &ctxIgnoringStore{mem: mem, writeErr: writeErr},
			)
			cache.Set(context.Background(), opts, []*RetrieveResult{{Memory: mem}})
			require.ErrorIs(t, write(wrapped, context.Background()), writeErr)

			_, cached := cache.Get(context.Background(), opts)
			require.True(t, cached, "a rejected write leaves the cached answer correct; do not evict it")
		})
	}
}

func TestCachingStoreInvalidationIsBoundedIndependentlyOfCaller(t *testing.T) {
	mr := miniredis.RunT(t)
	cache, err := NewRedisCache(context.Background(), &RedisCacheConfig{Address: mr.Addr()})
	require.NoError(t, err)
	t.Cleanup(func() { _ = cache.Close() })

	// Hold the actual socket response beyond the detached invalidation budget.
	// A context deadline alone does not bound go-redis I/O unless its client
	// enables ContextTimeoutEnabled.
	entered, release := make(chan struct{}, 1), make(chan struct{})
	defer close(release)
	mr.Server().SetPreHook(func(_ *server.Peer, cmd string, _ ...string) bool {
		if strings.EqualFold(cmd, "SMEMBERS") {
			select {
			case entered <- struct{}{}:
			default:
			}
			<-release
		}
		return false
	})
	wrapped := NewCachingStore(&ctxIgnoringStore{}, cache, "milvus")
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	done := make(chan error, 1)
	go func() { done <- wrapped.Store(ctx, &Memory{UserID: "bounded-invalidation"}) }()
	select {
	case <-entered:
	case <-time.After(time.Second):
		t.Fatal("invalidation never reached Redis")
	}
	select {
	case err := <-done:
		require.NoError(t, err, "a cache timeout must not fail a committed write")
	case <-time.After(cacheInvalidateTimeout + 500*time.Millisecond):
		t.Fatal("Redis I/O outlived the invalidation budget")
	}
}
