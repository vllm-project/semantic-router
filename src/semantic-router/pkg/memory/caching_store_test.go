package memory

import (
	"context"
	"fmt"
	"testing"
	"time"

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
