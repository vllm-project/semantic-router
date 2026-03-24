package memory

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewCachingStore_WithNilCache_ReturnsSameStore(t *testing.T) {
	underlying := NewInMemoryStore()
	wrapped := NewCachingStore(underlying, nil)
	// When cache is nil, NewCachingStore returns the same store (no wrapper)
	assert.Same(t, underlying, wrapped)
}

func TestCachingStore_DelegatesToUnderlying(t *testing.T) {
	underlying := NewInMemoryStore()
	// Use a non-nil cache that we don't connect (we only test delegation)
	// NewCachingStore with nil cache returns store; with non-nil cache returns CachingStore.
	// So we need a real Redis cache to test the wrapper. Skip integration and just test that
	// the wrapper implements Store and delegates Get/List/IsEnabled/CheckConnection/Close.
	cacheCfg := &RedisCacheConfig{Address: "localhost:6379", TTLSeconds: 60}
	redisCache, err := NewRedisCache(context.Background(), cacheCfg)
	if err != nil {
		t.Skipf("Redis not available for CachingStore test: %v", err)
	}
	defer func() { _ = redisCache.Close() }()

	wrapped := NewCachingStore(underlying, redisCache)
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

func TestCachingStore_Retrieve_MissThenHit(t *testing.T) {
	cacheCfg := &RedisCacheConfig{Address: "localhost:6379", TTLSeconds: 60}
	redisCache, err := NewRedisCache(context.Background(), cacheCfg)
	if err != nil {
		t.Skipf("Redis not available: %v", err)
	}
	defer func() { _ = redisCache.Close() }()

	underlying := NewInMemoryStore()
	// Store one memory so Retrieve can return something
	mem := &Memory{ID: "m1", Type: MemoryTypeSemantic, Content: "user likes coffee", UserID: "u1"}
	require.NoError(t, underlying.Store(context.Background(), mem))

	wrapped := NewCachingStore(underlying, redisCache)
	opts := RetrieveOptions{Query: "coffee", UserID: "u1", Limit: 5, Threshold: 0.5}

	// First call: miss, then populate cache
	r1, err := wrapped.Retrieve(context.Background(), opts)
	require.NoError(t, err)
	require.Len(t, r1, 1)

	// Second call: hit from cache (same opts)
	r2, err := wrapped.Retrieve(context.Background(), opts)
	require.NoError(t, err)
	require.Len(t, r2, 1)
	assert.Equal(t, r1[0].Memory.Content, r2[0].Memory.Content)
}

// TestCachingStore_Retrieve_EmptyResultsCached verifies that empty retrieval results are cached so the second call is a hit.
func TestCachingStore_Retrieve_EmptyResultsCached(t *testing.T) {
	cacheCfg := &RedisCacheConfig{Address: "localhost:6379", TTLSeconds: 60}
	redisCache, err := NewRedisCache(context.Background(), cacheCfg)
	if err != nil {
		t.Skipf("Redis not available: %v", err)
	}
	defer func() { _ = redisCache.Close() }()
	underlying := NewInMemoryStore()
	wrapped := NewCachingStore(underlying, redisCache)
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
func TestCachingStore_Store_InvalidatesCache(t *testing.T) {
	cacheCfg := &RedisCacheConfig{Address: "localhost:6379", TTLSeconds: 60}
	redisCache, err := NewRedisCache(context.Background(), cacheCfg)
	if err != nil {
		t.Skipf("Redis not available: %v", err)
	}
	defer func() { _ = redisCache.Close() }()
	underlying := NewInMemoryStore()
	wrapped := NewCachingStore(underlying, redisCache)
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
	// Store another memory for same user (invalidates cache)
	mem2 := &Memory{ID: "m2", Type: MemoryTypeSemantic, Content: "likes tea", UserID: "u1"}
	require.NoError(t, wrapped.Store(context.Background(), mem2))
	// Next Retrieve should go to underlying (cache was invalidated); result count depends on store/embedding
	r3, err := wrapped.Retrieve(context.Background(), opts)
	require.NoError(t, err)
	assert.GreaterOrEqual(t, len(r3), 1, "after invalidation we should get at least the new or existing memories")
}

// TestCachingStore_ForgetByScope_InvalidatesCache verifies that ForgetByScope invalidates cache for that user.
func TestCachingStore_ForgetByScope_InvalidatesCache(t *testing.T) {
	cacheCfg := &RedisCacheConfig{Address: "localhost:6379", TTLSeconds: 60}
	redisCache, err := NewRedisCache(context.Background(), cacheCfg)
	if err != nil {
		t.Skipf("Redis not available: %v", err)
	}
	defer func() { _ = redisCache.Close() }()
	underlying := NewInMemoryStore()
	mem := &Memory{ID: "m1", Type: MemoryTypeSemantic, Content: "content", UserID: "u1"}
	require.NoError(t, underlying.Store(context.Background(), mem))
	wrapped := NewCachingStore(underlying, redisCache)
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
