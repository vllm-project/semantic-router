package memory

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewRedisCache_NilConfig_ReturnsNil(t *testing.T) {
	cache, err := NewRedisCache(context.Background(), nil)
	assert.NoError(t, err)
	assert.Nil(t, cache)
}

func TestNewRedisCache_EmptyAddress_ReturnsNil(t *testing.T) {
	cache, err := NewRedisCache(context.Background(), &RedisCacheConfig{Address: ""})
	assert.NoError(t, err)
	assert.Nil(t, cache)
}

func TestCacheKey_StableHash_SameOptsSameKey(t *testing.T) {
	opts := RetrieveOptions{Query: "coffee", UserID: "u1", Limit: 5, Threshold: 0.6}
	key1 := cacheKey("mem:", opts.UserID, opts)
	key2 := cacheKey("mem:", opts.UserID, opts)
	assert.Equal(t, key1, key2)
	assert.Contains(t, key1, "u1:")
	assert.Contains(t, key1, "mem:")
}

func TestCacheKey_DifferentOpts_DifferentKeys(t *testing.T) {
	prefix := "mem:"
	userID := "u1"
	k1 := cacheKey(prefix, userID, RetrieveOptions{Query: "a", UserID: userID, Limit: 5, Threshold: 0.5})
	k2 := cacheKey(prefix, userID, RetrieveOptions{Query: "b", UserID: userID, Limit: 5, Threshold: 0.5})
	k3 := cacheKey(prefix, userID, RetrieveOptions{Query: "a", UserID: userID, Limit: 10, Threshold: 0.5})
	assert.NotEqual(t, k1, k2)
	assert.NotEqual(t, k1, k3)
}

func TestCacheKey_IncludesProjectIDAndTypes(t *testing.T) {
	opts1 := RetrieveOptions{Query: "q", UserID: "u1", ProjectID: "p1", Limit: 5, Threshold: 0.5}
	opts2 := RetrieveOptions{Query: "q", UserID: "u1", ProjectID: "p2", Limit: 5, Threshold: 0.5}
	k1 := cacheKey("m:", opts1.UserID, opts1)
	k2 := cacheKey("m:", opts2.UserID, opts2)
	assert.NotEqual(t, k1, k2)
}

func TestRedisCache_Get_NilReceiver_ReturnsFalse(t *testing.T) {
	var c *RedisCache
	results, ok := c.Get(context.Background(), RetrieveOptions{UserID: "u1", Query: "q"})
	assert.Nil(t, results)
	assert.False(t, ok)
}

func TestRedisCache_Set_NilReceiver_NoPanic(t *testing.T) {
	var c *RedisCache
	c.Set(context.Background(), RetrieveOptions{UserID: "u1", Query: "q"}, nil)
	c.Set(context.Background(), RetrieveOptions{UserID: "u1", Query: "q"}, []*RetrieveResult{})
}

func TestRedisCache_InvalidateByUser_NilReceiver_NoPanic(t *testing.T) {
	var c *RedisCache
	c.InvalidateByUser(context.Background(), "u1")
}

func TestRedisCache_InvalidateByUser_EmptyUser_NoPanic(t *testing.T) {
	// When Redis is not available we can't create a cache; so test only the empty user path
	// by calling on a nil cache (already tested) or we need a real Redis. For empty user,
	// the implementation does nothing when userID == "".
	cacheCfg := &RedisCacheConfig{Address: "localhost:6379"}
	cache, err := NewRedisCache(context.Background(), cacheCfg)
	if err != nil {
		t.Skipf("Redis not available: %v", err)
	}
	defer func() { _ = cache.Close() }()
	cache.InvalidateByUser(context.Background(), "")
}

func TestRedisCache_Close_NilReceiver_NoError(t *testing.T) {
	var c *RedisCache
	assert.NoError(t, c.Close())
}

// TestRedisCache_InvalidateByUser_DeletesTrackedKeys verifies that Set registers
// each value key in the user's index set and that InvalidateByUser deletes every
// tracked value key plus the index set itself (no keyspace scan).
func TestRedisCache_InvalidateByUser_DeletesTrackedKeys(t *testing.T) {
	cacheCfg := &RedisCacheConfig{Address: "localhost:6379", TTLSeconds: 60}
	cache, err := NewRedisCache(context.Background(), cacheCfg)
	if err != nil {
		t.Skipf("Redis not available: %v", err)
	}
	defer func() { _ = cache.Close() }()
	ctx := context.Background()

	user := "u_inv_idx"
	cache.InvalidateByUser(ctx, user) // clean slate from prior runs

	// Two distinct queries -> two distinct value keys for the same user.
	opts1 := RetrieveOptions{Query: "alpha", UserID: user, Limit: 5, Threshold: 0.5}
	opts2 := RetrieveOptions{Query: "beta", UserID: user, Limit: 5, Threshold: 0.5}
	cache.Set(ctx, opts1, []*RetrieveResult{})
	cache.Set(ctx, opts2, []*RetrieveResult{})

	// The index set tracks exactly the two value keys.
	idxKey := cache.userIndexKey(user)
	members, err := cache.client.SMembers(ctx, idxKey).Result()
	require.NoError(t, err)
	assert.ElementsMatch(t,
		[]string{cacheKey(cache.prefix, user, opts1), cacheKey(cache.prefix, user, opts2)},
		members,
	)

	// Both are cache hits before invalidation.
	_, ok1 := cache.Get(ctx, opts1)
	_, ok2 := cache.Get(ctx, opts2)
	require.True(t, ok1)
	require.True(t, ok2)

	cache.InvalidateByUser(ctx, user)

	// Value keys and the index set are all gone.
	_, ok1 = cache.Get(ctx, opts1)
	_, ok2 = cache.Get(ctx, opts2)
	assert.False(t, ok1)
	assert.False(t, ok2)
	exists, err := cache.client.Exists(ctx, idxKey).Result()
	require.NoError(t, err)
	assert.Equal(t, int64(0), exists, "index set should be deleted after invalidation")
}

// TestRedisCache_InvalidateByUser_ScopedToUser verifies that invalidating one
// user leaves another user's cached entries intact.
func TestRedisCache_InvalidateByUser_ScopedToUser(t *testing.T) {
	cacheCfg := &RedisCacheConfig{Address: "localhost:6379", TTLSeconds: 60}
	cache, err := NewRedisCache(context.Background(), cacheCfg)
	if err != nil {
		t.Skipf("Redis not available: %v", err)
	}
	defer func() { _ = cache.Close() }()
	ctx := context.Background()

	optsA := RetrieveOptions{Query: "q", UserID: "u_scope_a", Limit: 5, Threshold: 0.5}
	optsB := RetrieveOptions{Query: "q", UserID: "u_scope_b", Limit: 5, Threshold: 0.5}
	cache.InvalidateByUser(ctx, optsA.UserID)
	cache.InvalidateByUser(ctx, optsB.UserID)
	cache.Set(ctx, optsA, []*RetrieveResult{})
	cache.Set(ctx, optsB, []*RetrieveResult{})

	cache.InvalidateByUser(ctx, optsA.UserID)

	_, okA := cache.Get(ctx, optsA)
	_, okB := cache.Get(ctx, optsB)
	assert.False(t, okA, "invalidated user should miss")
	assert.True(t, okB, "other user's cache must be untouched")

	cache.InvalidateByUser(ctx, optsB.UserID) // cleanup
}

func TestRedisCache_SetThenGet_RoundTrip(t *testing.T) {
	cacheCfg := &RedisCacheConfig{Address: "localhost:6379", TTLSeconds: 60}
	cache, err := NewRedisCache(context.Background(), cacheCfg)
	if err != nil {
		t.Skipf("Redis not available: %v", err)
	}
	defer func() { _ = cache.Close() }()
	ctx := context.Background()
	opts := RetrieveOptions{Query: "round", UserID: "u_round", Limit: 2, Threshold: 0.5}
	results := []*RetrieveResult{
		{Memory: &Memory{ID: "1", Content: "a", UserID: "u_round", Type: MemoryTypeSemantic}, Score: 0.9},
	}
	cache.Set(ctx, opts, results)
	got, ok := cache.Get(ctx, opts)
	require.True(t, ok)
	require.Len(t, got, 1)
	assert.Equal(t, "1", got[0].Memory.ID)
	assert.Equal(t, "a", got[0].Memory.Content)
	assert.Equal(t, float32(0.9), got[0].Score)
}
