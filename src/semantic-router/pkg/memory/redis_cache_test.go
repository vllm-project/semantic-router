package memory

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/internal/testutil/storagetest"
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

func TestRedisCacheSetPreservesHitsAndTTL(t *testing.T) {
	mr := miniredis.RunT(t)
	cache, err := NewRedisCache(context.Background(), &RedisCacheConfig{
		Address: mr.Addr(), KeyPrefix: t.Name() + ":", TTLSeconds: 2,
	})
	require.NoError(t, err)
	t.Cleanup(func() { _ = cache.Close() })

	opts := RetrieveOptions{Query: "ttl", UserID: "ttl-user", Limit: 5, Threshold: 0.5}
	results := retrieveResultFor(&Memory{ID: "cached", UserID: opts.UserID, Content: "cached result"})
	cache.Set(context.Background(), opts, results)
	got, hit := cache.Get(context.Background(), opts)
	require.True(t, hit, "a normal cache read must hit after Set")
	require.Len(t, got, 1)
	assert.Equal(t, "cached", got[0].Memory.ID)

	for _, key := range []string{
		cacheKey(cache.prefix, opts.UserID, opts),
		cache.userIndexKey(opts.UserID),
		cache.userGenerationKey(opts.UserID),
	} {
		ttl, err := cache.client.PTTL(context.Background(), key).Result()
		require.NoError(t, err)
		assert.Greater(t, ttl, time.Duration(0), "%s must expire", key)
		assert.LessOrEqual(t, ttl, 2*time.Second, "%s must retain the configured TTL", key)
	}

	mr.FastForward(3 * time.Second)
	for _, key := range []string{
		cacheKey(cache.prefix, opts.UserID, opts),
		cache.userIndexKey(opts.UserID),
		cache.userGenerationKey(opts.UserID),
	} {
		exists, err := cache.client.Exists(context.Background(), key).Result()
		require.NoError(t, err)
		assert.Zero(t, exists, "%s should be reclaimed after the configured TTL", key)
	}
	_, hit = cache.Get(context.Background(), opts)
	assert.False(t, hit, "an expired cache value must remain a miss")
}

func TestRedisCacheConditionalRefillIsSafeAcrossClients(t *testing.T) {
	mr := miniredis.RunT(t)
	ctx := context.Background()
	cfg := &RedisCacheConfig{Address: mr.Addr(), KeyPrefix: t.Name() + ":", TTLSeconds: 30}
	reader, err := NewRedisCache(ctx, cfg)
	require.NoError(t, err)
	t.Cleanup(func() { _ = reader.Close() })
	writer, err := NewRedisCache(ctx, cfg)
	require.NoError(t, err)
	t.Cleanup(func() { _ = writer.Close() })

	opts := RetrieveOptions{Query: "shared-redis", UserID: "shared-user", Limit: 5, Threshold: 0.5}
	_, staleGeneration, hit := reader.GetWithGeneration(ctx, opts)
	require.False(t, hit)
	require.NotEmpty(t, staleGeneration)

	// A separate cache client commits a write and invalidates the shared user
	// namespace while the reader is still waiting on its backing-store query.
	require.NoError(t, writer.InvalidateByUser(ctx, opts.UserID))
	_, currentGeneration, hit := writer.GetWithGeneration(ctx, opts)
	require.False(t, hit)
	require.NotEmpty(t, currentGeneration)
	fresh := retrieveResultFor(&Memory{ID: "fresh", UserID: opts.UserID})
	stored, err := writer.SetIfGeneration(ctx, opts, fresh, currentGeneration)
	require.NoError(t, err)
	require.True(t, stored)

	stale := retrieveResultFor(&Memory{ID: "stale", UserID: opts.UserID})
	stored, err = reader.SetIfGeneration(ctx, opts, stale, staleGeneration)
	require.NoError(t, err)
	require.False(t, stored, "a token from before another client's invalidation must be rejected")
	got, hit := writer.Get(ctx, opts)
	require.True(t, hit)
	require.Equal(t, "fresh", got[0].Memory.ID, "a stale refill must not overwrite the new value")
}

func TestRedisCacheDoesNotReadLateV2RefillAndCleansItUp(t *testing.T) {
	mr := miniredis.RunT(t)
	ctx := context.Background()
	cache, err := NewRedisCache(ctx, &RedisCacheConfig{
		Address: mr.Addr(), KeyPrefix: t.Name() + ":", TTLSeconds: 30,
	})
	require.NoError(t, err)
	t.Cleanup(func() { _ = cache.Close() })

	opts := RetrieveOptions{Query: "rolling-upgrade", UserID: "rolling-user", Limit: 5, Threshold: 0.5}
	_, _, hit := cache.GetWithGeneration(ctx, opts)
	require.False(t, hit)
	require.NoError(t, cache.InvalidateByUser(ctx, opts.UserID))

	// Simulate an old instance finishing its v2 read-through after invalidation.
	// It writes a v2 result and registers it in the unchanged per-user index.
	v2Key := cacheKeyForVersion(cache.prefix, opts.UserID, opts, "v2:")
	stale, err := json.Marshal(retrieveResultFor(&Memory{ID: "late-v2", UserID: opts.UserID}))
	require.NoError(t, err)
	require.NoError(t, cache.client.Set(ctx, v2Key, stale, time.Minute).Err())
	require.NoError(t, cache.client.SAdd(ctx, cache.userIndexKey(opts.UserID), v2Key).Err())

	_, hit = cache.Get(ctx, opts)
	require.False(t, hit, "the new v3 reader must not observe an old instance's late v2 refill")
	_, err = cache.client.Get(ctx, v2Key).Result()
	require.NoError(t, err, "the simulated late v2 entry should exist until indexed cleanup")

	require.NoError(t, cache.InvalidateByUser(ctx, opts.UserID))
	exists, err := cache.client.Exists(ctx, v2Key, cache.userIndexKey(opts.UserID)).Result()
	require.NoError(t, err)
	require.Zero(t, exists, "the shared user index must still clean up late v2 entries")
}

func TestRedisCacheSetFailureCannotLeaveUntrackedValue(t *testing.T) {
	mr := miniredis.RunT(t)
	cache, err := NewRedisCache(context.Background(), &RedisCacheConfig{
		Address: mr.Addr(), KeyPrefix: t.Name() + ":", TTLSeconds: 30,
	})
	require.NoError(t, err)
	t.Cleanup(func() { _ = cache.Close() })

	opts := RetrieveOptions{Query: "wrong-index-type", UserID: "u1", Limit: 5}
	_, generation, hit := cache.GetWithGeneration(context.Background(), opts)
	require.False(t, hit)
	require.NotEmpty(t, generation)
	indexKey := cache.userIndexKey(opts.UserID)
	require.NoError(t, cache.client.Set(context.Background(), indexKey, "not-a-set", time.Minute).Err())

	stored, err := cache.SetIfGeneration(context.Background(), opts, retrieveResultFor(&Memory{ID: "untracked"}), generation)
	require.Error(t, err, "a corrupt index type must fail the cache write")
	assert.False(t, stored)
	exists, err := cache.client.Exists(context.Background(), cacheKey(cache.prefix, opts.UserID, opts)).Result()
	require.NoError(t, err)
	assert.Zero(t, exists, "a cache write error must not leave an untracked value")
}

func TestRedisCache_InvalidateByUser_NilReceiver_NoPanic(t *testing.T) {
	var c *RedisCache
	_ = c.InvalidateByUser(context.Background(), "u1")
}

// StorageIntegration: redis
func TestRedisCache_InvalidateByUser_EmptyUser_NoPanic(t *testing.T) {
	// When Redis is not available we can't create a cache; so test only the empty user path
	// by calling on a nil cache (already tested) or we need a real Redis. For empty user,
	// the implementation does nothing when userID == "".
	storagetest.Require(t, "redis")
	cacheCfg := &RedisCacheConfig{Address: storageRedisAddress()}
	cache, err := NewRedisCache(context.Background(), cacheCfg)
	if err != nil {
		storagetest.Unavailable(t, "redis", fmt.Sprintf("Redis not available: %v", err))
	}
	defer func() { _ = cache.Close() }()
	_ = cache.InvalidateByUser(context.Background(), "")
}

func TestRedisCache_Close_NilReceiver_NoError(t *testing.T) {
	var c *RedisCache
	assert.NoError(t, c.Close())
}

// TestRedisCache_InvalidateByUser_DeletesTrackedKeys verifies that Set registers
// each value key in the user's index set and that InvalidateByUser deletes every
// tracked value key plus the index set itself (no keyspace scan).
// StorageIntegration: redis
func TestRedisCache_InvalidateByUser_DeletesTrackedKeys(t *testing.T) {
	storagetest.Require(t, "redis")
	cacheCfg := &RedisCacheConfig{Address: storageRedisAddress(), TTLSeconds: 60}
	cache, err := NewRedisCache(context.Background(), cacheCfg)
	if err != nil {
		storagetest.Unavailable(t, "redis", fmt.Sprintf("Redis not available: %v", err))
	}
	defer func() { _ = cache.Close() }()
	ctx := context.Background()

	user := "u_inv_idx"
	_ = cache.InvalidateByUser(ctx, user) // clean slate from prior runs

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

	_ = cache.InvalidateByUser(ctx, user)

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
// StorageIntegration: redis
func TestRedisCache_InvalidateByUser_ScopedToUser(t *testing.T) {
	storagetest.Require(t, "redis")
	cacheCfg := &RedisCacheConfig{Address: storageRedisAddress(), TTLSeconds: 60}
	cache, err := NewRedisCache(context.Background(), cacheCfg)
	if err != nil {
		storagetest.Unavailable(t, "redis", fmt.Sprintf("Redis not available: %v", err))
	}
	defer func() { _ = cache.Close() }()
	ctx := context.Background()

	optsA := RetrieveOptions{Query: "q", UserID: "u_scope_a", Limit: 5, Threshold: 0.5}
	optsB := RetrieveOptions{Query: "q", UserID: "u_scope_b", Limit: 5, Threshold: 0.5}
	_ = cache.InvalidateByUser(ctx, optsA.UserID)
	_ = cache.InvalidateByUser(ctx, optsB.UserID)
	cache.Set(ctx, optsA, []*RetrieveResult{})
	cache.Set(ctx, optsB, []*RetrieveResult{})

	_ = cache.InvalidateByUser(ctx, optsA.UserID)

	_, okA := cache.Get(ctx, optsA)
	_, okB := cache.Get(ctx, optsB)
	assert.False(t, okA, "invalidated user should miss")
	assert.True(t, okB, "other user's cache must be untouched")

	_ = cache.InvalidateByUser(ctx, optsB.UserID) // cleanup
}

// StorageIntegration: redis
func TestRedisCache_SetThenGet_RoundTrip(t *testing.T) {
	storagetest.Require(t, "redis")
	cacheCfg := &RedisCacheConfig{Address: storageRedisAddress(), TTLSeconds: 60}
	cache, err := NewRedisCache(context.Background(), cacheCfg)
	if err != nil {
		storagetest.Unavailable(t, "redis", fmt.Sprintf("Redis not available: %v", err))
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

func TestCacheKeySeparatesRetrievalPolicies(t *testing.T) {
	baseline := RetrieveOptions{Query: "coffee", UserID: "u1", Limit: 5, Threshold: 0.5}
	for _, change := range []struct {
		name  string
		apply func(*RetrieveOptions)
	}{
		{"hybrid", func(o *RetrieveOptions) { o.HybridSearch = true }},
		{"fusion mode", func(o *RetrieveOptions) { o.HybridMode = "rrf" }},
		{"adaptive", func(o *RetrieveOptions) { o.AdaptiveThreshold = true }},
	} {
		t.Run(change.name, func(t *testing.T) {
			base := baseline
			if change.name == "fusion mode" {
				base.HybridSearch = true
				base.HybridMode = "weighted"
			}
			other := base
			change.apply(&other)
			assert.NotEqual(t, cacheKey("mem:", "u1", base), cacheKey("mem:", "u1", other))
		})
	}
}

func TestCacheKeyNormalizesEffectiveHybridMode(t *testing.T) {
	base := RetrieveOptions{Query: "coffee", UserID: "u1", Limit: 5, Threshold: 0.5}
	for _, hybrid := range []bool{false, true} {
		base.HybridSearch = hybrid
		explicit := base
		explicit.HybridMode = "weighted"
		assert.Equal(t, cacheKey("mem:", "u1", base), cacheKey("mem:", "u1", explicit))
		if !hybrid {
			explicit.HybridMode = "rrf"
			assert.Equal(t, cacheKey("mem:", "u1", base), cacheKey("mem:", "u1", explicit))
		}
	}
}

func TestCacheKeyVersionedEncodingPreservesThresholdPrecision(t *testing.T) {
	base := RetrieveOptions{Query: "coffee", UserID: "u1", Limit: 5, Threshold: 0.5}
	key := cacheKey("mem:", "u1", base)
	assert.Contains(t, key, "mem:v3:u1:")
	nearby := base
	nearby.Threshold = math.Nextafter32(base.Threshold, 1)
	assert.NotEqual(t, key, cacheKey("mem:", "u1", nearby), "distinct score floors must not round to the same key")
	project := base
	project.ProjectID = "another-project"
	assert.NotEqual(t, key, cacheKey("mem:", "u1", project))
	types := base
	types.Types = []MemoryType{MemoryTypeSemantic}
	assert.NotEqual(t, key, cacheKey("mem:", "u1", types))
	assert.NotEqual(t, key, cacheKey("mem:", "other-user", base))
}

func TestCacheKeyHybridDefaultsAgreeWithActualRerankers(t *testing.T) {
	candidates := func() []*RetrieveResult {
		return []*RetrieveResult{
			{Memory: &Memory{ID: "coffee", Content: "coffee preference"}, Score: 0.8},
			{Memory: &Memory{ID: "tea", Content: "tea preference"}, Score: 0.4},
		}
	}
	implicit := RetrieveOptions{Query: "coffee", HybridSearch: true}
	explicit := implicit
	explicit.HybridMode = "weighted"
	rrf := implicit
	rrf.HybridMode = "rrf"
	assert.Equal(t, hybridRerankCandidates(candidates(), implicit), hybridRerankCandidates(candidates(), explicit))
	assert.NotEqual(t, hybridRerankCandidates(candidates(), implicit), hybridRerankCandidates(candidates(), rrf))
	assert.Equal(t, cacheKey("m:", "u", implicit), cacheKey("m:", "u", explicit))
	assert.NotEqual(t, cacheKey("m:", "u", implicit), cacheKey("m:", "u", rrf))
}
