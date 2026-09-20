package memory

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/internal/testutil/storagetest"
)

// StorageIntegration: redis
func TestCachingStoreEquivalentPoliciesShareVersionedEntry(t *testing.T) {
	storagetest.Require(t, "redis")
	ctx := context.Background()
	for _, hybrid := range []bool{false, true} {
		t.Run(fmt.Sprintf("hybrid_%t", hybrid), func(t *testing.T) {
			cache, err := NewRedisCache(ctx, &RedisCacheConfig{Address: storageRedisAddress(), KeyPrefix: "equivalent:" + t.Name(), TTLSeconds: 60})
			require.NoError(t, err)
			t.Cleanup(func() { _ = cache.Close() })
			store := &retrievalPolicyStore{}
			wrapped := NewCachingStore(store, cache, "valkey")
			opts := RetrieveOptions{Query: "coffee", UserID: "equivalent-user", HybridSearch: hybrid, Limit: 5, Threshold: 0.5}
			cache.InvalidateByUser(ctx, opts.UserID)
			t.Cleanup(func() { cache.InvalidateByUser(ctx, opts.UserID) })
			first, err := wrapped.Retrieve(ctx, opts)
			require.NoError(t, err)
			require.Equal(t, opts, store.lastOptions, "normalizing the key must not change backing-store options")
			modes := []string{"weighted"}
			if !hybrid {
				modes = append(modes, "rrf")
			}
			for _, mode := range modes {
				equivalent := opts
				equivalent.HybridMode = mode
				cached, retrieveErr := wrapped.Retrieve(ctx, equivalent)
				require.NoError(t, retrieveErr)
				require.Equal(t, first, cached)
				require.Equal(t, 1, store.calls)
			}
			members, err := cache.client.SMembers(ctx, cache.userIndexKey(opts.UserID)).Result()
			require.NoError(t, err)
			require.Equal(t, []string{cacheKey(cache.prefix, opts.UserID, opts)}, members)
			t.Logf("equivalent hybrid=%t modes=%v share one Redis value and one backend call", hybrid, modes)
		})
	}
}

// StorageIntegration: redis
func TestRedisCacheVersionExcludesLegacyValuesAndPreservesLifecycle(t *testing.T) {
	storagetest.Require(t, "redis")
	ctx := context.Background()
	cache, err := NewRedisCache(ctx, &RedisCacheConfig{Address: storageRedisAddress(), KeyPrefix: "version:" + t.Name(), TTLSeconds: 60})
	require.NoError(t, err)
	t.Cleanup(func() { _ = cache.Close() })
	opts := RetrieveOptions{Query: "coffee", UserID: "version-user", Limit: 5, Threshold: 0.5, HybridSearch: true}
	cache.InvalidateByUser(ctx, opts.UserID)
	t.Cleanup(func() { cache.InvalidateByUser(ctx, opts.UserID) })
	stale, err := json.Marshal([]*RetrieveResult{{Memory: &Memory{ID: "legacy-vector-result", UserID: opts.UserID}}})
	require.NoError(t, err)
	// Reconstruct the deployed pre-policy key, then keep it in the unchanged
	// per-user index just as an existing Redis installation would.
	h := sha256.New()
	_, err = fmt.Fprintf(h, "%s\x00%s\x00%d\x00%.6f", opts.Query, opts.ProjectID, opts.Limit, opts.Threshold)
	require.NoError(t, err)
	legacy := cache.prefix + "v:" + opts.UserID + ":" + hex.EncodeToString(h.Sum(nil))[:16]
	_, err = fmt.Fprintf(h, "\x00%t\x00%s\x00%t", opts.HybridSearch, opts.HybridMode, opts.AdaptiveThreshold)
	require.NoError(t, err)
	unversionedPolicy := cache.prefix + "v:" + opts.UserID + ":" + hex.EncodeToString(h.Sum(nil))[:16]
	for _, oldKey := range []string{legacy, unversionedPolicy} {
		require.NoError(t, cache.client.Set(ctx, oldKey, stale, time.Minute).Err())
		require.NoError(t, cache.client.SAdd(ctx, cache.userIndexKey(opts.UserID), oldKey).Err())
	}
	_, hit := cache.Get(ctx, opts)
	require.False(t, hit, "v2 must not interpret an unversioned value")
	store := &retrievalPolicyStore{}
	wrapped := NewCachingStore(store, cache, "valkey")
	got, err := wrapped.Retrieve(ctx, opts)
	require.NoError(t, err)
	require.Equal(t, 1, store.calls)
	require.NotEqual(t, "legacy-vector-result", got[0].Memory.ID)
	key := cacheKey(cache.prefix, opts.UserID, opts)
	require.Contains(t, key, ":v2:")
	members, err := cache.client.SMembers(ctx, cache.userIndexKey(opts.UserID)).Result()
	require.NoError(t, err)
	require.ElementsMatch(t, []string{legacy, unversionedPolicy, key}, members)
	for _, k := range []string{key, cache.userIndexKey(opts.UserID)} {
		ttl, ttlErr := cache.client.PTTL(ctx, k).Result()
		require.NoError(t, ttlErr)
		require.Greater(t, ttl, 50*time.Second)
		require.LessOrEqual(t, ttl, time.Minute)
	}
	cache.InvalidateByUser(ctx, opts.UserID)
	remaining, err := cache.client.Exists(ctx, legacy, unversionedPolicy, key, cache.userIndexKey(opts.UserID)).Result()
	require.NoError(t, err)
	require.Zero(t, remaining, "invalidation must remove both old and new indexed values")
	cache.Set(ctx, opts, got)
	require.NoError(t, cache.client.PExpire(ctx, key, time.Millisecond).Err())
	require.Eventually(t, func() bool { _, hit := cache.Get(ctx, opts); return !hit }, time.Second, 5*time.Millisecond)
}
