//go:build !windows && cgo

package cache

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestValkeyCacheIntegration_ExactRoundTripAndPartitionIsolation(t *testing.T) {
	cache := setupValkeyCacheIntegration(t)
	defer func() { _ = cache.Close() }()

	fingerprint := fmt.Sprintf("exact-%d", time.Now().UnixNano())
	require.NoError(
		t,
		cache.AddExact(context.Background(), "tenant-a", fingerprint, []byte(`{"answer":"cached"}`), 60),
	)
	hit, err := cache.FindExact(context.Background(), "tenant-a", fingerprint)
	require.NoError(t, err)
	require.True(t, hit.Found)
	assert.JSONEq(t, `{"answer":"cached"}`, string(hit.ResponseBody))
	assert.Equal(t, float32(1), hit.Similarity)
	assert.True(t, hit.AgeKnown)
	assert.False(t, hit.StoredAt.IsZero())
	assert.WithinDuration(t, time.Now(), hit.StoredAt, 5*time.Second)
	assert.False(t, hit.ExpiresAt.IsZero())
	assert.WithinDuration(t, time.Now().Add(60*time.Second), hit.ExpiresAt, 5*time.Second)

	miss, err := cache.FindExact(context.Background(), "tenant-b", fingerprint)
	require.NoError(t, err)
	assert.False(t, miss.Found)
}

func TestValkeyCacheIntegration_ExactMaxAge(t *testing.T) {
	cache := setupValkeyCacheIntegration(t)
	defer func() { _ = cache.Close() }()

	service := NewResponseCacheService(
		NewLegacyBackendAdapter(cache, ValkeyCacheType),
		ResponseCacheServiceOptions{L1MaxEntries: -1},
	)

	fingerprint := fmt.Sprintf("exact-valkey-maxage-%d", time.Now().UnixNano())
	identity := CacheIdentity{
		Partition: CachePartition{
			RequestModel: "valkey-model",
			Protocol:     "openai:body",
		},
		ExactFingerprint: fingerprint,
	}

	err := service.StoreExact(context.Background(), CacheWrite{
		Identity:     identity,
		RequestID:    "req-valkey-1",
		ResponseBody: []byte(`{"answer":"valkey-hit"}`),
		TTL:          TTL(time.Hour),
	})
	require.NoError(t, err, "StoreExact")

	// Fresh hit with maxAge of 60s
	maxAge := 60 * time.Second
	result, err := service.LookupExact(context.Background(), ExactLookup{
		Identity: identity,
		MaxAge:   &maxAge,
	})
	require.NoError(t, err, "LookupExact fresh")
	assert.True(t, result.Found, "expected HIT from Valkey L2")
	assert.Equal(t, CacheSourceL2, result.Source)
	assert.True(t, result.AgeKnown, "AgeKnown should be true for Valkey exact hit")
	assert.LessOrEqual(t, result.Age, maxAge)

	// Stale miss with maxAge of 1ns
	time.Sleep(2 * time.Millisecond)
	staleMaxAge := time.Nanosecond
	staleResult, err := service.LookupExact(context.Background(), ExactLookup{
		Identity: identity,
		MaxAge:   &staleMaxAge,
	})
	require.NoError(t, err, "LookupExact stale")
	assert.False(t, staleResult.Found, "expected stale MISS from Valkey L2")
	assert.Equal(t, HitKindMiss, staleResult.HitKind)
}

func TestValkeyCacheIntegration_LegacyStringOverwrite(t *testing.T) {
	cache := setupValkeyCacheIntegration(t)
	defer func() { _ = cache.Close() }()

	ctx := context.Background()
	fingerprint := fmt.Sprintf("exact-valkey-legacy-%d", time.Now().UnixNano())
	rawKey := exactCacheStorageKey("tenant-legacy", fingerprint)

	// 1. Manually write a legacy raw string entry (as written by pre-PR versions)
	_, err := cache.client.CustomCommand(ctx, []string{"SET", rawKey, `{"legacy":"valkey-string"}`})
	require.NoError(t, err)

	// 2. Verify fallback read works but AgeKnown is false
	legacyHit, err := cache.FindExact(ctx, "tenant-legacy", fingerprint)
	require.NoError(t, err)
	require.True(t, legacyHit.Found)
	assert.JSONEq(t, `{"legacy":"valkey-string"}`, string(legacyHit.ResponseBody))
	assert.False(t, legacyHit.AgeKnown)

	// 3. Overwrite the legacy string entry with AddExact (must delete first and not fail with WRONGTYPE)
	require.NoError(t, cache.AddExact(ctx, "tenant-legacy", fingerprint, []byte(`{"modern":"valkey-hash"}`), 60))

	// 4. Verify updated entry is now a hash with AgeKnown=true and timing metadata
	modernHit, err := cache.FindExact(ctx, "tenant-legacy", fingerprint)
	require.NoError(t, err)
	require.True(t, modernHit.Found)
	assert.JSONEq(t, `{"modern":"valkey-hash"}`, string(modernHit.ResponseBody))
	assert.True(t, modernHit.AgeKnown)
	assert.False(t, modernHit.StoredAt.IsZero())
	assert.False(t, modernHit.ExpiresAt.IsZero())
}
