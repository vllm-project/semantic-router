//go:build !windows && cgo

package cache

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"testing"
	"time"

	"github.com/redis/go-redis/v9"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestRedisExactCacheIntegrationRoundTripAndPartitionIsolation(t *testing.T) {
	if os.Getenv("SKIP_REDIS_TESTS") == "true" {
		t.Skip("Redis integration tests disabled")
	}
	host := os.Getenv("REDIS_HOST")
	if host == "" {
		host = "localhost"
	}
	port := 6379
	if configured := os.Getenv("REDIS_PORT"); configured != "" {
		parsed, err := strconv.Atoi(configured)
		require.NoError(t, err)
		port = parsed
	}
	client := redis.NewClient(&redis.Options{
		Addr: fmt.Sprintf("%s:%d", host, port),
	})
	t.Cleanup(func() { _ = client.Close() })
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := client.Ping(ctx).Err(); err != nil {
		t.Skipf("Redis unavailable: %v", err)
	}
	cache := &RedisCache{
		client:     client,
		config:     &config.RedisConfig{},
		enabled:    true,
		ttlSeconds: 60,
	}

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

func TestRedisExactCacheIntegration_MaxAge(t *testing.T) {
	if os.Getenv("SKIP_REDIS_TESTS") == "true" {
		t.Skip("Redis integration tests disabled")
	}
	host := os.Getenv("REDIS_HOST")
	if host == "" {
		host = "localhost"
	}
	port := 6379
	if configured := os.Getenv("REDIS_PORT"); configured != "" {
		parsed, err := strconv.Atoi(configured)
		require.NoError(t, err)
		port = parsed
	}
	client := redis.NewClient(&redis.Options{
		Addr: fmt.Sprintf("%s:%d", host, port),
	})
	t.Cleanup(func() { _ = client.Close() })
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := client.Ping(ctx).Err(); err != nil {
		t.Skipf("Redis unavailable: %v", err)
	}
	backend := &RedisCache{
		client:     client,
		config:     &config.RedisConfig{},
		enabled:    true,
		ttlSeconds: 3600,
	}
	service := NewResponseCacheService(
		NewLegacyBackendAdapter(backend, RedisCacheType),
		ResponseCacheServiceOptions{L1MaxEntries: -1},
	)

	fingerprint := fmt.Sprintf("exact-maxage-%d", time.Now().UnixNano())
	identity := CacheIdentity{
		Partition: CachePartition{
			RequestModel: "redis-model",
			Protocol:     "openai:body",
		},
		ExactFingerprint: fingerprint,
	}

	err := service.StoreExact(ctx, CacheWrite{
		Identity:     identity,
		RequestID:    "req-redis-1",
		ResponseBody: []byte(`{"answer":"redis-hit"}`),
		TTL:          TTL(time.Hour),
	})
	require.NoError(t, err, "StoreExact")

	// Fresh hit with maxAge of 60s
	maxAge := 60 * time.Second
	result, err := service.LookupExact(ctx, ExactLookup{
		Identity: identity,
		MaxAge:   &maxAge,
	})
	require.NoError(t, err, "LookupExact fresh")
	assert.True(t, result.Found, "expected HIT from Redis L2")
	assert.Equal(t, CacheSourceL2, result.Source)
	assert.True(t, result.AgeKnown, "AgeKnown should be true for Redis exact hit")
	assert.LessOrEqual(t, result.Age, maxAge)

	// Stale miss with maxAge of 1ns
	time.Sleep(2 * time.Millisecond)
	staleMaxAge := time.Nanosecond
	staleResult, err := service.LookupExact(ctx, ExactLookup{
		Identity: identity,
		MaxAge:   &staleMaxAge,
	})
	require.NoError(t, err, "LookupExact stale")
	assert.False(t, staleResult.Found, "expected stale MISS from Redis L2")
	assert.Equal(t, HitKindMiss, staleResult.HitKind)
}

func TestRedisExactCacheIntegration_LegacyStringOverwrite(t *testing.T) {
	if os.Getenv("SKIP_REDIS_TESTS") == "true" {
		t.Skip("Redis integration tests disabled")
	}
	host := os.Getenv("REDIS_HOST")
	if host == "" {
		host = "localhost"
	}
	port := 6379
	if configured := os.Getenv("REDIS_PORT"); configured != "" {
		parsed, err := strconv.Atoi(configured)
		require.NoError(t, err)
		port = parsed
	}
	client := redis.NewClient(&redis.Options{
		Addr: fmt.Sprintf("%s:%d", host, port),
	})
	t.Cleanup(func() { _ = client.Close() })
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := client.Ping(ctx).Err(); err != nil {
		t.Skipf("Redis unavailable: %v", err)
	}
	cache := &RedisCache{
		client:     client,
		config:     &config.RedisConfig{},
		enabled:    true,
		ttlSeconds: 60,
	}

	fingerprint := fmt.Sprintf("exact-legacy-%d", time.Now().UnixNano())
	rawKey := exactCacheStorageKey("tenant-legacy", fingerprint)

	// 1. Manually write a legacy raw string entry (as written by pre-PR versions)
	require.NoError(t, client.Set(ctx, rawKey, []byte(`{"legacy":"string"}`), 0).Err())

	// 2. Verify fallback read works but AgeKnown is false
	legacyHit, err := cache.FindExact(ctx, "tenant-legacy", fingerprint)
	require.NoError(t, err)
	require.True(t, legacyHit.Found)
	assert.JSONEq(t, `{"legacy":"string"}`, string(legacyHit.ResponseBody))
	assert.False(t, legacyHit.AgeKnown)

	// 3. Overwrite the legacy string entry with AddExact (must not fail with WRONGTYPE)
	require.NoError(t, cache.AddExact(ctx, "tenant-legacy", fingerprint, []byte(`{"modern":"hash"}`), 60))

	// 4. Verify updated entry is now a hash with AgeKnown=true and timing metadata
	modernHit, err := cache.FindExact(ctx, "tenant-legacy", fingerprint)
	require.NoError(t, err)
	require.True(t, modernHit.Found)
	assert.JSONEq(t, `{"modern":"hash"}`, string(modernHit.ResponseBody))
	assert.True(t, modernHit.AgeKnown)
	assert.False(t, modernHit.StoredAt.IsZero())
	assert.False(t, modernHit.ExpiresAt.IsZero())
}
