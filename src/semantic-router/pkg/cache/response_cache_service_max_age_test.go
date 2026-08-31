package cache

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestResponseCacheServiceMaxAgeAgainstL2(t *testing.T) {
	ctx := context.Background()

	backend := NewInMemoryCache(InMemoryCacheOptions{
		Enabled:    true,
		MaxEntries: 100,
		TTLSeconds: 3600,
	})
	service := NewResponseCacheService(
		NewLegacyBackendAdapter(backend, InMemoryCacheType),
		ResponseCacheServiceOptions{L1MaxEntries: -1},
	)

	identity := CacheIdentity{
		Partition: CachePartition{
			RequestModel: "model-a",
			Protocol:     "openai:body",
		},
		ExactFingerprint: "fingerprint-1",
	}
	err := service.StoreExact(ctx, CacheWrite{
		Identity:     identity,
		RequestID:    "req-1",
		ResponseBody: []byte(`{"id":"cached"}`),
		TTL:          TTL(time.Hour),
	})
	require.NoError(t, err, "StoreExact")

	// Look up with maxAge of 60 seconds. Entry is fresh (created just now),
	// so it should hit L2 and not be treated as a stale miss.
	maxAge := 60 * time.Second
	result, err := service.LookupExact(ctx, ExactLookup{
		Identity: identity,
		MaxAge:   &maxAge,
	})
	require.NoError(t, err, "LookupExact")
	assert.True(t, result.Found, "expected HIT from L2, got MISS (hit_kind=%s, age_known=%v, age=%v)", result.HitKind, result.AgeKnown, result.Age)
	assert.Equal(t, CacheSourceL2, result.Source)
	assert.True(t, result.AgeKnown, "AgeKnown should be true")
	assert.LessOrEqual(t, result.Age, maxAge)
}

func TestResponseCacheServiceMaxAgeStaleMissAgainstL2(t *testing.T) {
	ctx := context.Background()

	backend := NewInMemoryCache(InMemoryCacheOptions{
		Enabled:    true,
		MaxEntries: 100,
		TTLSeconds: 3600,
	})
	service := NewResponseCacheService(
		NewLegacyBackendAdapter(backend, InMemoryCacheType),
		ResponseCacheServiceOptions{L1MaxEntries: -1},
	)

	identity := CacheIdentity{
		Partition: CachePartition{
			RequestModel: "model-a",
			Protocol:     "openai:body",
		},
		ExactFingerprint: "fingerprint-2",
	}
	err := service.StoreExact(ctx, CacheWrite{
		Identity:     identity,
		RequestID:    "req-2",
		ResponseBody: []byte(`{"id":"cached-2"}`),
		TTL:          TTL(time.Hour),
	})
	require.NoError(t, err, "StoreExact")

	// Wait briefly so entry age > 1ms, then lookup with maxAge of 1 Nanosecond (exceeded bound).
	time.Sleep(2 * time.Millisecond)
	maxAge := time.Nanosecond
	result, err := service.LookupExact(ctx, ExactLookup{
		Identity: identity,
		MaxAge:   &maxAge,
	})
	require.NoError(t, err, "LookupExact")
	assert.False(t, result.Found, "expected MISS due to max-age bound exceeded")
	assert.Equal(t, HitKindMiss, result.HitKind)

	stats, err := service.Stats(ctx)
	require.NoError(t, err, "Stats")
	assert.Equal(t, int64(1), stats.StaleMissCount, "stale miss count should be incremented")
}

func TestLegacyBackendAdapterAgePropagation(t *testing.T) {
	ctx := context.Background()

	backend := NewInMemoryCache(InMemoryCacheOptions{
		Enabled:    true,
		MaxEntries: 100,
		TTLSeconds: 3600,
	})
	adapter := NewLegacyBackendAdapter(backend, InMemoryCacheType)

	identity := CacheIdentity{
		Partition: CachePartition{
			RequestModel: "model-a",
			Protocol:     "openai:body",
		},
		ExactFingerprint: "fingerprint-adapter",
	}
	err := adapter.StoreExact(ctx, CacheWrite{
		Identity:     identity,
		RequestID:    "req-adapter",
		ResponseBody: []byte(`{"id":"adapter"}`),
		TTL:          TTL(time.Hour),
	})
	require.NoError(t, err, "StoreExact")

	res, err := adapter.LookupExact(ctx, ExactLookup{Identity: identity})
	require.NoError(t, err, "LookupExact")
	assert.True(t, res.Found)
	assert.True(t, res.AgeKnown)
	assert.False(t, res.ExpiresAt.IsZero())
	assert.GreaterOrEqual(t, res.Age, time.Duration(0))
}
