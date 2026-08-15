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
		cache.AddExact("tenant-a", fingerprint, []byte(`{"answer":"cached"}`), 60),
	)
	hit, err := cache.FindExact("tenant-a", fingerprint)
	require.NoError(t, err)
	require.True(t, hit.Found)
	assert.JSONEq(t, `{"answer":"cached"}`, string(hit.ResponseBody))
	assert.Equal(t, float32(1), hit.Similarity)

	miss, err := cache.FindExact("tenant-b", fingerprint)
	require.NoError(t, err)
	assert.False(t, miss.Found)
}
