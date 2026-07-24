//go:build !windows && cgo

package cache

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// refusedRedisConfig returns a Redis config pointed at a closed local port so
// the connection is refused quickly (deterministic, no external server).
func refusedRedisConfig() *config.RedisConfig {
	cfg := &config.RedisConfig{}
	cfg.Connection.Host = "127.0.0.1"
	cfg.Connection.Port = 1 // reserved; connection refused
	cfg.Connection.Timeout = 1
	cfg.Index.Name = "test_index"
	cfg.Index.Prefix = "doc:"
	cfg.Index.VectorField.Name = "embedding"
	cfg.Index.VectorField.MetricType = "COSINE"
	cfg.Index.IndexType = "HNSW"
	cfg.Index.Params.M = 16
	cfg.Index.Params.EfConstruction = 64
	cfg.Search.TopK = 1
	return cfg
}

// refusedValkeyConfig mirrors refusedRedisConfig for the Valkey backend.
func refusedValkeyConfig() *config.ValkeyConfig {
	cfg := &config.ValkeyConfig{}
	cfg.Connection.Host = "127.0.0.1"
	cfg.Connection.Port = 1
	cfg.Connection.Timeout = 1
	cfg.Index.Name = "test_index"
	cfg.Index.Prefix = "doc:"
	cfg.Index.VectorField.Name = "embedding"
	cfg.Index.VectorField.MetricType = "COSINE"
	cfg.Index.IndexType = "HNSW"
	cfg.Index.Params.M = 16
	cfg.Index.Params.EfConstruction = 64
	cfg.Search.TopK = 1
	cfg.Development.AutoCreateIndex = true
	return cfg
}

// #2473: a constructor whose connection check fails must Close the partially
// built client before returning, rather than leaking it. These tests drive the
// CheckConnection-failure path (connection refused) and assert the constructor
// returns an error and no usable cache handle — exercising the cleanup branch.
//
// Note: a refused dial leaks no goroutines (go-redis/glide build their pools
// lazily), so a goroutine-count assertion here would not distinguish the fix.
// The leak the Close() guards against is an established connection stranded
// when CheckConnection fails AFTER dialing (e.g. auth/handshake); that requires
// a live server seam and is out of scope for this unit test.

func TestNewRedisCacheClosesClientOnConnectFailure(t *testing.T) {
	c, err := NewRedisCache(RedisCacheOptions{
		Enabled: true,
		Config:  refusedRedisConfig(),
	})
	require.Error(t, err, "unreachable Redis must fail the constructor")
	assert.Nil(t, c, "constructor must not return a client alongside its error")
}

func TestNewValkeyCacheClosesClientOnConnectFailure(t *testing.T) {
	c, err := NewValkeyCache(ValkeyCacheOptions{
		Enabled: true,
		Config:  refusedValkeyConfig(),
	})
	require.Error(t, err, "unreachable Valkey must fail the constructor")
	assert.Nil(t, c, "constructor must not return a client alongside its error")
}
