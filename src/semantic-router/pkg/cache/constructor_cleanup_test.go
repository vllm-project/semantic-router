//go:build !windows && cgo

package cache

import (
	"errors"
	"testing"

	"github.com/redis/go-redis/v9"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	glide "github.com/valkey-io/valkey-glide/go/v2"

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

// #2473 acceptance: a constructor whose setup step fails must release the
// partially built client instead of leaking it.
//
// The cleanup has no black-box signal, so these tests assert the release call
// itself:
//   - the constructor already returned (nil, err) before the fix, so asserting
//     on return values alone passes either way;
//   - a network probe does not distinguish it either — against a server that
//     accepts the dial but never answers, go-redis drops the connection when the
//     command times out, so the peer sees EOF with or without our Close.
//
// Deleting a release call therefore has to fail here, or it fails nowhere.

func TestReleaseOnFailureReleasesOnlyWhenTheStepFails(t *testing.T) {
	stepErr := errors.New("setup failed")

	released := 0
	err := releaseOnFailure(func() error { return stepErr }, func() { released++ })
	require.ErrorIs(t, err, stepErr, "the step's error must reach the caller unchanged")
	assert.Equal(t, 1, released, "a failed step must release the client exactly once")

	released = 0
	err = releaseOnFailure(func() error { return nil }, func() { released++ })
	require.NoError(t, err)
	assert.Zero(t, released, "a successful step must keep the client open")
}

func TestNewRedisCacheClosesClientOnConnectFailure(t *testing.T) {
	closed := 0
	c, err := NewRedisCache(RedisCacheOptions{
		Enabled: true,
		Config:  refusedRedisConfig(),
		closeClient: func(client *redis.Client) {
			closed++
			_ = client.Close()
		},
	})
	require.Error(t, err, "unreachable Redis must fail the constructor")
	assert.Nil(t, c, "constructor must not return a client alongside its error")
	assert.Equal(t, 1, closed, "failed connection check must release the partially built client exactly once")
}

// Valkey's client builder connects eagerly, so a refused dial fails inside
// glide.NewClient and there is no client to release yet. Pinning that keeps the
// asymmetry with Redis explicit: the release path below the builder is covered
// by TestReleaseOnFailureReleasesOnlyWhenTheStepFails, and this test proves the
// pre-builder path cannot leak because nothing was built.
func TestNewValkeyCacheFailsBeforeBuildingAClientOnRefusedDial(t *testing.T) {
	closed := 0
	c, err := NewValkeyCache(ValkeyCacheOptions{
		Enabled: true,
		Config:  refusedValkeyConfig(),
		closeClient: func(client *glide.Client) {
			closed++
			client.Close()
		},
	})
	require.Error(t, err, "unreachable Valkey must fail the constructor")
	assert.Nil(t, c, "constructor must not return a client alongside its error")
	assert.Contains(t, err.Error(), "failed to create Valkey client",
		"a refused dial must fail in the client builder, before any client exists")
	assert.Zero(t, closed, "nothing was built, so nothing may be released")
}
