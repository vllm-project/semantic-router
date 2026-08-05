//go:build !windows && cgo

package cache

import (
	"testing"

	"github.com/redis/go-redis/v9"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// TestRedisExtractSearchResultUsesConfiguredMetric pins the wiring between the
// Redis backend and the shared distance conversion. The metric type is read at
// the call site rather than inside distanceToSimilarity, and several sibling
// string fields live on the same config struct, so passing the wrong one would
// compile cleanly and silently fall through to the default branch.
func TestRedisExtractSearchResultUsesConfiguredMetric(t *testing.T) {
	const distance = "0.75"

	tests := []struct {
		metric     string
		similarity float32
	}{
		{"COSINE", 0.625},  // 1 - d/2
		{"IP", 0.25},       // 1 - d
		{"L2", 1.0 / 1.75}, // 1 / (1 + d)
	}

	for _, tt := range tests {
		t.Run(tt.metric, func(t *testing.T) {
			cfg := &config.RedisConfig{}
			cfg.Index.VectorField.MetricType = tt.metric
			// A neighbouring string field with a different value: if the call
			// site ever reads this one instead, the assertions below break.
			cfg.Index.IndexType = "HNSW"

			cache := &RedisCache{config: cfg}
			similarity, responseBody, ok := cache.extractSearchResult(redis.Document{
				Fields: map[string]string{
					"vector_distance": distance,
					"response_body":   `{"result":"ok"}`,
				},
			})

			require.True(t, ok, "a document with both fields present must parse")
			assert.JSONEq(t, `{"result":"ok"}`, string(responseBody))
			assert.InDelta(t, tt.similarity, similarity, 0.001)
		})
	}
}

// TestRedisMetricTypeNormalization mirrors TestValkeyMetricTypeNormalization.
// initializeIndex matches the metric type against the uppercase spellings and
// falls back to a COSINE index otherwise, so a config that is not normalized
// builds one index and reads its distances back with another metric's formula.
func TestRedisMetricTypeNormalization(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{"lowercase cosine", "cosine", "COSINE"},
		{"lowercase l2", "l2", "L2"},
		{"mixed case Ip", "Ip", "IP"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &config.RedisConfig{}
			cfg.Connection.Host = "invalid-host-to-fail-fast"
			cfg.Connection.Port = 6379
			cfg.Index.Name = "test_idx"
			cfg.Index.Prefix = "doc:"
			cfg.Index.VectorField.Name = "embedding"
			cfg.Index.VectorField.MetricType = tt.input

			// NewRedisCache fails on CheckConnection, but normalization happens
			// before the connection attempt and mutates cfg in place.
			_, _ = NewRedisCache(RedisCacheOptions{
				Enabled: true,
				Config:  cfg,
			})

			assert.Equal(t, tt.expected, cfg.Index.VectorField.MetricType)
		})
	}
}

// TestRedisExtractSearchResultMissingDistance covers the parse failures that
// must be reported as "not ok" rather than as a similarity of zero, which the
// caller would otherwise store and expose as x-vsr-cache-similarity.
func TestRedisExtractSearchResultMissingDistance(t *testing.T) {
	cfg := &config.RedisConfig{}
	cfg.Index.VectorField.MetricType = "COSINE"
	cache := &RedisCache{config: cfg}

	t.Run("no vector_distance field", func(t *testing.T) {
		_, _, ok := cache.extractSearchResult(redis.Document{
			Fields: map[string]string{"response_body": `{"result":"ok"}`},
		})
		assert.False(t, ok)
	})

	t.Run("empty response_body", func(t *testing.T) {
		similarity, responseBody, ok := cache.extractSearchResult(redis.Document{
			Fields: map[string]string{"vector_distance": "0.0"},
		})
		assert.False(t, ok)
		assert.Nil(t, responseBody)
		assert.InDelta(t, float32(1.0), similarity, 0.001,
			"the similarity is still reported so the caller can log it")
	})
}
