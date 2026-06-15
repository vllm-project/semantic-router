//go:build !windows && cgo

package cache

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestValkeyCacheDisabled(t *testing.T) {
	cache, err := NewValkeyCache(ValkeyCacheOptions{
		Enabled: false,
	})
	assert.NoError(t, err)
	assert.NotNil(t, cache)
	assert.False(t, cache.IsEnabled())
}

func TestValkeyCacheConfigValidation(t *testing.T) {
	_, err := NewValkeyCache(ValkeyCacheOptions{
		Enabled: true,
		Config:  nil,
	})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "valkey config is required")
}

func TestValkeyCacheEmbeddingModel(t *testing.T) {
	valkeyConfig := &config.ValkeyConfig{}
	valkeyConfig.Connection.Host = "localhost"
	valkeyConfig.Connection.Port = 6379
	valkeyConfig.Index.Name = "test_index"
	valkeyConfig.Index.Prefix = "test:"
	valkeyConfig.Index.VectorField.Name = "embedding"
	valkeyConfig.Index.VectorField.MetricType = "COSINE"
	valkeyConfig.Index.IndexType = "HNSW"
	valkeyConfig.Index.Params.M = 16
	valkeyConfig.Index.Params.EfConstruction = 64
	valkeyConfig.Search.TopK = 1
	valkeyConfig.Development.AutoCreateIndex = true

	cache, err := NewValkeyCache(ValkeyCacheOptions{
		Enabled:             false,
		Config:              valkeyConfig,
		SimilarityThreshold: 0.8,
		TTLSeconds:          3600,
	})
	assert.NoError(t, err)
	assert.Empty(t, cache.embeddingModel)

	cache, err = NewValkeyCache(ValkeyCacheOptions{
		Enabled:             false,
		Config:              valkeyConfig,
		SimilarityThreshold: 0.8,
		TTLSeconds:          3600,
		EmbeddingModel:      "qwen3",
	})
	assert.NoError(t, err)
	assert.Empty(t, cache.embeddingModel)
}

func TestValkeyCacheGetStats(t *testing.T) {
	cache := &ValkeyCache{
		enabled:   false,
		hitCount:  10,
		missCount: 5,
	}

	stats := cache.GetStats()
	assert.Equal(t, int64(10), stats.HitCount)
	assert.Equal(t, int64(5), stats.MissCount)
	assert.Equal(t, 0.6666666666666666, stats.HitRatio)
}

func TestValkeyMetricTypeNormalization(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{"lowercase cosine", "cosine", "COSINE"},
		{"mixed case Cosine", "Cosine", "COSINE"},
		{"uppercase COSINE", "COSINE", "COSINE"},
		{"lowercase l2", "l2", "L2"},
		{"lowercase ip", "ip", "IP"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &config.ValkeyConfig{}
			cfg.Connection.Host = "invalid-host-to-fail-fast"
			cfg.Connection.Port = 6379
			cfg.Connection.Timeout = 1
			cfg.Index.Name = "test_idx"
			cfg.Index.Prefix = "doc:"
			cfg.Index.VectorField.Name = "embedding"
			cfg.Index.VectorField.MetricType = tt.input
			cfg.Index.IndexType = "HNSW"
			cfg.Index.Params.M = 16
			cfg.Index.Params.EfConstruction = 64
			cfg.Development.AutoCreateIndex = true

			// NewValkeyCache will fail to connect, but normalization
			// happens before the connection attempt, mutating cfg.
			_, _ = NewValkeyCache(ValkeyCacheOptions{
				Enabled: true,
				Config:  cfg,
			})
			assert.Equal(t, tt.expected, cfg.Index.VectorField.MetricType)
		})
	}
}

func TestDistanceToSimilarity(t *testing.T) {
	tests := []struct {
		name     string
		metric   string
		distance float64
		expected float32
	}{
		{"COSINE zero distance", "COSINE", 0.0, 1.0},
		{"COSINE max distance", "COSINE", 2.0, 0.0},
		{"IP passthrough", "IP", 0.75, 0.75},
		{"L2 zero distance", "L2", 0.0, 1.0},
		// Lowercase inputs hit the default branch (1 - distance).
		// After normalization these should never occur, but verify
		// the fallback is sane.
		{"lowercase cosine falls to default", "cosine", 0.0, 1.0},
		{"lowercase ip falls to default", "ip", 0.75, 0.25},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := distanceToSimilarity(tt.metric, tt.distance)
			assert.InDelta(t, tt.expected, result, 0.001)
		})
	}
}

func TestEscapeTagValue(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{"simple alphanumeric", "abc123", "abc123"},
		{"UUID with hyphens", "req-abc-123", `req\-abc\-123`},
		{"full UUID", "550e8400-e29b-41d4-a716-446655440000", `550e8400\-e29b\-41d4\-a716\-446655440000`},
		{"prefix with underscore", "req_test_123", "req_test_123"},
		{"dots", "v1.2.3", `v1\.2\.3`},
		{"colons", "ns:key:1", `ns\:key\:1`},
		{"spaces", "hello world", `hello\ world`},
		{"empty string", "", ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := escapeTagValue(tt.input)
			assert.Equal(t, tt.expected, result)
		})
	}
}
