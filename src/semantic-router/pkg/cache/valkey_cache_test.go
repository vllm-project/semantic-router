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
