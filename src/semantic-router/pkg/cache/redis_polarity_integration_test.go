//go:build !windows && cgo && !riscv64

package cache

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"testing"
	"time"

	"github.com/redis/go-redis/v9"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/internal/testutil/storagetest"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// StorageIntegration: redis
func TestRedisSemanticPolarityIntegration(t *testing.T) {
	storagetest.Require(t, "redis")
	host := os.Getenv("REDIS_HOST")
	if host == "" {
		host = "localhost"
	}
	port := 6379
	if raw := os.Getenv("REDIS_PORT"); raw != "" {
		value, err := strconv.Atoi(raw)
		require.NoError(t, err)
		port = value
	}
	cfg := &config.RedisConfig{}
	cfg.Connection.Host, cfg.Connection.Port = host, port
	cfg.Connection.Timeout = 5
	cfg.Index.Name = fmt.Sprintf("polarity_redis_%d", time.Now().UnixNano())
	cfg.Index.Prefix = cfg.Index.Name + ":"
	cfg.Index.VectorField.Name, cfg.Index.VectorField.Dimension = "embedding", 384
	cfg.Index.VectorField.MetricType, cfg.Index.IndexType = "COSINE", "FLAT"
	cfg.Search.TopK = 4
	cfg.Development.AutoCreateIndex = true
	backend, err := NewRedisCache(RedisCacheOptions{
		Enabled: true, Config: cfg, TTLSeconds: 60, SimilarityThreshold: .8,
		EmbeddingModel: "bert", EmbeddingProvider: redisValkeyPolarityVectors(),
	})
	if err != nil {
		storagetest.Unavailable(t, "redis", fmt.Sprintf("Redis vector search unavailable: %v", err))
	}
	t.Cleanup(func() {
		_ = backend.client.FTDropIndexWithArgs(context.Background(), cfg.Index.Name, &redis.FTDropIndexOptions{DeleteDocs: true}).Err()
		_ = backend.Close()
	})
	runRedisValkeyStoredPolarityCases(t, backend, RedisCacheType)
}
