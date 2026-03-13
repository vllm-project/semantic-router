//go:build !windows && cgo

package cache

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestValidateCacheConfigAcceptsValkeyInlineRedisConfig(t *testing.T) {
	redisConfig := &config.RedisConfig{}
	redisConfig.Connection.Host = "valkey.default.svc"
	redisConfig.Connection.Port = 6379

	cacheConfig := CacheConfig{
		BackendType:         ValkeyCacheType,
		Enabled:             true,
		SimilarityThreshold: 0.8,
		TTLSeconds:          3600,
		EmbeddingModel:      "bert",
		Redis:               redisConfig,
	}

	if err := ValidateCacheConfig(cacheConfig); err != nil {
		t.Fatalf("expected valkey config to validate, got error: %v", err)
	}
}
