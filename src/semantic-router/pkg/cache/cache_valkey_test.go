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

func TestGetAvailableCacheBackendsIncludesValkey(t *testing.T) {
	backends := GetAvailableCacheBackends()
	if len(backends) != 4 {
		t.Fatalf("expected 4 backends, got %d", len(backends))
	}

	valkeyBackend := backends[3]
	if valkeyBackend.Type != ValkeyCacheType {
		t.Fatalf("expected backend type %q, got %q", ValkeyCacheType, valkeyBackend.Type)
	}
	if valkeyBackend.Name != "Valkey Vector Database" {
		t.Fatalf("unexpected backend name: %s", valkeyBackend.Name)
	}
	if len(valkeyBackend.Features) == 0 {
		t.Fatal("expected valkey backend features to be populated")
	}
}
