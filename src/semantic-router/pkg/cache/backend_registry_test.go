package cache

import (
	"testing"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestBackendRegistryCapabilityMatrix(t *testing.T) {
	for _, backend := range []CacheBackendType{
		InMemoryCacheType,
		RedisCacheType,
		ValkeyCacheType,
		MilvusCacheType,
		QdrantCacheType,
		HybridCacheType,
	} {
		capabilities := CapabilitiesForBackend(backend)
		if capabilities.Backend != backend {
			t.Fatalf("%s capabilities backend = %s", backend, capabilities.Backend)
		}
		if !capabilities.Exact || !capabilities.Semantic || !capabilities.PerEntryTTL {
			t.Fatalf("%s capabilities are incomplete: %#v", backend, capabilities)
		}
	}
}

func TestBackendRegistryRejectsUnselectedBackendConfiguration(t *testing.T) {
	config := GetDefaultCacheConfig()
	config.BackendType = InMemoryCacheType
	config.Redis = &routerconfig.RedisConfig{}
	if err := validateRegisteredBackend(config); err == nil {
		t.Fatal("validateRegisteredBackend() accepted unselected redis fields")
	}
}
