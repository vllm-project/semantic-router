package cache

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

func physicalNamespace(cfg CacheConfig) []string {
	switch normalizedBackendType(cfg.BackendType) {
	case RedisCacheType:
		return []string{cfg.Redis.Index.Name, cfg.Redis.Index.Prefix}
	case ValkeyCacheType:
		return []string{cfg.Valkey.Index.Name, cfg.Valkey.Index.Prefix}
	case MilvusCacheType, HybridCacheType:
		return []string{cfg.Milvus.Collection.Name}
	case QdrantCacheType:
		return []string{cfg.Qdrant.CollectionName}
	default:
		return nil
	}
}

func namespaceFixture(backend CacheBackendType, dimension int) CacheConfig {
	cfg := CacheConfig{Enabled: true, BackendType: backend, EmbeddingModel: "mmbert"}
	switch backend {
	case RedisCacheType:
		cfg.Redis = &config.RedisConfig{}
		cfg.Redis.Index.Name = "original_idx"
		cfg.Redis.Index.Prefix = "original:"
		cfg.Redis.Index.VectorField.Dimension = dimension
		cfg.Redis.Development.DropIndexOnStartup = true
	case ValkeyCacheType:
		cfg.Valkey = &config.ValkeyConfig{}
		cfg.Valkey.Index.Name = "original_idx"
		cfg.Valkey.Index.Prefix = "original:"
		cfg.Valkey.Index.VectorField.Dimension = dimension
		cfg.Valkey.Development.DropIndexOnStartup = true
	case MilvusCacheType, HybridCacheType:
		cfg.Milvus = &config.MilvusConfig{}
		cfg.Milvus.Collection.Name = "original_collection"
		cfg.Milvus.Collection.VectorField.Dimension = dimension
		cfg.Milvus.Development.DropCollectionOnStartup = true
	case QdrantCacheType:
		cfg.Qdrant = &config.QdrantConfig{Host: "localhost", CollectionName: "original_collection"}
	}
	return cfg
}

func TestEmbeddingNamespaceIsolatesPhysicalIndexesAndPrefixesBeforeOpening(t *testing.T) {
	for _, backend := range []CacheBackendType{RedisCacheType, ValkeyCacheType, MilvusCacheType, HybridCacheType, QdrantCacheType} {
		t.Run(string(backend), func(t *testing.T) {
			original := namespaceFixture(backend, 768)
			logical := physicalNamespace(original)
			resolve := func(settings embedding.ConsumerSettings) (embedding.ContentIdentity, error) {
				if settings.Layer != 0 || settings.Dimension != 768 {
					t.Fatalf("wrong effective settings: %#v", settings)
				}
				return embedding.ContentIdentity{Fingerprint: "old-weights-768"}, nil
			}
			first, identity, err := PrepareEmbeddingNamespace(original, resolve)
			if err != nil || identity == "" {
				t.Fatalf("bind: %v", err)
			}
			again, _, err := PrepareEmbeddingNamespace(original, resolve)
			if err != nil || !reflect.DeepEqual(physicalNamespace(first), physicalNamespace(again)) {
				t.Fatal("same identity was not stable")
			}
			changed, _, err := PrepareEmbeddingNamespace(original, func(embedding.ConsumerSettings) (embedding.ContentIdentity, error) {
				return embedding.ContentIdentity{Fingerprint: "new-weights-768"}, nil
			})
			if err != nil {
				t.Fatal(err)
			}
			for index, name := range physicalNamespace(first) {
				if name == logical[index] || name == physicalNamespace(changed)[index] {
					t.Fatal("old/new physical namespace reused")
				}
			}
			if !reflect.DeepEqual(logical, physicalNamespace(original)) {
				t.Fatal("user configuration mutated")
			}
			if backend != QdrantCacheType {
				reduced, _, err := PrepareEmbeddingNamespace(namespaceFixture(backend, 256), func(settings embedding.ConsumerSettings) (embedding.ContentIdentity, error) {
					if settings.Dimension != 256 {
						t.Fatalf("reduced dimension ignored: %#v", settings)
					}
					return embedding.ContentIdentity{Fingerprint: fmt.Sprintf("new-weights-%d", settings.Dimension)}, nil
				})
				if err != nil {
					t.Fatal(err)
				}
				if reflect.DeepEqual(physicalNamespace(changed), physicalNamespace(reduced)) {
					t.Fatal("dimension change opened old fixed-shape collection")
				}
			}
			if first.Redis != nil && (!first.Redis.Development.DropIndexOnStartup || !original.Redis.Development.DropIndexOnStartup) {
				t.Fatal("drop flag/config mutation")
			}
			if first.Valkey != nil && (!first.Valkey.Development.DropIndexOnStartup || !original.Valkey.Development.DropIndexOnStartup) {
				t.Fatal("drop flag/config mutation")
			}
			if first.Milvus != nil && (!first.Milvus.Development.DropCollectionOnStartup || !original.Milvus.Development.DropCollectionOnStartup) {
				t.Fatal("drop flag/config mutation")
			}
		})
	}
}

func TestEmbeddingNamespaceLeavesUnsupportedProvidersAndDisabledCachesUntouched(t *testing.T) {
	for _, enabled := range []bool{false, true} {
		cfg := namespaceFixture(RedisCacheType, 768)
		cfg.Enabled = enabled
		cfg.EmbeddingModel = "bert"
		got, identity, err := PrepareEmbeddingNamespace(cfg, func(embedding.ConsumerSettings) (embedding.ContentIdentity, error) {
			t.Fatal("unsupported provider resolved")
			return embedding.ContentIdentity{}, nil
		})
		if err != nil || identity != "" || !reflect.DeepEqual(got, cfg) {
			t.Fatal("legacy behavior changed")
		}
	}
}
