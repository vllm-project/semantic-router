package cache

import (
	"context"
	"fmt"
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/internal/testutil/storagetest"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

func TestOmniCachePreservesFullPreparedDimension(t *testing.T) {
	for _, size := range []int{384, 768} {
		provider := storagetest.Vectors{Size: size}
		cache := NewInMemoryCache(InMemoryCacheOptions{Enabled: true, EmbeddingModel: "multimodal", EmbeddingProvider: provider})
		t.Cleanup(func() { _ = cache.Close() })
		vector, err := cache.computeEmbedding(context.Background(), "full vector")
		if err != nil || len(vector) != size {
			t.Fatalf("dimension %d: len=%d err=%v", size, len(vector), err)
		}
		settings, ok := LocalEmbeddingSettings(cache)
		if !ok || settings.Dimension != size || settings.Layer != 0 {
			t.Fatalf("wrong settings: %+v", settings)
		}
		if got, err := resolveCacheDimension(0, provider); err != nil || got != size {
			t.Fatalf("resolve=%d %v", got, err)
		}
		wrong := 384
		if size == 384 {
			wrong = 768
		}
		if _, err := resolveCacheDimension(wrong, provider); err == nil {
			t.Fatal("wrong explicit width accepted")
		}
	}
	if got, err := resolveCacheDimension(128, nil); err != nil || got != 128 {
		t.Fatalf("explicit exact-only schema rejected: %d %v", got, err)
	}
	if _, err := resolveCacheDimension(0, nil); err == nil {
		t.Fatal("missing semantic provider guessed a dimension")
	}
}

func TestOmniCacheIdentityResolvesDimensionAndIsolatesEqualWidthArtifacts(t *testing.T) {
	for _, backend := range []CacheBackendType{InMemoryCacheType, RedisCacheType, ValkeyCacheType, MilvusCacheType, HybridCacheType, QdrantCacheType} {
		for _, size := range []int{384, 768} {
			t.Run(fmt.Sprintf("%s-%d", backend, size), func(t *testing.T) {
				original := namespaceFixture(backend, 0)
				original.EmbeddingModel = "multimodal"
				original.EmbeddingProvider = storagetest.Vectors{Size: size}
				before := physicalNamespace(original)
				resolve := func(name string) func(embedding.ConsumerSettings) (embedding.ContentIdentity, error) {
					return func(settings embedding.ConsumerSettings) (embedding.ContentIdentity, error) {
						if settings.Dimension != size || settings.Layer != 0 || settings.ModelType != "multimodal" {
							t.Fatalf("unexpected settings: %+v", settings)
						}
						return embedding.ContentIdentity{Fingerprint: name, Descriptor: embedding.RuntimeDescriptor{Dimension: size}}, nil
					}
				}
				a, identity, err := PrepareEmbeddingNamespace(original, resolve("artifact-a"))
				if err != nil || identity != "artifact-a" {
					t.Fatal(err)
				}
				b, _, err := PrepareEmbeddingNamespace(original, resolve("artifact-b"))
				if err != nil {
					t.Fatal(err)
				}
				if backend != InMemoryCacheType && reflect.DeepEqual(physicalNamespace(a), physicalNamespace(b)) {
					t.Fatal("same-dimension different weights shared a persistent index")
				}
				if !reflect.DeepEqual(before, physicalNamespace(original)) {
					t.Fatal("logical config changed")
				}
				switch backend {
				case RedisCacheType:
					if a.Redis.Index.VectorField.Dimension != size || original.Redis.Index.VectorField.Dimension != 0 {
						t.Fatal("redis width not resolved independently")
					}
				case ValkeyCacheType:
					if a.Valkey.Index.VectorField.Dimension != size || original.Valkey.Index.VectorField.Dimension != 0 {
						t.Fatal("valkey width not resolved independently")
					}
				case MilvusCacheType, HybridCacheType:
					if a.Milvus.Collection.VectorField.Dimension != size || original.Milvus.Collection.VectorField.Dimension != 0 {
						t.Fatal("milvus width not resolved independently")
					}
				}
			})
		}
	}
	bad := namespaceFixture(RedisCacheType, 384)
	bad.EmbeddingModel = "multimodal"
	bad.EmbeddingProvider = storagetest.Vectors{Size: 768}
	if _, _, err := PrepareEmbeddingNamespace(bad, func(embedding.ConsumerSettings) (embedding.ContentIdentity, error) {
		return embedding.ContentIdentity{Fingerprint: "mini", Descriptor: embedding.RuntimeDescriptor{Dimension: 768}}, nil
	}); err == nil {
		t.Fatal("explicit wrong Mini width accepted")
	}
}
