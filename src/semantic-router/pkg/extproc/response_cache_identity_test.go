package extproc

import (
	"context"
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

func TestResponseCacheBindsActualLocalEmbeddingAfterInitialization(t *testing.T) {
	cfg := &config.RouterConfig{}
	backend := cache.NewInMemoryCache(cache.InMemoryCacheOptions{Enabled: true, EmbeddingDeployment: config.EmbeddingDefaultDeployment, EmbeddingDimension: 256, EmbeddingLayer: 6})
	called := false
	identity, err := responseCacheEmbeddingIdentity(cfg, backend, func(settings embedding.ConsumerSettings) (embedding.ContentIdentity, error) {
		called = true
		if settings.Layer != 6 || settings.Dimension != 256 || settings.InputPolicy == "" {
			t.Fatalf("wrong effective consumer: %#v", settings)
		}
		// The actual loader, not requested path, supplies identity.
		return embedding.ContentIdentity{Fingerprint: "actually-loaded-space"}, nil
	})
	if err != nil || !called || identity != "actually-loaded-space" {
		t.Fatalf("bind: %s %v called=%v", identity, err, called)
	}
	if _, err = responseCacheEmbeddingIdentity(cfg, backend, func(embedding.ConsumerSettings) (embedding.ContentIdentity, error) {
		return embedding.ContentIdentity{}, errors.New("unavailable")
	}); err == nil {
		t.Fatal("an enabled cache adopted an untagged namespace after descriptor failure")
	}
}

// A provider without a content identity (a remote endpoint) leaves the cache
// untagged.
func TestResponseCacheDoesNotInventIdentityForRemoteProviders(t *testing.T) {
	cfg := &config.RouterConfig{}
	backend := cache.NewInMemoryCache(cache.InMemoryCacheOptions{Enabled: true, EmbeddingDeployment: "remote-embedding"})
	identity, err := responseCacheEmbeddingIdentity(cfg, backend, func(embedding.ConsumerSettings) (embedding.ContentIdentity, error) {
		return embedding.ContentIdentity{}, embedding.ErrIdentityUnsupported
	})
	if identity != "" || err != nil {
		t.Fatalf("a remote provider tagged the cache: %s %v", identity, err)
	}
}

func TestResponseCacheBindsPreparedOmniRepresentation(t *testing.T) {
	for _, size := range []int{384, 768} {
		provider, err := embedding.NewFuncProvider("test", size, func(context.Context, string) ([]float32, error) {
			t.Fatal("identity resolution ran inference")
			return nil, nil
		})
		if err != nil {
			t.Fatal(err)
		}
		backend := cache.NewInMemoryCache(cache.InMemoryCacheOptions{Enabled: true, EmbeddingDeployment: "omni", EmbeddingProvider: provider})
		t.Cleanup(func() { _ = backend.Close() })
		identity, err := responseCacheEmbeddingIdentity(&config.RouterConfig{}, backend, func(settings embedding.ConsumerSettings) (embedding.ContentIdentity, error) {
			if settings.Dimension != size || settings.Layer != 0 || settings.ModelType != "omni" {
				t.Fatalf("wrong Omni representation: %+v", settings)
			}
			return embedding.ContentIdentity{Fingerprint: "prepared-artifact", Descriptor: embedding.RuntimeDescriptor{Dimension: size}}, nil
		})
		if err != nil || identity != "prepared-artifact" {
			t.Fatalf("identity=%s err=%v", identity, err)
		}
	}
}
