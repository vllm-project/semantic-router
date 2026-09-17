package extproc

import (
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

func TestResponseCacheBindsActualLocalEmbeddingAfterInitialization(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.EmbeddingModels.MmBertModelPath = "models/requested"
	cfg.EmbeddingModels.UseCPU = true
	backend := cache.NewInMemoryCache(cache.InMemoryCacheOptions{Enabled: true, EmbeddingModel: "mmbert"})
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
		t.Fatal("enabled mmbert adopted an untagged cache after descriptor failure")
	}
}

func TestResponseCacheDoesNotInventIdentityForOtherProviders(t *testing.T) {
	cfg := &config.RouterConfig{}
	for _, model := range []string{"bert", "gemma", "qwen3", "multimodal"} {
		backend := cache.NewInMemoryCache(cache.InMemoryCacheOptions{Enabled: true, EmbeddingModel: model})
		identity, err := responseCacheEmbeddingIdentity(cfg, backend, func(embedding.ConsumerSettings) (embedding.ContentIdentity, error) {
			t.Fatal("unsupported provider was initialized for identity")
			return embedding.ContentIdentity{}, nil
		})
		if identity != "" || err != nil {
			t.Fatalf("%s changed legacy behavior: %s %v", model, identity, err)
		}
	}
}
