package extproc

import (
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

func (r *OpenAIRouter) responseCacheService() *cache.ResponseCacheService {
	if r == nil || r.Cache == nil {
		return nil
	}
	r.responseCacheMu.Lock()
	defer r.responseCacheMu.Unlock()
	if r.ResponseCache != nil {
		return r.ResponseCache
	}
	service, err := newResponseCacheService(r.Config, r.Cache, "", r.Embeddings)
	if err != nil {
		logging.Errorf("Response cache content identity unavailable: %v", err)
		return nil
	}
	r.ResponseCache = service
	return service
}

func newResponseCacheService(cfg *config.RouterConfig, backend cache.CacheBackend, boundIdentity string, embeddings *embedding.Set) (*cache.ResponseCacheService, error) {
	backendType, options := (&OpenAIRouter{Config: cfg}).responseCacheServiceConfig()
	var provider embedding.Provider
	if cfg != nil && cfg.NeedsSemanticResponseCache() && embeddings != nil {
		var err error
		provider, err = embeddings.Get(detectSemanticCacheEmbeddingModel(cfg), 0, 0)
		if err != nil && backend != nil && backend.IsEnabled() {
			return nil, err
		}
	}
	identity := boundIdentity
	if identity == "" && (cfg == nil || cfg.NeedsSemanticResponseCache()) {
		var err error
		identity, err = responseCacheEmbeddingIdentity(cfg, backend, func(settings embedding.ConsumerSettings) (embedding.ContentIdentity, error) {
			return embedding.ResolveProviderIdentity(provider, settings)
		})
		if err != nil {
			return nil, err
		}
	}
	options.EmbeddingIdentity = identity
	adapter := cache.NewLegacyBackendAdapter(backend, backendType).WithEmbeddingProvider(provider)
	if cfg != nil {
		adapter.WithEmbeddingModel(detectSemanticCacheEmbeddingModel(cfg))
	}
	return cache.NewResponseCacheService(adapter, options), nil
}

type localEmbeddingIdentityInitializer func(embedding.ConsumerSettings) (embedding.ContentIdentity, error)

func responseCacheEmbeddingIdentity(cfg *config.RouterConfig, backend cache.CacheBackend, initialize localEmbeddingIdentityInitializer) (string, error) {
	if cfg == nil || backend == nil || !backend.IsEnabled() {
		return "", nil
	}
	settings, supported := cache.LocalEmbeddingSettings(backend)
	if !supported {
		return "", nil
	}
	identity, err := initialize(settings)
	if err != nil {
		return "", fmt.Errorf("initialize semantic cache embedding identity: %w", err)
	}
	if identity.Fingerprint == "" {
		return "", fmt.Errorf("semantic cache embedding identity is empty")
	}
	return identity.Fingerprint, nil
}

func (r *OpenAIRouter) responseCacheServiceConfig() (
	cache.CacheBackendType,
	cache.ResponseCacheServiceOptions,
) {
	options := cache.DefaultResponseCacheServiceOptions()
	if r.Config == nil {
		return cache.InMemoryCacheType, options
	}
	store := r.Config.SemanticCache
	options.L1MaxEntries = boundedL1Entries(store.MaxEntries, options.L1MaxEntries)
	options.L1TTL = boundedL1TTL(store.TTLSeconds, options.L1TTL)
	return cache.CacheBackendType(store.BackendType), options
}

func boundedL1Entries(configured, maximum int) int {
	if configured > 0 && configured < maximum {
		return configured
	}
	return maximum
}

func boundedL1TTL(configuredSeconds int, maximum time.Duration) time.Duration {
	if configuredSeconds <= 0 {
		return maximum
	}
	configured := time.Duration(configuredSeconds) * time.Second
	if configured < maximum {
		return configured
	}
	return maximum
}
