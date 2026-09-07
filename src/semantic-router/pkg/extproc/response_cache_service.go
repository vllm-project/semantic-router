package extproc

import (
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
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
	backendType, options := r.responseCacheServiceConfig()
	r.ResponseCache = cache.NewResponseCacheService(
		cache.NewLegacyBackendAdapter(r.Cache, backendType),
		options,
	)
	return r.ResponseCache
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
