package extproc

import (
	"strings"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/authz"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/fallback"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func testNeutralRequest(model, text string) *llmprotocol.Request {
	return &llmprotocol.Request{
		Generation: 1,
		Model:      model,
		Messages: []llmprotocol.Message{{
			Role:    llmprotocol.RoleUser,
			Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: text}},
		}},
	}
}

func headerValuesByName(headers []*core.HeaderValueOption) map[string]string {
	result := make(map[string]string, len(headers))
	for _, header := range headers {
		result[header.Header.Key] = string(header.Header.RawValue)
	}
	return result
}

func immediateHeaderValue(response *ext_proc.ProcessingResponse, key string) string {
	if response == nil || response.GetImmediateResponse() == nil || response.GetImmediateResponse().Headers == nil {
		return ""
	}
	for _, header := range response.GetImmediateResponse().Headers.SetHeaders {
		if strings.EqualFold(header.Header.Key, key) {
			if len(header.Header.RawValue) > 0 {
				return string(header.Header.RawValue)
			}
			return header.Header.Value
		}
	}
	return ""
}

// CreateTestRouter creates a properly initialized router for testing.
func CreateTestRouter(cfg *config.RouterConfig) (*OpenAIRouter, error) {
	classifierCfg := cloneRouterConfigForTest(cfg)
	semanticCache, err := newTestSemanticCache(classifierCfg)
	if err != nil {
		return nil, err
	}

	classifier, err := classification.NewClassifier(classifierCfg)
	if err != nil {
		_ = semanticCache.Close()
		return nil, err
	}

	fallbackPolicy := fallback.DefaultPolicy()
	circuitBreaker := fallback.NewBackendCircuitBreaker(fallbackPolicy.CircuitBreaker)
	orchestrator := fallback.NewOrchestrator(fallbackPolicy, circuitBreaker)

	return &OpenAIRouter{
		Config:               cfg,
		CategoryDescriptions: cfg.GetCategoryDescriptions(),
		Classifier:           classifier,
		Cache:                semanticCache,
		ResponseAPIFilter:    newTestResponseAPIFilter(cfg),
		CredentialResolver:   newTestCredentialResolver(cfg),
		FallbackOrchestrator: orchestrator,
	}, nil
}

func cloneRouterConfigForTest(cfg *config.RouterConfig) *config.RouterConfig {
	if cfg == nil {
		return nil
	}
	clone := *cfg
	return &clone
}

func newTestSemanticCache(cfg *config.RouterConfig) (cache.CacheBackend, error) {
	return cache.NewCacheBackend(cache.CacheConfig{
		BackendType:         cache.InMemoryCacheType,
		Enabled:             cfg.Enabled,
		SimilarityThreshold: cfg.GetCacheSimilarityThreshold(),
		MaxEntries:          cfg.MaxEntries,
		TTLSeconds:          cfg.TTLSeconds,
		EvictionPolicy:      cache.EvictionPolicyType(cfg.EvictionPolicy),
	})
}

func newTestResponseAPIFilter(cfg *config.RouterConfig) *ResponseAPIFilter {
	if !cfg.ResponseAPI.Enabled {
		return nil
	}
	return NewResponseAPIFilter(NewMockResponseStore())
}

func newTestCredentialResolver(cfg *config.RouterConfig) *authz.CredentialResolver {
	credResolver := authz.NewCredentialResolver(
		authz.NewHeaderInjectionProvider(authz.DefaultHeaderMap()),
		authz.NewStaticConfigProvider(cfg),
	)
	credResolver.SetFailOpen(true)
	return credResolver
}
