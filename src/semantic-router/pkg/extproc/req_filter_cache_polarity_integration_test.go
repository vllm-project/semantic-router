package extproc

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

// Exercise the real cache, adapter and service so a synthetic miss cannot hide
// a verifier failure that accidentally serves an unverified response.
func TestPolarityVerifierFailureThroughCacheAndExtProc(t *testing.T) {
	for _, unavailable := range []bool{false, true} {
		name := "verifier error"
		if unavailable {
			name = "missing verifier"
		}
		t.Run(name, func(t *testing.T) {
			provider, err := embedding.NewFuncProvider("test", 3, func(context.Context, string) ([]float32, error) {
				return []float32{1, 0, 0}, nil
			})
			require.NoError(t, err)
			backend := cache.NewInMemoryCache(cache.InMemoryCacheOptions{
				Enabled: true, MaxEntries: 16, SimilarityThreshold: 0.8,
				EmbeddingProvider: provider,
				PolarityGuard:     cache.PolarityGuardOptions{UseNLI: true, ContradictionThreshold: 0.5},
			})
			t.Cleanup(func() { _ = backend.Close() })
			service := cache.NewResponseCacheService(cache.NewLegacyBackendAdapter(backend, cache.InMemoryCacheType), cache.DefaultResponseCacheServiceOptions())
			decision := config.Decision{
				Name: "polarity-route", ModelRefs: []config.ModelRef{{Model: "m"}},
				Plugins: []config.DecisionPlugin{{Type: config.DecisionPluginResponseCache, Configuration: config.MustStructuredPayload(map[string]interface{}{
					"enabled": true, "scope": "global",
				})}},
			}
			cfg := &config.RouterConfig{}
			cfg.Enabled = true
			cfg.Decisions = []config.Decision{decision}
			router := &OpenAIRouter{Config: cfg, Cache: backend, ResponseCache: service}
			request := func(query string) *RequestContext {
				return &RequestContext{
					RequestID: query, StartTime: time.Now(), TraceContext: context.Background(),
					SemanticRequest: testNeutralRequest("test-model", query), VSRSelectedDecision: &decision,
				}
			}
			prime := request("How do I enable two-factor authentication?")
			router.handleCaching(prime, decision.Name)
			require.NoError(t, service.StoreSemantic(context.Background(), cache.CacheWrite{
				RequestID: "prime", Identity: prime.CacheIdentity,
				ResponseBody: []byte(`{"choices":[{"message":{"role":"assistant","content":"CACHED-ANSWER"}}]}`),
				TTL:          cache.TTL(time.Minute),
			}))

			calls := 0
			backend.SetPolarityVerifier(func(context.Context, string, string) (float32, error) {
				calls++
				return 0, errors.New("nli backend unavailable")
			})
			if unavailable {
				backend.SetPolarityVerifier(nil)
			}
			ctx := request("How can I enable two-factor authentication?")
			response, shortCircuit := router.handleCaching(ctx, decision.Name)
			require.Nil(t, response)
			require.False(t, shortCircuit, "an unverified candidate must continue upstream")
			require.False(t, ctx.VSRCacheHit)
			require.Equal(t, float32(1), ctx.VSRCacheSimilarity)
			if !unavailable {
				require.Equal(t, 1, calls, "the production lookup must invoke the failing verifier")
			}

			// Recover only this cache's verifier. The same candidate must now hit,
			// proving the miss was caused by verification rather than setup.
			backend.SetPolarityVerifier(func(context.Context, string, string) (float32, error) { return 0, nil })
			recovered := request("How can I enable two-factor authentication?")
			response, shortCircuit = router.handleCaching(recovered, decision.Name)
			require.True(t, shortCircuit)
			require.NotNil(t, response)
			require.True(t, recovered.VSRCacheHit)
		})
	}
}
