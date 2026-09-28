package extproc

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel/attribute"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"go.uber.org/zap/zaptest/observer"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/tracing"
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

// The lexical guard accepts German and Chinese negations, so a served hit must
// report whether the guard could judge its pair.
func TestSemanticHitReportsNegationGuardScope(t *testing.T) {
	cases := []struct {
		name, stored, incoming, want string
	}{
		{"same words", "How do I enable two-factor authentication?", "how do I enable two-factor authentication", "checked"},
		{"German negation", "Ist es sicher, Ibuprofen mit Alkohol zu nehmen?", "Ist es nicht sicher, Ibuprofen mit Alkohol zu nehmen?", "not_applicable"},
		{"Chinese negation", "这个药可以和酒一起吃吗？", "这个药不可以和酒一起吃吗？", "not_applicable"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			recorder := routingTraceRecorder(t)
			logCore, logs := observer.New(zapcore.InfoLevel)
			t.Cleanup(zap.ReplaceGlobals(zap.New(logCore)))
			provider, err := embedding.NewFuncProvider("test", 3, func(context.Context, string) ([]float32, error) {
				return []float32{1, 0, 0}, nil
			})
			require.NoError(t, err)
			backend := cache.NewInMemoryCache(cache.InMemoryCacheOptions{
				Enabled: true, MaxEntries: 16, SimilarityThreshold: 0.8, EmbeddingProvider: provider,
			})
			t.Cleanup(func() { _ = backend.Close() })
			service := cache.NewResponseCacheService(cache.NewLegacyBackendAdapter(backend, cache.InMemoryCacheType), cache.DefaultResponseCacheServiceOptions())
			decision := config.Decision{
				Name: "negation-scope-route", ModelRefs: []config.ModelRef{{Model: "m"}},
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
			prime := request(tc.stored)
			router.handleCaching(prime, decision.Name)
			require.NoError(t, service.StoreSemantic(context.Background(), cache.CacheWrite{
				RequestID: "prime", Identity: prime.CacheIdentity,
				ResponseBody: []byte(`{"choices":[{"message":{"role":"assistant","content":"CACHED-ANSWER"}}]}`),
				TTL:          cache.TTL(time.Minute),
			}))

			ctx := request(tc.incoming)
			_, shortCircuit := router.handleCaching(ctx, decision.Name)
			require.True(t, shortCircuit, "the lexical guard accepts this pair, so the cached answer is served")

			var attrs map[string]attribute.Value
			for _, span := range recorder.Ended() {
				if spanAttrs := traceAttributes(span); spanAttrs[tracing.AttrPluginResult].AsString() == "cache_hit" {
					attrs = spanAttrs
				}
			}
			require.NotNil(t, attrs, "the served lookup must end a response_cache span")
			require.Equal(t, tc.want, attrs["cache.negation_guard"].AsString(), "response_cache span attributes: %v", attrs)

			var hitFields map[string]interface{}
			for _, entry := range logs.FilterMessage("cache_hit").All() {
				if fields := entry.ContextMap(); fields["request_id"] == ctx.RequestID {
					hitFields = fields
				}
			}
			require.Equal(t, tc.want, hitFields["negation_guard"], "cache_hit event fields: %v", hitFields)
		})
	}
}
