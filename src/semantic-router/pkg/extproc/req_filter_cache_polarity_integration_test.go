package extproc

import (
	"context"
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
