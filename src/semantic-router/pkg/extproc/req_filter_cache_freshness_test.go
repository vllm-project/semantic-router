package extproc

import (
	"context"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Fixed lookup metadata avoids sleeps, embeddings, and external services.
type discoveryFreshnessBackend struct {
	mockStreamingCache
	result                    cache.LookupResult
	exactCalls, semanticCalls int
}

func (b *discoveryFreshnessBackend) FindExact(context.Context, string, string) (cache.LookupResult, error) {
	b.exactCalls++
	return b.result, nil
}

func (b *discoveryFreshnessBackend) LookupSimilarWithThreshold(context.Context, string, string, float32) (cache.LookupResult, error) {
	b.semanticCalls++
	return b.result, nil
}

func TestDiscoveryCacheRequestFreshness(t *testing.T) {
	for _, mode := range []string{"exact", "semantic", "exact_then_semantic"} {
		for _, tc := range []struct {
			name      string
			age       time.Duration
			directive string
			wantHit   bool
		}{
			{"fresh_control", time.Second, "max-age=10", true},
			{"unbounded_control", time.Minute, "", true},
			{"old_entry", time.Minute, "max-age=10", false},
		} {
			t.Run(mode+"/"+tc.name, func(t *testing.T) {
				backend := &discoveryFreshnessBackend{result: cache.LookupResult{
					Found: true, ResponseBody: []byte(exactCacheHitBody), Similarity: 1,
					Age: tc.age, AgeKnown: true, ExpiresAt: time.Now().Add(time.Hour),
				}}
				decision := config.Decision{
					Name:      "freshness-probe",
					ModelRefs: []config.ModelRef{{Model: "gpt-4o"}},
					Plugins: []config.DecisionPlugin{{
						Type: config.DecisionPluginResponseCache,
						Configuration: config.MustStructuredPayload(map[string]interface{}{
							"enabled": true, "mode": mode,
							"request_controls": map[string]interface{}{
								"enabled": true, "allowed": []string{"max-age"},
							},
						}),
					}},
				}
				router := &OpenAIRouter{
					Cache: backend,
					Config: &config.RouterConfig{
						SemanticCache:      config.SemanticCache{Enabled: true},
						IntelligentRouting: config.IntelligentRouting{Decisions: []config.Decision{decision}},
					},
					ResponseCache: cache.NewResponseCacheService(
						cache.NewLegacyBackendAdapter(backend, cache.InMemoryCacheType),
						cache.ResponseCacheServiceOptions{L1MaxEntries: -1},
					),
				}
				ctx := &RequestContext{
					Headers: map[string]string{
						"x-authz-user-id":         "cache-test-user",
						defaultCacheControlHeader: tc.directive,
					},
					RequestID:           "freshness-probe",
					SemanticRequest:     testNeutralRequest("gpt-4o", "hello"),
					VSRSelectedDecision: &router.Config.Decisions[0],
				}
				response, hit := router.handleCaching(ctx, decision.Name, "gpt-4o")
				if tc.directive != "" && (ctx.CacheMaxAgeSeconds == nil || *ctx.CacheMaxAgeSeconds != 10) {
					t.Fatal("request freshness directive was not applied")
				}
				t.Logf("hit=%v kind=%s age_seconds=%v exact_calls=%d semantic_calls=%d", hit, ctx.VSRCacheHitKind, ctx.VSRCacheEntryAgeSeconds, backend.exactCalls, backend.semanticCalls)
				if hit != tc.wantHit {
					t.Errorf("cache hit=%v, want %v for entry age=%s and directive=%q", hit, tc.wantHit, tc.age, tc.directive)
				}
				if hit && response.GetImmediateResponse() == nil {
					t.Error("cache hit did not produce an immediate response")
				}
			})
		}
	}
}
