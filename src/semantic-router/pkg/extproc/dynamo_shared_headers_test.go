package extproc

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

var sharedProviderHeaders = []string{
	headers.DynamoTenantID, headers.DynamoWorkerInstanceIDLegacy,
	headers.DynamoPrefillInstanceIDLegacy, headers.DynamoDPRankLegacy,
	headers.DynamoDataParallelRankLegacy, headers.DynamoPrefillDPRankLegacy,
}

func TestSharedHeadersAllowNonDynamoDispatch(t *testing.T) {
	for _, name := range sharedProviderHeaders {
		for _, source := range []string{"client", "profile"} {
			t.Run(name+"/"+source, func(t *testing.T) {
				value := "provider-specific-value"
				if name == headers.DynamoTenantID {
					value = strings.Repeat("t", llmprotocol.DefaultPolicy().Limits.DynamoNVExtStringBytes+1)
				}
				router, model := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
				router.Config.VLLMEndpoints[0].Type = "vllm"
				request := testNeutralRequest(model, "hello")
				ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
				if source == "client" {
					ctx.Headers = map[string]string{strings.ToUpper(name): value}
				} else {
					profile := router.Config.ProviderProfiles["provider"]
					profile.ExtraHeaders = map[string]string{name: value}
					router.Config.ProviderProfiles["provider"] = profile
				}
				if err := validateDynamoRoutingHeaders(ctx); err != nil {
					t.Fatal(err)
				}
				if err := validateDynamoBackendPool(router.Config, model, ctx, ctx.ProtocolEnvelope); err != nil {
					t.Fatal(err)
				}
				dispatch, err := router.prepareProviderDispatch(request, model, "", false, ctx)
				if err != nil {
					t.Fatal(err)
				}
				response := router.buildProviderDispatchResponse(dispatch, ctx)
				finalized, finalizeErr := router.finalizeProviderDispatchResponse(dispatch, response, ctx)
				if finalizeErr != nil {
					t.Fatal(finalizeErr)
				}
				mutation := finalized.GetRequestBody().GetResponse().GetHeaderMutation()
				effective, snapshotErr := snapshotEffectiveDynamoRoutingHeaders(ctx, mutation)
				if snapshotErr != nil || len(effective) != 0 {
					t.Fatalf("shared header treated as Dynamo: %v, %v", effective, snapshotErr)
				}
				got := headerValueCI(ctx, name)
				for _, removed := range mutation.GetRemoveHeaders() {
					if strings.EqualFold(removed, name) {
						t.Fatalf("shared header removed: %s", name)
					}
				}
				for _, option := range mutation.GetSetHeaders() {
					if strings.EqualFold(option.GetHeader().GetKey(), name) {
						got = string(option.GetHeader().GetRawValue())
					}
				}
				if got != value {
					t.Fatalf("header value changed: got %q, want %q", got, value)
				}
			})
		}
	}
}

func TestSharedHeadersAllowResponseCache(t *testing.T) {
	for _, name := range sharedProviderHeaders {
		t.Run(name, func(t *testing.T) {
			decision := config.Decision{Name: "shared-header-cache", Plugins: []config.DecisionPlugin{{
				Type:          config.DecisionPluginResponseCache,
				Configuration: config.MustStructuredPayload(map[string]interface{}{"enabled": true, "mode": "exact_then_semantic", "scope": "global"}),
			}}}
			cache := &mockStreamingCache{exactHit: true, exactResponse: []byte(exactCacheHitBody)}
			router := &OpenAIRouter{Cache: cache, Config: &config.RouterConfig{
				SemanticCache:      config.SemanticCache{Enabled: true},
				IntelligentRouting: config.IntelligentRouting{Decisions: []config.Decision{decision}},
			}}
			ctx := &RequestContext{
				Headers:             map[string]string{name: "provider-value"},
				SemanticRequest:     testNeutralRequest("model-a", "hello"),
				VSRSelectedDecision: &decision,
			}
			response, hit := router.handleCaching(ctx, decision.Name, "model-a")
			if !hit || response.GetImmediateResponse() == nil {
				t.Fatal("expected shared-header request to hit the cache")
			}
			if !cache.exactFindCalled || ctx.CacheReadBypass || ctx.CacheWriteBypass {
				t.Fatalf("shared header bypassed cache: lookup=%v read=%v write=%v", cache.exactFindCalled, ctx.CacheReadBypass, ctx.CacheWriteBypass)
			}
		})
	}
}

func TestSharedHeadersAllowShadowDispatch(t *testing.T) {
	for _, name := range sharedProviderHeaders {
		for _, source := range []string{"client", "profile"} {
			t.Run(name+"/"+source, func(t *testing.T) {
				backend := newShadowTestBackend(t)
				router, model := newShadowTestRouter(t, backend)
				if source == "profile" {
					profile := router.Config.ProviderProfiles["provider"]
					profile.ExtraHeaders = map[string]string{name: "provider-value"}
					router.Config.ProviderProfiles["provider"] = profile
				}
				plugin := shadowTestPluginConfig()
				plugin.ForwardHeaders = []string{name}
				run := runShadowRequest(t, router, model, plugin, func(ctx *RequestContext) {
					if source == "client" {
						ctx.Headers[name] = "provider-value"
					}
				})
				waitForShadow(t, router)
				if backend.requestCount() != 1 {
					t.Fatalf("shadow requests = %d, want 1", backend.requestCount())
				}
				if outcome := singleShadowOutcome(t, run); outcome.Verdict != shadowVerdictCompleted {
					t.Fatalf("outcome = %+v", outcome)
				}
				// Primary provider headers do not belong to the shadow provider.
				// Their presence must allow shadow dispatch without copying them.
				if got := backend.headers[0].Get(name); source == "profile" && got != "" {
					t.Fatalf("primary provider header leaked to shadow: %q", got)
				}
			})
		}
	}
}
