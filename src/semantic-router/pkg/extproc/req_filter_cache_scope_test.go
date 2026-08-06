package extproc

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestHandleCaching_SkipsGlobalCacheWhenDecisionsConfiguredButNoDecisionMatched(t *testing.T) {
	mockCache := &mockStreamingCache{}
	cfg := &config.RouterConfig{
		SemanticCache: config.SemanticCache{Enabled: true},
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{
				{
					Name:      "default-route",
					ModelRefs: []config.ModelRef{{Model: "test"}},
				},
			},
		},
	}
	router := &OpenAIRouter{Cache: mockCache, Config: cfg}
	ctx := &RequestContext{
		RequestID:           "req-1",
		OriginalRequestBody: []byte(`{"model":"MoM","messages":[{"role":"user","content":"hello"}]}`),
	}

	resp, hit := router.handleCaching(ctx, "")
	assert.Nil(t, resp)
	assert.False(t, hit)
	assert.False(t, mockCache.findSimilarCalled, "should not perform cache lookup when decisions exist but none matched")
	assert.False(t, mockCache.addPendingCalled, "should not enqueue cache write when decisions exist but none matched")
}

func TestHandleCaching_UsesGlobalCacheWhenNoDecisionsConfigured(t *testing.T) {
	mockCache := &mockStreamingCache{}
	cfg := &config.RouterConfig{
		SemanticCache: config.SemanticCache{Enabled: true},
	}
	router := &OpenAIRouter{Cache: mockCache, Config: cfg}
	ctx := &RequestContext{
		RequestID:           "req-1",
		OriginalRequestBody: []byte(`{"model":"MoM","messages":[{"role":"user","content":"hello"}]}`),
	}

	resp, hit := router.handleCaching(ctx, "")
	assert.Nil(t, resp)
	assert.False(t, hit)
	assert.True(t, mockCache.findSimilarCalled, "should preserve global cache lookup when no decisions are configured")
	assert.False(t, mockCache.addPendingCalled, "cache misses must remain request-local until a response succeeds")
}

func TestHandleCaching_PartitionsNamedRecipeWithoutChangingSemanticQuery(t *testing.T) {
	mockCache := &mockStreamingCache{}
	router := &OpenAIRouter{
		Cache:  mockCache,
		Config: &config.RouterConfig{SemanticCache: config.SemanticCache{Enabled: true}},
	}
	ctx := &RequestContext{
		RequestID:           "req-1",
		OriginalRequestBody: []byte(`{"model":"MoM","messages":[{"role":"user","content":"hello"}]}`),
	}
	ctx.Routing.SelectRecipe(&config.RoutingRecipe{Name: "privacy"})

	resp, hit := router.handleCaching(ctx, "")
	assert.Nil(t, resp)
	assert.False(t, hit)
	assert.True(t, strings.HasPrefix(mockCache.findSimilarModel, "v2:"))
	assert.Len(t, mockCache.findSimilarModel, 67)
	assert.NotContains(t, mockCache.findSimilarModel, "privacy")
	assert.Equal(t, "hello", ctx.CacheQuery, "recipe identity must not pollute the embedding input")
}

func TestSemanticCachePartition_PreservesDefaultRecipeKey(t *testing.T) {
	ctx := &RequestContext{}
	ctx.Routing.SelectRecipe(&config.RoutingRecipe{Name: config.DefaultRecipeName})

	assert.Equal(t, "MoM", semanticCachePartition(ctx, "MoM"))
}

func TestHandleCaching_ExactHitSkipsSemanticEmbeddingLookup(t *testing.T) {
	mockCache := &mockStreamingCache{
		exactHit:      true,
		exactResponse: []byte(`{"id":"cached","choices":[{"message":{"role":"assistant","content":"cached"}}]}`),
	}
	decision := config.Decision{
		Name:      "exact-route",
		ModelRefs: []config.ModelRef{{Model: "frontier"}},
		Plugins: []config.DecisionPlugin{{
			Type: "semantic-cache",
			Configuration: config.MustStructuredPayload(map[string]interface{}{
				"enabled": true,
				"mode":    "exact_then_semantic",
			}),
		}},
	}
	router := &OpenAIRouter{
		Cache: mockCache,
		Config: &config.RouterConfig{
			SemanticCache: config.SemanticCache{Enabled: true},
			IntelligentRouting: config.IntelligentRouting{
				Decisions: []config.Decision{decision},
			},
		},
	}
	ctx := &RequestContext{
		RequestID:           "req-exact",
		OriginalRequestBody: []byte(`{"model":"MoM","messages":[{"role":"user","content":"hello"}]}`),
		VSRSelectedDecision: &router.Config.Decisions[0],
	}

	resp, hit := router.handleCaching(ctx, decision.Name, "frontier")
	assert.NotNil(t, resp)
	assert.True(t, hit)
	assert.False(t, mockCache.findSimilarCalled)
	assert.True(t, ctx.VSRCacheHit)
}

func TestHandleCaching_BypassesAnthropicClientUntilCacheHitEmitterExists(t *testing.T) {
	mockCache := &mockStreamingCache{exactHit: true}
	decision := config.Decision{
		Name:      "anthropic-route",
		ModelRefs: []config.ModelRef{{Model: "claude"}},
		Plugins: []config.DecisionPlugin{{
			Type: "semantic-cache",
			Configuration: config.MustStructuredPayload(map[string]interface{}{
				"enabled": true,
				"mode":    "exact_then_semantic",
			}),
		}},
	}
	router := &OpenAIRouter{
		Cache: mockCache,
		Config: &config.RouterConfig{
			SemanticCache: config.SemanticCache{Enabled: true},
			IntelligentRouting: config.IntelligentRouting{
				Decisions: []config.Decision{decision},
			},
		},
	}
	ctx := &RequestContext{
		ClientProtocol:      config.ClientProtocolAnthropic,
		RequestID:           "req-anthropic",
		OriginalRequestBody: []byte(`{"model":"claude","messages":[{"role":"user","content":"hello"}]}`),
		VSRSelectedDecision: &router.Config.Decisions[0],
	}

	resp, hit := router.handleCaching(ctx, decision.Name, "claude")

	assert.Nil(t, resp)
	assert.False(t, hit)
	assert.False(t, mockCache.exactFindCalled)
	assert.False(t, mockCache.findSimilarCalled)
}

func TestHandleCaching_NoCacheControlSkipsReadsButKeepsWritePath(t *testing.T) {
	mockCache := &mockStreamingCache{
		exactHit:      true,
		exactResponse: []byte(`{"id":"cached","choices":[]}`),
	}
	decision := config.Decision{
		Name:      "controlled-cache",
		ModelRefs: []config.ModelRef{{Model: "frontier"}},
		Plugins: []config.DecisionPlugin{{
			Type: "semantic-cache",
			Configuration: config.MustStructuredPayload(map[string]interface{}{
				"enabled":                true,
				"mode":                   "exact_then_semantic",
				"allow_request_controls": true,
			}),
		}},
	}
	router := &OpenAIRouter{
		Cache: mockCache,
		Config: &config.RouterConfig{
			SemanticCache: config.SemanticCache{Enabled: true},
			IntelligentRouting: config.IntelligentRouting{
				Decisions: []config.Decision{decision},
			},
		},
	}
	ctx := &RequestContext{
		Headers:             map[string]string{"x-vsr-cache-control": "no-cache"},
		RequestID:           "req-control",
		OriginalRequestBody: []byte(`{"model":"MoM","messages":[{"role":"user","content":"hello"}]}`),
		VSRSelectedDecision: &router.Config.Decisions[0],
	}

	resp, hit := router.handleCaching(ctx, decision.Name, "frontier")
	assert.Nil(t, resp)
	assert.False(t, hit)
	assert.False(t, mockCache.exactFindCalled)
	assert.False(t, mockCache.findSimilarCalled)
	assert.False(t, mockCache.addPendingCalled)
	assert.True(t, ctx.CacheReadBypass)
	assert.False(t, ctx.CacheWriteBypass)
}

func TestHandleCaching_HardPartitionsTenantSelectedModelAndCompatibility(t *testing.T) {
	t.Setenv("USER_SCOPE_NAMESPACE_SECRET", "cache-scope-test-secret")
	mockCache := &mockStreamingCache{}
	router := &OpenAIRouter{
		Cache:  mockCache,
		Config: &config.RouterConfig{SemanticCache: config.SemanticCache{Enabled: true}},
	}
	ctx := &RequestContext{
		Headers: map[string]string{
			"x-authz-user-id": "alice",
		},
		RequestID: "req-1",
		OriginalRequestBody: []byte(
			`{"model":"MoM","messages":[{"role":"system","content":"be precise"},{"role":"user","content":"hello"}]}`,
		),
	}
	ctx.Routing.SelectRecipe(&config.RoutingRecipe{Name: "privacy"})

	resp, hit := router.handleCaching(ctx, "", "frontier")
	assert.Nil(t, resp)
	assert.False(t, hit)
	assert.True(t, strings.HasPrefix(mockCache.findSimilarModel, "v2:"))
	assert.Len(t, mockCache.findSimilarModel, 67)
	assert.NotContains(t, mockCache.findSimilarModel, "alice")
	assert.NotContains(t, mockCache.findSimilarModel, "privacy")
	assert.NotContains(t, mockCache.findSimilarModel, "frontier")
	assert.NotContains(t, mockCache.findSimilarModel, "|none|")

	alicePartition := mockCache.findSimilarModel
	bobCtx := &RequestContext{
		Headers: map[string]string{
			"x-authz-user-id": "bob",
		},
		RequestID:           "req-2",
		OriginalRequestBody: ctx.OriginalRequestBody,
	}
	bobCtx.Routing.SelectRecipe(&config.RoutingRecipe{Name: "privacy"})
	_, _ = router.handleCaching(bobCtx, "", "frontier")
	assert.NotEqual(t, alicePartition, mockCache.findSimilarModel)
}
