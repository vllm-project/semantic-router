package extproc

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestHandleCaching_SkipsGlobalCacheWhenDecisionsConfiguredButNoDecisionMatched(t *testing.T) {
	mockCache := &mockStreamingCache{}
	cfg := &config.RouterConfig{
		SemanticCache: config.SemanticCache{Enabled: true},
		BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{
			"test": {ResourceID: "model-test", ResourceRevision: 1},
		}},
		Recipes: []config.RoutingRecipe{{
			ID: "recipe-cache-policy", Revision: 1, Name: "cache-policy",
			Profile: config.RoutingProfile{Decisions: []config.Decision{
				{
					ID:   "decision-default-route",
					Name: "default-route",
				},
			}},
		}},
		Entrypoints: []config.EntrypointMapping{{
			ID: "entrypoint-cache-policy", Revision: 1, Name: "router/cache", ModelNames: []string{"router/cache"},
			Rules: []config.EntrypointRule{{
				ID: "rule-cache-policy", Name: "default",
				Action: config.EntrypointRuleAction{
					RecipeID: "recipe-cache-policy", RecipeRevision: 1, Recipe: "cache-policy",
					Assignments: map[string]config.RoutingAssignmentSet{
						"decision-default-route": {Models: []config.RoutingModelAssignment{{
							ModelID: "model-test", ModelRevision: 1, ModelName: "test", Weight: "1",
						}}},
					},
				},
			}},
		}},
	}
	if err := cfg.PrepareEntrypointRecipes(); err != nil {
		t.Fatalf("prepare cache-policy Entrypoint: %v", err)
	}
	recipe, ok := cfg.RecipeForRequestModel("router/cache")
	if !ok {
		t.Fatal("resolve cache-policy Entrypoint")
	}
	router := &OpenAIRouter{Cache: mockCache, Config: cfg}
	ctx := &RequestContext{
		RequestID:       "req-1",
		SemanticRequest: testNeutralRequest("router/cache", "hello"),
	}
	ctx.Routing.SelectRecipe(recipe)

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
		RequestID:       "req-1",
		SemanticRequest: testNeutralRequest("MoM", "hello"),
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
		RequestID:       "req-1",
		SemanticRequest: testNeutralRequest("MoM", "hello"),
	}
	ctx.Routing.SelectRecipe(&config.RoutingRecipe{Name: "privacy"})

	resp, hit := router.handleCaching(ctx, "")
	assert.Nil(t, resp)
	assert.False(t, hit)
	assert.Contains(t, mockCache.findSimilarModel, "privacy")
	assert.Contains(t, mockCache.findSimilarModel, "MoM")
	assert.Equal(t, "hello", ctx.CacheQuery, "recipe identity must not pollute the embedding input")
}

func TestResponseCachePartitionIncludesDefaultRecipeAndProtocol(t *testing.T) {
	ctx := &RequestContext{}
	ctx.Routing.SelectRecipe(&config.RoutingRecipe{Name: config.DefaultRecipeName})

	partition := semanticCachePartition(ctx, "MoM")
	assert.Contains(t, partition, "default")
	assert.Contains(t, partition, "MoM")
	assert.Contains(t, partition, "openai%3Abody")
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
			Type: config.DecisionPluginResponseCache,
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
		InferenceAccess:     testInferenceRequestAccess("cache-test-user", ""),
		RequestID:           "req-exact",
		SemanticRequest:     testNeutralRequest("MoM", "hello"),
		VSRSelectedDecision: &router.Config.Decisions[0],
	}

	resp, hit := router.handleCaching(ctx, decision.Name, "frontier")
	assert.NotNil(t, resp)
	assert.True(t, hit)
	assert.False(t, mockCache.findSimilarCalled)
	assert.True(t, ctx.VSRCacheHit)
}

func TestHandleCaching_ReplaysExactHitForAnthropicClient(t *testing.T) {
	mockCache := &mockStreamingCache{
		exactHit: true,
		exactResponse: []byte(`{
			"id":"msg_cached","type":"message","role":"assistant","model":"claude",
			"content":[{"type":"text","text":"cached"}],"stop_reason":"end_turn",
			"usage":{"input_tokens":3,"output_tokens":1}
		}`),
	}
	decision := config.Decision{
		Name:      "anthropic-route",
		ModelRefs: []config.ModelRef{{Model: "claude"}},
		Plugins: []config.DecisionPlugin{{
			Type: config.DecisionPluginResponseCache,
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
		InferenceAccess:     testInferenceRequestAccess("cache-test-user", ""),
		SourceFormat:        llmprotocol.AnthropicMessagesV1,
		RequestID:           "req-anthropic",
		SemanticRequest:     testNeutralRequest("claude", "hello"),
		VSRSelectedDecision: &router.Config.Decisions[0],
	}

	resp, hit := router.handleCaching(ctx, decision.Name, "claude")

	assert.NotNil(t, resp)
	assert.True(t, hit)
	assert.True(t, mockCache.exactFindCalled)
	assert.False(t, mockCache.findSimilarCalled)
	body := string(resp.GetImmediateResponse().GetBody())
	assert.Contains(t, body, `"type":"message"`)
	assert.Contains(t, body, `"text":"cached"`)
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
			Type: config.DecisionPluginResponseCache,
			Configuration: config.MustStructuredPayload(map[string]interface{}{
				"enabled": true,
				"mode":    "exact_then_semantic",
				"request_controls": map[string]interface{}{
					"enabled": true,
				},
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
		SemanticRequest:     testNeutralRequest("MoM", "hello"),
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
	decision := config.Decision{
		Name:      "tenant-route",
		ModelRefs: []config.ModelRef{{Model: "frontier"}},
		Plugins: []config.DecisionPlugin{{
			Type: config.DecisionPluginResponseCache,
			Configuration: config.MustStructuredPayload(map[string]interface{}{
				"enabled": true,
				"scope":   "user",
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
		InferenceAccess:     testInferenceRequestAccess("alice", ""),
		RequestID:           "req-1",
		VSRSelectedDecision: &router.Config.Decisions[0],
		SemanticRequest:     testNeutralRequest("MoM", "hello"),
	}
	ctx.Routing.SelectRecipe(&config.RoutingRecipe{Name: "privacy"})

	resp, hit := router.handleCaching(ctx, "", "frontier")
	assert.Nil(t, resp)
	assert.False(t, hit)
	assert.NotContains(t, mockCache.findSimilarModel, "alice")
	assert.Contains(t, mockCache.findSimilarModel, "privacy")
	assert.Contains(t, mockCache.findSimilarModel, "frontier")

	alicePartition := mockCache.findSimilarModel
	bobCtx := &RequestContext{
		InferenceAccess:     testInferenceRequestAccess("bob", ""),
		RequestID:           "req-2",
		SemanticRequest:     testNeutralRequest("MoM", "hello"),
		VSRSelectedDecision: &router.Config.Decisions[0],
	}
	bobCtx.Routing.SelectRecipe(&config.RoutingRecipe{Name: "privacy"})
	_, _ = router.handleCaching(bobCtx, "", "frontier")
	assert.NotEqual(t, alicePartition, mockCache.findSimilarModel)
}
