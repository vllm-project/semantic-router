package extproc

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

type replayIdentityCache struct {
	mockStreamingCache
	partition string
}

func (c *replayIdentityCache) LookupSimilarWithThreshold(
	_ context.Context, partition, _ string, _ float32,
) (cache.LookupResult, error) {
	c.partition = partition
	return cache.LookupResult{Found: true, ResponseBody: c.exactResponse, Similarity: 1}, nil
}

func (c *replayIdentityCache) FindExact(
	_ context.Context, partition, _ string,
) (cache.LookupResult, error) {
	c.partition = partition
	return cache.LookupResult{Found: true, ResponseBody: c.exactResponse, Similarity: 1}, nil
}

func TestHandleCaching_ReplayUsesSelectedBackendProvenance(t *testing.T) {
	for _, mode := range []string{config.ResponseCacheModeSemantic, config.ResponseCacheModeExact} {
		for _, responseModel := range []string{"backend", "provider/backend-version"} {
			t.Run(mode+"/"+responseModel, func(t *testing.T) {
				body, err := json.Marshal(map[string]interface{}{
					"id": "cached", "model": responseModel,
					"choices": []interface{}{map[string]interface{}{
						"index": 0, "finish_reason": "stop",
						"message": map[string]interface{}{"role": "assistant", "content": "hello"},
					}},
					"usage": map[string]int{"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500},
				})
				require.NoError(t, err)
				backend := &replayIdentityCache{mockStreamingCache: mockStreamingCache{exactResponse: body}}
				decision := config.Decision{
					Name: "cached-route", ModelRefs: []config.ModelRef{{Model: "backend"}, {Model: "baseline"}},
					Plugins: []config.DecisionPlugin{{
						Type: config.DecisionPluginResponseCache,
						Configuration: config.MustStructuredPayload(map[string]interface{}{
							"enabled": true, "mode": mode, "scope": "global",
						}),
					}},
				}
				recorder := routerreplay.NewRecorder(store.NewMemoryStore(10, 0))
				router := &OpenAIRouter{
					Cache: backend, ReplayRecorder: recorder,
					Config: &config.RouterConfig{
						SemanticCache:      config.SemanticCache{Enabled: true},
						IntelligentRouting: config.IntelligentRouting{Decisions: []config.Decision{decision}},
						BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{
							"backend": {
								ExternalModelIDs: map[string]string{"provider": "provider/backend-version"},
								Pricing:          config.ModelPricing{Currency: "USD", PromptPer1M: 1, CompletionPer1M: 2},
							},
							"baseline": {Pricing: config.ModelPricing{Currency: "USD", PromptPer1M: 4, CompletionPer1M: 8}},
						}},
					},
				}
				replayConfig := config.DefaultRouterReplayPluginConfig()
				replayConfig.Enabled = true
				ctx := &RequestContext{
					RequestID: "cached-request", TraceContext: context.Background(),
					SourceFormat:        llmprotocol.OpenAIChatV1,
					SemanticRequest:     testNeutralRequest("public-entrypoint", "hello"),
					VSRSelectedDecision: &decision, RouterReplayPluginConfig: &replayConfig,
				}
				ctx.Routing.SelectRecipe(&config.RoutingRecipe{Name: "recipe"})

				response, hit := router.handleCaching(ctx, decision.Name, "backend")

				require.True(t, hit)
				require.NotNil(t, response.GetImmediateResponse())
				require.NotEmpty(t, ctx.RouterReplayID)
				record, found := recorder.GetRecord(ctx.RouterReplayID)
				require.True(t, found)
				assert.Equal(t, "public-entrypoint", record.OriginalModel)
				assert.Equal(t, "backend", record.SelectedModel)
				require.NotNil(t, record.RouteDiagnostics)
				assert.Equal(t, "backend", record.RouteDiagnostics.SelectedModel)
				assert.Equal(t, "backend", record.RouteDiagnostics.ProposalModel)
				assert.Empty(t, record.SelectionMethod, "cache replay must not invent a fresh algorithm selection")
				assert.True(t, router.Config.ModelNameMatches(record.SelectedModel, responseModel))
				assert.Equal(t, "backend", ctx.CacheIdentity.Partition.SelectedModel)
				identity := router.responseCacheService().ResolveIdentity(ctx.CacheIdentity)
				wantPartition := identity.Partition.Key()
				if mode == config.ResponseCacheModeSemantic {
					wantPartition = identity.SemanticPartitionKey()
				}
				assert.Equal(t, wantPartition, backend.partition)
				assert.True(t, record.FromCache)
				assert.Equal(t, routerreplay.LifecycleCompleted, record.LifecycleState)
				assertApproxFloat64(t, record.ActualCost, 0)
				assertApproxFloat64(t, record.BaselineCost, 0.008)
				assertApproxFloat64(t, record.CostSavings, 0.008)
				require.NotNil(t, record.BaselineModel)
				assert.Equal(t, "baseline", *record.BaselineModel)
				nonCacheUsage := router.buildReplayUsageCost(ctx, responseUsageMetrics{promptTokens: 1000, completionTokens: 500})
				assertApproxFloat64(t, nonCacheUsage.ActualCost, 0.002)
			})
		}
	}
}
