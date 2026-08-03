package extproc

import (
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
	assert.True(t, mockCache.addPendingCalled, "should preserve global cache write path when no decisions are configured")
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
	assert.Equal(t, "privacy::MoM", mockCache.findSimilarModel)
	assert.Equal(t, mockCache.findSimilarModel, mockCache.addPendingModel)
	assert.Equal(t, "hello", ctx.CacheQuery, "recipe identity must not pollute the embedding input")
}

func TestSemanticCachePartition_PreservesDefaultRecipeKey(t *testing.T) {
	ctx := &RequestContext{}
	ctx.Routing.SelectRecipe(&config.RoutingRecipe{Name: config.DefaultRecipeName})

	assert.Equal(t, "MoM", semanticCachePartition(ctx, "MoM"))
}
