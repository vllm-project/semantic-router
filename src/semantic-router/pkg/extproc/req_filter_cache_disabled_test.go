package extproc

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestHandleCachingWithoutBackendSkipsIdentityButKeepsModelAndQuery(t *testing.T) {
	decision := config.Decision{
		Name:      "plain-decision",
		ModelRefs: []config.ModelRef{{Model: "m"}},
		Plugins: []config.DecisionPlugin{
			{Type: config.DecisionPluginResponseCache, Configuration: config.MustStructuredPayload(map[string]interface{}{
				"enabled": true,
				"scope":   "global",
			})},
		},
	}
	cfg := &config.RouterConfig{}
	cfg.Enabled = true
	cfg.Decisions = []config.Decision{decision}

	request := testNeutralRequest("test-model", "What is 2+2?")
	identity, err := cache.BuildSemanticRequestIdentity(*request)
	require.NoError(t, err)

	for name, backend := range map[string]cache.CacheBackend{
		"nil":      nil,
		"disabled": cache.NewInMemoryCache(cache.InMemoryCacheOptions{Enabled: false}),
	} {
		t.Run(name, func(t *testing.T) {
			router := &OpenAIRouter{Config: cfg, Cache: backend}
			ctx := &RequestContext{
				RequestID:           "cache-disabled-" + name,
				StartTime:           time.Now(),
				SemanticRequest:     testNeutralRequest("test-model", "What is 2+2?"),
				TraceContext:        context.Background(),
				VSRSelectedDecision: &decision,
			}

			resp, hit := router.handleCaching(ctx, decision.Name)

			assert.Nil(t, resp)
			assert.False(t, hit)
			assert.Equal(t, identity.Model, ctx.RequestModel)
			assert.Equal(t, identity.Query, ctx.RequestQuery)
			assert.Equal(t, identity.Model, ctx.CacheRequestModel)
			assert.Equal(t, identity.Query, ctx.CacheQuery)
			assert.Empty(t, ctx.CacheExactFingerprint)
			assert.Empty(t, ctx.CacheCompatibilityFingerprint)
			assert.Empty(t, ctx.CacheIdentity.ExactFingerprint)
		})
	}
}

func TestHandleCachingWithoutBackendStillAppliesRequestControls(t *testing.T) {
	decision := config.Decision{
		Name:      "controls-decision",
		ModelRefs: []config.ModelRef{{Model: "m"}},
		Plugins: []config.DecisionPlugin{
			{Type: config.DecisionPluginResponseCache, Configuration: config.MustStructuredPayload(map[string]interface{}{
				"enabled":                true,
				"scope":                  "global",
				"allow_request_controls": true,
			})},
		},
	}
	cfg := &config.RouterConfig{}
	cfg.Enabled = true
	cfg.Decisions = []config.Decision{decision}
	router := &OpenAIRouter{Config: cfg, Cache: cache.NewInMemoryCache(cache.InMemoryCacheOptions{Enabled: false})}
	ctx := &RequestContext{
		RequestID:           "cache-disabled-controls",
		Headers:             map[string]string{defaultCacheControlHeader: "no-store"},
		SemanticRequest:     testNeutralRequest("test-model", "What is 2+2?"),
		TraceContext:        context.Background(),
		VSRSelectedDecision: &decision,
	}

	resp, hit := router.handleCaching(ctx, decision.Name)

	assert.Nil(t, resp)
	assert.False(t, hit)
	assert.True(t, ctx.CacheWriteBypass)
	assert.False(t, ctx.CacheReadBypass)
	assert.Empty(t, ctx.CacheExactFingerprint)
}

func TestHandleCachingWithoutBackendStillSkipsPersonalizingDecisions(t *testing.T) {
	decision := config.Decision{
		Name:      "memory-decision",
		ModelRefs: []config.ModelRef{{Model: "m"}},
		Plugins: []config.DecisionPlugin{
			{Type: config.DecisionPluginMemory, Configuration: config.MustStructuredPayload(map[string]interface{}{
				"enabled": true,
			})},
		},
	}
	cfg := &config.RouterConfig{}
	cfg.Enabled = true
	cfg.Decisions = []config.Decision{decision}
	router := &OpenAIRouter{Config: cfg, Cache: cache.NewInMemoryCache(cache.InMemoryCacheOptions{Enabled: false})}
	ctx := &RequestContext{
		RequestID:           "cache-disabled-memory",
		SemanticRequest:     testNeutralRequest("test-model", "Remember this"),
		TraceContext:        context.Background(),
		VSRSelectedDecision: &decision,
	}

	resp, hit := router.handleCaching(ctx, decision.Name)

	assert.Nil(t, resp)
	assert.False(t, hit)
	assert.Empty(t, ctx.RequestQuery)
	assert.Empty(t, ctx.CacheExactFingerprint)
}

func TestHandleLooperCacheSkipWithoutBackendKeepsModelAndQuery(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	ctx := &RequestContext{
		RequestID:       "looper-cache-disabled",
		LooperRequest:   true,
		SemanticRequest: testNeutralRequest("test-model", "Summarize this"),
		TraceContext:    context.Background(),
	}

	resp, hit := router.handleCaching(ctx, "")

	assert.Nil(t, resp)
	assert.False(t, hit)
	assert.Equal(t, "test-model", ctx.RequestModel)
	assert.Equal(t, "Summarize this", ctx.RequestQuery)
	assert.Empty(t, ctx.CacheExactFingerprint)
}
