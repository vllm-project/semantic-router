package extproc

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

// responseCachedDecision adds an enabled personalized exact response cache to
// decision, so only sticky selection can keep a request away from it.
func responseCachedDecision(t *testing.T, decision config.Decision) config.Decision {
	t.Helper()
	decision.Plugins = append(decision.Plugins, config.DecisionPlugin{
		Type: config.DecisionPluginResponseCache,
		Configuration: config.MustStructuredPayload(&config.ResponseCachePluginConfig{
			Enabled: true, Scope: "user", Personalized: &config.ResponseCachePersonalizedConfig{Mode: "exact"},
		}),
	})
	return decision
}

func cacheTestContext(decision config.Decision) *RequestContext {
	return &RequestContext{
		RequestID:           "sticky-cache",
		StartTime:           time.Now(),
		Headers:             map[string]string{headers.AuthzUserID: "user-a"},
		SemanticRequest:     testNeutralRequest("model", "what is the weather"),
		VSRSelectedDecision: &decision,
		MemoryContext:       "user history",
	}
}

// Neither the ordinary nor the personalized response-cache read may serve a
// sticky decision, and its response is never written back.
func TestStickyDecisionBypassesRouterResponseCache(t *testing.T) {
	plain := responseCachedDecision(t, stickyDecision(t, "cached", &config.ToolSelectionPluginConfig{
		Enabled: true, Mode: config.ToolSelectionModeAdd,
	}, supportedStickyToolsConfig()))
	sticky := responseCachedDecision(t, supportedStickyDecision(t, "cached", config.ToolSelectionModeAdd))

	require.True(t, canUsePersonalizedExactCache(cacheTestContext(plain)), "control: the personalized cache applies without sticky")
	require.False(t, canUsePersonalizedExactCache(cacheTestContext(sticky)), "the personalized cache must not serve a sticky decision")

	cfg := &config.RouterConfig{}
	cfg.Enabled = true
	spy := &spyCache{shouldHit: true, hitResponse: []byte(`{"choices":[]}`)}
	router := &OpenAIRouter{Config: cfg, Cache: spy}
	ctx := cacheTestContext(sticky)
	ctx.MemoryContext = ""
	response, hit := router.handleCaching(ctx, "cached", "model")
	require.Nil(t, response)
	require.False(t, hit)
	require.False(t, spy.findCalled, "an ordinary cache read must not run for a sticky decision")
	require.False(t, spy.pendingAdded, "no pending cache write for a sticky decision")

	writer := &mockStreamingCache{}
	writeRouter := &OpenAIRouter{Cache: writer, Config: &config.RouterConfig{SemanticCache: config.SemanticCache{Enabled: true}}}
	for name, decision := range map[string]config.Decision{"control": plain, "sticky": sticky} {
		writer.addEntryCalled = false
		write := cacheTestContext(decision)
		write.MemoryContext = ""
		write.RequestModel, write.RequestQuery = "model", "what is the weather"
		writeRouter.updateResponseCache(write, []byte(`{"choices":[]}`))
		require.Equal(t, name == "control", writer.addEntryCalled, "%s cache write", name)
	}
}
