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

type polarityMissCache struct {
	spyCache
	result cache.LookupResult
}

func (c *polarityMissCache) LookupSimilarWithThreshold(
	_ context.Context,
	_ string,
	query string,
	_ float32,
) (cache.LookupResult, error) {
	c.findCalled = true
	c.findQuery = query
	return c.result, nil
}

func TestPolarityUnverifiedMissContinuesUpstream(t *testing.T) {
	const candidateSimilarity = float32(0.97)
	backend := &polarityMissCache{
		result: cache.LookupResult{Similarity: candidateSimilarity},
	}
	decision := config.Decision{
		Name:      "polarity-route",
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

	router := &OpenAIRouter{Config: cfg, Cache: backend}
	ctx := &RequestContext{
		RequestID:           "polarity-verifier-error",
		StartTime:           time.Now(),
		SemanticRequest:     testNeutralRequest("test-model", "How do I disable two-factor authentication?"),
		TraceContext:        context.Background(),
		VSRSelectedDecision: &decision,
	}

	// pkg/cache pins verifier failures to a miss while retaining the rejected
	// candidate's score. Exercise that result through the production cache
	// adapter, service, and ExtProc plugin contract: it must not short-circuit
	// the request that should continue to the selected upstream model.
	response, shortCircuit := router.handleCaching(ctx, decision.Name)

	require.True(t, backend.findCalled, "the semantic lookup must reach the configured cache backend")
	assert.Nil(t, response, "an unverified candidate must not produce a cached response")
	assert.False(t, shortCircuit, "a verifier failure degraded to a miss must continue upstream")
	assert.False(t, ctx.VSRCacheHit, "the rejected candidate must not be reported as a cache hit")
	assert.Equal(t, candidateSimilarity, ctx.VSRCacheSimilarity, "the rejected candidate score should remain observable")
}
