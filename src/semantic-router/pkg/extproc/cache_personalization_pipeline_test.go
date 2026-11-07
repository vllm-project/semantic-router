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

type spyCache struct {
	findCalled   bool
	findQuery    string
	hitResponse  []byte
	shouldHit    bool
	pendingAdded bool
	pendingQuery string
}

func (s *spyCache) IsEnabled() bool                              { return true }
func (s *spyCache) CheckConnection(context.Context) error        { return nil }
func (s *spyCache) Close() error                                 { return nil }
func (s *spyCache) GetStats() cache.CacheStats                   { return cache.CacheStats{} }
func (s *spyCache) UpdateWithResponse(string, []byte, int) error { return nil }
func (s *spyCache) AddEntry(context.Context, string, string, string, []byte, []byte, int) error {
	return nil
}

func (s *spyCache) FindSimilar(string, string) ([]byte, bool, error) {
	return nil, false, nil
}

func (s *spyCache) FindSimilarWithThreshold(string, string, float32) ([]byte, bool, error) {
	return nil, false, nil
}

func (s *spyCache) AddPendingRequest(_ string, _ string, query string, _ []byte, _ int) error {
	s.pendingAdded = true
	s.pendingQuery = query
	return nil
}

func (s *spyCache) LookupSimilar(_ context.Context, _ string, query string, _ cache.SimilarityBounds) (cache.SimilarEntries, error) {
	s.findCalled = true
	s.findQuery = query
	if s.shouldHit {
		return similarEntries(cache.LookupResult{ResponseBody: s.hitResponse, Found: true}, query), nil
	}
	return cache.SimilarEntries{}, nil
}

// --- Core cache bypass tests: prove decisionWillPersonalize works ---

func TestCacheBypassWhenRAGEnabled(t *testing.T) {
	spy := &spyCache{shouldHit: true, hitResponse: []byte(`{"choices":[]}`)}

	decision := config.Decision{
		Name:      "rag-decision",
		ModelRefs: []config.ModelRef{{Model: "m"}},
		Plugins: []config.DecisionPlugin{
			{Type: config.DecisionPluginContext, Configuration: config.MustStructuredPayload(map[string]interface{}{
				"enabled": true,
				"sources": []interface{}{map[string]interface{}{"type": "rag", "backend": "external_api"}},
			})},
		},
	}

	cfg := &config.RouterConfig{}
	cfg.Enabled = true
	cfg.Decisions = []config.Decision{decision}

	router := &OpenAIRouter{Config: cfg, Cache: spy}
	ctx := &RequestContext{
		RequestID:           "test-bypass-1",
		StartTime:           time.Now(),
		SemanticRequest:     testNeutralRequest("test-model", "What are my recent orders?"),
		VSRSelectedDecision: &decision,
	}

	resp, hit := router.handleCaching(ctx, "rag-decision")

	assert.False(t, spy.findCalled, "cache FindSimilarWithThreshold must NOT be called when RAG is enabled")
	assert.Nil(t, resp, "no cached response should be returned")
	assert.False(t, hit, "should not report a cache hit")
	assert.False(t, spy.pendingAdded, "no pending cache write when personalization skips entire cache path")
}

func TestCacheBypassWhenMemoryEnabledGlobally(t *testing.T) {
	spy := &spyCache{shouldHit: true, hitResponse: []byte(`{"choices":[]}`)}

	decision := config.Decision{
		Name:      "memory-decision",
		ModelRefs: []config.ModelRef{{Model: "m"}},
	}

	cfg := &config.RouterConfig{}
	cfg.Enabled = true
	cfg.Memory.Enabled = true
	cfg.Decisions = []config.Decision{decision}

	router := &OpenAIRouter{Config: cfg, Cache: spy}
	ctx := &RequestContext{
		RequestID:           "test-bypass-memory-1",
		StartTime:           time.Now(),
		SemanticRequest:     testNeutralRequest("test-model", "What did we discuss yesterday?"),
		VSRSelectedDecision: &decision,
	}

	resp, hit := router.handleCaching(ctx, "memory-decision")

	assert.False(t, spy.findCalled, "cache read must be skipped when global memory is enabled")
	assert.Nil(t, resp)
	assert.False(t, hit)
	assert.False(t, spy.pendingAdded, "no pending cache write when personalization skips entire cache path")
}

func TestCacheBypassWithBothRAGAndMemory(t *testing.T) {
	spy := &spyCache{shouldHit: true, hitResponse: []byte(`{"choices":[]}`)}

	decision := config.Decision{
		Name:      "full-personalization",
		ModelRefs: []config.ModelRef{{Model: "m"}},
		Plugins: []config.DecisionPlugin{
			{Type: config.DecisionPluginContext, Configuration: config.MustStructuredPayload(map[string]interface{}{
				"enabled": true,
				"sources": []interface{}{map[string]interface{}{"type": "rag", "backend": "external_api"}, map[string]interface{}{"type": "memory"}},
			})},
		},
	}

	cfg := &config.RouterConfig{}
	cfg.Enabled = true
	cfg.Memory.Enabled = true
	cfg.Decisions = []config.Decision{decision}

	router := &OpenAIRouter{Config: cfg, Cache: spy}
	ctx := &RequestContext{
		Headers:             make(map[string]string),
		RequestID:           "test-both-1",
		StartTime:           time.Now(),
		SemanticRequest:     testNeutralRequest("m", "Summarize my project status"),
		VSRSelectedDecision: &decision,
	}

	resp, hit := router.handleCaching(ctx, "full-personalization")

	assert.False(t, spy.findCalled, "cache must be bypassed when both RAG and memory are enabled")
	assert.Nil(t, resp)
	assert.False(t, hit)
}

func TestCacheBypassWithPerDecisionMemoryOverride(t *testing.T) {
	spy := &spyCache{shouldHit: true, hitResponse: []byte(`{"choices":[]}`)}

	decision := config.Decision{
		Name:      "per-decision-mem",
		ModelRefs: []config.ModelRef{{Model: "m"}},
		Plugins: []config.DecisionPlugin{
			{Type: config.DecisionPluginContext, Configuration: config.MustStructuredPayload(map[string]interface{}{
				"enabled": true,
				"sources": []interface{}{map[string]interface{}{"type": "memory"}},
			})},
		},
	}

	cfg := &config.RouterConfig{}
	cfg.Enabled = true
	cfg.Memory.Enabled = false
	cfg.Decisions = []config.Decision{decision}

	router := &OpenAIRouter{Config: cfg, Cache: spy}
	ctx := &RequestContext{
		Headers:             make(map[string]string),
		RequestID:           "per-decision-mem-1",
		StartTime:           time.Now(),
		SemanticRequest:     testNeutralRequest("m", "Recall our chat"),
		VSRSelectedDecision: &decision,
	}

	resp, hit := router.handleCaching(ctx, "per-decision-mem")

	assert.False(t, spy.findCalled,
		"cache must be bypassed when per-decision memory overrides global disabled")
	assert.Nil(t, resp)
	assert.False(t, hit)
}

// --- Cache hit test: prove cache WORKS when no personalization ---

func TestCacheWorksNormallyWithoutPersonalization(t *testing.T) {
	spy := &spyCache{shouldHit: true, hitResponse: []byte(`{"choices":[]}`)}

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

	router := &OpenAIRouter{Config: cfg, Cache: spy}
	ctx := &RequestContext{
		RequestID:           "test-normal-cache-1",
		StartTime:           time.Now(),
		SemanticRequest:     testNeutralRequest("test-model", "What is 2+2?"),
		TraceContext:        context.Background(),
		VSRSelectedDecision: &decision,
	}

	resp, hit := router.handleCaching(ctx, "plain-decision")

	require.True(t, spy.findCalled,
		"cache FindSimilarWithThreshold MUST be called when no personalization plugins exist")
	assert.NotNil(t, resp, "cached response should be returned on cache hit")
	assert.True(t, hit, "should report a cache hit")
	assert.True(t, ctx.VSRCacheHit, "context should reflect cache hit")
}

func TestNoCacheBypassWhenMemoryExplicitlyDisabledPerDecision(t *testing.T) {
	spy := &spyCache{shouldHit: true, hitResponse: []byte(`{"choices":[]}`)}

	decision := config.Decision{
		Name:      "mem-disabled",
		ModelRefs: []config.ModelRef{{Model: "m"}},
		Plugins: []config.DecisionPlugin{
			{Type: config.DecisionPluginResponseCache, Configuration: config.MustStructuredPayload(map[string]interface{}{
				"enabled": true,
				"scope":   "global",
			})},
			{Type: config.DecisionPluginContext, Configuration: config.MustStructuredPayload(map[string]interface{}{
				"enabled": false,
			})},
		},
	}

	cfg := &config.RouterConfig{}
	cfg.Enabled = true
	cfg.Memory.Enabled = true
	cfg.Decisions = []config.Decision{decision}

	router := &OpenAIRouter{Config: cfg, Cache: spy}
	ctx := &RequestContext{
		RequestID:           "mem-disabled-1",
		StartTime:           time.Now(),
		SemanticRequest:     testNeutralRequest("m", "Quick question"),
		TraceContext:        context.Background(),
		VSRSelectedDecision: &decision,
	}

	resp, hit := router.handleCaching(ctx, "mem-disabled")

	require.True(t, spy.findCalled,
		"cache MUST be used when per-decision memory explicitly disables it")
	assert.NotNil(t, resp, "should return cached response")
	assert.True(t, hit, "should report a cache hit")
}

// --- decisionWillPersonalize predicate tests ---

func TestDecisionWillPersonalize_RAGEnabled(t *testing.T) {
	decision := config.Decision{
		Name:      "rag-dec",
		ModelRefs: []config.ModelRef{{Model: "m"}},
		Plugins: []config.DecisionPlugin{
			{Type: config.DecisionPluginContext, Configuration: config.MustStructuredPayload(map[string]interface{}{
				"enabled": true,
				"sources": []interface{}{map[string]interface{}{"type": "rag", "backend": "milvus"}},
			})},
		},
	}

	ctx := &RequestContext{VSRSelectedDecision: &decision}
	assert.True(t, decisionWillPersonalize(ctx, &config.RouterConfig{}))
}

func TestDecisionWillPersonalize_MemoryEnabled(t *testing.T) {
	decision := config.Decision{
		Name:      "mem-dec",
		ModelRefs: []config.ModelRef{{Model: "m"}},
	}

	cfg := &config.RouterConfig{}
	cfg.Memory.Enabled = true

	ctx := &RequestContext{VSRSelectedDecision: &decision}
	assert.True(t, decisionWillPersonalize(ctx, cfg))
}

func TestDecisionWillPersonalize_NoPersonalization(t *testing.T) {
	decision := config.Decision{
		Name:      "plain",
		ModelRefs: []config.ModelRef{{Model: "m"}},
	}

	ctx := &RequestContext{VSRSelectedDecision: &decision}
	assert.False(t, decisionWillPersonalize(ctx, &config.RouterConfig{}))
}

func TestDecisionWillPersonalize_NilDecision(t *testing.T) {
	ctx := &RequestContext{VSRSelectedDecision: nil}
	assert.False(t, decisionWillPersonalize(ctx, &config.RouterConfig{}))
}

func TestDecisionWillPersonalize_PerDecisionMemoryDisabledOverridesGlobal(t *testing.T) {
	decision := config.Decision{
		Name:      "mem-off",
		ModelRefs: []config.ModelRef{{Model: "m"}},
		Plugins: []config.DecisionPlugin{
			{Type: config.DecisionPluginContext, Configuration: config.MustStructuredPayload(map[string]interface{}{
				"enabled": false,
			})},
		},
	}

	cfg := &config.RouterConfig{}
	cfg.Memory.Enabled = true

	ctx := &RequestContext{VSRSelectedDecision: &decision}
	assert.False(t, decisionWillPersonalize(ctx, cfg),
		"per-decision memory disabled must override global enabled")
}

// --- RAG plugin resolution tests ---

// resolveBodyMutation captures the body-rewrite condition exercised by this
// personalization test helper.
func resolveBodyMutation(ctx *RequestContext, upstreamModel, model string) bool {
	needs := upstreamModel != model
	if !needs && (ctx.RAGRetrievedContext != "" || ctx.MemoryContext != "") {
		needs = true
	}
	return needs
}

// TestRAGBodyMutationForcedForSpecifiedModel verifies that when RAG context
// is injected and the upstream model name matches the requested model (so no
// model-name rewriting is needed), the body mutation is still forced so that
// Envoy sends the RAG-modified body to the upstream LLM.
func TestRAGBodyMutationForcedForSpecifiedModel(t *testing.T) {
	model := "gpt-4"
	upstreamModel := "gpt-4"

	t.Run("body mutation forced when RAG context present", func(t *testing.T) {
		ctx := &RequestContext{RAGRetrievedContext: "User project is 85% complete"}
		assert.Equal(t, model, upstreamModel, "model names match, no rewriting needed")
		assert.True(t, resolveBodyMutation(ctx, upstreamModel, model),
			"body mutation MUST be forced when RAG context is present")
	})

	t.Run("body mutation forced when memory context present", func(t *testing.T) {
		ctx := &RequestContext{MemoryContext: "Previous conversation context..."}
		assert.True(t, resolveBodyMutation(ctx, upstreamModel, model),
			"body mutation MUST be forced when memory context is present")
	})

	t.Run("no forced mutation when no personalized context", func(t *testing.T) {
		ctx := &RequestContext{}
		assert.False(t, resolveBodyMutation(ctx, upstreamModel, model),
			"no mutation needed when no personalized context and model names match")
	})

	t.Run("rewriting already triggers mutation regardless", func(t *testing.T) {
		differentUpstream := "gpt-4-turbo"
		ctx := &RequestContext{RAGRetrievedContext: "Some RAG context"}
		assert.NotEqual(t, model, differentUpstream, "model rewriting already forces mutation")
		assert.True(t, resolveBodyMutation(ctx, differentUpstream, model), "mutation still true")
	})
}
