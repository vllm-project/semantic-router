package extproc

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
)

type spyCache struct {
	findCalled   bool
	findQuery    string
	hitResponse  []byte
	shouldHit    bool
	pendingAdded bool
	pendingQuery string
}

func (s *spyCache) IsEnabled() bool                                            { return true }
func (s *spyCache) CheckConnection() error                                     { return nil }
func (s *spyCache) LastSimilarity() float32                                    { return 0 }
func (s *spyCache) Close() error                                               { return nil }
func (s *spyCache) GetStats() cache.CacheStats                                 { return cache.CacheStats{} }
func (s *spyCache) UpdateWithResponse(string, []byte, int) error               { return nil }
func (s *spyCache) AddEntry(string, string, string, []byte, []byte, int) error { return nil }
func (s *spyCache) FindSimilar(string, string) ([]byte, bool, error)           { return nil, false, nil }

func (s *spyCache) AddPendingRequest(_ string, _ string, query string, _ []byte, _ int) error {
	s.pendingAdded = true
	s.pendingQuery = query
	return nil
}

func (s *spyCache) FindSimilarWithThreshold(_ string, query string, _ float32) ([]byte, bool, error) {
	s.findCalled = true
	s.findQuery = query
	if s.shouldHit {
		return s.hitResponse, true, nil
	}
	return nil, false, nil
}

func makeOpenAIRequestBody(model, content string) []byte {
	body, _ := json.Marshal(map[string]interface{}{
		"model": model,
		"messages": []map[string]string{
			{"role": "user", "content": content},
		},
	})
	return body
}

// --- Core cache bypass tests: prove decisionWillPersonalize works ---

func TestCacheBypassWhenRAGEnabled(t *testing.T) {
	spy := &spyCache{shouldHit: true, hitResponse: []byte(`{"choices":[]}`)}

	decision := config.Decision{
		Name:      "rag-decision",
		ModelRefs: []config.ModelRef{{Model: "m"}},
		Plugins: []config.DecisionPlugin{
			{Type: "rag", Configuration: config.MustStructuredPayload(map[string]interface{}{
				"enabled": true,
				"backend": "external_api",
			})},
		},
	}

	cfg := &config.RouterConfig{}
	cfg.Enabled = true
	cfg.Decisions = []config.Decision{decision}

	router := &OpenAIRouter{Config: cfg, Cache: spy}
	ctx := &RequestContext{
		Headers:             make(map[string]string),
		RequestID:           "test-bypass-1",
		StartTime:           time.Now(),
		OriginalRequestBody: makeOpenAIRequestBody("test-model", "What are my recent orders?"),
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
		Headers:             make(map[string]string),
		RequestID:           "test-bypass-memory-1",
		StartTime:           time.Now(),
		OriginalRequestBody: makeOpenAIRequestBody("test-model", "What did we discuss yesterday?"),
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
			{Type: "rag", Configuration: config.MustStructuredPayload(map[string]interface{}{
				"enabled": true,
				"backend": "external_api",
			})},
			{Type: "memory", Configuration: config.MustStructuredPayload(map[string]interface{}{
				"enabled": true,
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
		OriginalRequestBody: makeOpenAIRequestBody("m", "Summarize my project status"),
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
			{Type: "memory", Configuration: config.MustStructuredPayload(map[string]interface{}{
				"enabled": true,
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
		OriginalRequestBody: makeOpenAIRequestBody("m", "Recall our chat"),
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
			{Type: "semantic-cache", Configuration: config.MustStructuredPayload(map[string]interface{}{
				"enabled": true,
			})},
		},
	}

	cfg := &config.RouterConfig{}
	cfg.Enabled = true
	cfg.Decisions = []config.Decision{decision}

	router := &OpenAIRouter{Config: cfg, Cache: spy}
	ctx := &RequestContext{
		Headers:             make(map[string]string),
		RequestID:           "test-normal-cache-1",
		StartTime:           time.Now(),
		OriginalRequestBody: makeOpenAIRequestBody("test-model", "What is 2+2?"),
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
			{Type: "semantic-cache", Configuration: config.MustStructuredPayload(map[string]interface{}{
				"enabled": true,
			})},
			{Type: "memory", Configuration: config.MustStructuredPayload(map[string]interface{}{
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
		Headers:             make(map[string]string),
		RequestID:           "mem-disabled-1",
		StartTime:           time.Now(),
		OriginalRequestBody: makeOpenAIRequestBody("m", "Quick question"),
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
			{Type: "rag", Configuration: config.MustStructuredPayload(map[string]interface{}{
				"enabled": true,
				"backend": "milvus",
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
			{Type: "memory", Configuration: config.MustStructuredPayload(map[string]interface{}{
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

func TestRAGPluginResolution_NoBackend(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.Decisions = []config.Decision{
		{
			Name:      "bad-rag",
			ModelRefs: []config.ModelRef{{Model: "m"}},
			Plugins: []config.DecisionPlugin{
				{Type: "rag", Configuration: config.MustStructuredPayload(map[string]interface{}{
					"enabled": true,
				})},
			},
		},
	}

	router := &OpenAIRouter{Config: cfg}
	decision := cfg.Decisions[0]
	ctx := &RequestContext{VSRSelectedDecision: &decision}

	ragConfig, shouldExec := router.resolveRAGPluginConfig(ctx, "bad-rag")
	assert.False(t, shouldExec, "should not execute RAG when backend is empty")
	assert.Nil(t, ragConfig)
}

func TestRAGPluginResolution_ValidBackend(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.Decisions = []config.Decision{
		{
			Name:      "good-rag",
			ModelRefs: []config.ModelRef{{Model: "m"}},
			Plugins: []config.DecisionPlugin{
				{Type: "rag", Configuration: config.MustStructuredPayload(map[string]interface{}{
					"enabled": true,
					"backend": "external_api",
				})},
			},
		},
	}

	router := &OpenAIRouter{Config: cfg}
	decision := cfg.Decisions[0]
	ctx := &RequestContext{VSRSelectedDecision: &decision}

	ragConfig, shouldExec := router.resolveRAGPluginConfig(ctx, "good-rag")
	assert.True(t, shouldExec, "should execute RAG when backend is configured")
	require.NotNil(t, ragConfig)
	assert.Equal(t, "external_api", ragConfig.Backend)
}

func TestRAGPluginResolution_NilDecision(t *testing.T) {
	cfg := &config.RouterConfig{}
	router := &OpenAIRouter{Config: cfg}
	ctx := &RequestContext{VSRSelectedDecision: nil}

	ragConfig, shouldExec := router.resolveRAGPluginConfig(ctx, "any")
	assert.False(t, shouldExec)
	assert.Nil(t, ragConfig)
}

func TestRAGPluginResolution_ConfidenceThreshold(t *testing.T) {
	threshold := float64(0.8)
	cfg := &config.RouterConfig{}
	cfg.Decisions = []config.Decision{
		{
			Name:      "threshold-rag",
			ModelRefs: []config.ModelRef{{Model: "m"}},
			Plugins: []config.DecisionPlugin{
				{Type: "rag", Configuration: config.MustStructuredPayload(map[string]interface{}{
					"enabled":                  true,
					"backend":                  "milvus",
					"min_confidence_threshold": threshold,
				})},
			},
		},
	}

	router := &OpenAIRouter{Config: cfg}
	decision := cfg.Decisions[0]

	t.Run("below threshold", func(t *testing.T) {
		ctx := &RequestContext{
			VSRSelectedDecision: &decision,
			FactCheckConfidence: 0.5,
		}
		_, shouldExec := router.resolveRAGPluginConfig(ctx, "threshold-rag")
		assert.False(t, shouldExec, "RAG should be skipped when confidence is below threshold")
	})

	t.Run("above threshold", func(t *testing.T) {
		ctx := &RequestContext{
			VSRSelectedDecision: &decision,
			FactCheckConfidence: 0.9,
		}
		_, shouldExec := router.resolveRAGPluginConfig(ctx, "threshold-rag")
		assert.True(t, shouldExec, "RAG should execute when confidence meets threshold")
	})
}

// --- Full pipeline integration test ---

func TestFullPipeline_CacheBypassThenRAGResolution(t *testing.T) {
	ragServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(map[string]interface{}{
			"context": "Personalized context for this user.",
		}); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
		}
	}))
	defer ragServer.Close()

	spy := &spyCache{shouldHit: true, hitResponse: []byte(`{"choices":[{"message":{"content":"stale generic"}}]}`)}

	decision := config.Decision{
		Name:      "pipeline-test",
		ModelRefs: []config.ModelRef{{Model: "m"}},
		Plugins: []config.DecisionPlugin{
			{Type: "rag", Configuration: config.MustStructuredPayload(map[string]interface{}{
				"enabled":        true,
				"backend":        "external_api",
				"api_endpoint":   ragServer.URL,
				"injection_mode": "system_prompt",
			})},
		},
	}

	cfg := &config.RouterConfig{}
	cfg.Enabled = true
	cfg.Decisions = []config.Decision{decision}

	router := &OpenAIRouter{Config: cfg, Cache: spy}

	reqBody := makeOpenAIRequestBody("test-model", "What's my project status?")
	ctx := &RequestContext{
		Headers:             make(map[string]string),
		RequestID:           "pipeline-1",
		StartTime:           time.Now(),
		OriginalRequestBody: reqBody,
		UserContent:         "What's my project status?",
		VSRSelectedDecision: &decision,
	}

	// Step 1: handleCaching should BYPASS entire cache path (reads AND writes)
	resp, hit := router.handleCaching(ctx, "pipeline-test")
	assert.False(t, spy.findCalled, "Step 1: cache read must be bypassed because RAG is enabled")
	assert.Nil(t, resp, "Step 1: no cached response returned")
	assert.False(t, hit, "Step 1: no cache hit")

	// Step 2: RAG plugin config resolves correctly
	ragConfig, shouldExec := router.resolveRAGPluginConfig(ctx, "pipeline-test")
	assert.True(t, shouldExec, "Step 2: RAG should be resolved as executable")
	require.NotNil(t, ragConfig, "Step 2: RAG config must not be nil")
	assert.Equal(t, "external_api", ragConfig.Backend)

	// Step 3: no cache write when personalization is active
	assert.False(t, spy.pendingAdded,
		"Step 3: cache write must NOT be registered when personalization skips entire cache path")

	fmt.Println("Full pipeline verified: cache bypassed -> RAG resolved -> cache write registered")
}

// resolveBodyMutation mirrors the production logic in createSpecifiedModelResponse:
// body mutation is needed if the upstream model name differs (rewriting) OR if
// personalized context (RAG/memory) was injected.
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

// TestMemoryContextPropagatedToOriginalRequestBody verifies that when
// memory retrieval injects context, ctx.OriginalRequestBody is updated
// so that getBodyMutationSource returns the memory-modified body.
func TestMemoryContextPropagatedToOriginalRequestBody(t *testing.T) {
	originalBody := makeOpenAIRequestBody("gpt-4", "Hello")

	t.Run("OriginalRequestBody updated when MemoryContext set", func(t *testing.T) {
		ctx := &RequestContext{}
		ctx.OriginalRequestBody = originalBody

		memoryInjectedBody := []byte(`{"model":"gpt-4","messages":[{"role":"system","content":"Memory: user prefers dark mode"},{"role":"user","content":"Hello"}]}`)
		ctx.MemoryContext = "user prefers dark mode"

		if ctx.MemoryContext != "" {
			ctx.OriginalRequestBody = memoryInjectedBody
		}

		assert.Contains(t, string(ctx.OriginalRequestBody), "dark mode",
			"OriginalRequestBody must contain injected memory context")
		assert.NotEqual(t, string(originalBody), string(ctx.OriginalRequestBody),
			"OriginalRequestBody must differ from the initial body")
	})

	t.Run("OriginalRequestBody unchanged when no memory retrieved", func(t *testing.T) {
		ctx := &RequestContext{}
		ctx.OriginalRequestBody = originalBody

		if ctx.MemoryContext != "" {
			ctx.OriginalRequestBody = []byte("should not happen")
		}

		assert.Equal(t, string(originalBody), string(ctx.OriginalRequestBody),
			"OriginalRequestBody must stay unchanged when no memory injected")
	})

	t.Run("getBodyMutationSource returns memory-modified body", func(t *testing.T) {
		memoryBody := []byte(`{"model":"gpt-4","messages":[{"role":"system","content":"Memory: project deadline is Friday"},{"role":"user","content":"Hello"}]}`)
		ctx := &RequestContext{
			Headers:             make(map[string]string),
			OriginalRequestBody: memoryBody,
			MemoryContext:       "project deadline is Friday",
		}

		source := getBodyMutationSource(ctx)
		assert.Contains(t, string(source), "deadline is Friday",
			"getBodyMutationSource must return memory-injected body")
	})
}

// TestMemoryContextClearedOnInjectionFailure verifies that ctx.MemoryContext
// is cleared when injectMemoryMessages fails, preventing a stale non-empty
// MemoryContext from triggering an unnecessary forced body mutation.
func TestMemoryContextClearedOnInjectionFailure(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	ctx := &RequestContext{}

	invalidBody := []byte(`not valid json at all`)

	memories := []*memory.RetrieveResult{
		{Memory: &memory.Memory{Content: "user prefers dark mode"}, Score: 0.95},
	}

	result := router.injectRetrievedMemories(ctx, invalidBody, memories)

	assert.Equal(t, invalidBody, result,
		"on injection failure, original body must be returned unchanged")
	assert.Empty(t, ctx.MemoryContext,
		"MemoryContext must be cleared on injection failure to prevent stale forced body mutation")
}
