package extproc

import (
	"strings"
	"testing"

	"github.com/openai/openai-go"
	"github.com/stretchr/testify/assert"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// =====================================================================
// Mock
// =====================================================================

type mockStreamingCache struct {
	addEntryCalled     bool
	addPendingCalled   bool
	findSimilarCalled  bool
	updateCalled       bool
	updateRequestID    string
	updateResponseBody []byte
	addEntryErr        error
	addEntryModel      string
	addEntryQuery      string
	updateErr          error
	lastTTLSeconds     int
	addPendingModel    string
	findSimilarModel   string
	exactResponse      []byte
	exactHit           bool
	exactFindCalled    bool
	exactAdded         bool
}

func (m *mockStreamingCache) IsEnabled() bool { return true }

func (m *mockStreamingCache) CheckConnection() error { return nil }

func (m *mockStreamingCache) AddPendingRequest(
	_ string,
	model string,
	_ string,
	_ []byte,
	_ int,
) error {
	m.addPendingCalled = true
	m.addPendingModel = model
	return nil
}

func (m *mockStreamingCache) UpdateWithResponse(requestID string, responseBody []byte, ttlSeconds int) error {
	m.updateCalled = true
	m.updateRequestID = requestID
	m.updateResponseBody = append([]byte(nil), responseBody...)
	m.lastTTLSeconds = ttlSeconds
	return m.updateErr
}

func (m *mockStreamingCache) AddEntry(
	_ string,
	model string,
	query string,
	_ []byte,
	_ []byte,
	ttlSeconds int,
) error {
	m.addEntryCalled = true
	m.addEntryModel = model
	m.addEntryQuery = query
	m.lastTTLSeconds = ttlSeconds
	return m.addEntryErr
}

func (m *mockStreamingCache) FindSimilar(_ string, _ string) ([]byte, bool, error) {
	return nil, false, nil
}

func (m *mockStreamingCache) FindSimilarWithThreshold(
	model string,
	_ string,
	_ float32,
) ([]byte, bool, error) {
	m.findSimilarCalled = true
	m.findSimilarModel = model
	return nil, false, nil
}

func (m *mockStreamingCache) LookupSimilarWithThreshold(
	model string,
	_ string,
	_ float32,
) (cache.LookupResult, error) {
	m.findSimilarCalled = true
	m.findSimilarModel = model
	return cache.LookupResult{}, nil
}

func (m *mockStreamingCache) LastSimilarity() float32 { return 0 }

func (m *mockStreamingCache) FindExact(_ string, _ string) (cache.LookupResult, error) {
	m.exactFindCalled = true
	return cache.LookupResult{
		ResponseBody: m.exactResponse,
		Found:        m.exactHit,
		Similarity:   1,
	}, nil
}

func (m *mockStreamingCache) AddExact(
	_ string,
	_ string,
	_ []byte,
	ttlSeconds int,
) error {
	m.exactAdded = true
	m.lastTTLSeconds = ttlSeconds
	return nil
}

func (m *mockStreamingCache) Close() error { return nil }

func (m *mockStreamingCache) GetStats() cache.CacheStats { return cache.CacheStats{} }

func cacheRouterForDecision(decision config.Decision) (*mockStreamingCache, *OpenAIRouter, *config.Decision) {
	mockCache := &mockStreamingCache{}
	cfg := &config.RouterConfig{
		SemanticCache: config.SemanticCache{Enabled: true},
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{decision},
		},
	}
	return mockCache, &OpenAIRouter{Cache: mockCache, Config: cfg}, &cfg.Decisions[0]
}

func withSelectedDecision(ctx *RequestContext, decision *config.Decision) *RequestContext {
	ctx.VSRSelectedDecision = decision
	ctx.VSRSelectedDecisionName = decision.Name
	return ctx
}

func TestUpdateResponseCache_WritesExactEntryWithRequestIdentity(t *testing.T) {
	mockCache, router, decision := cacheRouterForDecision(config.Decision{
		Name:      "exact-cache-decision",
		ModelRefs: []config.ModelRef{{Model: "test"}},
		Plugins: []config.DecisionPlugin{{
			Type: "semantic-cache",
			Configuration: config.MustStructuredPayload(map[string]interface{}{
				"enabled": true,
				"mode":    "exact_then_semantic",
			}),
		}},
	})
	ctx := withSelectedDecision(&RequestContext{
		RequestID:                     "req-exact-write",
		CacheRequestModel:             "auto",
		CacheSelectedModel:            "test",
		CacheExactFingerprint:         "fingerprint",
		CacheCompatibilityFingerprint: "compatibility",
		CacheSemanticSafe:             true,
		CacheQuery:                    "hello",
		OriginalRequestBody:           []byte(`{"model":"auto","messages":[{"role":"user","content":"hello"}]}`),
	}, decision)

	router.updateResponseCache(ctx, []byte(`{"ok":true}`))

	assert.True(t, mockCache.addEntryCalled)
	assert.False(t, mockCache.updateCalled)
	assert.True(t, mockCache.exactAdded)
	assert.True(t, strings.HasPrefix(mockCache.addEntryModel, "v2:"))
	assert.Equal(t, "hello", mockCache.addEntryQuery)
}

func TestUpdateResponseCache_NoStoreControlSkipsSemanticAndExactWrites(t *testing.T) {
	mockCache, router, decision := cacheRouterForDecision(config.Decision{
		Name:      "controlled-cache-decision",
		ModelRefs: []config.ModelRef{{Model: "test"}},
		Plugins: []config.DecisionPlugin{{
			Type: "semantic-cache",
			Configuration: config.MustStructuredPayload(map[string]interface{}{
				"enabled": true,
				"mode":    "exact_then_semantic",
			}),
		}},
	})
	ctx := withSelectedDecision(&RequestContext{
		RequestID:                     "req-no-store",
		CacheRequestModel:             "auto",
		CacheSelectedModel:            "test",
		CacheExactFingerprint:         "fingerprint",
		CacheCompatibilityFingerprint: "compatibility",
		CacheSemanticSafe:             true,
		CacheWriteBypass:              true,
	}, decision)

	router.updateResponseCache(ctx, []byte(`{"ok":true}`))

	assert.False(t, mockCache.addEntryCalled)
	assert.False(t, mockCache.updateCalled)
	assert.False(t, mockCache.exactAdded)
}

// =====================================================================
// NON-STREAMING: updateResponseCache
// =====================================================================

func TestUpdateResponseCache_SkipsWhenDecisionCacheDisabled(t *testing.T) {
	mockCache, router, decision := cacheRouterForDecision(config.Decision{
		Name:      "no-cache-decision",
		ModelRefs: []config.ModelRef{{Model: "test"}},
	})
	ctx := withSelectedDecision(&RequestContext{
		RequestID:           "req-1",
		RequestModel:        "test",
		RequestQuery:        "hello",
		OriginalRequestBody: []byte(`{"model":"test","messages":[{"role":"user","content":"hello"}]}`),
	}, decision)

	router.updateResponseCache(ctx, []byte(`{"ok":true}`))
	assert.False(t, mockCache.addEntryCalled, "should not store response when decision has no semantic-cache plugin")
}

func TestUpdateResponseCache_SkipsWhenDecisionCacheExplicitlyDisabled(t *testing.T) {
	mockCache, router, decision := cacheRouterForDecision(config.Decision{
		Name:      "disabled-cache-decision",
		ModelRefs: []config.ModelRef{{Model: "test"}},
		Plugins: []config.DecisionPlugin{{
			Type:          "semantic-cache",
			Configuration: config.MustStructuredPayload(map[string]interface{}{"enabled": false}),
		}},
	})
	ctx := withSelectedDecision(&RequestContext{
		RequestID:           "req-1",
		RequestModel:        "test",
		RequestQuery:        "hello",
		OriginalRequestBody: []byte(`{"model":"test","messages":[{"role":"user","content":"hello"}]}`),
	}, decision)

	router.updateResponseCache(ctx, []byte(`{"ok":true}`))
	assert.False(t, mockCache.addEntryCalled, "should not store response when decision has semantic-cache disabled")
}

func TestUpdateResponseCache_StoresWhenDecisionCacheEnabled(t *testing.T) {
	mockCache, router, decision := cacheRouterForDecision(config.Decision{
		Name:      "cache-decision",
		ModelRefs: []config.ModelRef{{Model: "test"}},
		Plugins: []config.DecisionPlugin{{
			Type:          "semantic-cache",
			Configuration: config.MustStructuredPayload(map[string]interface{}{"enabled": true}),
		}},
	})
	ctx := withSelectedDecision(&RequestContext{
		RequestID:           "req-1",
		RequestModel:        "test",
		RequestQuery:        "hello",
		OriginalRequestBody: []byte(`{"model":"test","messages":[{"role":"user","content":"hello"}]}`),
	}, decision)

	router.updateResponseCache(ctx, []byte(`{"ok":true}`))
	assert.True(t, mockCache.addEntryCalled, "should store response when decision has semantic-cache enabled")
}

func TestUpdateResponseCache_StoresWhenNoDecisionSelectedAndNoDecisionsConfigured(t *testing.T) {
	mockCache := &mockStreamingCache{}
	cfg := &config.RouterConfig{
		SemanticCache: config.SemanticCache{Enabled: true},
	}
	router := &OpenAIRouter{Cache: mockCache, Config: cfg}
	ctx := &RequestContext{
		RequestID:               "req-1",
		RequestModel:            "test",
		RequestQuery:            "hello",
		OriginalRequestBody:     []byte(`{"model":"test","messages":[{"role":"user","content":"hello"}]}`),
		VSRSelectedDecisionName: "",
	}

	router.updateResponseCache(ctx, []byte(`{"ok":true}`))
	assert.True(t, mockCache.addEntryCalled, "should store response when no decision is selected (global cache applies)")
}

func TestUpdateResponseCache_SkipsWhenNoDecisionSelectedButDecisionsConfigured(t *testing.T) {
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
		RequestID:               "req-1",
		VSRSelectedDecisionName: "",
	}

	router.updateResponseCache(ctx, []byte(`{"ok":true}`))
	assert.False(t, mockCache.addEntryCalled, "should not store response when decisions exist but no decision matched")
}

// =====================================================================
// STREAMING: cacheReconstructedStreamingResponse
// =====================================================================

func TestCacheReconstructedStreamingResponseAddsCompletedEntry(t *testing.T) {
	mockCache := &mockStreamingCache{}
	router := &OpenAIRouter{Cache: mockCache}
	ctx := &RequestContext{
		RequestID:    "req-1",
		RequestModel: "test-model",
		RequestQuery: "hello",
	}

	err := router.cacheReconstructedStreamingResponse(ctx, []byte(`{"ok":true}`))
	assert.NoError(t, err)
	assert.True(t, mockCache.addEntryCalled, "streaming finalize must write one completed entry")
	assert.False(t, mockCache.updateCalled, "streaming cache must not depend on shared pending state")
}

// Regression for #2030: ctx.RequestModel is overwritten from the client alias
// to the resolved model by finalize time, so finalize must key off request_id.
func TestCacheReconstructedStreamingResponseUpdatesAfterModelResolved(t *testing.T) {
	mockCache := &mockStreamingCache{}
	router := &OpenAIRouter{Cache: mockCache}
	ctx := &RequestContext{
		RequestID:    "req-1",
		RequestModel: "doubao-1-5-lite-32k-250115",
		RequestQuery: "who discovered gravity?",
	}

	err := router.cacheReconstructedStreamingResponse(ctx, []byte(`{"ok":true}`))
	assert.NoError(t, err)
	assert.True(t, mockCache.addEntryCalled)
	assert.False(t, mockCache.updateCalled)
}

func TestCacheReconstructedStreamingResponseUpdatesWithoutQueryMetadata(t *testing.T) {
	mockCache := &mockStreamingCache{}
	router := &OpenAIRouter{Cache: mockCache}
	ctx := &RequestContext{RequestID: "req-1"}

	err := router.cacheReconstructedStreamingResponse(ctx, []byte(`{"ok":true}`))
	assert.NoError(t, err)
	assert.False(t, mockCache.addEntryCalled)
	assert.False(t, mockCache.updateCalled)
}

func TestCacheReconstructedStreamingResponseSkipsWithoutRequestID(t *testing.T) {
	mockCache := &mockStreamingCache{}
	router := &OpenAIRouter{Cache: mockCache}
	ctx := &RequestContext{RequestModel: "test-model", RequestQuery: "hello"}

	err := router.cacheReconstructedStreamingResponse(ctx, []byte(`{"ok":true}`))
	assert.NoError(t, err)
	assert.False(t, mockCache.addEntryCalled)
	assert.False(t, mockCache.updateCalled)
}

// =====================================================================
// STREAMING: cacheReconstructedStreamingResponse — per-decision
// =====================================================================

func TestCacheReconstructedStreamingResponse_SkipsWhenDecisionCacheDisabled(t *testing.T) {
	mockCache, router, decision := cacheRouterForDecision(config.Decision{
		Name:      "no-cache-decision",
		ModelRefs: []config.ModelRef{{Model: "test"}},
	})
	ctx := withSelectedDecision(&RequestContext{
		RequestID:    "req-1",
		RequestModel: "test-model",
		RequestQuery: "hello",
	}, decision)

	err := router.cacheReconstructedStreamingResponse(ctx, []byte(`{"ok":true}`))
	assert.NoError(t, err)
	assert.False(t, mockCache.addEntryCalled, "should not call AddEntry when decision has no semantic-cache plugin")
	assert.False(t, mockCache.updateCalled, "should not call UpdateWithResponse when decision has no semantic-cache plugin")
}

func TestCacheReconstructedStreamingResponse_SkipsWhenDecisionCacheExplicitlyDisabled(t *testing.T) {
	mockCache, router, decision := cacheRouterForDecision(config.Decision{
		Name:      "disabled-cache-decision",
		ModelRefs: []config.ModelRef{{Model: "test"}},
		Plugins: []config.DecisionPlugin{{
			Type:          "semantic-cache",
			Configuration: config.MustStructuredPayload(map[string]interface{}{"enabled": false}),
		}},
	})
	ctx := withSelectedDecision(&RequestContext{
		RequestID:    "req-1",
		RequestModel: "test-model",
		RequestQuery: "hello",
	}, decision)

	err := router.cacheReconstructedStreamingResponse(ctx, []byte(`{"ok":true}`))
	assert.NoError(t, err)
	assert.False(t, mockCache.addEntryCalled, "should not call AddEntry when decision has semantic-cache disabled")
	assert.False(t, mockCache.updateCalled, "should not call UpdateWithResponse when decision has semantic-cache disabled")
}

func TestCacheReconstructedStreamingResponse_StoresWhenDecisionCacheEnabled(t *testing.T) {
	mockCache, router, decision := cacheRouterForDecision(config.Decision{
		Name:      "cache-decision",
		ModelRefs: []config.ModelRef{{Model: "test"}},
		Plugins: []config.DecisionPlugin{{
			Type:          "semantic-cache",
			Configuration: config.MustStructuredPayload(map[string]interface{}{"enabled": true}),
		}},
	})
	ctx := withSelectedDecision(&RequestContext{
		RequestID:    "req-1",
		RequestModel: "test-model",
		RequestQuery: "hello",
	}, decision)

	err := router.cacheReconstructedStreamingResponse(ctx, []byte(`{"ok":true}`))
	assert.NoError(t, err)
	assert.True(t, mockCache.addEntryCalled, "should write a completed cache entry")
	assert.False(t, mockCache.updateCalled)
}

// =====================================================================
// STREAMING: cacheStreamingResponse — personalized context
// =====================================================================

func TestCacheStreamingSkippedWhenRAGContextPresent(t *testing.T) {
	mockCache := &mockStreamingCache{}
	router := &OpenAIRouter{Cache: mockCache}
	ctx := &RequestContext{
		RequestID:           "req-rag",
		RequestModel:        "test-model",
		RequestQuery:        "what is our refund policy",
		StreamingComplete:   true,
		StreamingContent:    "Based on your documents...",
		StreamingMetadata:   map[string]interface{}{"id": "chatcmpl-1", "model": "test-model", "created": int64(1234567890)},
		RAGRetrievedContext: "Internal policy doc: refunds within 30 days",
	}

	err := router.cacheStreamingResponse(ctx)
	assert.NoError(t, err)
	assert.False(t, mockCache.addEntryCalled, "cache write must be skipped when RAG context is present")
	assert.False(t, mockCache.updateCalled, "cache update must be skipped when RAG context is present")
}

func TestCacheStreamingSkippedWhenMemoryContextPresent(t *testing.T) {
	mockCache := &mockStreamingCache{}
	router := &OpenAIRouter{Cache: mockCache}
	ctx := &RequestContext{
		RequestID:         "req-mem",
		RequestModel:      "test-model",
		RequestQuery:      "remind me what we discussed",
		StreamingComplete: true,
		StreamingContent:  "Last time you mentioned...",
		StreamingMetadata: map[string]interface{}{"id": "chatcmpl-2", "model": "test-model", "created": int64(1234567890)},
		MemoryContext:     "User previously discussed project deadlines",
	}

	err := router.cacheStreamingResponse(ctx)
	assert.NoError(t, err)
	assert.False(t, mockCache.addEntryCalled, "cache write must be skipped when memory context is present")
	assert.False(t, mockCache.updateCalled, "cache update must be skipped when memory context is present")
}

func TestCacheStreamingSkippedWhenPIIDetected(t *testing.T) {
	mockCache := &mockStreamingCache{}
	router := &OpenAIRouter{Cache: mockCache}
	ctx := &RequestContext{
		RequestID:         "req-pii",
		RequestModel:      "test-model",
		RequestQuery:      "parse this email",
		StreamingComplete: true,
		StreamingContent:  "The email contains...",
		StreamingMetadata: map[string]interface{}{"id": "chatcmpl-3", "model": "test-model", "created": int64(1234567890)},
		PIIDetected:       true,
	}

	err := router.cacheStreamingResponse(ctx)
	assert.NoError(t, err)
	assert.False(t, mockCache.addEntryCalled, "cache write must be skipped when PII is detected")
	assert.False(t, mockCache.updateCalled, "cache update must be skipped when PII is detected")
}

func TestCacheStreamingAllowedForGenericRequest(t *testing.T) {
	mockCache := &mockStreamingCache{}
	router := &OpenAIRouter{Cache: mockCache}
	ctx := &RequestContext{
		RequestID:         "req-generic",
		RequestModel:      "test-model",
		RequestQuery:      "explain quantum computing",
		StreamingComplete: true,
		StreamingContent:  "Quantum computing uses qubits...",
		StreamingMetadata: map[string]interface{}{"id": "chatcmpl-4", "model": "test-model", "created": int64(1234567890)},
	}

	err := router.cacheStreamingResponse(ctx)
	assert.NoError(t, err)
	assert.True(t, mockCache.addEntryCalled,
		"cache write must proceed for generic requests without personalized context")
	assert.False(t, mockCache.updateCalled)
}

func TestBuildReconstructedStreamingResponseIncludesToolCallsForReplay(t *testing.T) {
	ctx := &RequestContext{
		StreamingMetadata: map[string]interface{}{
			"id":            "chatcmpl-123",
			"model":         "test-model",
			"created":       int64(1234567890),
			"finish_reason": "tool_calls",
		},
		StreamingToolCalls: map[int]*StreamingToolCallState{
			0: {
				ID:        "call_weather",
				Name:      "get_weather",
				Arguments: "{\"location\":\"San Francisco\"}",
			},
		},
	}

	body, err := buildReconstructedStreamingResponse(ctx, openai.CompletionUsage{
		PromptTokens:     10,
		CompletionTokens: 2,
		TotalTokens:      12,
	}, true)
	assert.NoError(t, err)
	assert.JSONEq(t, `{
		"id": "chatcmpl-123",
		"object": "chat.completion",
		"created": 1234567890,
		"model": "test-model",
		"choices": [
			{
				"index": 0,
				"message": {
					"role": "assistant",
					"content": null,
					"tool_calls": [
						{
							"id": "call_weather",
							"type": "function",
							"function": {
								"name": "get_weather",
								"arguments": "{\"location\":\"San Francisco\"}"
							}
						}
					]
				},
				"finish_reason": "tool_calls"
			}
		],
		"usage": {
			"prompt_tokens": 10,
			"completion_tokens": 2,
			"total_tokens": 12
		}
	}`, string(body))
}

func TestBuildReconstructedStreamingResponseSkipsToolCallsOnlyForCache(t *testing.T) {
	ctx := &RequestContext{
		StreamingMetadata: map[string]interface{}{
			"id":      "chatcmpl-123",
			"model":   "test-model",
			"created": int64(1234567890),
		},
		StreamingToolCalls: map[int]*StreamingToolCallState{
			0: {
				ID:        "call_weather",
				Name:      "get_weather",
				Arguments: "{\"location\":\"San Francisco\"}",
			},
		},
	}

	_, err := buildReconstructedStreamingResponse(ctx, openai.CompletionUsage{}, false)
	assert.ErrorIs(t, err, errSkipStreamingCache)
}
