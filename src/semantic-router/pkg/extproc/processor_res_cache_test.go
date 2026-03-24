package extproc

import (
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// =====================================================================
// Mock
// =====================================================================

type mockStreamingCache struct {
	addEntryCalled      bool
	addPendingCalled    bool
	findSimilarCalled   bool
	updateCalled        bool
	addEntryRequestBody []byte
	addEntryErr         error
	updateErr           error
}

func (m *mockStreamingCache) IsEnabled() bool { return true }

func (m *mockStreamingCache) CheckConnection() error { return nil }

func (m *mockStreamingCache) AddPendingRequest(
	_ string,
	_ string,
	_ string,
	_ []byte,
	_ int,
) error {
	m.addPendingCalled = true
	return nil
}

func (m *mockStreamingCache) UpdateWithResponse(_ string, _ []byte, _ int) error {
	m.updateCalled = true
	return m.updateErr
}

func (m *mockStreamingCache) AddEntry(
	_ string,
	_ string,
	_ string,
	requestBody []byte,
	_ []byte,
	_ int,
) error {
	m.addEntryCalled = true
	m.addEntryRequestBody = append([]byte(nil), requestBody...)
	return m.addEntryErr
}

func (m *mockStreamingCache) FindSimilar(_ string, _ string) ([]byte, bool, error) {
	return nil, false, nil
}

func (m *mockStreamingCache) FindSimilarWithThreshold(
	_ string,
	_ string,
	_ float32,
) ([]byte, bool, error) {
	m.findSimilarCalled = true
	return nil, false, nil
}

func (m *mockStreamingCache) LastSimilarity() float32 { return 0 }

func (m *mockStreamingCache) Close() error { return nil }

func (m *mockStreamingCache) GetStats() cache.CacheStats { return cache.CacheStats{} }

// =====================================================================
// NON-STREAMING: updateResponseCache
// =====================================================================

func TestUpdateResponseCache_SkipsWhenDecisionCacheDisabled(t *testing.T) {
	mockCache := &mockStreamingCache{}
	cfg := &config.RouterConfig{
		SemanticCache: config.SemanticCache{Enabled: true},
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{
				{
					Name:      "no-cache-decision",
					ModelRefs: []config.ModelRef{{Model: "test"}},
				},
			},
		},
	}
	router := &OpenAIRouter{Cache: mockCache, Config: cfg}
	ctx := &RequestContext{
		RequestID:               "req-1",
		VSRSelectedDecisionName: "no-cache-decision",
	}

	router.updateResponseCache(ctx, []byte(`{"ok":true}`))
	assert.False(t, mockCache.updateCalled, "should not store response when decision has no semantic-cache plugin")
}

func TestUpdateResponseCache_SkipsWhenDecisionCacheExplicitlyDisabled(t *testing.T) {
	mockCache := &mockStreamingCache{}
	cfg := &config.RouterConfig{
		SemanticCache: config.SemanticCache{Enabled: true},
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{
				{
					Name:      "disabled-cache-decision",
					ModelRefs: []config.ModelRef{{Model: "test"}},
					Plugins: []config.DecisionPlugin{
						{
							Type:          "semantic-cache",
							Configuration: config.MustStructuredPayload(map[string]interface{}{"enabled": false}),
						},
					},
				},
			},
		},
	}
	router := &OpenAIRouter{Cache: mockCache, Config: cfg}
	ctx := &RequestContext{
		RequestID:               "req-1",
		VSRSelectedDecisionName: "disabled-cache-decision",
	}

	router.updateResponseCache(ctx, []byte(`{"ok":true}`))
	assert.False(t, mockCache.updateCalled, "should not store response when decision has semantic-cache disabled")
}

func TestUpdateResponseCache_StoresWhenDecisionCacheEnabled(t *testing.T) {
	mockCache := &mockStreamingCache{}
	cfg := &config.RouterConfig{
		SemanticCache: config.SemanticCache{Enabled: true},
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{
				{
					Name:      "cache-decision",
					ModelRefs: []config.ModelRef{{Model: "test"}},
					Plugins: []config.DecisionPlugin{
						{
							Type:          "semantic-cache",
							Configuration: config.MustStructuredPayload(map[string]interface{}{"enabled": true}),
						},
					},
				},
			},
		},
	}
	router := &OpenAIRouter{Cache: mockCache, Config: cfg}
	ctx := &RequestContext{
		RequestID:               "req-1",
		VSRSelectedDecisionName: "cache-decision",
	}

	router.updateResponseCache(ctx, []byte(`{"ok":true}`))
	assert.True(t, mockCache.updateCalled, "should store response when decision has semantic-cache enabled")
}

func TestUpdateResponseCache_StoresWhenNoDecisionSelectedAndNoDecisionsConfigured(t *testing.T) {
	mockCache := &mockStreamingCache{}
	cfg := &config.RouterConfig{
		SemanticCache: config.SemanticCache{Enabled: true},
	}
	router := &OpenAIRouter{Cache: mockCache, Config: cfg}
	ctx := &RequestContext{
		RequestID:               "req-1",
		VSRSelectedDecisionName: "",
	}

	router.updateResponseCache(ctx, []byte(`{"ok":true}`))
	assert.True(t, mockCache.updateCalled, "should store response when no decision is selected (global cache applies)")
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
	assert.False(t, mockCache.updateCalled, "should not store response when decisions exist but no decision matched")
}

// =====================================================================
// STREAMING: cacheReconstructedStreamingResponse
// =====================================================================

func TestCacheReconstructedStreamingResponseUsesAddEntry(t *testing.T) {
	mockCache := &mockStreamingCache{}
	router := &OpenAIRouter{Cache: mockCache}
	ctx := &RequestContext{
		RequestID:    "req-1",
		RequestModel: "test-model",
		RequestQuery: "hello",
	}

	err := router.cacheReconstructedStreamingResponse(ctx, []byte(`{"ok":true}`))
	assert.NoError(t, err)
	assert.True(t, mockCache.addEntryCalled)
	assert.False(t, mockCache.updateCalled)
	assert.JSONEq(t, `{}`, string(mockCache.addEntryRequestBody))
}

func TestCacheReconstructedStreamingResponseFallsBackToUpdate(t *testing.T) {
	mockCache := &mockStreamingCache{addEntryErr: errors.New("boom")}
	router := &OpenAIRouter{Cache: mockCache}
	ctx := &RequestContext{
		RequestID:           "req-1",
		RequestModel:        "test-model",
		RequestQuery:        "hello",
		OriginalRequestBody: []byte(`{"messages":[]}`),
	}

	err := router.cacheReconstructedStreamingResponse(ctx, []byte(`{"ok":true}`))
	assert.NoError(t, err)
	assert.True(t, mockCache.addEntryCalled)
	assert.True(t, mockCache.updateCalled)
	assert.JSONEq(t, `{"messages":[]}`, string(mockCache.addEntryRequestBody))
}

func TestCacheReconstructedStreamingResponseUpdatesWithoutQueryMetadata(t *testing.T) {
	mockCache := &mockStreamingCache{}
	router := &OpenAIRouter{Cache: mockCache}
	ctx := &RequestContext{RequestID: "req-1"}

	err := router.cacheReconstructedStreamingResponse(ctx, []byte(`{"ok":true}`))
	assert.NoError(t, err)
	assert.False(t, mockCache.addEntryCalled)
	assert.True(t, mockCache.updateCalled)
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
	mockCache := &mockStreamingCache{}
	cfg := &config.RouterConfig{
		SemanticCache: config.SemanticCache{Enabled: true},
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{
				{
					Name:      "no-cache-decision",
					ModelRefs: []config.ModelRef{{Model: "test"}},
				},
			},
		},
	}
	router := &OpenAIRouter{Cache: mockCache, Config: cfg}
	ctx := &RequestContext{
		RequestID:               "req-1",
		RequestModel:            "test-model",
		RequestQuery:            "hello",
		VSRSelectedDecisionName: "no-cache-decision",
	}

	err := router.cacheReconstructedStreamingResponse(ctx, []byte(`{"ok":true}`))
	assert.NoError(t, err)
	assert.False(t, mockCache.addEntryCalled, "should not call AddEntry when decision has no semantic-cache plugin")
	assert.False(t, mockCache.updateCalled, "should not call UpdateWithResponse when decision has no semantic-cache plugin")
}

func TestCacheReconstructedStreamingResponse_SkipsWhenDecisionCacheExplicitlyDisabled(t *testing.T) {
	mockCache := &mockStreamingCache{}
	cfg := &config.RouterConfig{
		SemanticCache: config.SemanticCache{Enabled: true},
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{
				{
					Name:      "disabled-cache-decision",
					ModelRefs: []config.ModelRef{{Model: "test"}},
					Plugins: []config.DecisionPlugin{
						{
							Type:          "semantic-cache",
							Configuration: config.MustStructuredPayload(map[string]interface{}{"enabled": false}),
						},
					},
				},
			},
		},
	}
	router := &OpenAIRouter{Cache: mockCache, Config: cfg}
	ctx := &RequestContext{
		RequestID:               "req-1",
		RequestModel:            "test-model",
		RequestQuery:            "hello",
		VSRSelectedDecisionName: "disabled-cache-decision",
	}

	err := router.cacheReconstructedStreamingResponse(ctx, []byte(`{"ok":true}`))
	assert.NoError(t, err)
	assert.False(t, mockCache.addEntryCalled, "should not call AddEntry when decision has semantic-cache disabled")
	assert.False(t, mockCache.updateCalled, "should not call UpdateWithResponse when decision has semantic-cache disabled")
}

func TestCacheReconstructedStreamingResponse_StoresWhenDecisionCacheEnabled(t *testing.T) {
	mockCache := &mockStreamingCache{}
	cfg := &config.RouterConfig{
		SemanticCache: config.SemanticCache{Enabled: true},
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{
				{
					Name:      "cache-decision",
					ModelRefs: []config.ModelRef{{Model: "test"}},
					Plugins: []config.DecisionPlugin{
						{
							Type:          "semantic-cache",
							Configuration: config.MustStructuredPayload(map[string]interface{}{"enabled": true}),
						},
					},
				},
			},
		},
	}
	router := &OpenAIRouter{Cache: mockCache, Config: cfg}
	ctx := &RequestContext{
		RequestID:               "req-1",
		RequestModel:            "test-model",
		RequestQuery:            "hello",
		VSRSelectedDecisionName: "cache-decision",
	}

	err := router.cacheReconstructedStreamingResponse(ctx, []byte(`{"ok":true}`))
	assert.NoError(t, err)
	assert.True(t, mockCache.addEntryCalled, "should call AddEntry when decision has semantic-cache enabled")
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
	assert.True(t, mockCache.addEntryCalled || mockCache.updateCalled,
		"cache write must proceed for generic requests without personalized context")
}
