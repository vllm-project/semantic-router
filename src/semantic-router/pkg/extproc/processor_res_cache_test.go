package extproc

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

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

func (m *mockStreamingCache) CheckConnection(context.Context) error { return nil }

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
	_ context.Context,
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
	_ context.Context,
	model string,
	_ string,
	_ float32,
) (cache.LookupResult, error) {
	m.findSimilarCalled = true
	m.findSimilarModel = model
	return cache.LookupResult{}, nil
}

func (m *mockStreamingCache) LastSimilarity() float32 { return 0 }

func (m *mockStreamingCache) FindExact(_ context.Context, _ string, _ string) (cache.LookupResult, error) {
	m.exactFindCalled = true
	return cache.LookupResult{
		ResponseBody: m.exactResponse,
		Found:        m.exactHit,
		Similarity:   1,
	}, nil
}

func (m *mockStreamingCache) AddExact(
	_ context.Context,
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
	if ctx.Headers == nil {
		ctx.Headers = map[string]string{"x-authz-user-id": "cache-test-user"}
	}
	return ctx
}

func TestUpdateResponseCache_WritesExactEntryWithRequestIdentity(t *testing.T) {
	mockCache, router, decision := cacheRouterForDecision(config.Decision{
		Name:      "exact-cache-decision",
		ModelRefs: []config.ModelRef{{Model: "test"}},
		Plugins: []config.DecisionPlugin{{
			Type: config.DecisionPluginResponseCache,
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
		SemanticRequest:               testNeutralRequest("auto", "hello"),
	}, decision)

	router.updateResponseCache(ctx, []byte(`{"ok":true}`))

	assert.True(t, mockCache.addEntryCalled)
	assert.False(t, mockCache.updateCalled)
	assert.True(t, mockCache.exactAdded)
	assert.Contains(t, mockCache.addEntryModel, "exact-cache-decision")
	assert.Equal(t, "hello", mockCache.addEntryQuery)
}

func TestUpdateResponseCache_NoStoreControlSkipsSemanticAndExactWrites(t *testing.T) {
	mockCache, router, decision := cacheRouterForDecision(config.Decision{
		Name:      "controlled-cache-decision",
		ModelRefs: []config.ModelRef{{Model: "test"}},
		Plugins: []config.DecisionPlugin{{
			Type: config.DecisionPluginResponseCache,
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
		RequestID:       "req-1",
		RequestModel:    "test",
		RequestQuery:    "hello",
		SemanticRequest: testNeutralRequest("test", "hello"),
	}, decision)

	router.updateResponseCache(ctx, []byte(`{"ok":true}`))
	assert.False(t, mockCache.addEntryCalled, "should not store response when decision has no semantic-cache plugin")
}

func TestUpdateResponseCache_SkipsWhenDecisionCacheExplicitlyDisabled(t *testing.T) {
	mockCache, router, decision := cacheRouterForDecision(config.Decision{
		Name:      "disabled-cache-decision",
		ModelRefs: []config.ModelRef{{Model: "test"}},
		Plugins: []config.DecisionPlugin{{
			Type:          config.DecisionPluginResponseCache,
			Configuration: config.MustStructuredPayload(map[string]interface{}{"enabled": false}),
		}},
	})
	ctx := withSelectedDecision(&RequestContext{
		RequestID:       "req-1",
		RequestModel:    "test",
		RequestQuery:    "hello",
		SemanticRequest: testNeutralRequest("test", "hello"),
	}, decision)

	router.updateResponseCache(ctx, []byte(`{"ok":true}`))
	assert.False(t, mockCache.addEntryCalled, "should not store response when decision has semantic-cache disabled")
}

func TestUpdateResponseCache_StoresWhenDecisionCacheEnabled(t *testing.T) {
	mockCache, router, decision := cacheRouterForDecision(config.Decision{
		Name:      "cache-decision",
		ModelRefs: []config.ModelRef{{Model: "test"}},
		Plugins: []config.DecisionPlugin{{
			Type:          config.DecisionPluginResponseCache,
			Configuration: config.MustStructuredPayload(map[string]interface{}{"enabled": true}),
		}},
	})
	ctx := withSelectedDecision(&RequestContext{
		RequestID:       "req-1",
		RequestModel:    "test",
		RequestQuery:    "hello",
		SemanticRequest: testNeutralRequest("test", "hello"),
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
		SemanticRequest:         testNeutralRequest("test", "hello"),
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

func TestCacheWriteContextSurvivesRequestCancellation(t *testing.T) {
	type ctxKey string
	const traceKey ctxKey = "trace"

	parent, cancel := context.WithCancel(context.WithValue(context.Background(), traceKey, "span-1"))
	writeCtx := cacheWriteContext(&RequestContext{TraceContext: parent})
	cancel()

	require.NoError(t, writeCtx.Err(), "a cancelled request must not cancel its cache write")
	assert.Equal(t, "span-1", writeCtx.Value(traceKey), "trace values must survive the detach")
	assert.ErrorIs(t, parent.Err(), context.Canceled, "the request context itself still cancels")
}

func TestCacheWriteContextFallsBackToBackground(t *testing.T) {
	assert.Equal(t, context.Background(), cacheWriteContext(nil))
	assert.Equal(t, context.Background(), cacheWriteContext(&RequestContext{}))
}
