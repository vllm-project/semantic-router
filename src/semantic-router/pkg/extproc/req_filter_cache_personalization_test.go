package extproc

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

func TestHandleCachingSkipsCacheForRAGEnabledDecision(t *testing.T) {
	mockCache := &mockPersonalizationCache{}
	router, decision := newCachePersonalizationTestRouter(
		mockCache,
		false,
		config.DecisionPlugin{
			Type: "rag",
			Configuration: config.MustStructuredPayload(map[string]interface{}{
				"enabled": true,
				"backend": "milvus",
			}),
		},
	)
	ctx := &RequestContext{
		RequestID:           "req-rag",
		OriginalRequestBody: []byte(`{"model":"test-model","messages":[{"role":"user","content":"hello"}]}`),
		VSRSelectedDecision: decision,
	}

	response, shouldReturn := router.handleCaching(ctx, decision.Name)

	assert.Nil(t, response)
	assert.False(t, shouldReturn)
	assert.Equal(t, "test-model", ctx.RequestModel)
	assert.Equal(t, "hello", ctx.RequestQuery)
	assert.False(t, mockCache.findSimilarCalled)
	assert.False(t, mockCache.addPendingCalled)
}

func TestHandleCachingSkipsCacheForGlobalMemoryEnabledWithUserID(t *testing.T) {
	mockCache := &mockPersonalizationCache{}
	router, decision := newCachePersonalizationTestRouter(mockCache, true)
	ctx := &RequestContext{
		Headers:             map[string]string{headers.AuthzUserID: "user-123"},
		RequestID:           "req-memory",
		OriginalRequestBody: []byte(`{"model":"test-model","messages":[{"role":"user","content":"what is my budget?"}]}`),
		VSRSelectedDecision: decision,
	}

	response, shouldReturn := router.handleCaching(ctx, decision.Name)

	assert.Nil(t, response)
	assert.False(t, shouldReturn)
	assert.False(t, mockCache.findSimilarCalled)
	assert.False(t, mockCache.addPendingCalled)
}

func TestHandleCachingDoesNotSkipCacheForGlobalMemoryWithoutUserID(t *testing.T) {
	mockCache := &mockPersonalizationCache{}
	router, decision := newCachePersonalizationTestRouter(mockCache, true)
	ctx := &RequestContext{
		RequestID:           "req-memory-anon",
		OriginalRequestBody: []byte(`{"model":"test-model","messages":[{"role":"user","content":"what is my budget?"}]}`),
		VSRSelectedDecision: decision,
	}

	response, shouldReturn := router.handleCaching(ctx, decision.Name)

	assert.Nil(t, response)
	assert.False(t, shouldReturn)
	assert.True(t, mockCache.findSimilarCalled)
	assert.True(t, mockCache.addPendingCalled)
}

func TestUpdateResponseCacheSkipsWhenMemoryContextPresent(t *testing.T) {
	mockCache := &mockPersonalizationCache{}
	router := &OpenAIRouter{Cache: mockCache}
	ctx := &RequestContext{
		RequestID:     "req-memory-context",
		MemoryContext: "remember this user preference",
	}

	router.updateResponseCache(ctx, []byte(`{"ok":true}`))

	assert.False(t, mockCache.updateCalled)
}

func TestUpdateResponseCacheUpdatesWhenNotPersonalized(t *testing.T) {
	mockCache := &mockPersonalizationCache{}
	router := &OpenAIRouter{Cache: mockCache}
	ctx := &RequestContext{
		RequestID: "req-normal",
	}

	router.updateResponseCache(ctx, []byte(`{"ok":true}`))

	assert.True(t, mockCache.updateCalled)
}

func TestUpdateResponseCacheUpdatesWhenGlobalMemoryHasNoUserID(t *testing.T) {
	mockCache := &mockPersonalizationCache{}
	router, decision := newCachePersonalizationTestRouter(mockCache, true)
	ctx := &RequestContext{
		RequestID:           "req-memory-no-user",
		RequestQuery:        "what is my budget?",
		VSRSelectedDecision: decision,
	}

	router.updateResponseCache(ctx, []byte(`{"ok":true}`))

	assert.True(t, mockCache.updateCalled)
}

func TestCacheReconstructedStreamingResponseSkipsWhenDecisionUsesRAG(t *testing.T) {
	mockCache := &mockPersonalizationCache{}
	router, decision := newCachePersonalizationTestRouter(
		mockCache,
		false,
		config.DecisionPlugin{
			Type: "rag",
			Configuration: config.MustStructuredPayload(map[string]interface{}{
				"enabled": true,
				"backend": "milvus",
			}),
		},
	)
	ctx := &RequestContext{
		RequestID:           "req-stream-rag",
		RequestModel:        "test-model",
		RequestQuery:        "hello",
		VSRSelectedDecision: decision,
	}

	err := router.cacheReconstructedStreamingResponse(ctx, []byte(`{"ok":true}`))

	assert.NoError(t, err)
	assert.False(t, mockCache.addEntryCalled)
	assert.False(t, mockCache.updateCalled)
}

func newCachePersonalizationTestRouter(
	cacheBackend cache.CacheBackend,
	globalMemoryEnabled bool,
	extraPlugins ...config.DecisionPlugin,
) (*OpenAIRouter, *config.Decision) {
	decision := config.Decision{
		Name: "test-decision",
		Plugins: append([]config.DecisionPlugin{
			{
				Type: "semantic-cache",
				Configuration: config.MustStructuredPayload(map[string]interface{}{
					"enabled": true,
				}),
			},
		}, extraPlugins...),
	}
	cfg := &config.RouterConfig{
		SemanticCache: config.SemanticCache{
			Enabled:    true,
			TTLSeconds: 60,
		},
		Memory: config.MemoryConfig{
			Enabled: globalMemoryEnabled,
		},
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{decision},
		},
	}
	return &OpenAIRouter{Config: cfg, Cache: cacheBackend}, &cfg.Decisions[0]
}

type mockPersonalizationCache struct {
	addPendingCalled  bool
	findSimilarCalled bool
	updateCalled      bool
	addEntryCalled    bool
}

func (m *mockPersonalizationCache) IsEnabled() bool { return true }

func (m *mockPersonalizationCache) CheckConnection() error { return nil }

func (m *mockPersonalizationCache) AddPendingRequest(
	_ string,
	_ string,
	_ string,
	_ []byte,
	_ int,
) error {
	m.addPendingCalled = true
	return nil
}

func (m *mockPersonalizationCache) UpdateWithResponse(_ string, _ []byte, _ int) error {
	m.updateCalled = true
	return nil
}

func (m *mockPersonalizationCache) AddEntry(
	_ string,
	_ string,
	_ string,
	_ []byte,
	_ []byte,
	_ int,
) error {
	m.addEntryCalled = true
	return nil
}

func (m *mockPersonalizationCache) FindSimilar(_ string, _ string) ([]byte, bool, error) {
	return nil, false, nil
}

func (m *mockPersonalizationCache) FindSimilarWithThreshold(
	_ string,
	_ string,
	_ float32,
) ([]byte, bool, error) {
	m.findSimilarCalled = true
	return nil, false, nil
}

func (m *mockPersonalizationCache) Close() error { return nil }

func (m *mockPersonalizationCache) GetStats() cache.CacheStats { return cache.CacheStats{} }
