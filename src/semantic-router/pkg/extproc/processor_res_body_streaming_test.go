package extproc

import (
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// TestParseStreamingChunk tests the parseStreamingChunk function
func TestParseStreamingChunk(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := &RequestContext{
		StreamingMetadata: make(map[string]interface{}),
	}

	// Test chunk with content
	chunk1 := `data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"test-model","choices":[{"index":0,"delta":{"content":"Hello "},"finish_reason":null}]}

`
	router.parseStreamingChunk(chunk1, ctx)

	// Verify metadata extracted
	assert.Equal(t, "chatcmpl-123", ctx.StreamingMetadata["id"])
	assert.Equal(t, "test-model", ctx.StreamingMetadata["model"])
	assert.Equal(t, int64(1234567890), ctx.StreamingMetadata["created"])

	// Verify content accumulated
	assert.Equal(t, "Hello ", ctx.StreamingContent)

	// Test chunk with more content
	chunk2 := `data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"test-model","choices":[{"index":0,"delta":{"content":"world"},"finish_reason":null}]}

`
	router.parseStreamingChunk(chunk2, ctx)
	assert.Equal(t, "Hello world", ctx.StreamingContent)

	// Test final chunk with finish_reason and usage
	chunk3 := `data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"test-model","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}}

`
	router.parseStreamingChunk(chunk3, ctx)
	assert.Equal(t, "stop", ctx.StreamingMetadata["finish_reason"])
	assert.NotNil(t, ctx.StreamingMetadata["usage"])

	// Verify usage was extracted
	usage, ok := ctx.StreamingMetadata["usage"].(map[string]interface{})
	assert.True(t, ok, "Usage should be extracted")
	if ok {
		assert.Equal(t, float64(10), usage["prompt_tokens"])
		assert.Equal(t, float64(2), usage["completion_tokens"])
		assert.Equal(t, float64(12), usage["total_tokens"])
	}
}

// TestParseStreamingChunk_SkipDoneMarker tests that [DONE] marker is skipped
func TestParseStreamingChunk_SkipDoneMarker(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := &RequestContext{
		StreamingMetadata: make(map[string]interface{}),
		StreamingContent:  "Existing content",
	}

	// Test [DONE] marker
	chunk := `data: [DONE]

`
	router.parseStreamingChunk(chunk, ctx)

	// Content should not change
	assert.Equal(t, "Existing content", ctx.StreamingContent)
}

// TestParseStreamingChunk_MalformedJSON tests that malformed JSON is skipped
func TestParseStreamingChunk_MalformedJSON(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := &RequestContext{
		StreamingMetadata: make(map[string]interface{}),
		StreamingContent:  "Existing content",
	}

	// Test malformed JSON
	chunk := `data: {invalid json}

`
	router.parseStreamingChunk(chunk, ctx)

	// Content should not change
	assert.Equal(t, "Existing content", ctx.StreamingContent)
}

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

type mockStreamingCache struct {
	addEntryCalled      bool
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
	return nil, false, nil
}

func (m *mockStreamingCache) Close() error { return nil }

func (m *mockStreamingCache) GetStats() cache.CacheStats { return cache.CacheStats{} }

func TestCacheReconstructedStreamingResponse_SkipsWhenDecisionCacheDisabled(t *testing.T) {
	mockCache := &mockStreamingCache{}
	cfg := &config.RouterConfig{
		SemanticCache: config.SemanticCache{Enabled: true},
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{
				{
					Name:      "no-cache-decision",
					ModelRefs: []config.ModelRef{{Model: "test"}},
					// No semantic-cache plugin
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
							Configuration: map[string]interface{}{"enabled": false},
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
							Configuration: map[string]interface{}{"enabled": true},
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
