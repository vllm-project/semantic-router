package extproc

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
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

func TestParseStreamingChunk_CapturesToolCalls(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := &RequestContext{
		StreamingMetadata:  make(map[string]interface{}),
		StreamingToolCalls: make(map[int]*StreamingToolCallState),
	}

	chunk1 := `data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"test-model","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_weather","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\""}}]},"finish_reason":null}]}

`
	router.parseStreamingChunk(chunk1, ctx)

	chunk2 := "data: {\"id\":\"chatcmpl-123\",\"object\":\"chat.completion.chunk\",\"created\":1234567890,\"model\":\"test-model\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"San Francisco\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n"
	router.parseStreamingChunk(chunk2, ctx)

	if assert.Contains(t, ctx.StreamingToolCalls, 0) {
		assert.Equal(t, "call_weather", ctx.StreamingToolCalls[0].ID)
		assert.Equal(t, "get_weather", ctx.StreamingToolCalls[0].Name)
		assert.JSONEq(t, `{"location":"San Francisco"}`, ctx.StreamingToolCalls[0].Arguments)
	}
	assert.Equal(t, "tool_calls", ctx.StreamingMetadata["finish_reason"])
}

func TestFinalizeStreamingResponseRecordsReplayUsageForToolCallsOnly(t *testing.T) {
	recorder := routerreplay.NewRecorder(store.NewMemoryStore(10, 0))
	recorder.SetCapturePolicy(false, true, 4096)

	replayID, err := recorder.AddRecord(routerreplay.RoutingRecord{
		ID:        "replay-stream-tool-call",
		RequestID: "req-stream-tool-call",
		Decision:  "default_route",
	})
	if !assert.NoError(t, err) {
		return
	}

	router := &OpenAIRouter{
		ReplayRecorder: recorder,
		Config: &config.RouterConfig{
			BackendModels: config.BackendModels{
				ModelConfig: map[string]config.ModelParams{
					"test-model": {
						Pricing: config.ModelPricing{
							Currency:        "USD",
							PromptPer1M:     1,
							CompletionPer1M: 2,
						},
					},
					"expensive-model": {
						Pricing: config.ModelPricing{
							Currency:        "USD",
							PromptPer1M:     4,
							CompletionPer1M: 8,
						},
					},
				},
			},
		},
	}
	ctx := &RequestContext{
		RequestID:            "req-stream-tool-call",
		RequestModel:         "test-model",
		RouterReplayID:       replayID,
		RouterReplayRecorder: recorder,
		StreamingMetadata: map[string]interface{}{
			"id":            "chatcmpl-123",
			"model":         "test-model",
			"created":       int64(1234567890),
			"finish_reason": "tool_calls",
			"usage": map[string]interface{}{
				"prompt_tokens":     float64(10),
				"completion_tokens": float64(2),
				"total_tokens":      float64(12),
			},
		},
		StreamingToolCalls: map[int]*StreamingToolCallState{
			0: {
				ID:        "call_weather",
				Name:      "get_weather",
				Arguments: "{\"location\":\"San Francisco\"}",
			},
		},
	}

	router.finalizeStreamingResponse(ctx)

	record, found := recorder.GetRecord(replayID)
	if !assert.True(t, found) {
		return
	}

	assert.NotEmpty(t, record.ResponseBody)
	assert.Contains(t, record.ResponseBody, `"tool_calls"`)
	assert.Contains(t, record.ResponseBody, `"finish_reason":"tool_calls"`)
	if assert.NotNil(t, record.PromptTokens) {
		assert.Equal(t, 10, *record.PromptTokens)
	}
	if assert.NotNil(t, record.CompletionTokens) {
		assert.Equal(t, 2, *record.CompletionTokens)
	}
	if assert.NotNil(t, record.TotalTokens) {
		assert.Equal(t, 12, *record.TotalTokens)
	}
	if assert.NotNil(t, record.ActualCost) {
		assert.Greater(t, *record.ActualCost, 0.0)
	}
	if assert.NotNil(t, record.ToolTrace) {
		assert.Equal(t, "LLM Tool Call", record.ToolTrace.Flow)
		assert.Len(t, record.ToolTrace.Steps, 1)
	}
}
