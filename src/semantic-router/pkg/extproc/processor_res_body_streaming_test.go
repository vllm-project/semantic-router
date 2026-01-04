package extproc

import (
	"testing"

	"github.com/stretchr/testify/assert"
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

// Note: Integration tests for cacheStreamingResponse should be added to extproc_test.go
// using CreateTestRouter. These unit tests focus on parseStreamingChunk which is
// the core parsing logic that can be tested independently.
