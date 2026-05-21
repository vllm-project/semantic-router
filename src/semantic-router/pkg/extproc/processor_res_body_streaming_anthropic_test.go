package extproc

import (
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/anthropic"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestHandleAnthropicStreamingResponseBody_TranslatesSSE(t *testing.T) {
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			SemanticCache: config.SemanticCache{Enabled: false},
		},
	}
	ctx := &RequestContext{
		APIFormat:           config.APIFormatAnthropic,
		RequestModel:        "claude-sonnet-4-5",
		AnthropicStream:     anthropic.NewStreamState(),
		StreamingMetadata:   make(map[string]interface{}),
		ProcessingStartTime: time.Now().Add(-10 * time.Millisecond),
	}
	anthropicChunk := strings.Join([]string{
		"event: message_start",
		`data: {"type":"message_start","message":{"id":"msg_01","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-5","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":3,"output_tokens":0}}}`,
		"",
		"event: content_block_start",
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		"",
		"event: content_block_delta",
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}`,
		"",
		"event: message_delta",
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":1}}`,
		"",
		"event: message_stop",
		`data: {"type":"message_stop"}`,
		"",
	}, "\n")

	resp := router.handleAnthropicStreamingResponseBody([]byte(anthropicChunk), ctx)
	require.NotNil(t, resp)
	body := resp.GetResponseBody().GetResponse().GetBodyMutation().GetBody()
	require.NotEmpty(t, body)
	assert.Contains(t, string(body), "chat.completion.chunk")
	assert.Contains(t, string(body), "Hi")
	assert.True(t, ctx.StreamingComplete)
	assert.Equal(t, "Hi", ctx.StreamingContent)
}
