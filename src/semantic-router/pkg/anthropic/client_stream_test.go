package anthropic

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/openai/openai-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestWithStreamingRequestBody_SetsStreamFlag(t *testing.T) {
	req := &openai.ChatCompletionNewParams{
		Model:    "claude-sonnet-4-5",
		Messages: simpleUserMsg("hi"),
	}

	body, err := ToAnthropicRequestBody(req)
	require.NoError(t, err)

	streamingBody, err := WithStreamingRequestBody(body)
	require.NoError(t, err)

	var parsed map[string]interface{}
	require.NoError(t, json.Unmarshal(streamingBody, &parsed))
	assert.Equal(t, true, parsed["stream"])
}

func TestBuildStreamingRequestHeaders_Accept(t *testing.T) {
	headers := BuildStreamingRequestHeaders("key", 10, "")
	accept := ""
	for _, h := range headers {
		if h.Key == "accept" {
			accept = h.Value
		}
	}
	assert.Equal(t, "text/event-stream", accept)
}

func TestTransformSSEChunkToOpenAI_TextStream(t *testing.T) {
	state := NewStreamState()
	events := []string{
		`{"type":"message_start","message":{"id":"msg_01","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-5","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":0}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":"Hello"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":2}}`,
		`{"type":"message_stop"}`,
	}
	chunk := buildAnthropicSSE(events...)

	out, done, err := TransformSSEChunkToOpenAI([]byte(chunk), state, "claude-sonnet-4-5")
	require.NoError(t, err)
	assert.True(t, done)
	assert.Contains(t, string(out), "data: ")
	assert.Contains(t, string(out), "Hello")
	assert.Contains(t, string(out), " world")
	assert.Contains(t, string(out), "data: [DONE]")
	assert.Contains(t, string(out), `"role":"assistant"`)
	assert.Contains(t, string(out), `"finish_reason":"stop"`)
}

func TestTransformSSEChunkToOpenAI_ToolUseStream(t *testing.T) {
	state := NewStreamState()
	events := []string{
		`{"type":"message_start","message":{"id":"msg_tool","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-5","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":0}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_01","name":"get_weather","input":{}}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"city\":"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"\"Paris\"}"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":12}}`,
		`{"type":"message_stop"}`,
	}
	chunk := buildAnthropicSSE(events...)

	out, done, err := TransformSSEChunkToOpenAI([]byte(chunk), state, "claude-sonnet-4-5")
	require.NoError(t, err)
	assert.True(t, done)
	body := string(out)
	assert.Contains(t, body, "tool_calls")
	assert.Contains(t, body, "toolu_01")
	assert.Contains(t, body, "get_weather")
	assert.Contains(t, body, `"finish_reason":"tool_calls"`)
	assert.Contains(t, body, "Paris")
}

func TestTransformSSEChunkToOpenAI_AccumulatesToolArguments(t *testing.T) {
	state := NewStreamState()
	start := buildAnthropicSSE(
		`{"type":"message_start","message":{"id":"msg_tool","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-5","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":0}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_01","name":"get_weather","input":{}}}`,
	)
	_, _, err := TransformSSEChunkToOpenAI([]byte(start), state, "claude-sonnet-4-5")
	require.NoError(t, err)

	delta := buildAnthropicSSE(
		`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"city\":\"Paris\"}"}}`,
	)
	out, _, err := TransformSSEChunkToOpenAI([]byte(delta), state, "claude-sonnet-4-5")
	require.NoError(t, err)
	assert.Contains(t, string(out), "Paris")
}

func buildAnthropicSSE(events ...string) string {
	var b strings.Builder
	for _, event := range events {
		b.WriteString("event: ")
		var typed struct {
			Type string `json:"type"`
		}
		_ = json.Unmarshal([]byte(event), &typed)
		b.WriteString(typed.Type)
		b.WriteString("\ndata: ")
		b.WriteString(event)
		b.WriteString("\n\n")
	}
	return b.String()
}

func TestTransformSSEChunkToOpenAI_PassthroughOpenAIFormat(t *testing.T) {
	state := NewStreamState()
	openAIChunk := []byte(`data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"hi"}}]}

data: [DONE]

`)
	out, done, err := TransformSSEChunkToOpenAI(openAIChunk, state, "claude-sonnet-4-5")
	require.NoError(t, err)
	assert.True(t, done)
	assert.Equal(t, string(openAIChunk), string(out))
}

func TestMapAnthropicStopReasonToOpenAI(t *testing.T) {
	assert.Equal(t, "stop", mapAnthropicStopReasonToOpenAI(anthropic.StopReasonEndTurn))
	assert.Equal(t, "length", mapAnthropicStopReasonToOpenAI(anthropic.StopReasonMaxTokens))
	assert.Equal(t, "tool_calls", mapAnthropicStopReasonToOpenAI(anthropic.StopReasonToolUse))
}
