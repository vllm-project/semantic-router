package anthropic

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/ir"
)

// buildOpenAISSE builds an OpenAI SSE byte stream from one or more
// chunk JSON strings, in the wire shape vsr's response pipeline
// produces.
func buildOpenAISSE(chunks ...string) []byte {
	var b strings.Builder
	for _, c := range chunks {
		b.WriteString("data: ")
		b.WriteString(c)
		b.WriteString("\n\n")
	}
	return []byte(b.String())
}

// TestEmitAnthropicSSEChunk_TextOnly asserts the minimal happy path:
// one OpenAI chunk with text content followed by one chunk with
// finish_reason produces the spec-correct Anthropic event sequence.
func TestEmitAnthropicSSEChunk_TextOnly(t *testing.T) {
	state := NewStreamState()
	input := buildOpenAISSE(
		`{"id":"chatcmpl-1","object":"chat.completion.chunk","model":"claude-sonnet-4-5","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}`,
		`{"id":"chatcmpl-1","object":"chat.completion.chunk","model":"claude-sonnet-4-5","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}`,
		`{"id":"chatcmpl-1","object":"chat.completion.chunk","model":"claude-sonnet-4-5","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":2,"total_tokens":6}}`,
	)

	out, done, err := EmitAnthropicSSEChunk(input, state, nil, "claude-sonnet-4-5")
	require.NoError(t, err)
	assert.True(t, done)
	body := string(out)

	// Spec-correct event order.
	startIdx := strings.Index(body, "event: message_start")
	blockStartIdx := strings.Index(body, "event: content_block_start")
	blockDeltaIdx := strings.Index(body, "event: content_block_delta")
	blockStopIdx := strings.Index(body, "event: content_block_stop")
	messageDeltaIdx := strings.Index(body, "event: message_delta")
	messageStopIdx := strings.Index(body, "event: message_stop")

	require.GreaterOrEqual(t, startIdx, 0, "message_start present")
	require.Greater(t, blockStartIdx, startIdx, "content_block_start after message_start")
	require.Greater(t, blockDeltaIdx, blockStartIdx, "content_block_delta after content_block_start")
	require.Greater(t, blockStopIdx, blockDeltaIdx, "content_block_stop after deltas")
	require.Greater(t, messageDeltaIdx, blockStopIdx, "message_delta after content_block_stop")
	require.Greater(t, messageStopIdx, messageDeltaIdx, "message_stop last")

	assert.Contains(t, body, `"text":"Hello"`)
	assert.Contains(t, body, `"text":" world"`)
	assert.Contains(t, body, `"stop_reason":"end_turn"`)
	assert.True(t, state.MessageStartSent)
	assert.True(t, state.MessageStopSent)
}

// TestEmitAnthropicSSEChunk_ToolUseAccumulation asserts that
// fragmented tool-call argument deltas surface as repeated
// input_json_delta events whose partial_json strings concatenate into
// the original arguments.
func TestEmitAnthropicSSEChunk_ToolUseAccumulation(t *testing.T) {
	state := NewStreamState()
	input := buildOpenAISSE(
		`{"id":"c","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}`,
		`{"id":"c","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"ci"}}]},"finish_reason":null}]}`,
		`{"id":"c","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ty\":\"Paris\"}"}}]},"finish_reason":null}]}`,
		`{"id":"c","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`,
	)

	out, done, err := EmitAnthropicSSEChunk(input, state, nil, "claude-sonnet-4-5")
	require.NoError(t, err)
	assert.True(t, done)
	body := string(out)

	assert.Contains(t, body, `"type":"tool_use"`)
	assert.Contains(t, body, `"id":"call_1"`)
	assert.Contains(t, body, `"name":"get_weather"`)
	assert.Contains(t, body, `"partial_json":"{\"ci"`)
	assert.Contains(t, body, `"partial_json":"ty\":\"Paris\"}"`)
	assert.Contains(t, body, `"stop_reason":"tool_use"`)

	// Exactly one content_block_start for the tool_use block.
	assert.Equal(t, 1, strings.Count(body, "event: content_block_start"))
}

// TestEmitAnthropicSSEChunk_TextThenToolUse asserts that switching
// content types within a stream closes the previous block before
// opening the new one — a non-negotiable Anthropic spec invariant.
func TestEmitAnthropicSSEChunk_TextThenToolUse(t *testing.T) {
	state := NewStreamState()
	input := buildOpenAISSE(
		`{"id":"c","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"thinking..."},"finish_reason":null}]}`,
		`{"id":"c","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"f","arguments":"{}"}}]},"finish_reason":null}]}`,
		`{"id":"c","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`,
	)

	out, _, err := EmitAnthropicSSEChunk(input, state, nil, "claude-sonnet-4-5")
	require.NoError(t, err)
	body := string(out)

	// Two starts (text + tool_use), two stops.
	assert.Equal(t, 2, strings.Count(body, "event: content_block_start"))
	assert.Equal(t, 2, strings.Count(body, "event: content_block_stop"))

	// The text content_block_stop appears before the tool_use
	// content_block_start (no overlap).
	firstStop := strings.Index(body, "event: content_block_stop")
	toolStart := strings.Index(body, `"type":"tool_use"`)
	assert.Less(t, firstStop, toolStart, "text block must close before tool_use opens")
}

// TestEmitAnthropicSSEChunk_ReasoningContent asserts that
// reasoning_content on OpenAI delta surfaces as an Anthropic thinking
// block. This is the round-trip half that goes with commit 1's
// upstream thinking_delta → reasoning_content translation.
func TestEmitAnthropicSSEChunk_ReasoningContent(t *testing.T) {
	state := NewStreamState()
	input := buildOpenAISSE(
		`{"id":"c","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"I should compute"},"finish_reason":null}]}`,
		`{"id":"c","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"content":" the answer."},"finish_reason":null}]}`,
		`{"id":"c","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":3,"total_tokens":6}}`,
	)

	out, _, err := EmitAnthropicSSEChunk(input, state, nil, "claude-sonnet-4-5")
	require.NoError(t, err)
	body := string(out)

	assert.Contains(t, body, `"type":"thinking"`)
	assert.Contains(t, body, `"thinking":"I should compute"`)
	assert.Contains(t, body, `"text":" the answer."`)

	// Thinking block opens at index 0, text at index 1.
	thinkingStart := strings.Index(body, `"type":"thinking"`)
	textStart := strings.Index(body, `"type":"text"`)
	assert.Less(t, thinkingStart, textStart, "thinking block precedes text per spec")
}

// TestEmitAnthropicSSEChunk_SignatureReplay asserts that signatures
// captured by the inbound translator into IRExtensions are replayed as
// signature_delta events on the outbound side, then the thinking block
// is closed.
func TestEmitAnthropicSSEChunk_SignatureReplay(t *testing.T) {
	state := NewStreamState()
	ext := &ir.IRExtensions{
		ThinkingSignatures: map[string]string{
			thinkingBlockKey(0): "sig_xyz",
		},
	}
	input := buildOpenAISSE(
		`{"id":"c","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"reasoning"},"finish_reason":null}]}`,
		`{"id":"c","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`,
	)

	out, _, err := EmitAnthropicSSEChunk(input, state, ext, "claude-sonnet-4-5")
	require.NoError(t, err)
	body := string(out)

	assert.Contains(t, body, `"type":"signature_delta"`)
	assert.Contains(t, body, `"signature":"sig_xyz"`)
	// Signature is consumed (so we don't double-emit on later chunks).
	_, stillPresent := ext.ThinkingSignatures[thinkingBlockKey(0)]
	assert.False(t, stillPresent, "signature should be consumed after replay")
}

// TestEmitAnthropicSSEChunk_NilExt asserts the emitter tolerates a nil
// ext gracefully — the existing OpenAI-client cell may call without
// an IRExtensions sidecar.
func TestEmitAnthropicSSEChunk_NilExt(t *testing.T) {
	state := NewStreamState()
	input := buildOpenAISSE(
		`{"id":"c","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`,
	)

	out, done, err := EmitAnthropicSSEChunk(input, state, nil, "claude-sonnet-4-5")
	require.NoError(t, err)
	assert.True(t, done)
	assert.Contains(t, string(out), `"type":"text"`)
}

// TestEmitAnthropicSSEChunk_NilState asserts that a nil StreamState
// is initialized rather than panicking, matching the inbound
// translator's tolerance.
func TestEmitAnthropicSSEChunk_NilState(t *testing.T) {
	input := buildOpenAISSE(
		`{"id":"c","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"x"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`,
	)
	out, done, err := EmitAnthropicSSEChunk(input, nil, nil, "claude-sonnet-4-5")
	require.NoError(t, err)
	assert.True(t, done)
	assert.NotEmpty(t, out)
}

// TestEmitAnthropicSSEChunk_LengthFinish asserts length finish_reason
// maps to max_tokens stop_reason on the wire.
func TestEmitAnthropicSSEChunk_LengthFinish(t *testing.T) {
	state := NewStreamState()
	input := buildOpenAISSE(
		`{"id":"c","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"truncated"},"finish_reason":"length"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`,
	)
	out, _, err := EmitAnthropicSSEChunk(input, state, nil, "claude-sonnet-4-5")
	require.NoError(t, err)
	assert.Contains(t, string(out), `"stop_reason":"max_tokens"`)
}

// TestEmitAnthropicPingEvent asserts the standalone ping helper emits
// the spec-correct wire shape.
func TestEmitAnthropicPingEvent(t *testing.T) {
	out := string(EmitAnthropicPingEvent())
	assert.Contains(t, out, "event: ping")
	assert.Contains(t, out, `"type":"ping"`)
	assert.True(t, strings.HasSuffix(out, "\n\n"), "SSE event must terminate with blank line")
}

// TestFormatAnthropicSSEEvent asserts the SSE framing helper produces
// the wire shape Anthropic clients expect — event header, data line,
// terminating blank line.
func TestFormatAnthropicSSEEvent(t *testing.T) {
	out := string(formatAnthropicSSEEvent("message_start", []byte(`{"type":"message_start"}`)))
	assert.Equal(t, "event: message_start\ndata: {\"type\":\"message_start\"}\n\n", out)
}
