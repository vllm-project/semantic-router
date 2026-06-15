package anthropic

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/openai/openai-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/ir"
)

// TestNewStreamState_InitializesOutboundFields asserts that
// NewStreamState allocates the outbound-direction maps and leaves the
// outbound bookkeeping flags at zero values so the symmetric emitter can
// rely on them being non-nil and false on the first chunk.
func TestNewStreamState_InitializesOutboundFields(t *testing.T) {
	state := NewStreamState()

	// Inbound maps remain allocated as before.
	require.NotNil(t, state.BlockIndexToToolIdx)
	require.NotNil(t, state.BlockIndexToThinkingActive)

	// Outbound maps are allocated so the emitter can write without a
	// nil-map panic on first use.
	require.NotNil(t, state.ToolIdxToBlockIndex)

	// Outbound bookkeeping starts unset.
	assert.False(t, state.MessageStartSent)
	assert.False(t, state.MessageStopSent)
	assert.Equal(t, int64(0), state.NextBlockIndex)
	assert.Nil(t, state.OpenTextBlockIndex)
	assert.Nil(t, state.OpenThinkingBlockIdx)
	assert.True(t, state.LastChunkAt.IsZero())
}

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

	out, done, err := TransformSSEChunkToOpenAI([]byte(chunk), state, "claude-sonnet-4-5", nil)
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

	out, done, err := TransformSSEChunkToOpenAI([]byte(chunk), state, "claude-sonnet-4-5", nil)
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
	_, _, err := TransformSSEChunkToOpenAI([]byte(start), state, "claude-sonnet-4-5", nil)
	require.NoError(t, err)

	delta := buildAnthropicSSE(
		`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"city\":\"Paris\"}"}}`,
	)
	out, _, err := TransformSSEChunkToOpenAI([]byte(delta), state, "claude-sonnet-4-5", nil)
	require.NoError(t, err)
	assert.Contains(t, string(out), "Paris")
}

// TestTransformSSEChunkToOpenAI_ThinkingBlockStart asserts that an
// Anthropic content_block_start with type "thinking" surfaces a
// reasoning_content bootstrap on the OpenAI envelope instead of being
// silently dropped (pre-fix behavior).
func TestTransformSSEChunkToOpenAI_ThinkingBlockStart(t *testing.T) {
	state := NewStreamState()
	events := buildAnthropicSSE(
		`{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`,
	)
	out, _, err := TransformSSEChunkToOpenAI([]byte(events), state, "claude-sonnet-4-5", nil)
	require.NoError(t, err)
	assert.Contains(t, string(out), `"reasoning_content"`)
	// Assert the key is present before asserting its value: a bare map
	// read returns the bool zero value, so a missing key would silently
	// pass assert.True's negation. require.Contains fails loudly instead.
	require.Contains(t, state.BlockIndexToThinkingActive, int64(0), "block index 0 must be tracked as thinking")
	assert.True(t, state.BlockIndexToThinkingActive[0], "block index 0 should be marked thinking-active")
}

// TestTransformSSEChunkToOpenAI_ThinkingDelta asserts that thinking_delta
// events emit reasoning_content with the delta text.
func TestTransformSSEChunkToOpenAI_ThinkingDelta(t *testing.T) {
	state := NewStreamState()
	events := buildAnthropicSSE(
		`{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Let me think about this..."}}`,
	)
	out, _, err := TransformSSEChunkToOpenAI([]byte(events), state, "claude-sonnet-4-5", nil)
	require.NoError(t, err)
	body := string(out)
	assert.Contains(t, body, `"reasoning_content":"Let me think about this..."`)
}

// TestTransformSSEChunkToOpenAI_ThinkingDelta_WithoutBlockStart_Drops
// asserts that thinking_delta events arriving without a preceding
// content_block_start are dropped (cannot be routed to a block).
func TestTransformSSEChunkToOpenAI_ThinkingDelta_WithoutBlockStart_Drops(t *testing.T) {
	state := NewStreamState()
	events := buildAnthropicSSE(
		`{"type":"content_block_delta","index":7,"delta":{"type":"thinking_delta","thinking":"orphan"}}`,
	)
	out, _, err := TransformSSEChunkToOpenAI([]byte(events), state, "claude-sonnet-4-5", nil)
	require.NoError(t, err)
	assert.NotContains(t, string(out), "orphan")
}

// TestTransformSSEChunkToOpenAI_SignatureDelta asserts that signature_delta
// events capture the signature into IRExtensions.ThinkingSignatures.
func TestTransformSSEChunkToOpenAI_SignatureDelta(t *testing.T) {
	state := NewStreamState()
	ext := &ir.IRExtensions{}
	events := buildAnthropicSSE(
		`{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig_abc123"}}`,
	)
	_, _, err := TransformSSEChunkToOpenAI([]byte(events), state, "claude-sonnet-4-5", ext)
	require.NoError(t, err)
	assert.Equal(t, "sig_abc123", ext.ThinkingSignatures["content[0]"])
}

// TestTransformSSEChunkToOpenAI_SignatureDelta_NilExt_NoPanic asserts
// that the existing OpenAI-client cell (which passes nil ext) is
// unaffected by signature_delta events.
func TestTransformSSEChunkToOpenAI_SignatureDelta_NilExt_NoPanic(t *testing.T) {
	state := NewStreamState()
	events := buildAnthropicSSE(
		`{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig_xyz"}}`,
	)
	out, _, err := TransformSSEChunkToOpenAI([]byte(events), state, "claude-sonnet-4-5", nil)
	require.NoError(t, err)
	assert.NotContains(t, string(out), "sig_xyz", "signature should not leak onto OpenAI stream")
}

// TestTransformSSEChunkToOpenAI_ErrorEvent asserts that an
// `event: error` SSE line surfaces as an OpenAI chunk carrying both the
// error message in delta.content and finish_reason="error" — instead of
// being silently dropped (pre-fix behavior).
func TestTransformSSEChunkToOpenAI_ErrorEvent(t *testing.T) {
	state := NewStreamState()
	raw := "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"upstream is overloaded\"}}\n\n"
	out, _, err := TransformSSEChunkToOpenAI([]byte(raw), state, "claude-sonnet-4-5", nil)
	require.NoError(t, err)
	body := string(out)
	assert.Contains(t, body, "overloaded_error")
	assert.Contains(t, body, "upstream is overloaded")
	assert.Contains(t, body, `"finish_reason":"error"`)
}

// TestTransformSSEChunkToOpenAI_ErrorEvent_NoHeader asserts that an error
// payload sent without an `event: error` header (some transports) is
// still surfaced via the MessageStreamEventUnion error type arm.
func TestTransformSSEChunkToOpenAI_ErrorEvent_NoHeader(t *testing.T) {
	state := NewStreamState()
	raw := "data: {\"type\":\"error\",\"error\":{\"type\":\"rate_limit_error\",\"message\":\"slow down\"}}\n\n"
	out, _, err := TransformSSEChunkToOpenAI([]byte(raw), state, "claude-sonnet-4-5", nil)
	require.NoError(t, err)
	body := string(out)
	assert.Contains(t, body, "rate_limit_error")
	assert.Contains(t, body, "slow down")
}

// TestTransformSSEChunkToOpenAI_CapturesInitialUsage asserts that
// the inbound translator captures the message_start.usage block onto
// state.InitialUsage so the outbound emitter can echo it on its
// synthesized message_start event during round-trips.
func TestTransformSSEChunkToOpenAI_CapturesInitialUsage(t *testing.T) {
	state := NewStreamState()
	events := buildAnthropicSSE(
		`{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-5","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":42,"output_tokens":1,"cache_read_input_tokens":10,"cache_creation_input_tokens":5}}}`,
	)
	_, _, err := TransformSSEChunkToOpenAI([]byte(events), state, "claude-sonnet-4-5", nil)
	require.NoError(t, err)
	assert.Equal(t, int64(42), state.InitialUsage.InputTokens)
	assert.Equal(t, int64(10), state.InitialUsage.CacheReadInputTokens)
	assert.Equal(t, int64(5), state.InitialUsage.CacheCreationInputTokens)
}

// TestTransformSSEChunkToOpenAI_AnthropicOnlyStopReason asserts that
// pause_turn round-trips through IRExtensions because the OpenAI
// finish_reason alphabet has no equivalent.
func TestTransformSSEChunkToOpenAI_AnthropicOnlyStopReason(t *testing.T) {
	state := NewStreamState()
	ext := &ir.IRExtensions{}
	events := buildAnthropicSSE(
		`{"type":"message_delta","delta":{"stop_reason":"pause_turn","stop_sequence":null},"usage":{"output_tokens":50}}`,
	)
	_, _, err := TransformSSEChunkToOpenAI([]byte(events), state, "claude-sonnet-4-5", ext)
	require.NoError(t, err)
	assert.Equal(t, "pause_turn", ext.AnthropicStopReason)
}

// TestTransformSSEChunkToOpenAI_StandardStopReasonNotCaptured asserts
// that mappable stop reasons (end_turn, tool_use, etc.) are NOT
// stashed onto ext — only the Anthropic-only ones need the round-trip
// because the mappable ones survive OpenAI normalization.
func TestTransformSSEChunkToOpenAI_StandardStopReasonNotCaptured(t *testing.T) {
	state := NewStreamState()
	ext := &ir.IRExtensions{}
	events := buildAnthropicSSE(
		`{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":50}}`,
	)
	_, _, err := TransformSSEChunkToOpenAI([]byte(events), state, "claude-sonnet-4-5", ext)
	require.NoError(t, err)
	assert.Empty(t, ext.AnthropicStopReason, "mappable stop_reason should round-trip via standard mapping")
}

// TestTransformSSEChunkToOpenAI_RefusalCaptured asserts that a streaming
// Anthropic response with stop_reason "refusal" is captured onto ext so
// the outbound emitter can surface it verbatim. Before this fix only
// pause_turn was captured; refusal fell through to the default arm
// (mapped to "stop") and was lost on round-trips.
func TestTransformSSEChunkToOpenAI_RefusalCaptured(t *testing.T) {
	state := NewStreamState()
	ext := &ir.IRExtensions{}
	events := buildAnthropicSSE(
		`{"type":"message_delta","delta":{"stop_reason":"refusal","stop_sequence":null},"usage":{"output_tokens":12}}`,
	)
	_, _, err := TransformSSEChunkToOpenAI([]byte(events), state, "claude-sonnet-4-5", ext)
	require.NoError(t, err)
	assert.Equal(t, "refusal", ext.AnthropicStopReason, "refusal must be captured into ext")
	assert.Empty(t, ext.AnthropicStopSequence, "stop_sequence string must be empty for refusal")
}

// TestTransformSSEChunkToOpenAI_StopSequenceCaptured asserts that a
// streaming response with stop_reason "stop_sequence" captures both the
// reason and the stop_sequence string onto ext. The non-streaming sibling
// (captureAnthropicStopReasonIntoExt) captures the string; streaming must
// now match.
func TestTransformSSEChunkToOpenAI_StopSequenceCaptured(t *testing.T) {
	state := NewStreamState()
	ext := &ir.IRExtensions{}
	events := buildAnthropicSSE(
		`{"type":"message_delta","delta":{"stop_reason":"stop_sequence","stop_sequence":"DONE"},"usage":{"output_tokens":8}}`,
	)
	_, _, err := TransformSSEChunkToOpenAI([]byte(events), state, "claude-sonnet-4-5", ext)
	require.NoError(t, err)
	assert.Equal(t, "stop_sequence", ext.AnthropicStopReason, "stop_sequence reason must be captured")
	assert.Equal(t, "DONE", ext.AnthropicStopSequence, "stop_sequence string must be captured")
}

func TestExtractSSEDataLines_TracksEventHeader(t *testing.T) {
	raw := "event: error\ndata: {\"type\":\"error\"}\n\nevent: message_start\ndata: {\"type\":\"message_start\"}\n\n"
	lines := extractSSEDataLines([]byte(raw))
	require.Len(t, lines, 2)
	assert.True(t, lines[0].IsError, "first line should be tagged as error")
	assert.False(t, lines[1].IsError, "second line should not be tagged after blank-line reset")
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

func TestMapAnthropicStopReasonToOpenAI(t *testing.T) {
	assert.Equal(t, "stop", mapAnthropicStopReasonToOpenAI(anthropic.StopReasonEndTurn))
	assert.Equal(t, "length", mapAnthropicStopReasonToOpenAI(anthropic.StopReasonMaxTokens))
	assert.Equal(t, "tool_calls", mapAnthropicStopReasonToOpenAI(anthropic.StopReasonToolUse))
}
