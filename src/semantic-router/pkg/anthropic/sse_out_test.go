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

	// Exactly-once invariant: the emitter must never produce a second
	// message_stop regardless of how the upstream split finish_reason and usage.
	assert.Equal(t, 1, strings.Count(body, "event: message_stop"), "message_stop exactly once")

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

// TestEmitAnthropicSSEChunk_CacheUsageRoundTrip asserts that cache
// counters captured on IRExtensions (typically populated by the
// inbound translator for the Anthropic→Anthropic cell) land on the
// terminal message_delta.usage so an Anthropic client sees the same
// cache_read / cache_creation breakdown the upstream produced.
func TestEmitAnthropicSSEChunk_CacheUsageRoundTrip(t *testing.T) {
	state := NewStreamState()
	ext := &ir.IRExtensions{
		CacheReadInputTokens:     120,
		CacheCreationInputTokens: 80,
	}
	input := buildOpenAISSE(
		`{"id":"c","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"x"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`,
	)
	out, _, err := EmitAnthropicSSEChunk(input, state, ext, "claude-sonnet-4-5")
	require.NoError(t, err)
	body := string(out)
	assert.Contains(t, body, `"cache_read_input_tokens":120`)
	assert.Contains(t, body, `"cache_creation_input_tokens":80`)
}

// TestEmitAnthropicSSEChunk_AnthropicOnlyStopReasonOverride asserts
// that an Anthropic-only stop_reason captured on ext overrides the
// OpenAI-derived mapping on the outbound message_delta.
func TestEmitAnthropicSSEChunk_AnthropicOnlyStopReasonOverride(t *testing.T) {
	state := NewStreamState()
	ext := &ir.IRExtensions{
		AnthropicStopReason: "pause_turn",
	}
	input := buildOpenAISSE(
		`{"id":"c","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"x"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`,
	)
	out, _, err := EmitAnthropicSSEChunk(input, state, ext, "claude-sonnet-4-5")
	require.NoError(t, err)
	assert.Contains(t, string(out), `"stop_reason":"pause_turn"`)
}

// TestEmitAnthropicSSEChunk_StopSequenceVariant asserts that when
// ext.AnthropicStopReason is "stop_sequence" the outbound message_delta
// carries both stop_reason:"stop_sequence" and a non-nil stop_sequence
// field with the sequence string. This exercises mapOpenAIFinishReasonToAnthropic's
// stop_sequence branch end-to-end.
func TestEmitAnthropicSSEChunk_StopSequenceVariant(t *testing.T) {
	state := NewStreamState()
	ext := &ir.IRExtensions{
		AnthropicStopReason:   "stop_sequence",
		AnthropicStopSequence: "DONE",
	}
	input := buildOpenAISSE(
		`{"id":"c","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"x"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`,
	)
	out, _, err := EmitAnthropicSSEChunk(input, state, ext, "claude-sonnet-4-5")
	require.NoError(t, err)
	body := string(out)
	assert.Contains(t, body, `"stop_reason":"stop_sequence"`)
	assert.Contains(t, body, `"stop_sequence":"DONE"`)
}

// TestEmitAnthropicSSEChunk_InitialUsageEcho asserts that an initial
// usage captured on the inbound side reaches the outbound
// message_start payload, preserving the input_tokens count clients use
// to seed their usage accumulators.
func TestEmitAnthropicSSEChunk_InitialUsageEcho(t *testing.T) {
	state := NewStreamState()
	state.InitialUsage.InputTokens = 99
	input := buildOpenAISSE(
		`{"id":"c","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"x"},"finish_reason":"stop"}],"usage":{"prompt_tokens":99,"completion_tokens":1,"total_tokens":100}}`,
	)
	out, _, err := EmitAnthropicSSEChunk(input, state, nil, "claude-sonnet-4-5")
	require.NoError(t, err)
	body := string(out)
	// message_start.usage carries InitialUsage.InputTokens.
	startIdx := strings.Index(body, "event: message_start")
	deltaIdx := strings.Index(body, "event: message_delta")
	require.Greater(t, deltaIdx, startIdx)
	messageStartBlock := body[startIdx:deltaIdx]
	assert.Contains(t, messageStartBlock, `"input_tokens":99`)
}

// TestFormatAnthropicSSEEvent asserts the SSE framing helper produces
// the wire shape Anthropic clients expect — event header, data line,
// terminating blank line.
func TestFormatAnthropicSSEEvent(t *testing.T) {
	out := string(formatAnthropicSSEEvent("message_start", []byte(`{"type":"message_start"}`)))
	assert.Equal(t, "event: message_start\ndata: {\"type\":\"message_start\"}\n\n", out)
}

// TestEmitAnthropicSSEChunk_SplitFinishReasonAndUsage asserts that when
// an OpenAI-compatible backend legally splits finish_reason and usage
// across two separate chunks, message_stop appears exactly once on the
// wire. Before the idempotency guard, the second chunk (usage-only)
// also satisfied isTerminalChunk, causing a second message_stop.
func TestEmitAnthropicSSEChunk_SplitFinishReasonAndUsage(t *testing.T) {
	state := NewStreamState()

	// Chunk 1: finish_reason present, no usage.
	chunk1 := buildOpenAISSE(
		`{"id":"c","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"hi"},"finish_reason":"stop"}]}`,
	)
	out1, done1, err := EmitAnthropicSSEChunk(chunk1, state, nil, "claude-sonnet-4-5")
	require.NoError(t, err)
	assert.True(t, done1, "finish_reason chunk must set streamDone")
	assert.Equal(t, 1, strings.Count(string(out1), "event: message_stop"), "exactly one message_stop after finish_reason chunk")

	// Chunk 2: usage-only, no finish_reason. The idempotency guard must
	// suppress a second call to emitTerminalEvents.
	chunk2 := buildOpenAISSE(
		`{"id":"c","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{}}],"usage":{"prompt_tokens":4,"completion_tokens":2,"total_tokens":6}}`,
	)
	out2, done2, err := EmitAnthropicSSEChunk(chunk2, state, nil, "claude-sonnet-4-5")
	require.NoError(t, err)
	assert.True(t, done2, "post-terminal usage chunk must still report streamDone")
	assert.Equal(t, 0, strings.Count(string(out2), "event: message_stop"), "no second message_stop from usage-only chunk")

	// Total across both chunks: exactly one message_stop.
	combined := string(out1) + string(out2)
	assert.Equal(t, 1, strings.Count(combined, "event: message_stop"), "exactly one message_stop total across split chunks")
}

// TestEmitAnthropicSSEChunk_UsageOnlyTerminal covers a backend that ends
// the stream with a usage-only chunk (no choices, no prior finish_reason).
// The emitter must treat usage as the terminal signal and fire message_stop,
// otherwise an Anthropic client hangs waiting for an end event. This differs
// from SplitFinishReasonAndUsage, where finish_reason already drove the
// terminal sequence and the usage chunk is a guarded no-op.
func TestEmitAnthropicSSEChunk_UsageOnlyTerminal(t *testing.T) {
	state := NewStreamState()

	// Chunk 1: content only, no finish_reason.
	chunk1 := buildOpenAISSE(
		`{"id":"c","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"hi"}}]}`,
	)
	_, done1, err := EmitAnthropicSSEChunk(chunk1, state, nil, "claude-sonnet-4-5")
	require.NoError(t, err)
	assert.False(t, done1, "content-only chunk must not end the stream")

	// Chunk 2: usage-only terminal — empty choices, usage present, never a
	// finish_reason. This is the sole terminal signal.
	chunk2 := buildOpenAISSE(
		`{"id":"c","object":"chat.completion.chunk","model":"m","choices":[],"usage":{"prompt_tokens":4,"completion_tokens":2,"total_tokens":6}}`,
	)
	out2, done2, err := EmitAnthropicSSEChunk(chunk2, state, nil, "claude-sonnet-4-5")
	require.NoError(t, err)
	assert.True(t, done2, "usage-only terminal chunk must end the stream")
	assert.Equal(t, 1, strings.Count(string(out2), "event: message_stop"), "usage-only terminal must emit exactly one message_stop")
	assert.True(t, state.MessageStopSent, "MessageStopSent must be set after usage-only terminal")
	// The open text block must be closed before the terminal events.
	assert.Equal(t, 1, strings.Count(string(out2), "event: content_block_stop"), "open text block must be closed by the usage-only terminal")
}

// TestEmitAnthropicSSEChunk_UsageOnlyTerminalClosesThinkingAndToolBlocks
// guards the other two branches of closeAllOpenBlocks reached on the
// usage-only terminal path: a thinking block and tool blocks left open
// when the stream ends with a usage-only chunk. Each open block must
// receive a content_block_stop before message_delta/message_stop, or an
// Anthropic SDK accumulator sees an unterminated block and breaks.
func TestEmitAnthropicSSEChunk_UsageOnlyTerminalClosesThinkingAndToolBlocks(t *testing.T) {
	state := NewStreamState()

	// Open a thinking block, then two tool blocks, all without a
	// finish_reason, then end with a usage-only terminal chunk.
	chunks := buildOpenAISSE(
		`{"id":"c","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"hmm"}}]}`,
		`{"id":"c","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"a","arguments":"{}"}}]}}]}`,
		`{"id":"c","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_2","type":"function","function":{"name":"b","arguments":"{}"}}]}}]}`,
		`{"id":"c","object":"chat.completion.chunk","model":"m","choices":[],"usage":{"prompt_tokens":4,"completion_tokens":3,"total_tokens":7}}`,
	)

	out, done, err := EmitAnthropicSSEChunk(chunks, state, nil, "claude-sonnet-4-5")
	require.NoError(t, err)
	assert.True(t, done, "usage-only terminal must end the stream")
	body := string(out)

	// One thinking block + two tool blocks = three content_block_start,
	// and all three must be closed (three content_block_stop) before the
	// single terminal message_stop.
	assert.Equal(t, 3, strings.Count(body, "event: content_block_start"), "thinking + two tool blocks must each open")
	assert.Equal(t, 3, strings.Count(body, "event: content_block_stop"), "every open block must be closed on the usage-only terminal")
	assert.Equal(t, 1, strings.Count(body, "event: message_stop"), "exactly one message_stop")
	lastStop := strings.LastIndex(body, "event: content_block_stop")
	msgStop := strings.Index(body, "event: message_stop")
	require.GreaterOrEqual(t, lastStop, 0)
	require.GreaterOrEqual(t, msgStop, 0)
	assert.Less(t, lastStop, msgStop, "all content_block_stop events must precede message_stop")
}

// TestEmitAnthropicSSEChunk_PreviewUsageChunkNotTerminal locks the fix for
// issue #2215: a gateway may send a usage-only "preview" chunk (empty
// choices, usage present, no finish_reason) as the FIRST chunk of a stream.
// That preview usage is not the stream's final summary — real content and a
// real terminal chunk (with the true usage) follow. The emitter must NOT
// treat the preview chunk as terminal, or it fires message_stop after the
// first chunk, truncates the response, and reports the preview's
// completion_tokens (4) instead of the real total (252).
//
// This is the counterpart to UsageOnlyTerminal: there the usage-only chunk
// trails content and IS terminal; here it precedes all content and is NOT.
func TestEmitAnthropicSSEChunk_PreviewUsageChunkNotTerminal(t *testing.T) {
	state := NewStreamState()

	// Chunk 1: preview usage-only chunk at stream start — empty choices,
	// usage present (small preview value), no finish_reason.
	chunk1 := buildOpenAISSE(
		`{"id":"c","object":"chat.completion.chunk","choices":[],"usage":{"prompt_tokens":37494,"completion_tokens":4,"total_tokens":37498}}`,
	)
	out1, done1, err := EmitAnthropicSSEChunk(chunk1, state, nil, "claude-opus-4-6")
	require.NoError(t, err)
	assert.False(t, done1, "preview usage chunk must NOT end the stream")
	assert.False(t, state.MessageStopSent, "preview usage chunk must not set MessageStopSent")
	assert.Equal(t, 0, strings.Count(string(out1), "event: message_stop"), "preview usage chunk must not emit message_stop")

	// Chunk 2: real content.
	chunk2 := buildOpenAISSE(
		`{"id":"c","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":"hello world"}}]}`,
	)
	out2, done2, err := EmitAnthropicSSEChunk(chunk2, state, nil, "claude-opus-4-6")
	require.NoError(t, err)
	assert.False(t, done2, "content chunk must not end the stream")
	assert.Contains(t, string(out2), `"text":"hello world"`, "content after the preview chunk must reach the client")

	// Chunk 3: real terminal chunk — finish_reason plus the true usage.
	chunk3 := buildOpenAISSE(
		`{"id":"c","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":37494,"completion_tokens":252,"total_tokens":37746}}`,
	)
	out3, done3, err := EmitAnthropicSSEChunk(chunk3, state, nil, "claude-opus-4-6")
	require.NoError(t, err)
	assert.True(t, done3, "the real finish_reason chunk must end the stream")
	body3 := string(out3)
	assert.Equal(t, 1, strings.Count(body3, "event: message_stop"), "exactly one message_stop, on the real terminal chunk")
	// The terminal usage must come from the real chunk (252), not the preview (4).
	assert.Contains(t, body3, `"output_tokens":252`, "message_delta usage must reflect the real total, not the preview")
	assert.NotContains(t, body3, `"output_tokens":4`, "preview completion_tokens must not be reported")

	// Across the whole stream: exactly one message_stop, after content.
	combined := string(out1) + string(out2) + body3
	assert.Equal(t, 1, strings.Count(combined, "event: message_stop"), "exactly one message_stop total")
}

// TestEmitAnthropicSSEChunk_ErrorPath covers the IsError emitter branch:
// an error event followed by terminal state flags and no second emission.
func TestEmitAnthropicSSEChunk_ErrorPath(t *testing.T) {
	state := NewStreamState()
	state.MessageStartSent = true // simulate mid-stream error

	// Synthesize an extractedSSELine with IsError=true by building the
	// wire bytes the way extractSSEDataLines produces for error events.
	errInput := []byte("event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"api_error\",\"message\":\"overloaded\"}}\n\n")

	out, streamDone, err := EmitAnthropicSSEChunk(errInput, state, nil, "claude-sonnet-4-5")
	require.NoError(t, err)
	body := string(out)

	assert.True(t, streamDone, "error path must set streamDone")
	assert.True(t, state.MessageStopSent, "MessageStopSent must be true after error emission")
	assert.Contains(t, body, "event: error", "error event must be emitted")
	assert.Contains(t, body, "event: message_stop", "message_stop must follow error event")
	assert.Equal(t, 1, strings.Count(body, "event: message_stop"), "exactly one message_stop after error")

	// A second call with IsError must not produce another message_stop.
	out2, streamDone2, err2 := EmitAnthropicSSEChunk(errInput, state, nil, "claude-sonnet-4-5")
	require.NoError(t, err2)
	assert.True(t, streamDone2)
	assert.Equal(t, 0, strings.Count(string(out2), "event: message_stop"), "no second message_stop on re-entry")
}

// TestEmitAnthropicSSEChunk_ErrorAsFirstChunk locks the contract for an
// error that arrives before any content (MessageStartSent == false): the
// emitter sends a standalone error event followed by message_stop, with
// NO message_start. This matches Anthropic's own streaming API, which
// surfaces an early request failure as a bare error event rather than
// opening a message lifecycle first. The test guards against a future
// change that would synthesize a spurious message_start here.
func TestEmitAnthropicSSEChunk_ErrorAsFirstChunk(t *testing.T) {
	state := NewStreamState() // MessageStartSent == false: error is the first chunk

	errInput := []byte("event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"api_error\",\"message\":\"overloaded\"}}\n\n")
	out, streamDone, err := EmitAnthropicSSEChunk(errInput, state, nil, "claude-sonnet-4-5")
	require.NoError(t, err)
	body := string(out)

	assert.True(t, streamDone, "error-first chunk must end the stream")
	assert.True(t, state.MessageStopSent, "MessageStopSent must be set after an error-first chunk")
	assert.Contains(t, body, "event: error", "error event must be emitted")
	assert.Equal(t, 1, strings.Count(body, "event: message_stop"), "exactly one message_stop")
	assert.NotContains(t, body, "event: message_start",
		"a pre-content error must not synthesize a message_start (matches Anthropic API)")
}

// TestEmitAnthropicSSEChunk_DeterministicToolClose asserts that
// closeAllOpenBlocks emits content_block_stop events in ascending
// block-index order even when the underlying map iteration is
// non-deterministic. Multi-tool streams previously produced an
// unpredictable content_block_stop sequence because Go map iteration
// order is randomised per program.
func TestEmitAnthropicSSEChunk_DeterministicToolClose(t *testing.T) {
	// Feed two tool calls and let the stream terminate. Run many times
	// to surface any non-determinism.
	for range 50 {
		state := NewStreamState()
		input := buildOpenAISSE(
			// Tool 0 opens first.
			`{"id":"c","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_0","type":"function","function":{"name":"alpha","arguments":""}}]},"finish_reason":null}]}`,
			// Tool 1 opens second.
			`{"id":"c","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_1","type":"function","function":{"name":"beta","arguments":""}}]},"finish_reason":null}]}`,
			// Terminal chunk closes both.
			`{"id":"c","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}`,
		)

		out, done, err := EmitAnthropicSSEChunk(input, state, nil, "claude-sonnet-4-5")
		require.NoError(t, err)
		assert.True(t, done)

		body := string(out)
		// Both blocks must have a stop event.
		assert.Equal(t, 2, strings.Count(body, "event: content_block_stop"), "both tool blocks must be closed")

		// The emitter assigns block indices sequentially starting from 0:
		// tool 0 → block 0, tool 1 → block 1. Within the content_block_stop
		// events, the stop for index 0 must appear before the stop for index 1.
		// Scope the search to the stop-events section because message_start also
		// carries "index":0 earlier in the stream.
		stopStart := strings.Index(body, "event: content_block_stop")
		require.GreaterOrEqual(t, stopStart, 0, "content_block_stop events must be present")
		stopSection := body[stopStart:]
		firstStopInSection := strings.Index(stopSection, `"index":0`)
		secondStopInSection := strings.Index(stopSection, `"index":1`)
		require.GreaterOrEqual(t, firstStopInSection, 0, "first tool block stop (index 0) must be present")
		require.Positive(t, secondStopInSection, "second tool block stop (index 1) must be present")
		assert.Less(t, firstStopInSection, secondStopInSection, "tool block stops must appear in block-index ascending order")
	}
}
