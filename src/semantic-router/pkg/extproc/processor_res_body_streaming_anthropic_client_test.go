package extproc

import (
	"strings"
	"testing"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/anthropic"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// newTestRouter returns a minimal OpenAIRouter suitable for unit tests
// that invoke response-body handlers. The router has semantic cache
// disabled so it does not attempt Redis lookups.
func newTestRouter() *OpenAIRouter {
	return &OpenAIRouter{
		Config: &config.RouterConfig{
			SemanticCache: config.SemanticCache{Enabled: false},
		},
	}
}

// newAnthropicStreamCtx returns a RequestContext wired for the Anthropic
// client streaming path with the supplied StreamState.
func newAnthropicStreamCtx(state *anthropic.StreamState) *RequestContext {
	return &RequestContext{
		APIFormat:           config.APIFormatAnthropic,
		ClientProtocol:      config.ClientProtocolAnthropic,
		RequestModel:        "claude-sonnet-4-5",
		AnthropicStream:     state,
		StreamingMetadata:   make(map[string]interface{}),
		ProcessingStartTime: time.Now().Add(-10 * time.Millisecond),
	}
}

// TestHandleAnthropicClientStreamingResponseBody_SuppressesUpstreamTail
// is the regression test for the duplicate message_stop bug observed in
// the double-Anthropic streaming cell (Anthropic client + Anthropic
// backend).
//
// Symptoms before the fix: the emitter synthesized a clean message_stop
// when the prior chunk's message_delta carried usage; the very last
// chunk (which contained only the upstream message_stop frame) made the
// emitter produce zero bytes; the handler then returned a nil
// BodyMutation, which made Envoy forward the upstream bytes verbatim,
// resulting in a second (raw, whitespace-formatted) message_stop on the
// wire.
//
// The fix replaces the body unconditionally so an empty-emission chunk
// suppresses the upstream tail. This test feeds the terminal Anthropic
// message_stop frame to the handler with state that already has
// MessageStopSent=true (simulating the earlier emission) and asserts
// the returned body is the empty mutation, not nil.
func TestHandleAnthropicClientStreamingResponseBody_SuppressesUpstreamTail(t *testing.T) {
	router := newTestRouter()
	state := anthropic.NewStreamState()
	state.MessageStartSent = true
	state.MessageStopSent = true
	state.LastChunkAt = time.Now()
	ctx := newAnthropicStreamCtx(state)

	// Terminal-only chunk from the upstream: message_stop with the
	// trailing-whitespace JSON formatting that some upstream backends emit.
	upstreamTail := strings.Join([]string{
		"event: message_stop",
		`data: {"type":"message_stop"         }`,
		"",
	}, "\n")

	resp := router.handleAnthropicClientStreamingResponseBody([]byte(upstreamTail), ctx)
	require.NotNil(t, resp)
	mutation := resp.GetResponseBody().GetResponse().GetBodyMutation()
	require.NotNil(t, mutation, "BodyMutation must be set so envoy does not forward upstream bytes")

	body := mutation.GetBody()
	bodyStr := string(body)
	assert.NotContains(t, bodyStr, "message_stop",
		"emitter must not let the upstream message_stop leak through after its own message_stop already fired")
	assert.NotContains(t, bodyStr, "         }",
		"trailing-whitespace upstream formatting must not appear on the wire")
}

// TestHandleAnthropicClientStreamingResponseBody_PingInterleave asserts
// that maybeEmitPing fires when the silence gap since the last chunk
// crosses anthropicPingCadence. The ping bytes must appear before the
// content bytes in the mutation body so the client's idle watchdog is
// satisfied before the next token arrives.
func TestHandleAnthropicClientStreamingResponseBody_PingInterleave(t *testing.T) {
	router := newTestRouter()
	state := anthropic.NewStreamState()
	// Set LastChunkAt far in the past to guarantee a ping on the next call.
	state.LastChunkAt = time.Now().Add(-(anthropicPingCadence + time.Second))
	ctx := newAnthropicStreamCtx(state)

	// OpenAI-shaped chunk (no Anthropic translation needed because
	// APIFormat is not set to Anthropic for this sub-test).
	ctx.APIFormat = config.APIFormatOpenAI
	chunk := `data: {"id":"c","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"hi"},"finish_reason":null}]}` + "\n\n"

	resp := router.handleAnthropicClientStreamingResponseBody([]byte(chunk), ctx)
	require.NotNil(t, resp)
	mutation := resp.GetResponseBody().GetResponse().GetBodyMutation()
	require.NotNil(t, mutation, "BodyMutation must always be set")

	body := string(mutation.GetBody())
	pingIdx := strings.Index(body, "event: ping")
	require.GreaterOrEqual(t, pingIdx, 0, "ping event must be present in output")

	// Ping must come before the content events in the same mutation body.
	contentIdx := strings.Index(body, "event: message_start")
	if contentIdx >= 0 {
		assert.Less(t, pingIdx, contentIdx, "ping must precede content events")
	}
}

// TestHandleAnthropicClientStreamingResponseBody_EmptyStream asserts that
// an empty byte slice does not crash the handler and returns a sensible
// (non-nil, empty body) BodyMutation so Envoy suppresses the empty
// upstream chunk rather than forwarding nil.
func TestHandleAnthropicClientStreamingResponseBody_EmptyStream(t *testing.T) {
	router := newTestRouter()
	ctx := newAnthropicStreamCtx(anthropic.NewStreamState())

	resp := router.handleAnthropicClientStreamingResponseBody([]byte{}, ctx)
	require.NotNil(t, resp, "handler must not return nil on empty input")
	mutation := resp.GetResponseBody().GetResponse().GetBodyMutation()
	require.NotNil(t, mutation, "BodyMutation must be set even for empty input")
	// Empty body is the correct suppression signal; nil would leak upstream bytes.
	assert.Equal(t, []byte{}, mutation.GetBody(), "empty input must yield empty (not nil) mutation body")
}

// TestHandleAnthropicClientStreamingResponseBody_UnparsableChunk asserts
// that an unparsable upstream chunk is handled gracefully: the emitter
// emits no bytes (and returns no error — bad JSON is swallowed, not
// surfaced), and the handler still returns a non-nil response carrying a
// non-nil empty BodyMutation so Envoy suppresses the upstream chunk
// rather than forwarding the raw bytes.
//
// Note: this exercises the empty-emission path, NOT the err != nil
// branch. EmitAnthropicSSEChunk does not return an error for malformed
// data; it returns empty output. The handler's error branches are
// covered separately where a real translator error can be induced.
func TestHandleAnthropicClientStreamingResponseBody_UnparsableChunk(t *testing.T) {
	router := newTestRouter()
	ctx := newAnthropicStreamCtx(anthropic.NewStreamState())
	// OpenAI format bypasses the Anthropic translator, feeding the bytes
	// straight to the emitter, which cannot parse them and emits nothing.
	ctx.APIFormat = config.APIFormatOpenAI
	badChunk := []byte("data: not-valid-json\n\n")

	resp := router.handleAnthropicClientStreamingResponseBody(badChunk, ctx)
	require.NotNil(t, resp, "handler must not return nil on unparsable input")
	mutation := resp.GetResponseBody().GetResponse().GetBodyMutation()
	require.NotNil(t, mutation, "BodyMutation must be set even for unparsable input")
	assert.Equal(t, []byte{}, mutation.GetBody(), "unparsable input must yield empty (not nil) mutation body")
}

// TestHandleAnthropicClientStreamingResponseBody_PreviewUsageChunk is the
// end-to-end regression test for issue #2215. An upstream OpenAI gateway
// sends a usage-only "preview" chunk first (completion_tokens:4, empty
// choices), then content, then the real terminal chunk
// (completion_tokens:252), then [DONE].
//
// Before the fix the emitter treated the preview chunk as terminal: it
// fired message_stop after chunk 1, the handler called
// finalizeStreamingResponse early, the StreamingComplete idempotency guard
// then froze the recorded usage at the preview value (4), and all later
// content was dropped from the cache/replay. After the fix finalization
// happens only at [DONE], so the recorded usage is the real total (252) and
// the content is preserved.
func TestHandleAnthropicClientStreamingResponseBody_PreviewUsageChunk(t *testing.T) {
	router := newTestRouter()
	ctx := newAnthropicStreamCtx(anthropic.NewStreamState())
	// Issue #2215 config: OpenAI upstream, Anthropic client. OpenAI bytes
	// pass straight through translateUpstreamToOpenAI to the emitter.
	ctx.APIFormat = config.APIFormatOpenAI
	ctx.RequestModel = "claude-opus-4-6"

	feed := func(chunk string) string {
		resp := router.handleAnthropicClientStreamingResponseBody([]byte(chunk), ctx)
		require.NotNil(t, resp)
		return string(resp.GetResponseBody().GetResponse().GetBodyMutation().GetBody())
	}

	// Chunk 1: preview usage-only chunk (the trigger).
	out1 := feed(`data: {"id":"chatcmpl-x","object":"chat.completion.chunk","choices":[],"usage":{"prompt_tokens":37494,"completion_tokens":4,"total_tokens":37498}}` + "\n\n")
	assert.NotContains(t, out1, "event: message_stop", "preview usage chunk must not emit a premature message_stop")
	assert.False(t, ctx.StreamingComplete, "preview usage chunk must not finalize the stream")

	// Chunk 2 + 3: real content.
	feed(`data: {"id":"chatcmpl-x","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":"hello "}}]}` + "\n\n")
	feed(`data: {"id":"chatcmpl-x","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"world"}}]}` + "\n\n")
	assert.False(t, ctx.StreamingComplete, "stream must stay open while content flows")

	// Chunk 4: real terminal chunk with finish_reason and the true usage.
	out4 := feed(`data: {"id":"chatcmpl-x","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":37494,"completion_tokens":252,"total_tokens":37746}}` + "\n\n")
	assert.Contains(t, out4, "event: message_stop", "the real terminal chunk must emit message_stop")

	// Chunk 5: [DONE] marker.
	feed("data: [DONE]\n\n")

	require.True(t, ctx.StreamingComplete, "stream must be finalized after [DONE]")
	assert.Equal(t, "hello world", ctx.StreamingContent, "all content after the preview chunk must be preserved")

	// The recorded usage must be the real total (252), not the preview (4).
	usage := extractStreamingUsage(ctx)
	assert.Equal(t, int64(252), usage.CompletionTokens, "recorded completion_tokens must be the real total, not the preview value")
	assert.Equal(t, int64(37746), usage.TotalTokens, "recorded total_tokens must come from the real terminal chunk")
}

// TestHandleResponseBody_StreamingDispatchMatrix locks the four-cell
// streaming dispatch in handleResponseBody:
//
//	ClientProtocol × APIFormat → handler → wire shape
//	-------------------------------------------------------------
//	anthropic  × anthropic  → AnthropicClient streamer → Anthropic SSE
//	anthropic  × openai     → AnthropicClient streamer → Anthropic SSE
//	openai     × anthropic  → legacy Anthropic streamer → OpenAI SSE
//	openai     × openai     → generic OpenAI streamer   → pass-through
//
// The branch ordering at processor_res_body.go matters: the Anthropic
// *client* branch must precede the legacy Anthropic *backend* branch, or
// the double-Anthropic cell would fall through and emit OpenAI SSE to an
// Anthropic client. This test asserts each cell reaches the right handler
// by inspecting the observable wire shape, so a future reordering or a
// missing ClientProtocol guard fails loudly.
func TestHandleResponseBody_StreamingDispatchMatrix(t *testing.T) {
	openAIChunk := `data: {"id":"c","object":"chat.completion.chunk","model":"claude-sonnet-4-5","choices":[{"index":0,"delta":{"role":"assistant","content":"hi"},"finish_reason":null}]}` + "\n\n"
	anthropicChunk := strings.Join([]string{
		"event: content_block_delta",
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}`,
		"", "",
	}, "\n")

	// Each cell routes to a distinct handler with an observably distinct
	// result, so the assertions positively distinguish the four handlers
	// rather than checking only for the absence of a shape:
	//   - anthropic-client cells: handler replaces the body with Anthropic
	//     SSE (non-nil mutation containing "event:", never OpenAI chunk shape).
	//   - openai client + anthropic backend: legacy handler emits the
	//     translated OpenAI chunk (non-nil mutation containing
	//     "chat.completion.chunk").
	//   - openai client + openai backend: generic streamer passes through
	//     with a nil BodyMutation (no rewrite).
	const (
		wireAnthropic = "anthropic-sse"   // non-nil mutation, Anthropic events
		wireOpenAI    = "openai-chunk"    // non-nil mutation, OpenAI chunk shape
		wirePassThru  = "passthrough-nil" // nil mutation (no rewrite)
	)
	cases := []struct {
		name           string
		clientProtocol string
		apiFormat      string
		body           string
		wantWire       string
	}{
		{
			name:           "anthropic client + anthropic backend → Anthropic SSE",
			clientProtocol: config.ClientProtocolAnthropic,
			apiFormat:      config.APIFormatAnthropic,
			body:           anthropicChunk,
			wantWire:       wireAnthropic,
		},
		{
			name:           "anthropic client + openai backend → Anthropic SSE",
			clientProtocol: config.ClientProtocolAnthropic,
			apiFormat:      config.APIFormatOpenAI,
			body:           openAIChunk,
			wantWire:       wireAnthropic,
		},
		{
			name:           "openai client + anthropic backend → OpenAI SSE",
			clientProtocol: "",
			apiFormat:      config.APIFormatAnthropic,
			body:           anthropicChunk,
			wantWire:       wireOpenAI,
		},
		{
			name:           "openai client + openai backend → OpenAI pass-through",
			clientProtocol: "",
			apiFormat:      config.APIFormatOpenAI,
			body:           openAIChunk,
			wantWire:       wirePassThru,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			router := newTestRouter()
			ctx := &RequestContext{
				IsStreamingResponse: true,
				ClientProtocol:      tc.clientProtocol,
				APIFormat:           tc.apiFormat,
				RequestModel:        "claude-sonnet-4-5",
				StreamingMetadata:   make(map[string]interface{}),
				ProcessingStartTime: time.Now().Add(-10 * time.Millisecond),
			}
			v := &ext_proc.ProcessingRequest_ResponseBody{
				ResponseBody: &ext_proc.HttpBody{Body: []byte(tc.body)},
			}

			resp, err := router.handleResponseBody(v, ctx)
			require.NoError(t, err)
			require.NotNil(t, resp)

			mutation := resp.GetResponseBody().GetResponse().GetBodyMutation()

			switch tc.wantWire {
			case wireAnthropic:
				require.NotNil(t, mutation, "anthropic-client cell must replace the body")
				wire := string(mutation.GetBody())
				assert.Contains(t, wire, "event:", "anthropic-client cell must emit Anthropic SSE to the wire")
				assert.NotContains(t, wire, "chat.completion.chunk", "anthropic-client cell must not leak OpenAI chunk shape")
			case wireOpenAI:
				require.NotNil(t, mutation, "openai-client + anthropic-backend cell must emit the translated chunk")
				wire := string(mutation.GetBody())
				assert.Contains(t, wire, "chat.completion.chunk", "legacy Anthropic handler must emit OpenAI chunk shape")
				assert.NotContains(t, wire, "event: content_block", "openai-client cell must not emit Anthropic content_block events")
			case wirePassThru:
				assert.Nil(t, mutation, "openai client + openai backend must pass through with a nil BodyMutation (no rewrite)")
			}
		})
	}
}
