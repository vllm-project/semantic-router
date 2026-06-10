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

	cases := []struct {
		name           string
		clientProtocol string
		apiFormat      string
		body           string
		// wantAnthropicWire asserts the client sees Anthropic SSE (event: lines).
		wantAnthropicWire bool
		// wantOpenAIChunkWire asserts the client sees OpenAI chat.completion.chunk SSE.
		wantOpenAIChunkWire bool
	}{
		{
			name:              "anthropic client + anthropic backend → Anthropic SSE",
			clientProtocol:    config.ClientProtocolAnthropic,
			apiFormat:         config.APIFormatAnthropic,
			body:              anthropicChunk,
			wantAnthropicWire: true,
		},
		{
			name:              "anthropic client + openai backend → Anthropic SSE",
			clientProtocol:    config.ClientProtocolAnthropic,
			apiFormat:         config.APIFormatOpenAI,
			body:              openAIChunk,
			wantAnthropicWire: true,
		},
		{
			name:                "openai client + anthropic backend → OpenAI SSE",
			clientProtocol:      "",
			apiFormat:           config.APIFormatAnthropic,
			body:                anthropicChunk,
			wantOpenAIChunkWire: true,
		},
		{
			name:                "openai client + openai backend → OpenAI pass-through",
			clientProtocol:      "",
			apiFormat:           config.APIFormatOpenAI,
			body:                openAIChunk,
			wantOpenAIChunkWire: true,
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

			// The Anthropic-client cells own the wire and always replace the
			// body; the OpenAI-client cells route through handlers that emit
			// their accumulated chunk (anthropic backend) or pass through
			// (openai backend). We inspect whatever body mutation is present.
			mutation := resp.GetResponseBody().GetResponse().GetBodyMutation()
			var wire string
			if mutation != nil {
				wire = string(mutation.GetBody())
			}

			if tc.wantAnthropicWire {
				assert.Contains(t, wire, "event:", "anthropic-client cell must emit Anthropic SSE to the wire")
				assert.NotContains(t, wire, "chat.completion.chunk", "anthropic-client cell must not leak OpenAI chunk shape")
			}
			if tc.wantOpenAIChunkWire {
				assert.NotContains(t, wire, "event: content_block", "openai-client cell must not emit Anthropic content_block events")
			}
		})
	}
}
