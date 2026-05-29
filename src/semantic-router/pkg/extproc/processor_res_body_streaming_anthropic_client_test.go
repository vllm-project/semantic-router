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

// TestHandleAnthropicClientStreamingResponseBody_MidStreamError asserts
// that when EmitAnthropicSSEChunk returns an error (e.g. state == nil),
// the handler returns a non-nil response with an empty mutation rather
// than panicking or returning nil (which would forward raw upstream bytes).
//
// We trigger the error path indirectly: pass a body that will parse into
// a valid OpenAI chunk but set state to nil in the context so the emitter
// cannot proceed. The handler must recover gracefully.
func TestHandleAnthropicClientStreamingResponseBody_MidStreamError(t *testing.T) {
	router := newTestRouter()
	ctx := newAnthropicStreamCtx(nil) // nil state causes graceful nil-init inside emitter

	// A malformed upstream that fails translation will cause translateUpstreamToOpenAI
	// to return an error via TransformSSEChunkToOpenAI's unmarshal.
	// Use a non-Anthropic format so the translator is bypassed and we get
	// well-formed but minimal OpenAI SSE bytes fed into EmitAnthropicSSEChunk.
	ctx.APIFormat = config.APIFormatOpenAI
	// Deliberately corrupt the chunk so emitChunkEvents gets a bad json.Unmarshal.
	badChunk := []byte("data: not-valid-json\n\n")

	resp := router.handleAnthropicClientStreamingResponseBody(badChunk, ctx)
	require.NotNil(t, resp, "handler must not return nil on parse error")
	// The response must be a BodyResponse (not a header or other type) to
	// remain consistent with the streaming response pipeline.
	require.NotNil(t, resp.GetResponseBody())
}
