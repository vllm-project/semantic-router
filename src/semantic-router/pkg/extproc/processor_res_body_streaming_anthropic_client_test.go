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
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			SemanticCache: config.SemanticCache{Enabled: false},
		},
	}
	state := anthropic.NewStreamState()
	state.MessageStartSent = true
	state.MessageStopSent = true
	state.LastChunkAt = time.Now()

	ctx := &RequestContext{
		APIFormat:           config.APIFormatAnthropic,
		ClientProtocol:      config.ClientProtocolAnthropic,
		RequestModel:        "claude-sonnet-4-5",
		AnthropicStream:     state,
		StreamingMetadata:   make(map[string]interface{}),
		ProcessingStartTime: time.Now().Add(-10 * time.Millisecond),
	}

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
