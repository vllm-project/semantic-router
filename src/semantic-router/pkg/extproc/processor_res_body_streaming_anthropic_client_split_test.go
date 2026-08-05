package extproc

import (
	"strconv"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/anthropic"
)

// reasoningStreamFrames returns a realistic reasoning-model upstream
// Anthropic SSE sequence (message_start → thinking block → text block →
// message_delta → message_stop), one complete frame per element.
func reasoningStreamFrames() []string {
	frame := func(event, data string) string {
		return "event: " + event + "\n" + "data: " + data + "\n\n"
	}
	return []string{
		frame("message_start", `{"type":"message_start","message":{"id":"msg_2316","type":"message","role":"assistant","model":"test-model","content":[],"stop_reason":null,"usage":{"input_tokens":10,"output_tokens":1}}}`),
		frame("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`),
		frame("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Let me think about the greeting."}}`),
		frame("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig_abc"}}`),
		frame("content_block_stop", `{"type":"content_block_stop","index":0}`),
		frame("content_block_start", `{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`),
		frame("content_block_delta", `{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"你好"}}`),
		frame("content_block_delta", `{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"！有什么可以帮你？"}}`),
		frame("content_block_stop", `{"type":"content_block_stop","index":1}`),
		frame("message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":8}}`),
		frame("message_stop", `{"type":"message_stop"}`),
	}
}

// feedDoubleAnthropicStream feeds the given wire chunks through the
// double-Anthropic client streaming handler and returns the concatenated
// bytes the client would observe on the wire.
func feedDoubleAnthropicStream(t *testing.T, chunks []string) string {
	t.Helper()
	router := newTestRouter()
	ctx := newAnthropicStreamCtx(anthropic.NewStreamState()) // APIFormat=Anthropic, ClientProtocol=Anthropic
	var wire strings.Builder
	for i, c := range chunks {
		resp := router.handleAnthropicClientStreamingResponseBody([]byte(c), ctx)
		if resp == nil {
			t.Fatalf("chunk %d: handler returned nil", i)
		}
		wire.Write(resp.GetResponseBody().GetResponse().GetBodyMutation().GetBody())
	}
	return wire.String()
}

func assertUsableAnthropicStream(t *testing.T, wire string) {
	t.Helper()
	for _, want := range []string{
		"event: message_start",
		`"type":"thinking_delta"`,
		`"type":"text_delta"`,
		"你好",
		"event: message_stop",
	} {
		assert.Contains(t, wire, want, "client stream must contain %q", want)
	}
}

// TestDoubleAnthropicStream_FrameAlignedDelivery is the baseline: when Envoy
// hands the handler one complete SSE frame per chunk, the double-Anthropic
// cell translates the full reasoning stream correctly.
func TestDoubleAnthropicStream_FrameAlignedDelivery(t *testing.T) {
	wire := feedDoubleAnthropicStream(t, reasoningStreamFrames())
	assertUsableAnthropicStream(t, wire)
}

// TestDoubleAnthropicStream_ByteSplitDelivery is the regression test for
// issue #2316. Envoy STREAMED mode delivers the response body split at
// arbitrary byte offsets that do NOT align to SSE frame boundaries. Before
// the reassembly fix, split frames failed to parse and the handler
// suppressed the raw upstream bytes, so the client received zero output and
// hung. After the fix, the handler reassembles frames across chunks and the
// client sees the complete stream regardless of where the splits fall.
func TestDoubleAnthropicStream_ByteSplitDelivery(t *testing.T) {
	whole := strings.Join(reasoningStreamFrames(), "")

	// Try several unaligned span sizes so the test does not depend on a
	// lucky boundary; small spans guarantee mid-frame splits.
	for _, span := range []int{1, 7, 32, 64, 200} {
		t.Run("span="+strconv.Itoa(span), func(t *testing.T) {
			var chunks []string
			for i := 0; i < len(whole); i += span {
				end := i + span
				if end > len(whole) {
					end = len(whole)
				}
				chunks = append(chunks, whole[i:end])
			}
			wire := feedDoubleAnthropicStream(t, chunks)
			assertUsableAnthropicStream(t, wire)
		})
	}
}
