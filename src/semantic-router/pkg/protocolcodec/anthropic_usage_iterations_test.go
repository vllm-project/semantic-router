package protocolcodec

import (
	"context"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// The Anthropic API emits a per-attempt token-accounting breakdown as an
// "iterations" array inside the usage object; the aggregate totals alongside it
// already cover the turn. The neutral usage model has no per-attempt bucket, so
// the codec must accept and ignore it while still decoding the top-level
// counters that drive accounting.

// usageIterations is the field under test: the per-attempt breakdown that must
// be tolerated inside any usage object.
const usageIterations = `"iterations":[{"type":"message","input_tokens":5,"output_tokens":7,"cache_read_input_tokens":0,"cache_creation_input_tokens":0}]`

func TestAnthropicStreamUsageIterationsAreAcceptedAndIgnored(t *testing.T) {
	decoder := AnthropicMessagesCodec{}.NewDecoder(
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"},
		llmprotocol.DefaultPolicy(),
	)
	// A minimal well-formed stream; the terminal message_delta carries the
	// usage object with the iterations breakdown embedded.
	terminalUsage := `{"input_tokens":5,"output_tokens":7,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,` + usageIterations + `}`
	payload := []byte(
		"event: message_start\ndata: " + `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"provider-model","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":5,"output_tokens":0}}}` + "\n\n" +
			"event: content_block_start\ndata: " + `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}` + "\n\n" +
			"event: content_block_delta\ndata: " + `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"done"}}` + "\n\n" +
			"event: content_block_stop\ndata: " + `{"type":"content_block_stop","index":0}` + "\n\n" +
			"event: message_delta\ndata: " + `{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":` + terminalUsage + `}` + "\n\n" +
			"event: message_stop\ndata: " + `{"type":"message_stop"}` + "\n\n",
	)
	events, _, err := decoder.Push(payload)
	if err != nil {
		t.Fatalf("streamed message_delta usage carrying iterations was rejected: %v", err)
	}
	terminal := events[len(events)-1]
	if terminal.Type != llmprotocol.EventResponseCompleted || terminal.Usage == nil ||
		tokenValue(terminal.Usage.InputTotal) != 5 ||
		tokenValue(terminal.Usage.OutputTotal) != 7 ||
		tokenValue(terminal.Usage.Total) != 12 {
		t.Fatalf("top-level usage counters did not decode: %+v", terminal.Usage)
	}
}

func TestAnthropicResponseUsageIterationsAreAcceptedAndIgnored(t *testing.T) {
	engine := NewBuiltinEngine()
	body := []byte(`{"id":"msg_1","type":"message","role":"assistant","model":"provider-model",` +
		`"content":[{"type":"text","text":"done"}],"stop_reason":"end_turn","stop_sequence":null,` +
		`"usage":{"input_tokens":5,"output_tokens":7,` + usageIterations + `}}`)
	response, _, diagnostics, err := engine.DecodeResponse(llmprotocol.AnthropicMessagesV1, body)
	if err != nil {
		t.Fatalf("non-streaming usage carrying iterations was rejected: %v", err)
	}
	if response.Usage.State != llmprotocol.UsageAvailable ||
		tokenValue(response.Usage.InputTotal) != 5 ||
		tokenValue(response.Usage.OutputTotal) != 7 {
		t.Fatalf("top-level usage counters did not decode: %+v", response.Usage)
	}
	dropped := false
	for _, diagnostic := range diagnostics {
		if diagnostic.Field == "usage.iterations" && diagnostic.Action == llmprotocol.DiagnosticDropped {
			dropped = true
		}
	}
	if !dropped {
		t.Fatalf("dropped per-attempt accounting was not diagnosed: %+v", diagnostics)
	}
}
