package protocolcodec

import (
	"context"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestAnthropicStopSequencePresenceConsistency(t *testing.T) {
	for _, present := range []bool{true, false} {
		name := "explicit_null_control"
		if !present {
			name = "absent_stop_sequence"
		}
		t.Run(name, func(t *testing.T) {
			extra := ""
			if present {
				extra = `,"stop_sequence":null`
			}
			body := []byte(`{"id":"msg1","type":"message","role":"assistant","model":"m","content":[{"type":"text","text":"OK"}],"stop_reason":"end_turn"` + extra + `,"usage":{"input_tokens":1,"output_tokens":1}}`)
			for _, target := range []llmprotocol.WireFormat{
				llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1, llmprotocol.AnthropicMessagesV1,
			} {
				result, err := NewBuiltinEngine().TranslateResponse(llmprotocol.AnthropicMessagesV1, target, body, nil)
				if err != nil {
					t.Fatalf("buffered %s: %v", target, err)
				}
				assertStopSequenceDiagnostic(t, result.Diagnostics, "stop_sequence", !present)
			}
			decoder := AnthropicMessagesCodec{}.NewDecoder(llmprotocol.StreamContext{Context: context.Background()}, llmprotocol.DefaultPolicy())
			start := []byte("event: message_start\ndata: " + `{"type":"message_start","message":{"id":"msg1","type":"message","role":"assistant","model":"m","content":[],"stop_reason":null` + extra + `,"usage":{"input_tokens":1,"output_tokens":0}}}` + "\n\n")
			_, startDiagnostics, err := decoder.Push(start)
			if err != nil {
				t.Fatal(err)
			}
			assertStopSequenceDiagnostic(t, startDiagnostics, "message.stop_sequence", !present)
			delta := []byte("event: message_delta\ndata: " + `{"type":"message_delta","delta":{"stop_reason":"end_turn"` + extra + `},"usage":{"input_tokens":1,"output_tokens":1}}` + "\n\n")
			_, deltaDiagnostics, err := decoder.Push(delta)
			if err != nil {
				t.Fatalf("nullable stop_sequence must be accepted consistently: %v", err)
			}
			assertStopSequenceDiagnostic(t, deltaDiagnostics, "delta.stop_sequence", !present)
		})
	}
}

func assertStopSequenceDiagnostic(t *testing.T, diagnostics llmprotocol.Diagnostics, field string, expected bool) {
	t.Helper()
	count := 0
	for _, diagnostic := range diagnostics {
		if diagnostic.Field != field {
			continue
		}
		count++
		if diagnostic.Source != llmprotocol.AnthropicMessagesV1 || diagnostic.Action != llmprotocol.DiagnosticApproximated || diagnostic.Reason == "" {
			t.Fatalf("missing compatibility diagnostic details: %+v", diagnostic)
		}
	}
	want := 0
	if expected {
		want = 1
	}
	if count != want {
		t.Fatalf("%s diagnostic count = %d, want %d: %+v", field, count, want, diagnostics)
	}
}

func TestAnthropicNullablePresenceDiagnosticsRespectBatchLimit(t *testing.T) {
	policy := llmprotocol.DefaultPolicy()
	policy.Limits.Diagnostics = 1
	decoder := AnthropicMessagesCodec{}.NewDecoder(llmprotocol.StreamContext{Context: context.Background()}, policy)
	frames := []byte("event: message_start\ndata: " + `{"type":"message_start","message":{"id":"msg1","type":"message","role":"assistant","model":"m","content":[],"stop_reason":null,"usage":{"input_tokens":1,"output_tokens":0}}}` + "\n\n" +
		"event: message_delta\ndata: " + `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":1,"output_tokens":1}}` + "\n\n")
	events, diagnostics, err := decoder.Push(frames)
	if err != nil || len(events) == 0 {
		t.Fatalf("compatible response did not decode: events=%d error=%v", len(events), err)
	}
	if len(diagnostics) != policy.Limits.Diagnostics {
		t.Fatalf("diagnostics not bounded: %+v", diagnostics)
	}
}

func TestAnthropicMessageDeltaStillRequiresStopReason(t *testing.T) {
	decoder := newProviderStreamDecoder(llmprotocol.AnthropicMessagesV1)
	body := []byte("event: message_delta\ndata: " + `{"type":"message_delta","delta":{"stop_sequence":null},"usage":{"output_tokens":1}}` + "\n\n")
	_, _, err := decoder.Push(body)
	assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "stream_required_field")
}

func TestAnthropicStopSequenceReasonRequiresMatchedValue(t *testing.T) {
	for _, suffix := range []string{"", `,"stop_sequence":null`, `,"stop_sequence":""`} {
		decoder := newProviderStreamDecoder(llmprotocol.AnthropicMessagesV1)
		body := []byte("event: message_delta\ndata: " + `{"type":"message_delta","delta":{"stop_reason":"stop_sequence"` + suffix + `},"usage":{"output_tokens":1}}` + "\n\n")
		_, _, err := decoder.Push(body)
		assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "anthropic_stop_sequence_required")
	}
}
