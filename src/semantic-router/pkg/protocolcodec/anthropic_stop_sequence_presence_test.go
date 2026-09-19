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
			_, bufferErr := NewBuiltinEngine().TranslateResponse(llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1, body, nil)
			decoder := AnthropicMessagesCodec{}.NewDecoder(llmprotocol.StreamContext{Context: context.Background()}, llmprotocol.DefaultPolicy())
			start := []byte("event: message_start\ndata: " + `{"type":"message_start","message":{"id":"msg1","type":"message","role":"assistant","model":"m","content":[],"stop_reason":null` + extra + `,"usage":{"input_tokens":1,"output_tokens":0}}}` + "\n\n")
			if _, _, err := decoder.Push(start); err != nil {
				t.Fatal(err)
			}
			delta := []byte("event: message_delta\ndata: " + `{"type":"message_delta","delta":{"stop_reason":"end_turn"` + extra + `},"usage":{"input_tokens":1,"output_tokens":1}}` + "\n\n")
			_, _, streamErr := decoder.Push(delta)
			t.Logf("buffered=%v streaming=%v", bufferErr, streamErr)
			if bufferErr != nil || streamErr != nil {
				t.Fatalf("nullable stop_sequence must be accepted consistently: buffered=%v stream=%v", bufferErr, streamErr)
			}
		})
	}
}

func TestAnthropicStopSequenceReasonRequiresMatchedValue(t *testing.T) {
	for _, suffix := range []string{"", `,"stop_sequence":null`, `,"stop_sequence":""`} {
		decoder := newProviderStreamDecoder(llmprotocol.AnthropicMessagesV1)
		body := []byte("event: message_delta\ndata: " + `{"type":"message_delta","delta":{"stop_reason":"stop_sequence"` + suffix + `},"usage":{"output_tokens":1}}` + "\n\n")
		_, _, err := decoder.Push(body)
		assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "anthropic_stop_sequence_required")
	}
}
