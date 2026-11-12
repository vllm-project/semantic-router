package protocolcodec

import (
	"context"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Anthropic reports the inference speed mode ("standard" or "fast") in usage,
// on buffered replies and in message_start. It has no neutral accounting
// bucket, so it decodes, the usage counts are untouched, and the drop is
// reported.
const anthropicSpeedResponseFixture = `{"id":"msg_1","type":"message","role":"assistant","model":"provider-model","content":[{"type":"text","text":"done"}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":12,"output_tokens":4,"service_tier":"standard","inference_geo":"global","speed":"fast"}}`

func TestAnthropicResponseAcceptsUsageSpeed(t *testing.T) {
	engine := NewBuiltinEngine()
	t.Run("same format replays the reply", func(t *testing.T) {
		translated, err := engine.TranslateResponse(llmprotocol.AnthropicMessagesV1, llmprotocol.AnthropicMessagesV1,
			[]byte(anthropicSpeedResponseFixture), nil)
		if err != nil {
			t.Fatalf("TranslateResponse() error = %v", err)
		}
		if string(translated.Body) != anthropicSpeedResponseFixture {
			t.Fatalf("same-format reply was not replayed byte-for-byte:\n%s", translated.Body)
		}
		assertDiagnosticFields(t, translated.Diagnostics, "usage.inference_geo", "usage.service_tier", "usage.speed")
	})
	t.Run("cross format keeps the usage", func(t *testing.T) {
		response, _, diagnostics, err := engine.DecodeResponse(llmprotocol.AnthropicMessagesV1, []byte(anthropicSpeedResponseFixture))
		if err != nil {
			t.Fatalf("DecodeResponse() error = %v", err)
		}
		if response.Usage.InputTotal.Value == nil || *response.Usage.InputTotal.Value != 12 ||
			response.Usage.OutputTotal.Value == nil || *response.Usage.OutputTotal.Value != 4 {
			t.Fatalf("usage changed: %+v", response.Usage)
		}
		if !hasDiagnostic(diagnostics, "usage.speed", llmprotocol.DiagnosticDropped) {
			t.Fatalf("speed omission was not reported: %+v", diagnostics)
		}
		translated, err := engine.TranslateResponse(llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1,
			[]byte(anthropicSpeedResponseFixture), nil)
		if err != nil || strings.Contains(string(translated.Body), "speed") {
			t.Fatalf("Chat reply = %s, error %v; want the reply without speed", translated.Body, err)
		}
	})
}

func TestAnthropicResponseUsageSpeedStaysStrict(t *testing.T) {
	engine := NewBuiltinEngine()
	t.Run("null speed reports nothing", func(t *testing.T) {
		body := strings.Replace(anthropicSpeedResponseFixture, `"speed":"fast"`, `"speed":null`, 1)
		_, _, diagnostics, err := engine.DecodeResponse(llmprotocol.AnthropicMessagesV1, []byte(body))
		if err != nil {
			t.Fatalf("DecodeResponse() error = %v", err)
		}
		if hasDiagnostic(diagnostics, "usage.speed", llmprotocol.DiagnosticDropped) {
			t.Fatalf("null speed was reported: %+v", diagnostics)
		}
	})
	t.Run("unrelated unknown usage field", func(t *testing.T) {
		body := strings.Replace(anthropicSpeedResponseFixture, `"speed":"fast"`, `"speed":"fast","x_unknown":1`, 1)
		_, _, _, err := conformanceEngine().DecodeResponse(llmprotocol.AnthropicMessagesV1, []byte(body))
		assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "invalid_upstream_json")
	})
}

func anthropicSpeedStream(startUsage string) []byte {
	return []byte("event: message_start\n" +
		`data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"provider-model","content":[],"stop_reason":null,"stop_sequence":null,"usage":` + startUsage + `}}` + "\n\n" +
		"event: content_block_start\n" +
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}` + "\n\n" +
		"event: content_block_delta\n" +
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"done"}}` + "\n\n" +
		"event: content_block_stop\n" +
		`data: {"type":"content_block_stop","index":0}` + "\n\n" +
		"event: message_delta\n" +
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":4}}` + "\n\n" +
		"event: message_stop\n" +
		`data: {"type":"message_stop"}` + "\n\n")
}

// message_start usage metadata is reported once per stream, like the same
// fields on a buffered reply.
func TestAnthropicStreamAcceptsUsageSpeed(t *testing.T) {
	decoder := AnthropicMessagesCodec{}.NewDecoder(llmprotocol.StreamContext{Context: context.Background()}, llmprotocol.DefaultPolicy())
	_, diagnostics, err := decoder.Push(anthropicSpeedStream(`{"input_tokens":12,"output_tokens":1,"service_tier":"standard","inference_geo":"global","speed":"standard"}`))
	if err != nil {
		t.Fatalf("stream with usage.speed was rejected: %v", err)
	}
	if _, _, err := decoder.Finalize(nil); err != nil {
		t.Fatal(err)
	}
	var fields []string
	for _, diagnostic := range diagnostics {
		if diagnostic.Action == llmprotocol.DiagnosticDropped {
			fields = append(fields, diagnostic.Field)
		}
	}
	want := "stream.message.usage.inference_geo,stream.message.usage.service_tier,stream.message.usage.speed"
	if strings.Join(fields, ",") != want {
		t.Fatalf("dropped fields = %v, want %s", fields, want)
	}
}

func TestAnthropicStreamUsageSpeedStaysStrict(t *testing.T) {
	decoder := AnthropicMessagesCodec{}.NewDecoder(llmprotocol.StreamContext{Context: context.Background()}, conformancePolicy())
	_, _, err := decoder.Push(anthropicSpeedStream(`{"input_tokens":12,"output_tokens":1,"speed":"standard","x_unknown":1}`))
	assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "invalid_upstream_json")
}
