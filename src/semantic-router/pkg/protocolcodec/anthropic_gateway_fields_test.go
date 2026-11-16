package protocolcodec

import (
	"context"
	"reflect"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A gateway that serves the Anthropic Messages format adds its own usage
// fields, such as a cost in exponent form, beside Anthropic's own speed,
// service tier and inference geography.
const anthropicGatewayResponseFixture = `{"id":"msg_1","type":"message","role":"assistant","model":"provider-model","content":[{"type":"text","text":"done"}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":12,"output_tokens":4,"service_tier":"standard","inference_geo":"global","speed":"standard","total_cost":3e-05}}`

func droppedFields(diagnostics llmprotocol.Diagnostics) []string {
	var fields []string
	for _, diagnostic := range diagnostics {
		if diagnostic.Action == llmprotocol.DiagnosticDropped && diagnostic.Reason == unknownUpstreamFieldReason {
			fields = append(fields, diagnostic.Field)
		}
	}
	return fields
}

func TestAnthropicGatewayUsageFieldsNeedNoSetting(t *testing.T) {
	engine := NewBuiltinEngine()
	response, envelope, diagnostics, err := engine.DecodeResponse(llmprotocol.AnthropicMessagesV1, []byte(anthropicGatewayResponseFixture))
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v", err)
	}
	if got := droppedFields(diagnostics); !reflect.DeepEqual(got, []string{"usage.total_cost"}) {
		t.Fatalf("unknown fields = %v, want only usage.total_cost and no renamed counters", got)
	}
	if !DiagnosticsDroppedUnknownFields(diagnostics) {
		t.Fatal("the drop is not marked, so a cache would persist the gateway's bytes")
	}
	if response.Usage.InputTotal.Value == nil || *response.Usage.InputTotal.Value != 12 ||
		response.Usage.OutputTotal.Value == nil || *response.Usage.OutputTotal.Value != 4 {
		t.Fatalf("usage changed: %+v", response.Usage)
	}
	if !strings.Contains(string(envelope.Response), "total_cost") {
		t.Fatalf("a same-format replay lost the gateway field: %s", envelope.Response)
	}

	chat, err := engine.TranslateResponse(llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1, []byte(anthropicGatewayResponseFixture), nil)
	if err != nil {
		t.Fatalf("TranslateResponse() error = %v", err)
	}
	if !strings.Contains(string(chat.Body), `"prompt_tokens":12,"completion_tokens":4`) || strings.Contains(string(chat.Body), "total_cost") {
		t.Fatalf("Chat reply = %s", chat.Body)
	}

	native, err := engine.TranslateResponse(llmprotocol.AnthropicMessagesV1, llmprotocol.AnthropicMessagesV1, []byte(anthropicGatewayResponseFixture), nil)
	if err != nil {
		t.Fatalf("TranslateResponse(same format) error = %v", err)
	}
	if string(native.Body) != anthropicGatewayResponseFixture {
		t.Fatalf("a same-format reply changed the upstream bytes: %s", native.Body)
	}
}

func TestAnthropicGatewayUsageFieldsFailConformance(t *testing.T) {
	_, _, _, err := AnthropicMessagesCodec{}.DecodeResponse([]byte(anthropicGatewayResponseFixture), conformancePolicy())
	assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "invalid_upstream_json")
}

// Known usage members keep their invariants: a negative counter still fails
// the reply, whatever unknown members ride beside it.
func TestAnthropicGatewayFieldsDoNotRelaxKnownCounters(t *testing.T) {
	body := strings.Replace(anthropicGatewayResponseFixture, `"output_tokens":4`, `"output_tokens":-4`, 1)
	_, _, _, err := NewBuiltinEngine().DecodeResponse(llmprotocol.AnthropicMessagesV1, []byte(body))
	assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "negative_usage")
}

func anthropicGatewayStream() []byte {
	return []byte("event: message_start\n" +
		`data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"provider-model","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":12,"output_tokens":1,"speed":"standard","total_cost":1e-06}}}` + "\n\n" +
		"event: content_block_start\n" +
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}` + "\n\n" +
		"event: content_block_delta\n" +
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"done"}}` + "\n\n" +
		"event: content_block_stop\n" +
		`data: {"type":"content_block_stop","index":0}` + "\n\n" +
		"event: message_delta\n" +
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":4,"total_cost":3e-05}}` + "\n\n" +
		"event: message_stop\n" +
		`data: {"type":"message_stop"}` + "\n\n")
}

func TestAnthropicGatewayStreamUsageFieldsNeedNoSetting(t *testing.T) {
	decoder := AnthropicMessagesCodec{}.NewDecoder(llmprotocol.StreamContext{Context: context.Background()}, llmprotocol.DefaultPolicy())
	events, diagnostics, err := decoder.Push(anthropicGatewayStream())
	if err != nil {
		t.Fatalf("gateway stream was rejected: %v", err)
	}
	final, _, err := decoder.Finalize(nil)
	if err != nil {
		t.Fatal(err)
	}
	if got := droppedFields(diagnostics); !reflect.DeepEqual(got, []string{"message.usage.total_cost", "usage.total_cost"}) {
		t.Fatalf("unknown fields = %v", got)
	}
	var usage *llmprotocol.Usage
	for _, event := range append(events, final...) {
		if event.Type == llmprotocol.EventResponseCompleted {
			usage = event.Usage
		}
	}
	if usage == nil || usage.InputTotal.Value == nil || *usage.InputTotal.Value != 12 ||
		usage.OutputTotal.Value == nil || *usage.OutputTotal.Value != 4 {
		t.Fatalf("settled usage = %+v", usage)
	}

	conformance := AnthropicMessagesCodec{}.NewDecoder(llmprotocol.StreamContext{Context: context.Background()}, conformancePolicy())
	_, _, err = conformance.Push(anthropicGatewayStream())
	assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "invalid_upstream_json")
}

// The monotonic usage merge still guards the counters the Router consumes.
func TestAnthropicGatewayStreamKeepsTheMonotonicUsageMerge(t *testing.T) {
	stream := strings.Replace(string(anthropicGatewayStream()), `"usage":{"input_tokens":12,"output_tokens":1,`, `"usage":{"input_tokens":12,"output_tokens":9,`, 1)
	stream = strings.Replace(stream, `"usage":{"output_tokens":4,"total_cost"`, `"usage":{"output_tokens":3,"total_cost"`, 1)
	decoder := AnthropicMessagesCodec{}.NewDecoder(llmprotocol.StreamContext{Context: context.Background()}, llmprotocol.DefaultPolicy())
	_, _, err := decoder.Push([]byte(stream))
	assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "usage_decreased")
}

func TestAnthropicTransportErrorDropsGatewayFields(t *testing.T) {
	body := []byte(`{"type":"error","error":{"type":"overloaded_error","message":"busy"},"gateway_trace_id":"trace-1"}`)
	transportError, diagnostics, err := AnthropicMessagesCodec{}.DecodeTransportError(body, llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatalf("DecodeTransportError() error = %v", err)
	}
	if transportError.Error == nil || transportError.Error.Code != "overloaded_error" {
		t.Fatalf("transport error = %+v", transportError)
	}
	if got := droppedFields(diagnostics); !reflect.DeepEqual(got, []string{"gateway_trace_id"}) {
		t.Fatalf("unknown fields = %v", got)
	}
	_, _, err = AnthropicMessagesCodec{}.DecodeTransportError(body, conformancePolicy())
	assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "invalid_upstream_json")
}

// Members inside content blocks are reported as well; tool input is opaque
// JSON and is never walked.
func TestAnthropicContentBlockUnknownFieldsAreReported(t *testing.T) {
	body := `{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[` +
		`{"type":"text","text":"done","gateway_annotation":{"x":1}},` +
		`{"type":"tool_use","id":"toolu_1","name":"lookup","input":{"surprise":true}}],` +
		`"stop_reason":"tool_use","stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":1}}`
	response, _, diagnostics, err := NewBuiltinEngine().DecodeResponse(llmprotocol.AnthropicMessagesV1, []byte(body))
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v", err)
	}
	if got := droppedFields(diagnostics); !reflect.DeepEqual(got, []string{"content[].gateway_annotation"}) {
		t.Fatalf("unknown fields = %v", got)
	}
	var arguments string
	for _, item := range response.Output {
		for _, content := range item.Content {
			if content.ToolCall != nil {
				arguments = content.ToolCall.Arguments
			}
		}
	}
	if !strings.Contains(arguments, "surprise") {
		t.Fatalf("tool input changed: %q", arguments)
	}
}
