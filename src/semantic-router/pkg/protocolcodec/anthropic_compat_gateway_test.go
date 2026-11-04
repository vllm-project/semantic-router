package protocolcodec

import (
	"context"
	"reflect"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// An enterprise gateway that serves the Anthropic Messages format adds its own
// usage fields, such as a cost in exponent form.
const anthropicGatewayResponseFixture = `{"id":"msg_1","type":"message","role":"assistant","model":"provider-model","content":[{"type":"text","text":"done"}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":12,"output_tokens":4,"total_cost":3e-05}}`

func anthropicGatewayEngine(t *testing.T, vendor llmprotocol.ResponseVendor) *Engine {
	t.Helper()
	policy := llmprotocol.DefaultPolicy()
	policy.ResponseVendor = vendor
	engine, err := NewEngine(NewBuiltinRegistry(), policy)
	if err != nil {
		t.Fatal(err)
	}
	return engine
}

func droppedFields(diagnostics llmprotocol.Diagnostics) []string {
	var fields []string
	for _, diagnostic := range diagnostics {
		if diagnostic.Action == llmprotocol.DiagnosticDropped {
			fields = append(fields, diagnostic.Field)
		}
	}
	return fields
}

func TestAnthropicCompatGatewayDropsGatewayUsageFields(t *testing.T) {
	engine := anthropicGatewayEngine(t, llmprotocol.ResponseVendorAnthropicCompatGateway)
	response, envelope, diagnostics, err := engine.DecodeResponse(llmprotocol.AnthropicMessagesV1, []byte(anthropicGatewayResponseFixture))
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v", err)
	}
	if got := droppedFields(diagnostics); !reflect.DeepEqual(got, []string{"usage.total_cost"}) {
		t.Fatalf("dropped fields = %v, want only usage.total_cost and no renamed counters", got)
	}
	if !DiagnosticsDroppedVendorExtensions(diagnostics) {
		t.Fatal("the drop is not marked as a vendor extension, so a cache would persist the gateway's bytes")
	}
	if response.Usage.InputTotal.Value == nil || *response.Usage.InputTotal.Value != 12 ||
		response.Usage.OutputTotal.Value == nil || *response.Usage.OutputTotal.Value != 4 {
		t.Fatalf("usage changed: %+v", response.Usage)
	}
	if strings.Contains(string(envelope.Response), "total_cost") {
		t.Fatalf("the envelope kept the gateway field for a same-format replay: %s", envelope.Response)
	}

	chat, err := engine.TranslateResponse(llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1, []byte(anthropicGatewayResponseFixture), nil)
	if err != nil {
		t.Fatalf("TranslateResponse() error = %v", err)
	}
	if !strings.Contains(string(chat.Body), `"prompt_tokens":12,"completion_tokens":4`) || strings.Contains(string(chat.Body), "total_cost") {
		t.Fatalf("Chat reply = %s", chat.Body)
	}
}

func TestAnthropicGatewayUsageFieldsStayStrictWithoutOptIn(t *testing.T) {
	for _, vendor := range []llmprotocol.ResponseVendor{"", llmprotocol.ResponseVendorSnowflake} {
		_, _, _, err := anthropicGatewayEngine(t, vendor).DecodeResponse(llmprotocol.AnthropicMessagesV1, []byte(anthropicGatewayResponseFixture))
		assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "invalid_upstream_json")
	}
}

func anthropicGatewayStream() []byte {
	return []byte("event: message_start\n" +
		`data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"provider-model","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":12,"output_tokens":1,"total_cost":1e-06}}}` + "\n\n" +
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

func TestAnthropicCompatGatewayStreamDropsGatewayUsageFields(t *testing.T) {
	policy := llmprotocol.DefaultPolicy()
	policy.ResponseVendor = llmprotocol.ResponseVendorAnthropicCompatGateway
	decoder := AnthropicMessagesCodec{}.NewDecoder(llmprotocol.StreamContext{Context: context.Background()}, policy)
	events, diagnostics, err := decoder.Push(anthropicGatewayStream())
	if err != nil {
		t.Fatalf("gateway stream was rejected: %v", err)
	}
	final, _, err := decoder.Finalize(nil)
	if err != nil {
		t.Fatal(err)
	}
	if got := droppedFields(diagnostics); !reflect.DeepEqual(got, []string{"message.usage.total_cost", "usage.total_cost"}) {
		t.Fatalf("dropped fields = %v", got)
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

	strict := AnthropicMessagesCodec{}.NewDecoder(llmprotocol.StreamContext{Context: context.Background()}, llmprotocol.DefaultPolicy())
	_, _, err = strict.Push(anthropicGatewayStream())
	assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "invalid_upstream_json")
}

func TestAnthropicCompatGatewayTransportErrorDropsGatewayFields(t *testing.T) {
	body := []byte(`{"type":"error","error":{"type":"overloaded_error","message":"busy"},"gateway_trace_id":"trace-1"}`)
	transportError, diagnostics, err := AnthropicMessagesCodec{}.DecodeTransportError(body, func() llmprotocol.Policy {
		policy := llmprotocol.DefaultPolicy()
		policy.ResponseVendor = llmprotocol.ResponseVendorAnthropicCompatGateway
		return policy
	}())
	if err != nil {
		t.Fatalf("DecodeTransportError() error = %v", err)
	}
	if transportError.Error == nil || transportError.Error.Code != "overloaded_error" {
		t.Fatalf("transport error = %+v", transportError)
	}
	if got := droppedFields(diagnostics); !reflect.DeepEqual(got, []string{"gateway_trace_id"}) {
		t.Fatalf("dropped fields = %v", got)
	}
	_, _, err = AnthropicMessagesCodec{}.DecodeTransportError(body, llmprotocol.DefaultPolicy())
	assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "invalid_upstream_json")
}
