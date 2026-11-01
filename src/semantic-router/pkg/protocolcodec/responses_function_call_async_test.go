package protocolcodec

import (
	"context"
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// The function_call item as vLLM 0.32 writes it: the official async, caller
// and namespace fields are present and null.
const functionCallWithAsync = `{"arguments":"{\"command\": \"ls /tmp\"}","call_id":"call_1","name":"bash","type":"function_call","id":"fc_1","async":null,"caller":null,"namespace":null,"status":"completed"}`

func TestResponsesFunctionCallAsyncDecodesAsDroppedMetadata(t *testing.T) {
	body := `{"id":"resp_1","object":"response","created_at":1,"model":"m","status":"completed","output":[` + functionCallWithAsync + `]}`
	response, _, diagnostics, err := (OpenAIResponsesCodec{}).DecodeResponse([]byte(body), llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v", err)
	}
	call := response.Output[0].Content[0].ToolCall
	if call == nil || call.Name != "bash" || call.ID != "call_1" {
		t.Fatalf("tool call = %+v", call)
	}
	if !hasDiagnostic(diagnostics, "output.async", llmprotocol.DiagnosticDropped) {
		t.Fatalf("diagnostics = %+v, want output.async dropped", diagnostics)
	}
}

func TestResponsesFunctionCallAsyncDecodesInAStream(t *testing.T) {
	added := `{"arguments":"","call_id":"call_1","name":"bash","type":"function_call","id":"fc_1","async":null,"caller":null,"namespace":null,"status":"in_progress"}`
	stream := []byte("event: response.created\ndata: {\"type\":\"response.created\",\"sequence_number\":0,\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"model\":\"m\",\"status\":\"in_progress\",\"output\":[]}}\n\n" +
		"event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"sequence_number\":1,\"output_index\":0,\"item\":" + added + "}\n\n" +
		"event: response.function_call_arguments.done\ndata: {\"type\":\"response.function_call_arguments.done\",\"sequence_number\":2,\"output_index\":0,\"item_id\":\"fc_1\",\"arguments\":\"{\\\"command\\\": \\\"ls /tmp\\\"}\"}\n\n" +
		"event: response.output_item.done\ndata: {\"type\":\"response.output_item.done\",\"sequence_number\":3,\"output_index\":0,\"item\":" + functionCallWithAsync + "}\n\n" +
		"event: response.completed\ndata: {\"type\":\"response.completed\",\"sequence_number\":4,\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"model\":\"m\",\"status\":\"completed\",\"output\":[" + functionCallWithAsync + "]}}\n\n")
	engine, err := NewEngine(NewBuiltinRegistry(), llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := engine.DecodeResponseStream(llmprotocol.OpenAIResponsesV1, stream,
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"}); err != nil {
		t.Fatalf("DecodeResponseStream() error = %v", err)
	}
}

// A client that sends async on an input item gets a typed refusal, as for
// caller, instead of a generic non-canonical field error.
func TestResponsesInputFunctionCallAsyncIsRefusedByName(t *testing.T) {
	request := `{"model":"m","input":[{"type":"function_call","call_id":"call_1","name":"bash","arguments":"{}","async":true},` +
		`{"type":"function_call_output","call_id":"call_1","output":"ok"}]}`
	_, _, _, err := (OpenAIResponsesCodec{}).DecodeRequest([]byte(request), llmprotocol.DefaultPolicy())
	var protocolError *llmprotocol.ProtocolError
	if !errors.As(err, &protocolError) || protocolError.Code != "unsupported_input_async" {
		t.Fatalf("DecodeRequest() error = %v, want unsupported_input_async", err)
	}
}
