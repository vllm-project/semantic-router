package protocolcodec

import (
	"context"
	"errors"
	"reflect"
	"sort"
	"strings"
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

// custom_tool_call carries the same official field.
func TestResponsesCustomToolCallAsyncDecodesAsDroppedMetadata(t *testing.T) {
	body := `{"id":"resp_1","object":"response","created_at":1,"model":"m","status":"completed","output":[` +
		`{"type":"custom_tool_call","id":"ctc_1","call_id":"call_1","name":"apply_patch","input":"diff","async":null,"status":"completed"}]}`
	response, _, diagnostics, err := (OpenAIResponsesCodec{}).DecodeResponse([]byte(body), llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v", err)
	}
	if call := response.Output[0].Content[0].ToolCall; call == nil || call.Name != "apply_patch" {
		t.Fatalf("tool call = %+v", call)
	}
	if !hasDiagnostic(diagnostics, "output.async", llmprotocol.DiagnosticDropped) {
		t.Fatalf("diagnostics = %+v, want output.async dropped", diagnostics)
	}
}

// The added item of a streamed call, as vLLM 0.32 writes it.
const functionCallWithAsyncAdded = `{"arguments":"","call_id":"call_1","name":"bash","type":"function_call","id":"fc_1","async":null,"caller":null,"namespace":null,"status":"in_progress"}`

const functionCallArgumentsDone = "event: response.function_call_arguments.done\ndata: {\"type\":\"response.function_call_arguments.done\",\"sequence_number\":2,\"output_index\":0,\"item_id\":\"fc_1\",\"arguments\":\"{\\\"command\\\": \\\"ls /tmp\\\"}\"}\n\n"

// responsesToolCallStream streams one tool call: its added item, its input done
// event, and its completed item, which the terminal response repeats.
func responsesToolCallStream(added, inputDone, done string) []byte {
	return []byte("event: response.created\ndata: {\"type\":\"response.created\",\"sequence_number\":0,\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"model\":\"m\",\"status\":\"in_progress\",\"output\":[]}}\n\n" +
		"event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"sequence_number\":1,\"output_index\":0,\"item\":" + added + "}\n\n" +
		inputDone +
		"event: response.output_item.done\ndata: {\"type\":\"response.output_item.done\",\"sequence_number\":3,\"output_index\":0,\"item\":" + done + "}\n\n" +
		"event: response.completed\ndata: {\"type\":\"response.completed\",\"sequence_number\":4,\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"model\":\"m\",\"status\":\"completed\",\"output\":[" + done + "]}}\n\n")
}

func TestResponsesFunctionCallAsyncDecodesInAStream(t *testing.T) {
	stream := responsesToolCallStream(functionCallWithAsyncAdded, functionCallArgumentsDone, functionCallWithAsync)
	engine, err := NewEngine(NewBuiltinRegistry(), llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatal(err)
	}
	_, diagnostics, err := engine.DecodeResponseStream(llmprotocol.OpenAIResponsesV1, stream,
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"})
	if err != nil {
		t.Fatalf("DecodeResponseStream() error = %v", err)
	}
	if !hasDiagnostic(diagnostics, "output.async", llmprotocol.DiagnosticDropped) {
		t.Fatalf("diagnostics = %+v, want output.async dropped", diagnostics)
	}
}

func droppedOutputFields(diagnostics llmprotocol.Diagnostics) []string {
	var fields []string
	for _, diagnostic := range diagnostics {
		if diagnostic.Action == llmprotocol.DiagnosticDropped && strings.HasPrefix(diagnostic.Field, "output.") {
			fields = append(fields, diagnostic.Field)
		}
	}
	sort.Strings(fields)
	return fields
}

// A streamed call loses the same item metadata as a buffered one, so it
// reports the same dropped fields, once each, whatever the client format.
func TestResponsesStreamedToolCallReportsDroppedMetadataLikeBuffered(t *testing.T) {
	allTargets := []llmprotocol.WireFormat{llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1}
	cases := []struct {
		name, added, inputDone, done string
		targets                      []llmprotocol.WireFormat
	}{
		{"function_call", functionCallWithAsyncAdded, functionCallArgumentsDone, functionCallWithAsync, allTargets},
		{
			"custom_tool_call",
			`{"type":"custom_tool_call","id":"ctc_1","call_id":"call_1","name":"apply_patch","input":"","async":null,"status":"in_progress"}`,
			"event: response.custom_tool_call_input.done\ndata: {\"type\":\"response.custom_tool_call_input.done\",\"sequence_number\":2,\"output_index\":0,\"item_id\":\"ctc_1\",\"input\":\"diff\"}\n\n",
			`{"type":"custom_tool_call","id":"ctc_1","call_id":"call_1","name":"apply_patch","input":"diff","async":null,"status":"completed"}`,
			// Anthropic Messages has no custom tools.
			[]llmprotocol.WireFormat{llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIChatV1},
		},
	}
	engine, err := NewEngine(NewBuiltinRegistry(), llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatal(err)
	}
	for _, item := range cases {
		buffered := `{"id":"resp_1","object":"response","created_at":1,"model":"m","status":"completed","output":[` + item.done + `]}`
		_, _, bufferedDiagnostics, err := (OpenAIResponsesCodec{}).DecodeResponse([]byte(buffered), llmprotocol.DefaultPolicy())
		if err != nil {
			t.Fatalf("%s: DecodeResponse() error = %v", item.name, err)
		}
		if !hasDiagnostic(bufferedDiagnostics, "output.async", llmprotocol.DiagnosticDropped) {
			t.Fatalf("%s: buffered diagnostics = %+v, want output.async dropped", item.name, bufferedDiagnostics)
		}
		want := droppedOutputFields(bufferedDiagnostics)
		for _, target := range item.targets {
			t.Run(item.name+"_to_"+string(target), func(t *testing.T) {
				stream, err := engine.NewStream(llmprotocol.OpenAIResponsesV1, target,
					llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"})
				if err != nil {
					t.Fatal(err)
				}
				_, _, diagnostics, err := stream.Push(responsesToolCallStream(item.added, item.inputDone, item.done))
				if err != nil {
					t.Fatalf("Push() error = %v", err)
				}
				_, _, finalDiagnostics, err := stream.Finalize(nil)
				if err != nil {
					t.Fatalf("Finalize() error = %v", err)
				}
				if got := droppedOutputFields(append(diagnostics, finalDiagnostics...)); !reflect.DeepEqual(got, want) {
					t.Fatalf("streamed dropped fields = %v, want %v", got, want)
				}
			})
		}
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
