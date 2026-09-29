package protocolcodec

import (
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestResponsesDispatchDoesNotInventInputItemIDs(t *testing.T) {
	engine := NewBuiltinEngine()
	for _, test := range []struct {
		name   string
		format llmprotocol.WireFormat
		body   string
		ids    []string
	}{
		{
			name: "Chat system and user", format: llmprotocol.OpenAIChatV1,
			body: `{"model":"m","messages":[{"role":"system","content":"instructions"},{"role":"user","content":"hello"}]}`,
			ids:  []string{"", ""},
		},
		{
			name: "Responses source ID", format: llmprotocol.OpenAIResponsesV1,
			body: `{"model":"m","input":[{"type":"message","id":"msg_user_1","role":"user","content":[{"type":"input_text","text":"first"}]},{"type":"message","role":"user","content":[{"type":"input_text","text":"second"}]}]}`,
			ids:  []string{"msg_user_1", ""},
		},
		{
			name: "Anthropic tool result has no source item ID", format: llmprotocol.AnthropicMessagesV1,
			body: `{"model":"m","max_tokens":64,"messages":[{"role":"user","content":"run lookup"},{"role":"assistant","content":[{"type":"tool_use","id":"call_1","name":"lookup","input":{}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"call_1","content":"ok"}]}]}`,
			ids:  []string{"", "", ""},
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			request, envelope, _, err := engine.DecodeRequestForMutation(test.format, []byte(test.body))
			if err != nil {
				t.Fatal(err)
			}
			request.Model = "routed-model"
			request.Generation++
			encoded, err := engine.EncodeRequest(llmprotocol.OpenAIResponsesV1, request, envelope)
			if err != nil {
				t.Fatal(err)
			}
			var wire struct {
				Input []map[string]json.RawMessage `json:"input"`
			}
			if err := json.Unmarshal(encoded.Body, &wire); err != nil {
				t.Fatal(err)
			}
			if len(wire.Input) != len(test.ids) {
				t.Fatalf("input item count = %d, want %d: %s", len(wire.Input), len(test.ids), encoded.Body)
			}
			for index, want := range test.ids {
				id, exists := wire.Input[index]["id"]
				if want == "" && exists {
					t.Fatalf("input[%d] invented an ID: %s", index, encoded.Body)
				}
				if want != "" && string(id) != `"`+want+`"` {
					t.Fatalf("input[%d] ID = %s, want %q: %s", index, id, want, encoded.Body)
				}
			}
		})
	}
}

func TestResponsesCompletedToolHistoryIsAcceptedAndStatusIsNotForwarded(t *testing.T) {
	const body = `{"model":"m","input":[
		{"type":"function_call","id":"fc_1","name":"bash","arguments":"{}","call_id":"call_1","status":"completed"},
		{"type":"function_call_output","call_id":"call_1","output":"ok","status":"completed"},
		{"type":"message","role":"user","content":[{"type":"input_text","text":"continue"}],"status":"completed"}
	]}`
	engine := NewBuiltinEngine()
	request, envelope, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIResponsesV1, []byte(body))
	if err != nil {
		t.Fatalf("completed tool history rejected: %v", err)
	}
	request.Model = "routed-model"
	request.Generation++
	encoded, err := engine.EncodeRequest(llmprotocol.OpenAIResponsesV1, request, envelope)
	if err != nil {
		t.Fatalf("completed tool history could not be dispatched: %v", err)
	}
	var wire struct {
		Input []map[string]json.RawMessage `json:"input"`
	}
	if err := json.Unmarshal(encoded.Body, &wire); err != nil {
		t.Fatal(err)
	}
	if len(wire.Input) != 3 {
		t.Fatalf("tool history lost an item: %s", encoded.Body)
	}
	for index, item := range wire.Input {
		if _, hasStatus := item["status"]; hasStatus {
			t.Fatalf("input[%d] forwarded an echoed output status: %s", index, encoded.Body)
		}
	}
	if string(wire.Input[0]["call_id"]) != `"call_1"` || string(wire.Input[1]["call_id"]) != `"call_1"` {
		t.Fatalf("tool call link changed: %s", encoded.Body)
	}
}
