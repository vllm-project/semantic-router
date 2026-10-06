package protocolcodec

import (
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestResponsesParallelToolHistoryBecomesOneChatAssistantTurn(t *testing.T) {
	messages := projectResponsesHistoryToChat(t, `[
		{"role":"user","content":"use both results"},
		{"type":"function_call","id":"fc_a","call_id":"call_a","name":"weather","arguments":"{\"city\":\"Paris\"}"},
		{"type":"function_call","id":"fc_b","call_id":"call_b","name":"weather","arguments":"{\"city\":\"London\"}"},
		{"type":"function_call_output","call_id":"call_a","output":"sunny"},
		{"type":"function_call_output","call_id":"call_b","output":"rainy"}
	]`)

	if len(messages) != 4 {
		t.Fatalf("Chat message count = %d, want 4: %+v", len(messages), messages)
	}
	assistant := messages[1]
	if assistant.Role != "assistant" || len(assistant.ToolCalls) != 2 {
		t.Fatalf("parallel calls were not grouped into one assistant turn: %+v", assistant)
	}
	if assistant.ToolCalls[0].ID != "call_a" || assistant.ToolCalls[1].ID != "call_b" {
		t.Fatalf("parallel call order or IDs changed: %+v", assistant.ToolCalls)
	}
	if messages[2].Role != "tool" || messages[2].ToolCallID != "call_a" ||
		messages[3].Role != "tool" || messages[3].ToolCallID != "call_b" {
		t.Fatalf("tool results did not follow the grouped assistant turn: %+v", messages)
	}
}

func TestResponsesToolHistoryGroupingControls(t *testing.T) {
	t.Run("single call", func(t *testing.T) {
		messages := projectResponsesHistoryToChat(t, `[
			{"role":"user","content":"use the result"},
			{"type":"function_call","call_id":"call_a","name":"weather","arguments":"{}"},
			{"type":"function_call_output","call_id":"call_a","output":"sunny"}
		]`)
		if len(messages) != 3 || len(messages[1].ToolCalls) != 1 || messages[1].ToolCalls[0].ID != "call_a" {
			t.Fatalf("single-call history changed: %+v", messages)
		}
	})

	t.Run("sequential calls", func(t *testing.T) {
		messages := projectResponsesHistoryToChat(t, `[
			{"role":"user","content":"run two steps"},
			{"type":"function_call","call_id":"call_a","name":"first","arguments":"{}"},
			{"type":"function_call_output","call_id":"call_a","output":"done"},
			{"type":"function_call","call_id":"call_b","name":"second","arguments":"{}"},
			{"type":"function_call_output","call_id":"call_b","output":"done"}
		]`)
		if len(messages) != 5 {
			t.Fatalf("sequential Chat message count = %d, want 5: %+v", len(messages), messages)
		}
		if len(messages[1].ToolCalls) != 1 || messages[1].ToolCalls[0].ID != "call_a" ||
			messages[2].Role != "tool" || messages[2].ToolCallID != "call_a" ||
			len(messages[3].ToolCalls) != 1 || messages[3].ToolCalls[0].ID != "call_b" ||
			messages[4].Role != "tool" || messages[4].ToolCallID != "call_b" {
			t.Fatalf("calls were merged across an intervening tool result: %+v", messages)
		}
	})
}

func projectResponsesHistoryToChat(t *testing.T, input string) []chatMessageWire {
	t.Helper()
	engine := NewBuiltinEngine()
	body := []byte(`{"model":"source-model","input":` + input + `}`)
	request, envelope, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIResponsesV1, body)
	if err != nil {
		t.Fatal(err)
	}
	request.Model = "routed-model"
	request.Generation++
	encoded, err := engine.EncodeRequest(llmprotocol.OpenAIChatV1, request, envelope)
	if err != nil {
		t.Fatal(err)
	}
	var wire chatRequestWire
	if err := json.Unmarshal(encoded.Body, &wire); err != nil {
		t.Fatal(err)
	}
	return wire.Messages
}
