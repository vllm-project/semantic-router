package protocolcodec

import (
	"encoding/json"
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// The second request of an Inferoa 0.14.19 tool loop: every tool result repeats
// the tool's name. Prompts are shortened.
const inferoaToolLoopBody = `{
	"model": "vllm-sr/auto",
	"messages": [
		{"role": "system", "content": "You are Inferoa, a loop-engineering coding agent."},
		{"role": "user", "content": "what do you know about vllm-sr"},
		{"role": "assistant", "content": "", "tool_calls": [
			{"id": "call_1", "type": "function", "function": {"name": "file_search", "arguments": "{\"query\":\"vllm-sr\"}"}},
			{"id": "call_2", "type": "function", "function": {"name": "run_command", "arguments": "{\"command\":\"ls\"}"}}
		]},
		{"role": "tool", "tool_call_id": "call_1", "name": "file_search", "content": "Found 50 matches"},
		{"role": "tool", "tool_call_id": "call_2", "name": "run_command", "content": "Command exited 0"},
		{"role": "user", "content": "Continue from the tool evidence."}
	],
	"tools": [
		{"type": "function", "function": {"name": "file_search", "parameters": {"type": "object"}}},
		{"type": "function", "function": {"name": "run_command", "parameters": {"type": "object"}}}
	]
}`

const namedParticipantsBody = `{
	"model": "client-model",
	"messages": [
		{"role": "system", "name": "planner", "content": "Plan before you answer."},
		{"role": "user", "name": "alice", "content": "Summarize the review."},
		{"role": "assistant", "name": "reviewer", "content": "Two issues remain."},
		{"role": "user", "name": "bob", "content": "Which is more urgent?"}
	]
}`

type namedChatBody struct {
	Messages []struct {
		Role       string  `json:"role"`
		Name       *string `json:"name"`
		ToolCallID string  `json:"tool_call_id"`
	} `json:"messages"`
}

func routedChatRequest(t *testing.T, body string, target llmprotocol.WireFormat) RequestResult {
	t.Helper()
	engine := NewBuiltinEngine()
	request, envelope, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIChatV1, []byte(body))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	request.Model = "routed-model"
	request.Generation++
	encoded, err := engine.EncodeRequest(target, request, envelope)
	if err != nil {
		t.Fatalf("encode to %s: %v", target, err)
	}
	return encoded
}

func TestInferoaToolLoopReachesChatBackends(t *testing.T) {
	encoded := routedChatRequest(t, inferoaToolLoopBody, llmprotocol.OpenAIChatV1)
	var dispatched namedChatBody
	if err := json.Unmarshal(encoded.Body, &dispatched); err != nil {
		t.Fatal(err)
	}
	var results []string
	for _, message := range dispatched.Messages {
		if message.Role != "tool" {
			continue
		}
		if message.Name != nil {
			t.Fatalf("tool result %s kept its name %q", message.ToolCallID, *message.Name)
		}
		results = append(results, message.ToolCallID)
	}
	if len(results) != 2 || results[0] != "call_1" || results[1] != "call_2" {
		t.Fatalf("tool results = %v, want call_1 and call_2", results)
	}
	if len(encoded.Diagnostics) != 0 {
		t.Fatalf("diagnostics = %+v, want none", encoded.Diagnostics)
	}
}

func TestChatParticipantNamesReachChatBackends(t *testing.T) {
	encoded := routedChatRequest(t, namedParticipantsBody, llmprotocol.OpenAIChatV1)
	var dispatched namedChatBody
	if err := json.Unmarshal(encoded.Body, &dispatched); err != nil {
		t.Fatal(err)
	}
	want := []string{"planner", "alice", "reviewer", "bob"}
	if len(dispatched.Messages) != len(want) {
		t.Fatalf("dispatched %d messages, want %d", len(dispatched.Messages), len(want))
	}
	for i, message := range dispatched.Messages {
		if message.Name == nil || *message.Name != want[i] {
			t.Fatalf("message %d (%s) name = %v, want %q", i, message.Role, message.Name, want[i])
		}
	}
}

func TestChatParticipantNamesAreDroppedWithDiagnosticForOtherFormats(t *testing.T) {
	for _, target := range []llmprotocol.WireFormat{llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIResponsesV1} {
		t.Run(string(target), func(t *testing.T) {
			encoded := routedChatRequest(t, namedParticipantsBody, target)
			dropped := false
			for _, diagnostic := range encoded.Diagnostics {
				if diagnostic.Field == "messages[].name" && diagnostic.Action == llmprotocol.DiagnosticDropped {
					dropped = true
				}
			}
			if !dropped {
				t.Fatalf("diagnostics = %+v, want messages[].name dropped", encoded.Diagnostics)
			}
		})
	}
}

func TestChatMessageNameMustBeAString(t *testing.T) {
	body := `{"model":"m","messages":[{"role":"user","name":7,"content":"hello"}]}`
	_, _, _, err := NewBuiltinEngine().DecodeRequest(llmprotocol.OpenAIChatV1, []byte(body))
	var protocolError *llmprotocol.ProtocolError
	if !errors.As(err, &protocolError) || protocolError.Category != llmprotocol.ErrorInvalidRequest ||
		protocolError.Code != "invalid_messages_name" {
		t.Fatalf("got %v, want invalid_messages_name", err)
	}
}
