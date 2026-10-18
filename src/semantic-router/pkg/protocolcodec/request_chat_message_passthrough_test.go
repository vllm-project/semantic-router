package protocolcodec

import (
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// mutatedChatPassthrough decodes body with the preserve policy, applies the
// mutation, re-encodes, and returns the encoded messages for assertions.
func mutatedChatPassthrough(
	t *testing.T,
	body []byte,
	mutate func(*llmprotocol.Request),
) []map[string]json.RawMessage {
	t.Helper()
	engine := newChatPassthroughEngine(t)
	request, envelope, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIChatV1, body)
	if err != nil {
		t.Fatal(err)
	}
	if mutate != nil {
		mutate(&request)
	}
	result, err := engine.EncodeRequest(llmprotocol.OpenAIChatV1, request, envelope)
	if err != nil {
		t.Fatal(err)
	}
	var encoded struct {
		Messages []map[string]json.RawMessage `json:"messages"`
	}
	if err := json.Unmarshal(result.Body, &encoded); err != nil {
		t.Fatal(err)
	}
	return encoded.Messages
}

func messageHasField(message map[string]json.RawMessage, key string) bool {
	_, ok := message[key]
	return ok
}

func messageRole(t *testing.T, message map[string]json.RawMessage) string {
	t.Helper()
	var role string
	if raw, ok := message["role"]; ok {
		if err := json.Unmarshal(raw, &role); err != nil {
			t.Fatal(err)
		}
	}
	return role
}

func messageContent(t *testing.T, message map[string]json.RawMessage) string {
	t.Helper()
	var content string
	if raw, ok := message["content"]; ok {
		if err := json.Unmarshal(raw, &content); err != nil {
			t.Fatal(err)
		}
	}
	return content
}

// assertUnknownFieldsStayOnTheirMessage checks that x_trace stayed on the
// message with the given content and appeared on no other message.
func assertUnknownFieldsStayOnTheirMessage(t *testing.T, messages []map[string]json.RawMessage, ownerContent string) {
	t.Helper()
	owners := 0
	for _, message := range messages {
		if messageHasField(message, "x_trace") {
			owners++
			if content := messageContent(t, message); content != ownerContent {
				t.Fatalf("x_trace landed on message with content %q, want %q (messages: %v)", content, ownerContent, messages)
			}
		}
	}
	if owners != 1 {
		t.Fatalf("x_trace found on %d messages, want exactly 1 on the message with content %q", owners, ownerContent)
	}
}

// A system prompt injection must not drag the user message's captured fields
// onto the injected system message.
func TestChatMessagePassthroughSurvivesSystemInjection(t *testing.T) {
	body := []byte(`{"model":"auto","messages":[{"role":"user","content":"hi","x_trace":"u1"}]}`)
	messages := mutatedChatPassthrough(t, body, func(request *llmprotocol.Request) {
		if !llmprotocol.SetSystemInstruction(request, "be helpful", "replace") {
			t.Fatal("system instruction not applied")
		}
		request.Model, request.Generation = "public/target", request.Generation+1
	})
	if len(messages) != 2 || messageRole(t, messages[0]) != "system" {
		t.Fatalf("expected injected system message first, got %v", messages)
	}
	assertUnknownFieldsStayOnTheirMessage(t, messages, "hi")
}

// A memory-style prepended user message must not inherit the original user
// message's captured fields.
func TestChatMessagePassthroughSurvivesMemoryPrepend(t *testing.T) {
	body := []byte(`{"model":"auto","messages":[{"role":"user","content":"hi","x_trace":"u1"}]}`)
	messages := mutatedChatPassthrough(t, body, func(request *llmprotocol.Request) {
		memoryMessage := llmprotocol.Message{
			Role:    llmprotocol.RoleUser,
			Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "remembered context"}},
		}
		request.Messages = append([]llmprotocol.Message{memoryMessage}, request.Messages...)
		request.Generation++
	})
	if len(messages) != 2 || messageContent(t, messages[0]) != "remembered context" {
		t.Fatalf("expected memory message first, got %v", messages)
	}
	assertUnknownFieldsStayOnTheirMessage(t, messages, "hi")
}

// System/developer messages are decoded into Instructions and encoded before
// the conversation messages, so a client system message that does not lead
// the array reorders the encoded list. Captured fields must follow their
// message, not the index.
func TestChatMessagePassthroughSurvivesInstructionReorder(t *testing.T) {
	body := []byte(`{"model":"auto","messages":[` +
		`{"role":"user","content":"u1","x_trace":"u1"},` +
		`{"role":"system","content":"sys","x_sys":"s1"},` +
		`{"role":"user","content":"u2"}]}`)
	messages := mutatedChatPassthrough(t, body, func(request *llmprotocol.Request) {
		request.Model, request.Generation = "public/target", request.Generation+1
	})
	if len(messages) != 3 || messageRole(t, messages[0]) != "system" {
		t.Fatalf("expected system message reordered first, got %v", messages)
	}
	assertUnknownFieldsStayOnTheirMessage(t, messages, "u1")
	if !messageHasField(messages[0], "x_sys") {
		t.Fatalf("system message lost its captured x_sys field: %v", messages[0])
	}
}

// A RAG-style tool message inserted after the last user message shifts the
// indices of nothing before it but exercises insertion with two captured
// messages on both sides of the insertion point.
func TestChatMessagePassthroughSurvivesToolInsertion(t *testing.T) {
	body := []byte(`{"model":"auto","messages":[` +
		`{"role":"user","content":"u1","x_trace":"u1"},` +
		`{"role":"user","content":"u2","x_trace":"u2"}]}`)
	messages := mutatedChatPassthrough(t, body, func(request *llmprotocol.Request) {
		callMessage := llmprotocol.Message{
			Role: llmprotocol.RoleAssistant,
			Content: []llmprotocol.Content{{Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{
				ID: "call_1", Name: "vsr_rag_context", Arguments: "{}",
			}}},
		}
		toolMessage := llmprotocol.Message{
			Role: llmprotocol.RoleTool,
			Content: []llmprotocol.Content{{Kind: llmprotocol.ContentToolResult, ToolResult: &llmprotocol.ToolResult{
				CallID:  "call_1",
				Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "result"}},
			}}},
		}
		lastUser := len(request.Messages) - 1
		inserted := append([]llmprotocol.Message{callMessage, toolMessage}, request.Messages[lastUser+1:]...)
		request.Messages = append(request.Messages[:lastUser+1], inserted...)
		request.Generation++
	})
	if len(messages) != 4 || messageRole(t, messages[2]) != "assistant" || messageRole(t, messages[3]) != "tool" {
		t.Fatalf("expected assistant call and tool result inserted after last user message, got %v", messages)
	}
	assertMessageFieldEquals(t, messages[0], "x_trace", `"u1"`)
	assertMessageFieldEquals(t, messages[1], "x_trace", `"u2"`)
}

func assertMessageFieldEquals(t *testing.T, message map[string]json.RawMessage, key, want string) {
	t.Helper()
	raw, ok := message[key]
	if !ok {
		t.Fatalf("message lost its captured %s field: %v", key, message)
	}
	if string(raw) != want {
		t.Fatalf("message %s = %s, want %s", key, raw, want)
	}
}

// Top-level preserved fields share the SourceEnvelopeBytes bound with the
// per-message capture: an oversized unknown field must fail loudly at decode
// instead of being forwarded.
func TestChatTopLevelPassthroughBudgetExceeded(t *testing.T) {
	engine := newChatPassthroughEngine(t)
	oversized := make([]byte, engine.policy.Limits.SourceEnvelopeBytes+1)
	for i := range oversized {
		oversized[i] = 'a'
	}
	body := []byte(`{"model":"auto","messages":[{"role":"user","content":"hi"}],"x_blob":"` + string(oversized) + `"}`)
	if _, _, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIChatV1, body); err == nil {
		t.Fatal("oversized top-level passthrough field must exceed the budget")
	}
}
