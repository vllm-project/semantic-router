package contextcompression

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// The semantic path must use the shared turn rule, so context transformations
// and topic-continuity evidence agree on which messages open a turn.
func TestMessageStartsTurnUsesTheSharedRule(t *testing.T) {
	text := func(value string) llmprotocol.Content {
		return llmprotocol.Content{Kind: llmprotocol.ContentText, Text: value}
	}
	toolResult := llmprotocol.Content{
		Kind:       llmprotocol.ContentToolResult,
		ToolResult: &llmprotocol.ToolResult{CallID: "c1", Content: []llmprotocol.Content{text("ok")}},
	}
	toolCall := llmprotocol.Content{
		Kind:     llmprotocol.ContentToolCall,
		ToolCall: &llmprotocol.ToolCall{ID: "c1", Name: "lookup"},
	}
	image := llmprotocol.Content{Kind: llmprotocol.ContentImage, URL: "https://example.invalid/a.png"}
	request := &llmprotocol.Request{Messages: []llmprotocol.Message{
		{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{text("leading assistant")}},
		{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{text("question")}},
		{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{toolCall}},
		{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{toolResult}},
		{Role: llmprotocol.RoleTool, Content: []llmprotocol.Content{toolResult}},
		{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{toolResult, text("and a follow-up")}},
		{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{image}},
		{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{text("")}},
		{Role: llmprotocol.RoleUser},
	}}
	ir := ParseSemanticRequest(request, Provenance{})
	for _, message := range ir.Messages {
		want := llmprotocol.StartsConversationTurn(request.Messages[message.Index])
		if got := ir.messageStartsTurn(message); got != want {
			t.Fatalf("message %d (%s): messageStartsTurn = %v, shared rule = %v",
				message.Index, message.Role, got, want)
		}
	}
}

// The legacy raw-map path has no neutral messages; it must still treat a user
// message carrying only tool_result blocks as a continuation.
func TestLegacyMessageStartsTurn(t *testing.T) {
	raw := map[string]interface{}{"messages": []interface{}{
		map[string]interface{}{"role": "user", "content": "question"},
		map[string]interface{}{"role": "user", "content": []interface{}{
			map[string]interface{}{"type": "tool_result", "tool_use_id": "c1", "content": "ok"},
		}},
		map[string]interface{}{"role": "user", "content": []interface{}{
			map[string]interface{}{"type": "tool_result", "tool_use_id": "c1", "content": "ok"},
			map[string]interface{}{"type": "text", "text": "and a follow-up"},
		}},
	}}
	ir := ParseRequestIR(raw, Provenance{})
	want := []bool{true, false, true}
	if len(ir.Messages) != len(want) {
		t.Fatalf("parsed %d messages, want %d", len(ir.Messages), len(want))
	}
	for i, message := range ir.Messages {
		if got := ir.messageStartsTurn(message); got != want[i] {
			t.Fatalf("legacy message %d: messageStartsTurn = %v, want %v", i, got, want[i])
		}
	}
}
