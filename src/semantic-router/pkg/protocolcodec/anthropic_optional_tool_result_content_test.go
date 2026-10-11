package protocolcodec

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestAnthropicToolResultAcceptsOmittedContent(t *testing.T) {
	raw := []byte(`{
		"model": "claude-3-5-sonnet-20241022",
		"max_tokens": 64,
		"messages": [
			{"role": "user", "content": "Run action"},
			{
				"role": "assistant",
				"content": [
					{"type": "tool_use", "id": "call_action_1", "name": "run_action", "input": {}}
				]
			},
			{
				"role": "user",
				"content": [
					{"type": "tool_result", "tool_use_id": "call_action_1"}
				]
			}
		]
	}`)

	engine, err := NewEngine(NewBuiltinRegistry(), llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatal(err)
	}

	targets := []llmprotocol.WireFormat{
		llmprotocol.AnthropicMessagesV1,
		llmprotocol.OpenAIChatV1,
		llmprotocol.OpenAIResponsesV1,
	}

	for _, target := range targets {
		t.Run("TranslateTo/"+string(target), func(t *testing.T) {
			translated, err := engine.TranslateRequest(llmprotocol.AnthropicMessagesV1, target, raw, nil)
			if err != nil {
				t.Fatalf("unexpected translation failure for target %s: %v", target, err)
			}
			if len(translated.Body) == 0 {
				t.Fatal("expected non-empty translated body")
			}

			// Verify neutral tool result representation
			userMsg := translated.Request.Messages[len(translated.Request.Messages)-1]
			if len(userMsg.Content) != 1 {
				t.Fatalf("expected 1 content block, got %d", len(userMsg.Content))
			}
			content := userMsg.Content[0]
			if content.Kind != llmprotocol.ContentToolResult {
				t.Fatalf("expected ContentToolResult, got %s", content.Kind)
			}
			if content.ToolResult == nil {
				t.Fatal("expected non-nil ToolResult")
			}
			if content.ToolResult.CallID != "call_action_1" {
				t.Fatalf("expected CallID 'call_action_1', got %q", content.ToolResult.CallID)
			}
		})
	}
}

func TestAnthropicToolResultStillRequiresToolUseID(t *testing.T) {
	raw := []byte(`{
		"model": "claude-3-5-sonnet-20241022",
		"max_tokens": 64,
		"messages": [
			{
				"role": "user",
				"content": [
					{"type": "tool_result"}
				]
			}
		]
	}`)

	engine, err := NewEngine(NewBuiltinRegistry(), llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatal(err)
	}

	_, err = engine.TranslateRequest(llmprotocol.AnthropicMessagesV1, llmprotocol.AnthropicMessagesV1, raw, nil)
	assertProtocolError(t, err, llmprotocol.ErrorInvalidRequest, "invalid_content_variant")
}
