package protocolcodec

import (
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestDiscoveryAnthropicOptionalToolResultContent(t *testing.T) {
	for _, target := range []llmprotocol.WireFormat{llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1} {
		for _, tc := range []struct {
			name   string
			suffix string
		}{
			{"text_control", `,"content":"done"`},
			{"empty_string_control", `,"content":""`},
			{"empty_array_control", `,"content":[]`},
			{"omitted_content", ``},
		} {
			t.Run(string(target)+"/"+tc.name, func(t *testing.T) {
				body := []byte(`{"model":"m","max_tokens":16,"messages":[{"role":"user","content":"Perform the action."},{"role":"assistant","content":[{"type":"tool_use","id":"call_a","name":"action","input":{}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"call_a"` + tc.suffix + `}]}]}`)
				result, err := NewBuiltinEngine().TranslateRequest(llmprotocol.AnthropicMessagesV1, target, body, nil)
				if err != nil {
					t.Fatalf("valid tool result rejected: %v", err)
				}
				t.Logf("translated=%s", result.Body)
				results := 0
				for _, message := range result.Request.Messages {
					for _, content := range message.Content {
						if content.Kind == llmprotocol.ContentToolResult && content.ToolResult != nil {
							results++
							if content.ToolResult.CallID != "call_a" {
								t.Fatal("lost tool call linkage")
							}
						}
					}
				}
				if results != 1 || !json.Valid(result.Body) {
					t.Fatal("tool result not retained")
				}
			})
		}
	}
	t.Run("missing_id_rejected_control", func(t *testing.T) {
		_, _, _, err := NewBuiltinEngine().DecodeRequestForMutation(llmprotocol.AnthropicMessagesV1,
			[]byte(`{"model":"m","max_tokens":16,"messages":[{"role":"user","content":[{"type":"tool_result","content":"done"}]}]}`))
		if err == nil {
			t.Fatal("required tool_use_id must still be rejected")
		}
		t.Logf("required-field rejection=%v", err)
	})

	t.Run("omitted_content_decodes_to_empty_tool_result", func(t *testing.T) {
		request, _, _, err := NewBuiltinEngine().DecodeRequestForMutation(llmprotocol.AnthropicMessagesV1,
			[]byte(`{"model":"m","max_tokens":16,"messages":[{"role":"user","content":"Perform the action."},{"role":"assistant","content":[{"type":"tool_use","id":"call_a","name":"action","input":{}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"call_a"}]}]}`))
		if err != nil {
			t.Fatalf("omitted content rejected: %v", err)
		}
		var result *llmprotocol.ToolResult
		for _, message := range request.Messages {
			for _, content := range message.Content {
				if content.Kind == llmprotocol.ContentToolResult && content.ToolResult != nil {
					result = content.ToolResult
				}
			}
		}
		if result == nil {
			t.Fatal("tool result not decoded")
		}
		if result.CallID != "call_a" {
			t.Fatalf("tool_use_id linkage lost: %q", result.CallID)
		}
		if len(result.Content) != 0 {
			t.Fatalf("omitted content must decode empty, got %d blocks", len(result.Content))
		}
		if result.IsError != nil && *result.IsError {
			t.Fatal("omitted content must not be marked as an error")
		}
	})
}
