package extproc

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func toolCall(id string) llmprotocol.Content {
	return llmprotocol.Content{Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{ID: id, Name: "lookup"}}
}

func toolText(id, text string) llmprotocol.Message {
	return llmprotocol.Message{
		Role: llmprotocol.RoleTool,
		Content: []llmprotocol.Content{{
			Kind:       llmprotocol.ContentToolResult,
			ToolResult: &llmprotocol.ToolResult{CallID: id, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: text}}},
		}},
	}
}

func TestContinuationToolResultsKeepsThisTurnsResults(t *testing.T) {
	messages := []llmprotocol.Message{
		{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "start"}}},
		{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{toolCall("call_1")}},
		toolText("call_1", "earlier output"),
		{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "again"}}},
		{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{toolCall("call_2")}},
		toolText("call_2", "second output"),
		toolText("call_3", "third output"),
	}
	if got, want := continuationToolResults(messages, toolContinuationLimit), "second output\n\nthird output"; got != want {
		t.Fatalf("continuationToolResults = %q, want %q", got, want)
	}
}

func TestContinuationToolResultsStopsAtTheLimit(t *testing.T) {
	messages := []llmprotocol.Message{
		{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{toolCall("call_1")}},
		toolText("call_1", strings.Repeat("a", 100)),
		toolText("call_2", strings.Repeat("b", 100)),
	}
	got := continuationToolResults(messages, 40)
	if len(got) != 40 {
		t.Fatalf("continuationToolResults length = %d, want 40", len(got))
	}
	if strings.Contains(got, "b") {
		t.Fatalf("continuationToolResults read past the limit: %q", got)
	}
}
