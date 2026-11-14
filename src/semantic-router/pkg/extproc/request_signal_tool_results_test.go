package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A tool-continuation turn carries the results instead of user text, so the
// snapshot reads them: a selection on such a turn has content to read, and a
// request with user text keeps reading the user text alone.
func TestSnapshotCarriesToolResults(t *testing.T) {
	request := &llmprotocol.Request{Model: "general", Messages: []llmprotocol.Message{
		{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "find the file"}}},
		{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{ID: "c1", Name: "search"}}}},
		{Role: llmprotocol.RoleTool, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentToolResult, ToolResult: &llmprotocol.ToolResult{
			CallID:  "c1",
			Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "3 files matched: a, b, c"}},
		}}}},
	}}
	snapshot := extractSemanticRequestSignals(request)
	if len(snapshot.ToolResults) != 1 || snapshot.ToolResults[0] != "3 files matched: a, b, c" {
		t.Fatalf("tool results = %q", snapshot.ToolResults)
	}
	if snapshot.UserContent != "find the file" {
		t.Fatalf("user content = %q", snapshot.UserContent)
	}
}
