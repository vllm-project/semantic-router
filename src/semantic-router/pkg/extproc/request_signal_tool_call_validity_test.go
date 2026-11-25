package extproc

import (
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestExtractSemanticRequestSignalsInvalidToolCalls(t *testing.T) {
	request := &llmprotocol.Request{
		Model: "test-model",
		Tools: []llmprotocol.Tool{{
			Name:        "bash",
			InputSchema: json.RawMessage(`{"type":"object","required":["command"],"properties":{"command":{"type":"string"}}}`),
		}},
		Messages: []llmprotocol.Message{
			{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "run tests"}}},
			{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{ID: "call_1", Name: "bash", Arguments: `{"command":"ls"}`}}}},
		},
	}
	snapshot := extractSemanticRequestSignals(request)
	if snapshot.InvalidToolCalls != 1 {
		t.Fatalf("snapshot.InvalidToolCalls = %d, want 1", snapshot.InvalidToolCalls)
	}
}
