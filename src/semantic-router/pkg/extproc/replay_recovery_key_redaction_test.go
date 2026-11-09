package extproc

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A recovery key grants access to the stored content it names, so Replay may
// record that the retrieval tool was offered and called, but never which key:
// not in the tool definitions, the request body, or the tool trace.
func TestReplayNeverRecordsContextRecoveryKeys(t *testing.T) {
	const key = "ctx_zqissuedkey0123456789"
	schema, err := contextRecoveryToolSchema([]string{key})
	if err != nil {
		t.Fatal(err)
	}
	callerSchema := json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}}}`)
	request := llmprotocol.Request{
		Tools: []llmprotocol.Tool{
			{Name: "lookup", InputSchema: callerSchema},
			{Name: contextcompression.RetrieveToolName, InputSchema: schema},
		},
		Messages: []llmprotocol.Message{
			{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "live question"}}},
			{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{
				Kind:     llmprotocol.ContentToolCall,
				ToolCall: &llmprotocol.ToolCall{ID: "call-1", Name: contextcompression.RetrieveToolName, Arguments: `{"key":"` + key + `"}`},
			}}},
		},
	}

	sanitized := replaySafeSemanticRequest(request)
	body, err := json.Marshal(sanitized)
	if err != nil {
		t.Fatal(err)
	}
	_, definitions := extractSemanticPromptAndTools(&sanitized)
	for name, recorded := range map[string]string{"request body": string(body), "tool definitions": definitions} {
		if strings.Contains(recorded, key) {
			t.Fatalf("the %s records the recovery key", name)
		}
		if !strings.Contains(recorded, contextcompression.RetrieveToolName) {
			t.Fatalf("the %s no longer shows that the retrieval tool was used", name)
		}
	}
	if string(sanitized.Tools[0].InputSchema) != string(callerSchema) {
		t.Fatal("a caller-supplied tool was changed")
	}
	if !strings.Contains(string(request.Tools[1].InputSchema), key) ||
		!strings.Contains(request.Messages[1].Content[0].ToolCall.Arguments, key) {
		t.Fatal("sanitizing for Replay must not modify the live request")
	}

	step := replayToolCallStep(replayToolSourceResponse, "assistant", "openai.chat.v1", llmprotocol.ToolCall{
		ID: "call-1", Name: contextcompression.RetrieveToolName, Arguments: `{"key":"` + key + `"}`,
	})
	if step.ToolName != contextcompression.RetrieveToolName || step.Arguments != "" || step.RawArguments != "" {
		t.Fatalf("the tool trace records the retrieval key: %+v", step)
	}
	other := replayToolCallStep(replayToolSourceResponse, "assistant", "openai.chat.v1", llmprotocol.ToolCall{
		ID: "call-2", Name: "lookup", Arguments: `{"city":"Paris"}`,
	})
	if other.Arguments != `{"city":"Paris"}` {
		t.Fatalf("an ordinary tool call lost its arguments: %+v", other)
	}
}
