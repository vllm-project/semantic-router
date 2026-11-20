package masking

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A server tool's call and the citations it attached to text are sent as
// written, so their string leaves are masked like the safeguards'.
func TestApply_ServerToolBlocksMasked(t *testing.T) {
	request := &llmprotocol.Request{Messages: []llmprotocol.Message{{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{
		{Kind: llmprotocol.ContentServerTool, Raw: json.RawMessage(`{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{"query":"alice@example.com"}}`)},
		{Kind: llmprotocol.ContentText, Text: "Found alice@example.com.", Raw: json.RawMessage(`[{"type":"web_search_result_location","cited_text":"mail alice@example.com"}]`)},
	}}}}
	result, err := Apply(request, NewAllocator(defaultCfg()), scanForValue("EMAIL_ADDRESS", "alice@example.com"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	content := request.Messages[0].Content
	for _, got := range []string{string(content[0].Raw), content[1].Text, string(content[1].Raw)} {
		if !result.Changed || strings.Contains(got, "alice@example.com") || !strings.Contains(got, "[EMAIL_ADDRESS_0]") {
			t.Fatalf("server tool content = %+v, want the address masked", content)
		}
	}
}
