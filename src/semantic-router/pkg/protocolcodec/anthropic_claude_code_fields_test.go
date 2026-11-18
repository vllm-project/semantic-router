package protocolcodec

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Claude Code in auto mode sends safeguards for Anthropic's classifier review,
// and announces a deferred tool with a tool_addition block in a system message.
const anthropicClaudeCodeFieldsRequest = `{
	"model":"m","max_tokens":64,
	"safeguards":[{"type":"auto_mode","classifier_context":{"v":1,"permission_mode":"auto"}}],
	"tools":[{"name":"Read","input_schema":{"type":"object"}}],
	"messages":[
		{"role":"user","content":"first turn"},
		{"role":"system","content":[
			{"type":"text","text":"New tools are available."},
			{"type":"tool_addition","tool":{"type":"tool_definition","definition":{"name":"mcp__docs__search","description":"Search the docs","input_schema":{"type":"object","properties":{"q":{"type":"string"}}}}}},
			{"type":"tool_addition","tool":{"type":"tool_definition","definition":{"name":"Read","input_schema":{"type":"object"}}}}
		]},
		{"role":"system","content":[
			{"type":"tool_addition","tool":{"type":"tool_definition","definition":{"name":"mcp__docs__fetch","input_schema":{"type":"object"}}},"cache_control":{"type":"ephemeral"}}
		]},
		{"role":"user","content":"second turn"}
	]}`

func TestAnthropicClaudeCodeSafeguardsAndToolAdditions(t *testing.T) {
	engine := NewBuiltinEngine()
	request, envelope, _, err := engine.DecodeRequestForMutation(llmprotocol.AnthropicMessagesV1, []byte(anthropicClaudeCodeFieldsRequest))
	if err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, tool := range request.Tools {
		names = append(names, tool.Name)
	}
	if strings.Join(names, ",") != "Read,mcp__docs__search,mcp__docs__fetch" {
		t.Fatalf("tools = %v, want the declared tool, then each added one once", names)
	}
	if cache := request.Tools[2].Cache; cache == nil || cache.Type != "ephemeral" {
		t.Fatalf("the added tool lost its block's cache breakpoint: %+v", cache)
	}
	if len(request.Messages) != 3 || request.Messages[1].Role != llmprotocol.RoleSystem ||
		len(request.Messages[1].Content) != 1 || request.Messages[1].Content[0].Text != "New tools are available." {
		t.Fatalf("messages = %+v, want the system text kept and the addition-only message gone", request.Messages)
	}
	// Routing names a concrete model, so the request is encoded again.
	request.Model = "routed-model"
	request.Generation++

	encoded, err := engine.EncodeRequest(llmprotocol.AnthropicMessagesV1, request, envelope)
	if err != nil {
		t.Fatal(err)
	}
	var same struct {
		Safeguards json.RawMessage     `json:"safeguards"`
		Tools      []anthropicToolWire `json:"tools"`
	}
	if err := json.Unmarshal(encoded.Body, &same); err != nil {
		t.Fatal(err)
	}
	if string(same.Safeguards) != `[{"type":"auto_mode","classifier_context":{"v":1,"permission_mode":"auto"}}]` || len(same.Tools) != 3 {
		t.Fatalf("Anthropic dispatch changed safeguards or tools: %s", encoded.Body)
	}

	for _, format := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1} {
		t.Run(string(format), func(t *testing.T) {
			projected, encodeErr := engine.EncodeRequest(format, request, envelope)
			if encodeErr != nil {
				t.Fatal(encodeErr)
			}
			if strings.Contains(string(projected.Body), "classifier_context") {
				t.Fatalf("safeguards reached %s: %s", format, projected.Body)
			}
			if !hasDiagnostic(projected.Diagnostics, "safeguards", llmprotocol.DiagnosticDropped) {
				t.Fatalf("missing safeguards diagnostic: %+v", projected.Diagnostics)
			}
			if !strings.Contains(string(projected.Body), "mcp__docs__fetch") {
				t.Fatalf("an added tool did not reach %s: %s", format, projected.Body)
			}
		})
	}
}

func TestAnthropicToolAdditionNeedsAToolDefinition(t *testing.T) {
	body := `{"model":"m","max_tokens":8,"messages":[{"role":"system","content":[{"type":"tool_addition","tool":{"type":"toolset","definition":{"name":"x","input_schema":{"type":"object"}}}}]},{"role":"user","content":"hi"}]}`
	_, _, _, err := NewBuiltinEngine().DecodeRequestForMutation(llmprotocol.AnthropicMessagesV1, []byte(body))
	if err == nil || !strings.Contains(err.Error(), "tool_definition") {
		t.Fatalf("err = %v, want a refusal that names tool_definition", err)
	}
}

func TestAnthropicSafeguardsMustBeAnArray(t *testing.T) {
	body := `{"model":"m","max_tokens":8,"safeguards":{"type":"auto_mode"},"messages":[{"role":"user","content":"hi"}]}`
	if _, _, _, err := NewBuiltinEngine().DecodeRequestForMutation(llmprotocol.AnthropicMessagesV1, []byte(body)); err == nil {
		t.Fatal("an object in safeguards was accepted")
	}
}
