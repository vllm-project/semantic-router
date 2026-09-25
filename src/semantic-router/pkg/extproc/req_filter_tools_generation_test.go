package extproc

import (
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/tools"
)

func TestApplySelectedToolsPreservesGenerationForUnchangedDefinitions(t *testing.T) {
	request := &llmprotocol.Request{
		Generation: 7,
		Tools: []llmprotocol.Tool{{
			Name:        "lookup",
			Description: "Look up a value",
			InputSchema: json.RawMessage(`{"type":"object"}`),
		}},
	}
	selected := append([]llmprotocol.Tool(nil), request.Tools...)

	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	if err := router.applySelectedTools(request, selected, "sticky", 1, 0, "", nil); err != nil {
		t.Fatalf("apply selected tools: %v", err)
	}
	if request.Generation != 7 {
		t.Fatalf("generation=%d, want unchanged generation 7", request.Generation)
	}
	if !toolDefinitionsEqual(request.Tools, selected) {
		t.Fatalf("selected definitions changed: %#v", request.Tools)
	}
}

func TestApplySelectedToolsVersionsDefinitionAndOrderChanges(t *testing.T) {
	base := []llmprotocol.Tool{
		{Name: "lookup", InputSchema: json.RawMessage(`{"type":"object"}`)},
		{Name: "search", InputSchema: json.RawMessage(`{"type":"object"}`)},
	}
	tests := []struct {
		name     string
		selected []llmprotocol.Tool
	}{
		{
			name: "order",
			selected: []llmprotocol.Tool{
				{Name: "search", InputSchema: json.RawMessage(`{"type":"object"}`)},
				{Name: "lookup", InputSchema: json.RawMessage(`{"type":"object"}`)},
			},
		},
		{
			name: "definition",
			selected: []llmprotocol.Tool{
				{Name: "lookup", InputSchema: json.RawMessage(`{"type":"object","required":["id"]}`)},
				{Name: "search", InputSchema: json.RawMessage(`{"type":"object"}`)},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			request := &llmprotocol.Request{Generation: 7, Tools: base}
			router := &OpenAIRouter{Config: &config.RouterConfig{}}
			if err := router.applySelectedTools(request, test.selected, "sticky", 1, 0, "", nil); err != nil {
				t.Fatalf("apply selected tools: %v", err)
			}
			if request.Generation != 8 {
				t.Fatalf("generation=%d, want 8", request.Generation)
			}
		})
	}
}

func TestApplySelectedToolsVersionsProviderVisibleSchemaBytes(t *testing.T) {
	request := &llmprotocol.Request{
		Generation: 7,
		Tools: []llmprotocol.Tool{{
			Name:        "lookup",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"id":{"type":"string"}}}`),
		}},
	}
	selected := []llmprotocol.Tool{{
		Name:        "lookup",
		InputSchema: json.RawMessage("{\n  \"properties\": {\"id\": {\"type\": \"string\"}},\n  \"type\": \"object\"\n}"),
	}}

	if tools.ToolDefinitionFingerprint(request.Tools[0]) != tools.ToolDefinitionFingerprint(selected[0]) {
		t.Fatal("test schemas must remain semantically equivalent")
	}
	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	if err := router.applySelectedTools(request, selected, "sticky", 1, 0, "", nil); err != nil {
		t.Fatalf("apply selected tools: %v", err)
	}
	if request.Generation != 8 {
		t.Fatalf("generation=%d, want 8 for changed provider-visible schema bytes", request.Generation)
	}
}
