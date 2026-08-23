package tools

import (
	"encoding/json"
	"testing"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/param"
)

func TestSemanticToolProjectsTypedFields(t *testing.T) {
	tool, err := SemanticTool(openai.ChatCompletionToolParam{
		Type: "function",
		Function: openai.FunctionDefinitionParam{
			Name:        "lookup",
			Description: param.NewOpt("Look up a record"),
			Strict:      param.NewOpt(true),
			Parameters: openai.FunctionParameters{
				"type": "object",
				"properties": map[string]any{
					"id": map[string]any{"type": "string"},
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("SemanticTool() error = %v", err)
	}
	if tool.Name != "lookup" || tool.Description != "Look up a record" || tool.Strict == nil || !*tool.Strict {
		t.Fatalf("SemanticTool() = %+v", tool)
	}
	var schema map[string]any
	if err := json.Unmarshal(tool.InputSchema, &schema); err != nil {
		t.Fatalf("InputSchema is invalid JSON: %v", err)
	}
	if schema["type"] != "object" {
		t.Fatalf("InputSchema = %s", tool.InputSchema)
	}
}

func TestSemanticToolDefaultsOmittedSchema(t *testing.T) {
	tool, err := SemanticTool(openai.ChatCompletionToolParam{
		Function: openai.FunctionDefinitionParam{Name: "ping"},
	})
	if err != nil {
		t.Fatalf("SemanticTool() error = %v", err)
	}
	if got := string(tool.InputSchema); got != `{"type":"object"}` {
		t.Fatalf("InputSchema = %s", got)
	}
	if tool.Strict != nil {
		t.Fatalf("Strict = %v, want nil", tool.Strict)
	}
}

func TestSemanticToolRejectsOpaqueProviderOverrides(t *testing.T) {
	overridden := param.Override[openai.ChatCompletionToolParam](json.RawMessage(`{
		"type":"function",
		"function":{"name":"hidden","parameters":{"type":"object"}}
	}`))
	if _, err := SemanticTool(overridden); err == nil {
		t.Fatal("SemanticTool() accepted an opaque provider override")
	}
}

func TestSemanticToolRejectsUnsupportedType(t *testing.T) {
	_, err := SemanticTool(openai.ChatCompletionToolParam{
		Type:     "custom",
		Function: openai.FunctionDefinitionParam{Name: "custom"},
	})
	if err == nil {
		t.Fatal("SemanticTool() accepted an unsupported tool type")
	}
}
