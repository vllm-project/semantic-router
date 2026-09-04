package classification

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestResolveOpenAIToolChoiceFacts(t *testing.T) {
	tests := []struct {
		name         string
		toolChoice   string
		functionCall string
		want         ToolChoiceFacts
	}{
		{name: "required", toolChoice: `"required"`, want: ToolChoiceFacts{Required: true}},
		{name: "none", toolChoice: `"none"`, want: ToolChoiceFacts{None: true}},
		{name: "auto", toolChoice: `"auto"`, want: ToolChoiceFacts{}},
		{name: "chat named function", toolChoice: `{"type":"function","function":{"name":"lookup"}}`, want: ToolChoiceFacts{Required: true}},
		{name: "responses named function", toolChoice: `{"type":"function","name":"lookup"}`, want: ToolChoiceFacts{Required: true}},
		{name: "required allowed tools", toolChoice: `{"type":"allowed_tools","mode":"required","tools":[{"type":"function","name":"lookup"}]}`, want: ToolChoiceFacts{Required: true}},
		{name: "empty allowed tools", toolChoice: `{"type":"allowed_tools","mode":"required","tools":[]}`, want: ToolChoiceFacts{}},
		{name: "blank named function", toolChoice: `{"type":"function","function":{"name":" "}}`, want: ToolChoiceFacts{}},
		{name: "legacy none", functionCall: `"none"`, want: ToolChoiceFacts{None: true}},
		{name: "legacy named", functionCall: `{"name":"lookup"}`, want: ToolChoiceFacts{Required: true}},
		{name: "modern field wins", toolChoice: `"none"`, functionCall: `{"name":"lookup"}`, want: ToolChoiceFacts{None: true}},
		{name: "malformed modern field blocks legacy", toolChoice: `{`, functionCall: `{"name":"lookup"}`, want: ToolChoiceFacts{}},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := ResolveOpenAIToolChoiceFacts(
				json.RawMessage(test.toolChoice),
				json.RawMessage(test.functionCall),
			)
			assert.Equal(t, test.want, got)
		})
	}
}

func TestResolveAnthropicToolChoiceFacts(t *testing.T) {
	tests := []struct {
		name string
		raw  string
		want ToolChoiceFacts
	}{
		{name: "any", raw: `{"type":"any"}`, want: ToolChoiceFacts{Required: true}},
		{name: "named tool", raw: `{"type":"tool","name":"lookup"}`, want: ToolChoiceFacts{Required: true}},
		{name: "none", raw: `{"type":"none"}`, want: ToolChoiceFacts{None: true}},
		{name: "auto", raw: `{"type":"auto"}`, want: ToolChoiceFacts{}},
		{name: "unnamed tool", raw: `{"type":"tool"}`, want: ToolChoiceFacts{}},
		{name: "blank named tool", raw: `{"type":"tool","name":" "}`, want: ToolChoiceFacts{}},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := ResolveAnthropicToolChoiceFacts(json.RawMessage(test.raw))
			assert.Equal(t, test.want, got)
		})
	}
}
