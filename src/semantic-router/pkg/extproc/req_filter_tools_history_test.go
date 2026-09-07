package extproc

import (
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestStripSemanticToolPolicyRemovesControlsAndLinkedHistory(t *testing.T) {
	parallel := true
	request := &llmprotocol.Request{
		Generation:        1,
		Tools:             []llmprotocol.Tool{{Name: "lookup", InputSchema: json.RawMessage(`{"type":"object"}`)}},
		ToolChoice:        llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceAuto},
		ParallelToolCalls: &parallel,
		Messages: []llmprotocol.Message{
			{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "Inspect the service."}}},
			{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{
				{Kind: llmprotocol.ContentText, Text: "Keep this explanation."},
				{Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{ID: "call_1", Name: "lookup", Arguments: `{}`}},
			}},
			{Role: llmprotocol.RoleTool, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentToolResult, ToolResult: &llmprotocol.ToolResult{CallID: "call_1"}}}},
			{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "Summarize safely."}}},
		},
	}

	changed, removed := stripSemanticToolPolicy(request, true)
	if !changed || removed != 2 {
		t.Fatalf("changed=%v removed=%d, want true/2", changed, removed)
	}
	if len(request.Tools) != 0 || request.ToolChoice.Mode != "" || request.ParallelToolCalls != nil {
		t.Fatalf("tool controls remain: %+v", request)
	}
	if len(request.Messages) != 3 || len(request.Messages[1].Content) != 1 ||
		request.Messages[1].Content[0].Text != "Keep this explanation." {
		t.Fatalf("non-tool history was not preserved: %+v", request.Messages)
	}
}

func TestStripSemanticToolPolicyCanKeepHistory(t *testing.T) {
	request := &llmprotocol.Request{
		Generation: 1,
		Tools:      []llmprotocol.Tool{{Name: "lookup", InputSchema: json.RawMessage(`{"type":"object"}`)}},
		Messages: []llmprotocol.Message{{
			Role: llmprotocol.RoleAssistant,
			Content: []llmprotocol.Content{{Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{
				ID: "call_1", Name: "lookup", Arguments: `{}`,
			}}},
		}},
	}
	changed, removed := stripSemanticToolPolicy(request, false)
	if !changed || removed != 0 || len(request.Messages) != 1 {
		t.Fatalf("changed=%v removed=%d messages=%+v", changed, removed, request.Messages)
	}
}

func TestApplyPreDispatchToolsPolicyMutatesNeutralRequestOnce(t *testing.T) {
	payload, err := config.NewStructuredPayload(config.ToolsPluginConfig{
		Enabled: true, Mode: config.ToolsPluginModeNone, StripToolHistory: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	request := &llmprotocol.Request{
		Generation: 7,
		Tools:      []llmprotocol.Tool{{Name: "lookup", InputSchema: json.RawMessage(`{"type":"object"}`)}},
		Messages: []llmprotocol.Message{{Role: llmprotocol.RoleTool, Content: []llmprotocol.Content{{
			Kind: llmprotocol.ContentToolResult, ToolResult: &llmprotocol.ToolResult{CallID: "call_1"},
		}}}},
	}
	ctx := &RequestContext{
		SemanticRequest: request,
		VSRSelectedDecision: &config.Decision{Name: "private", Plugins: []config.DecisionPlugin{{
			Type: config.DecisionPluginTools, Configuration: payload,
		}}},
	}
	changed, err := (&OpenAIRouter{}).applyPreDispatchToolsPolicy(ctx)
	if err != nil || !changed {
		t.Fatalf("changed=%v err=%v", changed, err)
	}
	if request.Generation != 8 {
		t.Fatalf("generation=%d, want 8", request.Generation)
	}
	if len(request.Tools) != 0 || len(request.Messages) != 0 {
		t.Fatalf("neutral policy was not applied: %+v", request)
	}
}
