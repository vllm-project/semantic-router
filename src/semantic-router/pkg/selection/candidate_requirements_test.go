package selection

import (
	"encoding/json"
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestCandidateRequirementsBudgetAndMetadata(t *testing.T) {
	requirements := &config.CandidateRequirements{Capabilities: config.CandidateCapabilitiesDeclared, Context: config.CandidateContextKnownLimits}
	base := config.ModelParams{Capabilities: []string{"chat"}, ContextWindowSize: 100, MaxOutputTokens: 100}
	request := &llmprotocol.Request{Messages: []llmprotocol.Message{{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "1234"}}}}, Sampling: llmprotocol.Sampling{MaxOutputTokens: llmprotocol.Int64(95)}}
	demand := DemandForRequest(request)
	if demand.InputTokens != 5 {
		t.Fatalf("input=%d", demand.InputTokens)
	}
	if err := ValidateCandidateRequirements(requirements, "m", base, demand); err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		name   string
		params config.ModelParams
		demand CandidateDemand
	}{
		{"capacity is not default", base, func() CandidateDemand { d := demand; d.MaxOutputTokens = nil; return d }()},
		{"over full budget", base, func() CandidateDemand { d := demand; d.MaxOutputTokens = llmprotocol.Int64(96); return d }()},
		{"over output limit", func() config.ModelParams { p := base; p.MaxOutputTokens = 90; return p }(), demand},
		{"missing capabilities", func() config.ModelParams { p := base; p.Capabilities = nil; return p }(), demand},
		{"missing output metadata", func() config.ModelParams { p := base; p.MaxOutputTokens = 0; return p }(), demand},
		{"unknown request", base, CandidateDemand{}},
	} {
		t.Run(test.name, func(t *testing.T) {
			if err := ValidateCandidateRequirements(requirements, "m", test.params, test.demand); !errors.Is(err, ErrNoEligibleCandidates) {
				t.Fatalf("got %v", err)
			}
		})
	}
	if err := ValidateCandidateRequirements(nil, "legacy", config.ModelParams{}, CandidateDemand{}); err != nil {
		t.Fatal(err)
	}
	if err := ValidateCandidateRequirements(&config.CandidateRequirements{Context: config.CandidateContextKnownLimits}, "context only", config.ModelParams{ContextWindowSize: 100, MaxOutputTokens: 100}, demand); err != nil {
		t.Fatal(err)
	}
}

func TestEffectiveCandidateDemandStripsToolsWithoutMutatingIngress(t *testing.T) {
	tools, _ := config.NewStructuredPayload(map[string]any{"enabled": true, "mode": "none", "strip_tool_history": true})
	params, _ := config.NewStructuredPayload(map[string]any{"default_max_tokens": 32})
	decision := &config.Decision{Plugins: []config.DecisionPlugin{{Type: "tools", Configuration: tools}, {Type: "request_params", Configuration: params}}}
	request := &llmprotocol.Request{Messages: []llmprotocol.Message{
		{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{ID: "c", Name: "lookup", Arguments: `{}`}}, {Kind: llmprotocol.ContentText, Text: "retained"}}},
		{Role: llmprotocol.RoleTool, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentToolResult, ToolResult: &llmprotocol.ToolResult{CallID: "c", Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "result"}}}}}},
		{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "next"}}},
	}, Tools: []llmprotocol.Tool{{Name: "lookup", InputSchema: []byte(`{}`)}}}
	before, _ := json.Marshal(request)
	raw := DemandForRequest(request)
	if !raw.ModelCapabilities.Supports(llmprotocol.CapabilityTools) {
		t.Fatal("missing tool demand")
	}
	effective, err := EffectiveCandidateDemand(request, decision)
	if err != nil {
		t.Fatal(err)
	}
	after, _ := json.Marshal(request)
	if string(before) != string(after) {
		t.Fatal("selection changed ingress")
	}
	if effective.ModelCapabilities.Supports(llmprotocol.CapabilityTools) || effective.InputTokens >= raw.InputTokens || effective.MaxOutputTokens == nil || *effective.MaxOutputTokens != 32 {
		t.Fatalf("demand=%+v", effective)
	}
	if err := ValidateCandidateRequirements(&config.CandidateRequirements{Capabilities: config.CandidateCapabilitiesDeclared}, "text", config.ModelParams{Capabilities: []string{"chat"}}, effective); err != nil {
		t.Fatal(err)
	}
}

func TestCandidateMetadataRequiresUnambiguousConfiguredIdentity(t *testing.T) {
	inventory := map[string]config.ModelParams{
		"a": {ContextWindowSize: 100, ExternalModelIDs: map[string]string{"provider": "shared"}},
		"b": {ContextWindowSize: 200, ExternalModelIDs: map[string]string{"provider": "shared"}},
	}
	if _, ok := CandidateModelParams(inventory, nil, "shared"); ok {
		t.Fatal("ambiguous external identity admitted")
	}
	refs := []config.ModelRef{{Model: "a", LoRAName: "adapter"}}
	if params, ok := CandidateModelParams(inventory, refs, "adapter"); !ok || params.ContextWindowSize != 100 {
		t.Fatal("explicit adapter metadata not resolved")
	}
	if _, ok := CandidateModelParams(inventory, nil, "adapter"); ok {
		t.Fatal("undeclared adapter admitted")
	}
	delete(inventory, "b")
	if params, ok := CandidateModelParams(inventory, nil, "shared"); !ok || params.ContextWindowSize != 100 {
		t.Fatal("unique exact external ID unresolved")
	}
}

func TestDisabledReasoningDoesNotRequireModelReasoning(t *testing.T) {
	requirements := &config.CandidateRequirements{Capabilities: config.CandidateCapabilitiesDeclared}
	textModel := config.ModelParams{Capabilities: []string{"chat"}}
	for _, test := range []struct {
		name     string
		effort   string
		mode     llmprotocol.ReasoningMode
		requires bool
	}{
		{"none", "none", "", false}, {"disabled", "", llmprotocol.ReasoningModeDisabled, false}, {"high", "high", "", true}, {"enabled", "", llmprotocol.ReasoningModeEnabled, true},
	} {
		t.Run(test.name, func(t *testing.T) {
			request := &llmprotocol.Request{Messages: []llmprotocol.Message{{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "hello"}}}}, ReasoningEffort: test.effort, ReasoningMode: test.mode}
			demand := DemandForRequest(request)
			if demand.ModelCapabilities.Supports(llmprotocol.CapabilityReasoning) != test.requires {
				t.Fatalf("model demand=%v", demand.ModelCapabilities.Names())
			}
			if test.effort != "" && !demand.Capabilities.Supports(llmprotocol.CapabilityReasoningEffort) {
				t.Fatal("transport effort support was dropped")
			}
			if (ValidateCandidateRequirements(requirements, "text", textModel, demand) != nil) != test.requires {
				t.Fatal("reasoning model admission disagrees")
			}
		})
	}
}
