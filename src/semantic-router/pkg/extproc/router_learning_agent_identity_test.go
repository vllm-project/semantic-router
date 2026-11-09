package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

// Claude Code sends one session id for a run and its subagents, and an agent
// id only on subagent requests. Without an agent header, session scope holds a
// subagent on its parent's model; with one, each agent is held on its own.
func TestProtectionAgentIdentityHoldsEachAgentOnItsOwnModel(t *testing.T) {
	const agentHeader = "x-claude-code-agent-id"
	for _, tc := range []struct {
		name          string
		agentHeader   string
		wantSubagent  string
		wantAgentPart bool
	}{
		{name: "without_agent_header", wantSubagent: "frontier"},
		{name: "with_agent_header", agentHeader: agentHeader, wantSubagent: "cheap", wantAgentPart: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			sessiontelemetry.ResetRouterSessionMemoryForTesting()
			t.Cleanup(sessiontelemetry.ResetRouterSessionMemoryForTesting)
			registry := selection.NewRegistry()
			cfg := routerLearningProtectionOnlyTestConfig(config.RouterLearningScopeSession)
			if tc.agentHeader != "" {
				cfg.RouterLearning.Protection.Identity.Headers.Agent = stringPtr(tc.agentHeader)
			}
			router := &OpenAIRouter{Config: cfg, ModelSelector: registry}
			both := []config.ModelRef{{Model: "cheap"}, {Model: "frontier"}}
			selectRequest := func(agent, proposal string, tool bool) (string, *RequestContext) {
				t.Helper()
				ctx := routerLearningRequestContext("run-1", "")
				if agent != "" {
					ctx.Headers[agentHeader] = agent
				}
				ctx.VSRSelectedDecision = &config.Decision{Name: "choice"}
				ctx.VSRConversationFacts = classification.ConversationFacts{LastMessageToolResult: tool}
				registry.Register(selection.MethodStatic, selectionResultSelector{result: &selection.SelectionResult{
					SelectedModel: proposal, Score: 1, Confidence: 1, Method: selection.MethodStatic,
					AllScores: map[string]float64{proposal: 1},
				}})
				selCtx := router.buildSelectionContext(both, "choice", "", nil, "", nil, ctx)
				selected, _, err := router.selectModelFromCandidates(selCtx, nil, ctx)
				if err != nil || selected == nil {
					t.Fatalf("selection failed for agent=%q: selected=%+v err=%v", agent, selected, err)
				}
				if err := commitAgenticSessionDecision(ctx); err != nil {
					t.Fatal(err)
				}
				return selected.Model, ctx
			}

			if model, _ := selectRequest("", "frontier", false); model != "frontier" {
				t.Fatalf("main agent did not establish frontier: %q", model)
			}
			model, ctx := selectRequest("agent-a", "cheap", false)
			if model != tc.wantSubagent {
				t.Fatalf("subagent first call got %q, want %q", model, tc.wantSubagent)
			}
			if part, ok := ctx.VSRLearningPolicy.ToMap()["identity"].(map[string]interface{})["agent"]; ok != tc.wantAgentPart {
				t.Fatalf("agent identity diagnostics present=%v, want %v: %#v", ok, tc.wantAgentPart, part)
			}
			if model, _ := selectRequest("agent-a", "frontier", true); model != tc.wantSubagent {
				t.Errorf("subagent tool continuation got %q, want it held on %q", model, tc.wantSubagent)
			}
			if model, _ := selectRequest("", "cheap", true); model != "frontier" {
				t.Errorf("main agent got %q after its subagent, want it held on frontier", model)
			}
		})
	}
}
