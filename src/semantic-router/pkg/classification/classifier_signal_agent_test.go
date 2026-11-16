package classification

import (
	"slices"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Agent rules route by the agent a request belongs to: a Claude Code Explore
// subagent and a Codex review child each reach their own decision, with the
// model and effort it names, and the main agent the default.
func TestAgentRulesRouteEachAgentToItsDecision(t *testing.T) {
	cfg := &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Signals: config.Signals{AgentRules: []config.AgentRule{
				{Name: "explore", Role: config.AgentRoleSubagent, Types: []string{"explore"}},
				{Name: "codex_review", Harness: config.AgentHarnessCodex, Types: []string{"review"}},
				{Name: "main", Role: config.AgentRoleMain},
				{Name: "housekeeping", RequestClasses: []string{config.AgentRequestClassCompaction, config.AgentRequestClassAuxiliary}},
			}},
			Decisions: []config.Decision{
				{
					Name: "explore", Priority: 100,
					Rules:     config.RuleNode{Type: config.SignalTypeAgent, Name: "explore"},
					ModelRefs: []config.ModelRef{{Model: "cheap", ModelReasoningControl: config.ModelReasoningControl{ReasoningEffort: "low"}}},
				},
				{
					Name: "review", Priority: 100,
					Rules:     config.RuleNode{Type: config.SignalTypeAgent, Name: "codex_review"},
					ModelRefs: []config.ModelRef{{Model: "strong", ModelReasoningControl: config.ModelReasoningControl{ReasoningEffort: "high"}}},
				},
				{
					Name: "housekeeping", Priority: 200,
					Rules:     config.RuleNode{Type: config.SignalTypeAgent, Name: "housekeeping"},
					ModelRefs: []config.ModelRef{{Model: "cheap", ModelReasoningControl: config.ModelReasoningControl{ReasoningEffort: "low"}}},
				},
				{
					Name: "main", Priority: 50,
					Rules:     config.RuleNode{Type: config.SignalTypeAgent, Name: "main"},
					ModelRefs: []config.ModelRef{{Model: "default"}},
				},
			},
		},
	}
	classifier := &Classifier{Config: cfg}
	for _, tc := range []struct {
		name     string
		facts    AgentFacts
		matched  []string
		decision string
		effort   string
	}{
		{"claude code explore", AgentFacts{Role: config.AgentRoleSubagent, Harness: config.AgentHarnessClaudeCode, Type: "Explore"}, []string{"explore"}, "explore", "low"},
		{"codex review child", AgentFacts{Role: config.AgentRoleSubagent, Harness: config.AgentHarnessCodex, Type: "review"}, []string{"codex_review"}, "review", "high"},
		{"claude code review type is not codex", AgentFacts{Role: config.AgentRoleSubagent, Harness: config.AgentHarnessClaudeCode, Type: "review"}, nil, "", ""},
		{"main agent", AgentFacts{Role: config.AgentRoleMain, Harness: config.AgentHarnessCodex}, []string{"main"}, "main", ""},
		{"claude code compaction", AgentFacts{Role: config.AgentRoleMain, Harness: config.AgentHarnessClaudeCode, RequestClass: "compaction"}, []string{"main", "housekeeping"}, "housekeeping", "low"},
		{"a subagent's side request", AgentFacts{Role: config.AgentRoleSubagent, Harness: config.AgentHarnessClaudeCode, RequestClass: "auxiliary"}, []string{"housekeeping"}, "housekeeping", "low"},
		{"no harness", AgentFacts{}, nil, "", ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			results, err := classifier.EvaluateAllSignalsWithHeaders(SignalEvaluationInput{
				Text: "hello", ContextText: "hello", CurrentUserText: "hello",
				RequestFacts: RequestFacts{Agent: tc.facts},
			})
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(results.MatchedAgentRules, tc.matched) {
				t.Fatalf("matched agent rules = %v, want %v", results.MatchedAgentRules, tc.matched)
			}
			result, err := classifier.EvaluateDecisionWithEngine(results)
			if tc.decision == "" {
				if err == nil && result != nil && result.Decision != nil {
					t.Fatalf("decision = %q, want none", result.Decision.Name)
				}
				return
			}
			if err != nil || result == nil || result.Decision.Name != tc.decision {
				t.Fatalf("decision = %+v (%v), want %q", result, err, tc.decision)
			}
			if effort := result.Decision.ModelRefs[0].ReasoningEffort; effort != tc.effort {
				t.Fatalf("effort = %q, want %q", effort, tc.effort)
			}
		})
	}
}
