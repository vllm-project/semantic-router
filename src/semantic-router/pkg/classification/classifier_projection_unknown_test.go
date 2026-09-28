package classification

import (
	"errors"
	"slices"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
)

func projectionFailureClassifier() *Classifier {
	return &Classifier{Config: &config.RouterConfig{IntelligentRouting: config.IntelligentRouting{
		Projections: config.Projections{
			Scores: []config.ProjectionScore{
				{Name: "downstream", Method: "weighted_sum", Inputs: []config.ProjectionScoreInput{
					{Type: "projection", Name: "low", ValueSource: "confidence", Weight: 1},
				}},
				{Name: "risk", Method: "weighted_sum", Inputs: []config.ProjectionScoreInput{
					{Type: "safety", Name: "screen", ValueSource: "raw", Weight: 1},
				}},
				{Name: "raw_copy", Method: "weighted_sum", Inputs: []config.ProjectionScoreInput{
					{Type: "projection", Name: "risk", ValueSource: "raw", Weight: 1},
				}},
				{Name: "independent", Method: "weighted_sum", Inputs: []config.ProjectionScoreInput{
					{Type: "conversation", Name: "tools", Weight: 1},
				}},
			},
			Mappings: []config.ProjectionMapping{
				{Name: "risk_band", Source: "risk", Outputs: []config.ProjectionMappingOutput{
					{Name: "low", LT: float64Ptr(0.5)}, {Name: "high", GTE: float64Ptr(0.5)},
				}},
				{Name: "downstream_band", Source: "downstream", Outputs: []config.ProjectionMappingOutput{{Name: "derived_low", GTE: float64Ptr(0)}}},
				{Name: "raw_band", Source: "raw_copy", Outputs: []config.ProjectionMappingOutput{{Name: "raw_low", LT: float64Ptr(0.5)}}},
				{Name: "tool_band", Source: "independent", Outputs: []config.ProjectionMappingOutput{{Name: "tool_route", GTE: float64Ptr(1)}}},
			},
		},
		Strategy: config.RoutingStrategyPriority,
		Decisions: []config.Decision{
			{Name: "guard", Priority: 20, Rules: config.RuleNode{Type: "projection", Name: "high", OnUnknown: config.RuleOnUnknownFailRequest}},
			{Name: "private", Priority: 10},
		},
	}}}
}

func TestProjectionFailureReachesDecisionUnknownPolicy(t *testing.T) {
	c := projectionFailureClassifier()
	results := c.applyProjections(&SignalResults{
		SignalErrors:             map[string]string{"safety:screen": "safety_classification_failed"},
		MatchedConversationRules: []string{"tools"},
	})
	for _, name := range []string{"risk", "raw_copy", "downstream"} {
		if value, ok := results.ProjectionScores[name]; ok {
			t.Fatalf("failed score %q fabricated a value: %v", name, value)
		}
	}
	for _, name := range []string{"risk", "low", "high", "downstream", "derived_low", "raw_copy", "raw_low"} {
		key := "projection:" + name
		if results.SignalErrors[key] == "" || slices.Contains(results.MatchedProjectionRules, name) {
			t.Fatalf("failed projection %q lost unknown state: errors=%v matches=%v", name, results.SignalErrors, results.MatchedProjectionRules)
		}
		if _, ok := results.SignalConfidences[key]; ok {
			t.Fatalf("failed projection %q fabricated confidence", name)
		}
	}
	if results.ProjectionScores["independent"] != 1 || !slices.Contains(results.MatchedProjectionRules, "tool_route") {
		t.Fatal("unrelated projection was suppressed")
	}
	if len(results.ProjectionTrace.Scores) != 1 || results.ProjectionTrace.Scores[0].Name != "independent" {
		t.Fatalf("trace reported numeric values for failed scores: %+v", results.ProjectionTrace.Scores)
	}
	if len(results.ProjectionTrace.Mappings) != 1 || results.ProjectionTrace.Mappings[0].SelectedOutput != "tool_route" {
		t.Fatalf("trace reported failed mapping outputs: %+v", results.ProjectionTrace.Mappings)
	}
	got, traces, err := c.EvaluateDecisionWithEngineAndTrace(results)
	if got != nil || !errors.Is(err, decision.ErrDecisionUnresolved) || len(traces) == 0 || traces[0].State != "unknown" {
		t.Fatalf("failed safety projection did not fail closed: traces=%+v err=%v", traces, err)
	}
	for _, tc := range []struct {
		policy config.UnknownPolicy
		want   string
	}{{config.RuleOnUnknownMatch, "guard"}, {config.RuleOnUnknownNoMatch, "private"}} {
		c.Config.Decisions[0].Rules.OnUnknown = tc.policy
		got, err := c.EvaluateDecisionWithEngine(results)
		if err != nil || got == nil || got.Decision.Name != tc.want {
			t.Fatalf("policy %q lost its fallback contract: result=%+v err=%v", tc.policy, got, err)
		}
	}
}

func TestProjectionNoMatchRemainsAnObservedZero(t *testing.T) {
	c := projectionFailureClassifier()
	results := c.applyProjections(&SignalResults{SignalValues: map[string]float64{"safety:screen": 0}})
	if value, ok := results.ProjectionScores["risk"]; !ok || value != 0 || !slices.Contains(results.MatchedProjectionRules, "low") {
		t.Fatalf("valid zero observation was not mapped: %+v", results)
	}
	if len(results.SignalErrors) != 0 {
		t.Fatalf("valid non-match became unknown: %v", results.SignalErrors)
	}
	got, err := c.EvaluateDecisionWithEngine(results)
	if err != nil || got == nil || got.Decision.Name != "private" {
		t.Fatalf("valid low risk should reach the default: result=%+v err=%v", got, err)
	}
}
