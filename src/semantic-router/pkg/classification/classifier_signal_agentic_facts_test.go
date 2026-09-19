package classification

import (
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestEvaluateAgenticFactsSignal(t *testing.T) {
	reviewer := "reviewer"
	classifier := &Classifier{Config: &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Signals: config.Signals{AgenticFactsRules: []config.AgenticFactsRule{
				{
					Name:      "reviewer-role",
					Field:     config.AgenticFactsFieldDelegatedRole,
					Predicate: config.AgenticFactsPredicate{Equals: &reviewer},
				},
				{
					Name:      "execution-phase",
					Field:     config.AgenticFactsFieldTaskPhase,
					Predicate: config.AgenticFactsPredicate{In: []string{"execute", "verify"}},
				},
				{
					Name:      "not-used",
					Field:     config.AgenticFactsFieldDelegatedRole,
					Predicate: config.AgenticFactsPredicate{Equals: &reviewer},
				},
			}},
		},
	}}
	results := &SignalResults{
		SignalConfidences: map[string]float64{},
		SignalValues:      map[string]float64{},
		Metrics:           &SignalMetricsCollection{},
	}

	classifier.evaluateAgenticFactsSignal(results, &sync.Mutex{}, RequestFacts{
		AgenticFactsDelegatedRole: "reviewer",
		AgenticFactsTaskPhase:     "execute",
	}, map[string]bool{
		"agentic_facts:reviewer-role":   true,
		"agentic_facts:execution-phase": true,
		// "not-used" deliberately absent from usedSignals.
	})

	if len(results.MatchedAgenticFactsRules) != 2 {
		t.Fatalf("matched agentic facts rules = %v, want two matches", results.MatchedAgenticFactsRules)
	}
	if got := results.SignalConfidences["agentic_facts:reviewer-role"]; got != 1 {
		t.Fatalf("reviewer-role confidence = %v, want 1", got)
	}
}

func TestAgenticFactsRuleMatches(t *testing.T) {
	reviewer := "reviewer"
	equalsRule := config.AgenticFactsRule{
		Field:     config.AgenticFactsFieldDelegatedRole,
		Predicate: config.AgenticFactsPredicate{Equals: &reviewer},
	}
	inRule := config.AgenticFactsRule{
		Field:     config.AgenticFactsFieldTaskPhase,
		Predicate: config.AgenticFactsPredicate{In: []string{"execute", "verify"}},
	}
	unknownFieldRule := config.AgenticFactsRule{
		Field:     "lineage_depth",
		Predicate: config.AgenticFactsPredicate{Equals: &reviewer},
	}

	cases := []struct {
		name  string
		rule  config.AgenticFactsRule
		facts RequestFacts
		want  bool
	}{
		{"equals matches", equalsRule, RequestFacts{AgenticFactsDelegatedRole: "reviewer"}, true},
		{"equals does not match", equalsRule, RequestFacts{AgenticFactsDelegatedRole: "researcher"}, false},
		{"equals empty value never matches", equalsRule, RequestFacts{}, false},
		{"in matches", inRule, RequestFacts{AgenticFactsTaskPhase: "verify"}, true},
		{"in does not match", inRule, RequestFacts{AgenticFactsTaskPhase: "plan"}, false},
		{"unknown field never matches", unknownFieldRule, RequestFacts{AgenticFactsDelegatedRole: "reviewer"}, false},
	}
	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			if got := agenticFactsRuleMatches(tt.rule, tt.facts); got != tt.want {
				t.Fatalf("agenticFactsRuleMatches() = %v, want %v", got, tt.want)
			}
		})
	}
}
