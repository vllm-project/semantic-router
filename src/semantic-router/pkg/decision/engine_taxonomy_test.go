package decision

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

var taxonomyMatchTests = []struct {
	name             string
	decisions        []config.Decision
	taxonomyRules    []string
	expectedDecision string
}{
	{
		name: "taxonomy match triggers route",
		decisions: []config.Decision{
			taxonomyDecision("privacy-route", 200, "AND", "privacy_policy"),
		},
		taxonomyRules:    []string{"privacy_policy"},
		expectedDecision: "privacy-route",
	},
	{
		name: "taxonomy no match",
		decisions: []config.Decision{
			taxonomyDecision("privacy-route", 200, "AND", "privacy_policy"),
		},
		taxonomyRules:    []string{"local_standard"},
		expectedDecision: "",
	},
	{
		name: "taxonomy with OR keyword",
		decisions: []config.Decision{{
			Name:     "security-route",
			Priority: 300,
			Rules: config.RuleCombination{
				Operator: "OR",
				Conditions: []config.RuleCondition{
					{Type: "keyword", Name: "threat_kw"},
					{Type: "taxonomy", Name: "security_containment"},
				},
			},
			ModelRefs: []config.ModelRef{{Model: "guard-model"}},
		}},
		taxonomyRules:    []string{"security_containment"},
		expectedDecision: "security-route",
	},
	{
		name: "taxonomy AND keyword both required - keyword missing",
		decisions: []config.Decision{{
			Name:     "strict-route",
			Priority: 200,
			Rules: config.RuleCombination{
				Operator: "AND",
				Conditions: []config.RuleCondition{
					{Type: "taxonomy", Name: "privacy_policy"},
					{Type: "keyword", Name: "sensitive"},
				},
			},
			ModelRefs: []config.ModelRef{{Model: "local-model"}},
		}},
		taxonomyRules:    []string{"privacy_policy"},
		expectedDecision: "",
	},
	{
		name: "taxonomy priority ordering",
		decisions: []config.Decision{
			taxonomyDecision("low-priority", 100, "AND", "local_standard"),
			taxonomyDecision("high-priority", 300, "AND", "security_containment"),
		},
		taxonomyRules:    []string{"security_containment", "local_standard"},
		expectedDecision: "high-priority",
	},
}

func taxonomyDecision(name string, priority int, op string, taxonomyName string) config.Decision {
	return config.Decision{
		Name:     name,
		Priority: priority,
		Rules: config.RuleCombination{
			Operator: op,
			Conditions: []config.RuleCondition{
				{Type: "taxonomy", Name: taxonomyName},
			},
		},
		ModelRefs: []config.ModelRef{{Model: "model"}},
	}
}

func assertDecisionResult(t *testing.T, result *DecisionResult, err error, expected string) {
	t.Helper()
	if err != nil {
		t.Fatalf("EvaluateDecisionsWithSignals: %v", err)
	}
	if expected == "" {
		if result != nil {
			t.Errorf("expected no match, got %q", result.Decision.Name)
		}
		return
	}
	if result == nil {
		t.Fatalf("expected match %q, got nil", expected)
	}
	if result.Decision.Name != expected {
		t.Errorf("expected %q, got %q", expected, result.Decision.Name)
	}
}

func TestDecisionEngineTaxonomySignalMatching(t *testing.T) {
	for _, tt := range taxonomyMatchTests {
		t.Run(tt.name, func(t *testing.T) {
			engine := NewDecisionEngine(nil, nil, nil, tt.decisions, "priority")
			signals := &SignalMatches{TaxonomyRules: tt.taxonomyRules}
			result, err := engine.EvaluateDecisionsWithSignals(signals)
			assertDecisionResult(t, result, err, tt.expectedDecision)
		})
	}
}

func TestDecisionEngineTaxonomyWithProjectionRules(t *testing.T) {
	decisions := []config.Decision{{
		Name:     "override-route",
		Priority: 250,
		Rules: config.RuleCombination{
			Operator: "OR",
			Conditions: []config.RuleCondition{
				{Type: "taxonomy", Name: "privacy_policy"},
				{Type: "projection", Name: "privacy_override_active"},
			},
		},
		ModelRefs: []config.ModelRef{{Model: "local-model"}},
	}}

	engine := NewDecisionEngine(nil, nil, nil, decisions, "priority")

	t.Run("projection match alone", func(t *testing.T) {
		signals := &SignalMatches{ProjectionRules: []string{"privacy_override_active"}}
		result, err := engine.EvaluateDecisionsWithSignals(signals)
		assertDecisionResult(t, result, err, "override-route")
	})

	t.Run("taxonomy match alone", func(t *testing.T) {
		signals := &SignalMatches{TaxonomyRules: []string{"privacy_policy"}}
		result, err := engine.EvaluateDecisionsWithSignals(signals)
		assertDecisionResult(t, result, err, "override-route")
	})

	t.Run("neither matches", func(t *testing.T) {
		signals := &SignalMatches{
			TaxonomyRules:   []string{"local_standard"},
			ProjectionRules: []string{"some_other"},
		}
		result, err := engine.EvaluateDecisionsWithSignals(signals)
		assertDecisionResult(t, result, err, "")
	})
}
