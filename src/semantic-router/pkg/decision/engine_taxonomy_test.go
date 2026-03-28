package decision

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

var kbMatchTests = []struct {
	name             string
	decisions        []config.Decision
	kbRules          []string
	expectedDecision string
}{
	{
		name: "kb match triggers route",
		decisions: []config.Decision{
			kbDecision("privacy-route", 200, "AND", "privacy_policy"),
		},
		kbRules:          []string{"privacy_policy"},
		expectedDecision: "privacy-route",
	},
	{
		name: "kb no match",
		decisions: []config.Decision{
			kbDecision("privacy-route", 200, "AND", "privacy_policy"),
		},
		kbRules:          []string{"local_standard"},
		expectedDecision: "",
	},
	{
		name: "kb with OR keyword",
		decisions: []config.Decision{{
			Name:     "security-route",
			Priority: 300,
			Rules: config.RuleCombination{
				Operator: "OR",
				Conditions: []config.RuleCondition{
					{Type: "keyword", Name: "threat_kw"},
					{Type: "kb", Name: "security_containment"},
				},
			},
			ModelRefs: []config.ModelRef{{Model: "guard-model"}},
		}},
		kbRules:          []string{"security_containment"},
		expectedDecision: "security-route",
	},
	{
		name: "kb AND keyword both required - keyword missing",
		decisions: []config.Decision{{
			Name:     "strict-route",
			Priority: 200,
			Rules: config.RuleCombination{
				Operator: "AND",
				Conditions: []config.RuleCondition{
					{Type: "kb", Name: "privacy_policy"},
					{Type: "keyword", Name: "sensitive"},
				},
			},
			ModelRefs: []config.ModelRef{{Model: "local-model"}},
		}},
		kbRules:          []string{"privacy_policy"},
		expectedDecision: "",
	},
	{
		name: "kb priority ordering",
		decisions: []config.Decision{
			kbDecision("low-priority", 100, "AND", "local_standard"),
			kbDecision("high-priority", 300, "AND", "security_containment"),
		},
		kbRules:          []string{"security_containment", "local_standard"},
		expectedDecision: "high-priority",
	},
}

func kbDecision(name string, priority int, op string, kbName string) config.Decision {
	return config.Decision{
		Name:     name,
		Priority: priority,
		Rules: config.RuleCombination{
			Operator: op,
			Conditions: []config.RuleCondition{
				{Type: "kb", Name: kbName},
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

func TestDecisionEngineKBSignalMatching(t *testing.T) {
	for _, tt := range kbMatchTests {
		t.Run(tt.name, func(t *testing.T) {
			engine := NewDecisionEngine(nil, nil, nil, tt.decisions, "priority")
			signals := &SignalMatches{KBRules: tt.kbRules}
			result, err := engine.EvaluateDecisionsWithSignals(signals)
			assertDecisionResult(t, result, err, tt.expectedDecision)
		})
	}
}

func TestDecisionEngineKBWithProjectionRules(t *testing.T) {
	decisions := []config.Decision{{
		Name:     "override-route",
		Priority: 250,
		Rules: config.RuleCombination{
			Operator: "OR",
			Conditions: []config.RuleCondition{
				{Type: "kb", Name: "privacy_policy"},
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

	t.Run("kb match alone", func(t *testing.T) {
		signals := &SignalMatches{KBRules: []string{"privacy_policy"}}
		result, err := engine.EvaluateDecisionsWithSignals(signals)
		assertDecisionResult(t, result, err, "override-route")
	})

	t.Run("neither matches", func(t *testing.T) {
		signals := &SignalMatches{
			KBRules:         []string{"local_standard"},
			ProjectionRules: []string{"some_other"},
		}
		result, err := engine.EvaluateDecisionsWithSignals(signals)
		assertDecisionResult(t, result, err, "")
	})
}
