package decision

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

var categoryKBMatchTests = []struct {
	name             string
	decisions        []config.Decision
	categoryKBRules  []string
	expectedDecision string
}{
	{
		name: "category_kb match triggers route",
		decisions: []config.Decision{
			categoryKBDecision("privacy-route", 200, "AND", "proprietary_code"),
		},
		categoryKBRules:  []string{"proprietary_code"},
		expectedDecision: "privacy-route",
	},
	{
		name: "category_kb no match",
		decisions: []config.Decision{
			categoryKBDecision("privacy-route", 200, "AND", "proprietary_code"),
		},
		categoryKBRules:  []string{"generic_coding"},
		expectedDecision: "",
	},
	{
		name: "category_kb with OR keyword",
		decisions: []config.Decision{{
			Name:     "security-route",
			Priority: 300,
			Rules: config.RuleCombination{
				Operator: "OR",
				Conditions: []config.RuleCondition{
					{Type: "keyword", Name: "threat_kw"},
					{Type: "category_kb", Name: "prompt_injection"},
				},
			},
			ModelRefs: []config.ModelRef{{Model: "guard-model"}},
		}},
		categoryKBRules:  []string{"prompt_injection"},
		expectedDecision: "security-route",
	},
	{
		name: "category_kb AND keyword both required - keyword missing",
		decisions: []config.Decision{{
			Name:     "strict-route",
			Priority: 200,
			Rules: config.RuleCombination{
				Operator: "AND",
				Conditions: []config.RuleCondition{
					{Type: "category_kb", Name: "pii"},
					{Type: "keyword", Name: "sensitive"},
				},
			},
			ModelRefs: []config.ModelRef{{Model: "local-model"}},
		}},
		categoryKBRules:  []string{"pii"},
		expectedDecision: "",
	},
	{
		name: "category_kb priority ordering",
		decisions: []config.Decision{
			categoryKBDecision("low-priority", 100, "AND", "generic_coding"),
			categoryKBDecision("high-priority", 300, "AND", "prompt_injection"),
		},
		categoryKBRules:  []string{"prompt_injection", "generic_coding"},
		expectedDecision: "high-priority",
	},
}

func categoryKBDecision(name string, priority int, op string, kbName string) config.Decision {
	return config.Decision{
		Name:     name,
		Priority: priority,
		Rules: config.RuleCombination{
			Operator: op,
			Conditions: []config.RuleCondition{
				{Type: "category_kb", Name: kbName},
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

func TestDecisionEngine_CategoryKBSignalMatching(t *testing.T) {
	for _, tt := range categoryKBMatchTests {
		t.Run(tt.name, func(t *testing.T) {
			engine := NewDecisionEngine(nil, nil, nil, tt.decisions, "priority")
			signals := &SignalMatches{CategoryKBRules: tt.categoryKBRules}
			result, err := engine.EvaluateDecisionsWithSignals(signals)
			assertDecisionResult(t, result, err, tt.expectedDecision)
		})
	}
}

func TestDecisionEngine_CategoryKBWithProjectionRules(t *testing.T) {
	decisions := []config.Decision{{
		Name:     "override-route",
		Priority: 250,
		Rules: config.RuleCombination{
			Operator: "OR",
			Conditions: []config.RuleCondition{
				{Type: "category_kb", Name: "pii"},
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

	t.Run("category_kb match alone", func(t *testing.T) {
		signals := &SignalMatches{CategoryKBRules: []string{"pii"}}
		result, err := engine.EvaluateDecisionsWithSignals(signals)
		assertDecisionResult(t, result, err, "override-route")
	})

	t.Run("neither matches", func(t *testing.T) {
		signals := &SignalMatches{
			CategoryKBRules: []string{"generic_coding"},
			ProjectionRules: []string{"some_other"},
		}
		result, err := engine.EvaluateDecisionsWithSignals(signals)
		assertDecisionResult(t, result, err, "")
	})
}
