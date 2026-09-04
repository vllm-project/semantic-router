package classification

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestEvaluateDecisionWithEngineForDecisionsRestrictsCandidates(t *testing.T) {
	fusionDecision := config.Decision{
		Name:     "fusion-business",
		Priority: 10,
		Rules: config.RuleCombination{
			Type: config.SignalTypeDomain,
			Name: "business",
		},
		Algorithm: &config.AlgorithmConfig{Type: "fusion"},
	}
	classifier := &Classifier{
		Config: &config.RouterConfig{
			IntelligentRouting: config.IntelligentRouting{
				Strategy: "priority",
				Decisions: []config.Decision{
					{
						Name:     "static-business",
						Priority: 100,
						Rules: config.RuleCombination{
							Type: config.SignalTypeDomain,
							Name: "business",
						},
						Algorithm: &config.AlgorithmConfig{Type: "static"},
					},
					fusionDecision,
				},
			},
		},
	}
	signals := &SignalResults{MatchedDomainRules: []string{"business"}}

	result, err := classifier.EvaluateDecisionWithEngine(signals)
	if err != nil {
		t.Fatalf("EvaluateDecisionWithEngine failed: %v", err)
	}
	if result == nil || result.Decision.Name != "static-business" {
		t.Fatalf("expected all-decision evaluation to pick static-business, got %+v", result)
	}

	result, err = classifier.EvaluateDecisionWithEngineForDecisions(signals, []config.Decision{fusionDecision})
	if err != nil {
		t.Fatalf("EvaluateDecisionWithEngineForDecisions failed: %v", err)
	}
	if result == nil || result.Decision.Name != "fusion-business" {
		t.Fatalf("expected filtered evaluation to pick fusion-business, got %+v", result)
	}
}

func TestEvaluateDecisionWithEngineAppliesOnUnknown(t *testing.T) {
	threshold := 0.5
	classifier := &Classifier{Config: &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{{
				Name: "guarded",
				Rules: config.RuleCombination{
					Type:      config.SignalTypeClassifier,
					Name:      "risk",
					Label:     "RISKY",
					Predicate: &config.NumericPredicate{GTE: &threshold},
					OnUnknown: config.RuleOnUnknownFailRequest,
				},
			}},
		},
	}}
	signals := &SignalResults{SignalErrors: map[string]string{"classifier:risk": "timeout"}}

	_, err := classifier.EvaluateDecisionWithEngine(signals)
	if err == nil {
		t.Fatal("expected fail_request error")
	}
	if signals.AppliedUnknownPolicies["guarded"] != config.RuleOnUnknownFailRequest {
		t.Fatalf("applied policies = %v", signals.AppliedUnknownPolicies)
	}
}
