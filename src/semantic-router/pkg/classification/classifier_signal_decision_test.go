package classification

import (
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
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
	if signals.Diagnostics.AppliedUnknownPolicies["guarded"] != string(config.RuleOnUnknownFailRequest) {
		t.Fatalf("applied policies = %v", signals.Diagnostics.AppliedUnknownPolicies)
	}
}

func TestPIIToolResultMatchReachesDecisionEngine(t *testing.T) {
	classifier := &Classifier{Config: &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Strategy: "priority",
			Decisions: []config.Decision{{
				Name: "safe-route",
				Rules: config.RuleNode{Type: config.SignalTypePII, Name: "tool_data"},
			}},
		},
	}}

	result, err := classifier.EvaluateDecisionWithEngine(&SignalResults{
		MatchedPIIRules: []string{"tool_data"},
	})
	if err != nil {
		t.Fatalf("EvaluateDecisionWithEngine() error = %v", err)
	}
	if result == nil || result.Decision.Name != "safe-route" {
		t.Fatalf("result = %#v, want safe-route decision", result)
	}
}

func TestPIIToolResultErrorCanMatchConfiguredUnknownPolicy(t *testing.T) {
	classifier := &Classifier{Config: &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Strategy: "priority",
			Decisions: []config.Decision{{
				Name: "fail-closed",
				Rules: config.RuleNode{
					Type:      config.SignalTypePII,
					Name:      "tool_data",
					OnUnknown: config.RuleOnUnknownMatch,
				},
			}},
		},
	}}

	result, err := classifier.EvaluateDecisionWithEngine(&SignalResults{
		SignalErrors: map[string]string{"pii:tool_data": piiEvaluationIncompleteCode},
	})
	if err != nil {
		t.Fatalf("EvaluateDecisionWithEngine() error = %v", err)
	}
	if result == nil || result.Decision.Name != "fail-closed" {
		t.Fatalf("result = %#v, want fail-closed decision", result)
	}
}

func TestPIIToolResultErrorCanFailRequestWithUnknownPolicy(t *testing.T) {
	classifier := &Classifier{Config: &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Strategy: "priority",
			Decisions: []config.Decision{{
				Name: "strict-route",
				Rules: config.RuleNode{
					Type:      config.SignalTypePII,
					Name:      "tool_data",
					OnUnknown: config.RuleOnUnknownFailRequest,
				},
			}},
		},
	}}

	_, err := classifier.EvaluateDecisionWithEngine(&SignalResults{
		SignalErrors: map[string]string{"pii:tool_data": piiEvaluationFailedCode},
	})
	if !errors.Is(err, decision.ErrDecisionUnresolved) {
		t.Fatalf("error = %v, want ErrDecisionUnresolved", err)
	}
}
