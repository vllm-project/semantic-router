package classification

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func stagedDecisionClassifier() *Classifier {
	cfg := &config.RouterConfig{}
	cfg.JailbreakRules = []config.JailbreakRule{
		{Name: "unsafe_completion", Threshold: 0.5, Direction: config.SignalDirectionResponse},
	}
	cfg.Decisions = []config.Decision{
		{
			// Composed with a request signal under OR, this would win on its
			// own priority off the request-stage half of the rule alone.
			Name:     "escalate-on-unsafe-output",
			Priority: 100,
			Rules: config.RuleCombination{
				Operator: "OR",
				Conditions: []config.RuleCondition{
					{Type: config.SignalTypeDomain, Name: "business"},
					{Type: config.SignalTypeJailbreak, Name: "unsafe_completion"},
				},
			},
		},
		{
			Name:     "business",
			Priority: 10,
			Rules:    config.RuleCombination{Type: config.SignalTypeDomain, Name: "business"},
		},
	}
	return &Classifier{Config: cfg}
}

// A decision that reads a response-direction rule must not be selected while
// the request is still being routed: the model has not answered, so the signal
// was never evaluated.
func TestEvaluateDecisionSkipsResponseStageDecisionsAtRequestTime(t *testing.T) {
	classifier := stagedDecisionClassifier()

	result, err := classifier.EvaluateDecisionWithEngine(&SignalResults{
		MatchedDomainRules: []string{"business"},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result == nil || result.Decision.Name != "business" {
		t.Fatalf("request-stage evaluation selected %+v, want business", result)
	}
}

// Once the response-direction rule has been scored, the same decision resolves
// from the request-stage matches and the response observation together, and
// the request-stage decision is not re-selected.
func TestEvaluateResponseStageDecisionReadsResponseObservation(t *testing.T) {
	classifier := stagedDecisionClassifier()

	result, err := classifier.EvaluateResponseStageDecision(&SignalResults{
		MatchedDomainRules:    []string{"business"},
		MatchedJailbreakRules: []string{"unsafe_completion"},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result == nil || result.Decision.Name != "escalate-on-unsafe-output" {
		t.Fatalf("response-stage evaluation selected %+v, want escalate-on-unsafe-output", result)
	}

	clean, err := classifier.EvaluateResponseStageDecision(&SignalResults{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if clean != nil {
		t.Fatalf("a response-stage decision must not match without evidence, got %+v", clean)
	}
}

func TestEvaluateResponseStageDecisionWithoutResponseStageDecisions(t *testing.T) {
	classifier := &Classifier{Config: &config.RouterConfig{}}
	classifier.Config.Decisions = []config.Decision{{
		Name:  "business",
		Rules: config.RuleCombination{Type: config.SignalTypeDomain, Name: "business"},
	}}

	result, err := classifier.EvaluateResponseStageDecision(&SignalResults{MatchedDomainRules: []string{"business"}})
	if err != nil {
		t.Fatalf("no response-stage decision is not an error: %v", err)
	}
	if result != nil {
		t.Fatalf("request-stage decisions must not be re-selected at the response stage, got %+v", result)
	}
}
