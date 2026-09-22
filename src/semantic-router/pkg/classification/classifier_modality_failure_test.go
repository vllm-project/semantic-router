package classification

import (
	"context"
	"strings"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
)

// This classifier owns no native model in this fixture-free test. Its missing
// provider must propagate unknown instead of manufacturing an AR match.
func TestModalityClassifierFailureIsUnknownInsteadOfAR(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.ModalityRules = []config.ModalityRule{{Name: "AR"}, {Name: "DIFFUSION"}, {Name: "BOTH"}}
	cfg.ModalityDetector.Method = config.ModalityDetectionClassifier
	classifier := &Classifier{Config: cfg}
	results := newMetricScopeResults()
	var mu sync.Mutex
	classifier.evaluateModalitySignal(context.Background(), results, &mu, strings.Repeat("word ", 40000))
	if len(results.MatchedModalityRules) != 0 || len(results.SignalErrors) != 3 {
		t.Fatalf("failed model fabricated a verdict or hid failures: matches=%v errors=%v", results.MatchedModalityRules, results.SignalErrors)
	}
	signals := &decision.SignalMatches{SignalErrors: results.SignalErrors}
	for _, policy := range []config.UnknownPolicy{"", config.RuleOnUnknownMatch} {
		engine := decision.NewDecisionEngine(nil, nil, nil, []config.Decision{{
			Name: "route", Rules: config.RuleNode{
				Operator: "AND", OnUnknown: policy,
				Conditions: []config.RuleNode{{Type: config.SignalTypeModality, Name: "AR"}},
			},
		}}, "")
		result, err := engine.EvaluateDecisionsWithSignals(signals)
		if err != nil {
			t.Fatal(err)
		}
		matched := result != nil && result.Decision != nil
		if matched != (policy == config.RuleOnUnknownMatch) {
			t.Fatalf("unknown policy %q resolved failure incorrectly: matched=%v", policy, matched)
		}
	}
}

func TestModalityHybridRetainsExplicitKeywordFallback(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.ModalityRules = []config.ModalityRule{{Name: "AR"}, {Name: "DIFFUSION"}}
	cfg.ModalityDetector.Method = config.ModalityDetectionHybrid
	cfg.ModalityDetector.Keywords = []string{"draw a picture"}
	classifier := &Classifier{Config: cfg}
	results := newMetricScopeResults()
	var mu sync.Mutex
	classifier.evaluateModalitySignal(context.Background(), results, &mu, "draw a picture "+strings.Repeat("word ", 40000))
	if len(results.SignalErrors) != 0 || len(results.MatchedModalityRules) != 1 || results.MatchedModalityRules[0] != "DIFFUSION" {
		t.Fatalf("explicit hybrid keyword fallback changed: matches=%v errors=%v", results.MatchedModalityRules, results.SignalErrors)
	}
}
