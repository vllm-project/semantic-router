package decision

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func classifierObservationForTest(version string, availability SignalObservationAvailability, score float64) SignalObservation {
	return SignalObservation{
		Version: version, SignalType: config.SignalTypeClassifier, Name: "risk",
		Availability: availability,
		Provenance:   SignalObservationProvenance{Evaluator: config.SignalTypeClassifier, Backend: "local"},
		PayloadType:  config.RemoteClassifierContractLabelDistribution,
		Payload: &LabelDistributionObservation{
			Scores:        map[string]float64{"SAFE": 1 - score, "RISKY": score},
			MatchedLabels: []string{"RISKY"},
		},
	}
}

func TestClassifierConditionPrefersTypedObservation(t *testing.T) {
	threshold := 0.8
	engine := NewDecisionEngine(nil, nil, nil, []config.Decision{
		{Name: "typed", Priority: 100, Rules: config.RuleNode{
			Type: config.SignalTypeClassifier, Name: "risk", Label: "RISKY",
			Predicate: &config.NumericPredicate{GTE: &threshold},
		}},
		{Name: "fallback", Priority: 1, Rules: config.RuleNode{Operator: "AND"}},
	}, config.RoutingStrategyPriority)
	signals := &SignalMatches{
		Observations: []SignalObservation{classifierObservationForTest(SignalObservationVersionV1, SignalObservationAvailable, 0.9)},
		// Contradictory legacy fields prove the versioned handoff owns the result.
		ClassifierRules: []string{"risk:SAFE"},
		SignalValues:    map[string]float64{"classifier:risk:RISKY": 0.1},
	}

	result, err := engine.EvaluateDecisionsWithSignals(signals)
	if err != nil {
		t.Fatalf("EvaluateDecisionsWithSignals() error = %v", err)
	}
	if result == nil || result.Decision.Name != "typed" {
		t.Fatalf("decision = %+v, want typed", result)
	}
}

func TestClassifierObservationVersionFailsClosed(t *testing.T) {
	threshold := 0.8
	engine := NewDecisionEngine(nil, nil, nil, []config.Decision{
		{Name: "typed", Priority: 100, Rules: config.RuleNode{
			Type: config.SignalTypeClassifier, Name: "risk", Label: "RISKY",
			Predicate: &config.NumericPredicate{GTE: &threshold},
		}},
		{Name: "fallback", Priority: 1, Rules: config.RuleNode{Operator: "AND"}},
	}, config.RoutingStrategyPriority)
	signals := &SignalMatches{
		Observations: []SignalObservation{classifierObservationForTest("v2", SignalObservationAvailable, 0.9)},
		SignalValues: map[string]float64{"classifier:risk:RISKY": 1},
	}

	result, traces, _, err := engine.EvaluateDecisionsWithTraceAndDiagnostics(signals)
	if err != nil {
		t.Fatalf("EvaluateDecisionsWithTraceAndDiagnostics() error = %v", err)
	}
	if result == nil || result.Decision.Name != "fallback" {
		t.Fatalf("decision = %+v, want fallback", result)
	}
	if len(traces) == 0 || traces[0].RootTrace.SignalError != observationUnsupportedVersion {
		t.Fatalf("typed trace = %+v, want %q", traces, observationUnsupportedVersion)
	}
}

func TestUnavailableClassifierObservationFailsClosed(t *testing.T) {
	observation := classifierObservationForTest(SignalObservationVersionV1, SignalObservationUnavailable, 0.9)
	observation.ErrorCode = "classifier_evaluation_timeout"
	threshold := 0.8
	engine := NewDecisionEngine(nil, nil, nil, []config.Decision{
		{Name: "typed", Priority: 100, Rules: config.RuleNode{
			Type: config.SignalTypeClassifier, Name: "risk", Label: "RISKY",
			Predicate: &config.NumericPredicate{GTE: &threshold},
		}},
		{Name: "fallback", Priority: 1, Rules: config.RuleNode{Operator: "AND"}},
	}, config.RoutingStrategyPriority)

	result, err := engine.EvaluateDecisionsWithSignals(&SignalMatches{
		Observations: []SignalObservation{observation},
		SignalValues: map[string]float64{"classifier:risk:RISKY": 1},
	})
	if err != nil {
		t.Fatalf("EvaluateDecisionsWithSignals() error = %v", err)
	}
	if result == nil || result.Decision.Name != "fallback" {
		t.Fatalf("decision = %+v, want fallback", result)
	}
}
