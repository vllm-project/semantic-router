package classification

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
)

func TestIndependentClassifierRoutesWithoutSafetyAndPreservesUnknown(t *testing.T) {
	labels := []string{"one", "two"}
	threshold := float64(float32(.2))
	rule := config.ClassifierSignalRule{Name: "risk", Type: "local", Labels: labels}
	condition := config.RuleNode{Type: "classifier", Name: "risk", Label: "one", OnUnknown: config.RuleOnUnknownFailRequest}
	engine := decision.NewDecisionEngine(nil, nil, nil, []config.Decision{{Name: "independent-route", Rules: condition}}, config.RoutingStrategyPriority)
	cases := []struct {
		name               string
		scores, thresholds map[string]float64
		err                error
		matches            int
		unknown            bool
	}{
		{"both including exact FP32 tie", map[string]float64{"one": threshold, "two": .9}, map[string]float64{"one": threshold, "two": .7}, nil, 2, false},
		{"neither", map[string]float64{"one": .1, "two": .6}, map[string]float64{"one": threshold, "two": .7}, nil, 0, false},
		{"categorical unchanged", map[string]float64{"one": .2, "two": .8}, nil, nil, 1, false},
		{"incomplete scan", nil, nil, errors.New("missing tail"), 0, true},
		{"missing score", map[string]float64{"one": .9}, map[string]float64{"one": threshold, "two": .7}, nil, 0, true},
		{"missing threshold", map[string]float64{"one": .9, "two": .9}, map[string]float64{"one": threshold}, nil, 0, true},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			classifier := &Classifier{Config: &config.RouterConfig{IntelligentRouting: config.IntelligentRouting{Signals: config.Signals{ClassifierRules: []config.ClassifierSignalRule{rule}}}}, genericClassifiers: map[string]labelClassifier{"risk": fakeLabelClassifier{result: labelClassification{Scores: test.scores, Thresholds: test.thresholds, PolicyTrace: &ClassifierRuleMetrics{PolicySHA256: strings.Repeat("a", 64), Provider: "candle", Device: "cpu", Precision: "float32", InputTokens: 8, ProcessedTokens: 8, Windows: [][2]int{{0, 3}, {2, 5}, {4, 6}}, WindowBatchSize: 1}}, err: test.err}}}
			results := &SignalResults{Metrics: &SignalMetricsCollection{}, SignalValues: map[string]float64{}, SignalConfidences: map[string]float64{}, SignalErrors: map[string]string{}}
			classifier.evaluateGenericClassifierSignals(results, &sync.Mutex{}, "neutral", map[string]bool{"classifier:risk": true}, context.Background())
			if len(results.MatchedClassifierRules) != test.matches {
				t.Fatalf("matched %v", results.MatchedClassifierRules)
			}
			signal := &decision.SignalMatches{ClassifierRules: results.MatchedClassifierRules, SignalValues: results.SignalValues, SignalConfidences: results.SignalConfidences, SignalErrors: results.SignalErrors}
			routed, diagnostics, err := engine.EvaluateDecisionsWithDiagnostics(signal)
			if (err != nil) != test.unknown {
				t.Fatalf("unknown lost: %v %+v", err, diagnostics)
			}
			if test.matches == 2 && (routed == nil || routed.Decision.Name != "independent-route") {
				t.Fatalf("standalone scores did not route: %+v %v", routed, err)
			}
			if test.unknown && len(results.SignalValues) != 0 {
				t.Fatal("partial scores leaked into a decision")
			}
			if !test.unknown {
				raw, err := json.Marshal(results.Metrics)
				if err != nil || !strings.Contains(string(raw), `"policy_sha256"`) || !strings.Contains(string(raw), `"content_token_windows"`) {
					t.Fatalf("existing trace lost execution metadata: %s %v", raw, err)
				}
			}
		})
	}
	// A second recipe owns a separate classifier/result map even with the same
	// symbol. It cannot inherit the first recipe's positive labels or scores.
	second := &Classifier{Config: &config.RouterConfig{IntelligentRouting: config.IntelligentRouting{Signals: config.Signals{ClassifierRules: []config.ClassifierSignalRule{rule}}}}, genericClassifiers: map[string]labelClassifier{"risk": fakeLabelClassifier{err: errors.New("recipe-local unavailable")}}}
	results := &SignalResults{Metrics: &SignalMetricsCollection{}, SignalValues: map[string]float64{}, SignalConfidences: map[string]float64{}, SignalErrors: map[string]string{}}
	second.evaluateGenericClassifierSignals(results, &sync.Mutex{}, "neutral", map[string]bool{"classifier:risk": true}, context.Background())
	if len(results.MatchedClassifierRules) != 0 || len(results.SignalValues) != 0 {
		t.Fatal("foreign recipe matches reused")
	}
	if _, _, err := engine.EvaluateDecisionsWithDiagnostics(&decision.SignalMatches{SignalErrors: results.SignalErrors}); err == nil {
		t.Fatal("foreign recipe failure did not remain unknown")
	}
}
