package classification

import (
	"context"
	"math"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type safetyWindowFixture struct {
	result labelClassification
	calls  int
}

func (f *safetyWindowFixture) Classify(context.Context, string) (labelClassification, error) {
	f.calls++
	return f.result, nil
}

func TestSafetyWindowAggregationPreservesDistributionsAndSharedCalls(t *testing.T) {
	binary := &safetyWindowFixture{result: labelClassification{ScoreWindows: []map[string]float64{
		{"safe": .125, "a": .75, "b": .125},
		{"safe": .125, "a": .125, "b": .75},
	}}}
	hazard := &safetyWindowFixture{result: labelClassification{ScoreWindows: []map[string]float64{
		{"privacy": .4, "violence": .2}, {"privacy": .1, "violence": .9},
	}}}
	rules := []config.SafetyRule{
		{Name: "risk", Labels: []string{"safe", "a", "b"}, UnsafeLabels: []string{"a", "b"}, Threshold: .875},
		{Name: "above", Labels: []string{"safe", "a", "b"}, UnsafeLabels: []string{"a", "b"}, Threshold: .9},
		{
			Name: "violent", Labels: []string{"safe", "a", "b"}, UnsafeLabels: []string{"a", "b"}, Threshold: .875,
			Hazard: &config.SafetyHazardRule{Labels: []string{"privacy", "violence"}, Categories: []string{"violence"}, Threshold: .9},
		},
	}
	cfg := &config.RouterConfig{}
	cfg.SafetyRules = rules
	classifier := &Classifier{Config: cfg, safetyClassifiers: map[string]*safetyDetector{}}
	used := map[string]bool{}
	for _, rule := range rules {
		classifier.safetyClassifiers[rule.Name] = &safetyDetector{binary: binary, hazard: hazard, binaryKey: "binary", hazardKey: "hazard"}
		used["safety:"+rule.Name] = true
	}
	results := &SignalResults{SignalConfidences: map[string]float64{}, SignalValues: map[string]float64{}, SignalErrors: map[string]string{}, Metrics: &SignalMetricsCollection{}}
	classifier.evaluateSafetySignals(context.Background(), results, &sync.Mutex{}, "complete input", used)
	if binary.calls != 1 || hazard.calls != 1 {
		t.Fatalf("shared calls binary=%d hazard=%d", binary.calls, hazard.calls)
	}
	if len(results.MatchedSafetyRules) != 2 || results.MatchedSafetyRules[0] != "risk" || results.MatchedSafetyRules[1] != "violent" {
		t.Fatalf("window decisions: %v errors: %v", results.MatchedSafetyRules, results.SignalErrors)
	}
	if got := results.SignalValues["safety:risk"]; math.Abs(got-.875) > 1e-12 {
		t.Fatalf("invented softmax mass: %v", got)
	}
}

func TestSafetyRejectsIncompleteOrInvalidWindowScores(t *testing.T) {
	labels := []string{"safe", "unsafe"}
	for _, scores := range []map[string]float64{nil, {"safe": 1}, {"safe": .5, "unsafe": math.NaN()}, {"safe": -.1, "unsafe": 1.1}} {
		result := labelClassification{ScoreWindows: []map[string]float64{{"safe": .9, "unsafe": .1}, scores}}
		if safetyScoresFinite(labels, result) {
			t.Fatalf("invalid later window accepted: %v", scores)
		}
	}
	result := labelClassification{Scores: map[string]float64{"safe": .25, "unsafe": .75}}
	if !safetyScoresFinite(labels, result) || aggregateSafetyWindows(result, []string{"unsafe"}, selectedSafetyScore) != .75 {
		t.Fatal("whole-input path changed")
	}
}
