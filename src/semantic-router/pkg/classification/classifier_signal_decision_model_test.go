package classification

import (
	"context"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

type fakeDecider struct {
	mu       sync.Mutex
	calls    map[string]int
	requests map[string]modelservice.Request
	answers  map[string]modelservice.Answer
	failures map[string]error
}

func (f *fakeDecider) Decide(_ context.Context, deployment string, request modelservice.Request) (modelservice.Response, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.calls[deployment]++
	f.requests[deployment] = request
	if err := f.failures[deployment]; err != nil {
		return modelservice.Response{}, err
	}
	answers := map[string]modelservice.Answer{}
	for _, question := range request.Questions {
		if answer, ok := f.answers[question.ID]; ok {
			answers[question.ID] = answer
		}
	}
	return modelservice.Response{Answers: answers}, nil
}

func decisionRulesForTest() []config.DecisionSignalRule {
	high := 0.9
	level := 1.0
	return []config.DecisionSignalRule{
		{Name: "hard", Deployment: "kai", Question: config.DecisionQuestion{Type: "noul", Instructions: "Hard?"}},
		{Name: "strict", Deployment: "kai", Predicate: &config.NumericPredicate{GTE: &high}, Question: config.DecisionQuestion{Type: "noul", Instructions: "Hard?"}},
		{Name: "kind", Deployment: "kai", Question: config.DecisionQuestion{Type: "choice", Instructions: "Kind?", Choices: []config.DecisionChoice{{Key: "code"}, {Key: "math"}}}},
		{Name: "level", Deployment: "kai", Predicate: &config.NumericPredicate{GTE: &level}, Question: config.DecisionQuestion{Type: "score", Instructions: "Level?", Levels: []string{"a", "b", "c"}}},
		{Name: "broken", Deployment: "kai", Question: config.DecisionQuestion{Type: "noul", Instructions: "?"}},
		{Name: "remote", Deployment: "vega", Question: config.DecisionQuestion{Type: "noul", Instructions: "?"}},
	}
}

func newDecisionTestClassifier(decider modelservice.Decider) *Classifier {
	cfg := &config.RouterConfig{}
	cfg.DecisionRules = decisionRulesForTest()
	classifier := &Classifier{Config: cfg}
	classifier.SetDecisionDecider(decider)
	return classifier
}

func newSignalResults() *SignalResults {
	return &SignalResults{
		SignalConfidences: map[string]float64{},
		SignalValues:      map[string]float64{},
		SignalErrors:      map[string]string{},
		Metrics:           &SignalMetricsCollection{},
	}
}

func TestDecisionSignalsBatchPerDeploymentAndMatch(t *testing.T) {
	decider := &fakeDecider{
		calls:    map[string]int{},
		requests: map[string]modelservice.Request{},
		failures: map[string]error{"vega": modelservice.ErrUnavailable},
		answers: map[string]modelservice.Answer{
			"hard":   {Type: "noul", Noul: 0.7},
			"strict": {Type: "noul", Noul: 0.7},
			"kind":   {Type: "choice", Choice: "math", Probabilities: map[string]float64{"code": 0.2, "math": 0.8}},
			"level":  {Type: "score", Score: 1.4, Probabilities: map[string]float64{"0": 0.1, "1": 0.4, "2": 0.5}},
			"broken": {Type: "noul", Error: "max_length_exceeded"},
		},
	}
	classifier := newDecisionTestClassifier(decider)
	results := newSignalResults()
	used := map[string]bool{}
	for _, rule := range decisionRulesForTest() {
		used["decision:"+rule.Name] = true
	}
	classifier.evaluateDecisionModelSignals(context.Background(), results, &sync.Mutex{}, "merge two lists", used)

	if decider.calls["kai"] != 1 || decider.calls["vega"] != 1 {
		t.Fatalf("one call per deployment expected, got %v", decider.calls)
	}
	if got := len(decider.requests["kai"].Questions); got != 5 || decider.requests["kai"].State != "merge two lists" {
		t.Fatalf("kai request = %+v", decider.requests["kai"])
	}
	want := map[string]bool{"hard": true, "kind:math": true, "level": true}
	for _, name := range results.MatchedDecisionRules {
		if !want[name] {
			t.Fatalf("unexpected match %q in %v", name, results.MatchedDecisionRules)
		}
		delete(want, name)
	}
	if len(want) != 0 {
		t.Fatalf("missing matches %v (got %v)", want, results.MatchedDecisionRules)
	}
	if results.SignalValues["decision:kind:code"] != 0.2 || results.SignalValues["decision:kind"] != 0.8 {
		t.Fatalf("choice values = %v", results.SignalValues)
	}
	if results.SignalValues["decision:level"] != 1.4 || results.SignalConfidences["decision:level"] != 0.5 {
		t.Fatalf("score value/confidence = %v / %v", results.SignalValues["decision:level"], results.SignalConfidences["decision:level"])
	}
	if results.SignalErrors["decision:broken"] != "decision_max_length_exceeded" {
		t.Fatalf("per-question error = %q", results.SignalErrors["decision:broken"])
	}
	if results.SignalErrors["decision:remote"] != "decision_unavailable" {
		t.Fatalf("unavailable deployment must leave its signal unknown, got %q", results.SignalErrors["decision:remote"])
	}
	if _, failed := results.SignalErrors["decision:hard"]; failed {
		t.Fatal("answered signals carry no error")
	}
}

func TestUnusedDecisionSignalsAreNotAsked(t *testing.T) {
	decider := &fakeDecider{calls: map[string]int{}, requests: map[string]modelservice.Request{}, answers: map[string]modelservice.Answer{}}
	classifier := newDecisionTestClassifier(decider)
	classifier.evaluateDecisionModelSignals(context.Background(), newSignalResults(), &sync.Mutex{}, "x", map[string]bool{"decision:remote": true})
	if decider.calls["kai"] != 0 || decider.calls["vega"] != 1 {
		t.Fatalf("only used signals are asked: %v", decider.calls)
	}
}
