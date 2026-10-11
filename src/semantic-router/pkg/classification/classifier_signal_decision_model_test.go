package classification

import (
	"context"
	"slices"
	"strings"
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
	classifier.evaluateDecisionModelSignals(context.Background(), results, &sync.Mutex{}, "merge two lists", "merge two lists", nil, used)

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

func TestSetAndSpanAnswersMatchLabelsAndPublishEveryProbability(t *testing.T) {
	strict, low := 0.95, 0.5
	labels := []config.DecisionChoice{{Key: "billing"}, {Key: "shipping"}, {Key: "refunds"}}
	set := config.DecisionSignalRule{Name: "topics", Question: config.DecisionQuestion{Type: config.DecisionQuestionSet, Labels: labels}}
	setAnswer := modelservice.Answer{Type: "set", Probabilities: map[string]float64{"billing": 0.9, "shipping": 0.2, "refunds": 0.6}, Selected: []string{"billing", "refunds"}}
	span := config.DecisionSignalRule{Name: "places", Question: config.DecisionQuestion{Type: config.DecisionQuestionSpan, Labels: []config.DecisionChoice{{Key: "city"}, {Key: "country"}}}}
	spanAnswer := modelservice.Answer{Type: "span", Noul: 0.97, Spans: []modelservice.Span{
		{Label: "city", Text: "Paris", Probability: 0.97}, {Label: "city", Text: "Lyon", Probability: 0.4}, {Label: "street", Text: "Rue", Probability: 0.9},
	}}
	cases := []struct {
		name      string
		rule      config.DecisionSignalRule
		predicate *config.NumericPredicate
		answer    modelservice.Answer
		want      []string
	}{
		{name: "set: the labels the model selected", rule: set, answer: setAnswer, want: []string{"topics:billing", "topics:refunds"}},
		{name: "set: labels whose probability meets the predicate", rule: set, predicate: &config.NumericPredicate{GTE: &strict}, answer: setAnswer},
		{name: "set: a predicate replaces the selection", rule: set, predicate: &config.NumericPredicate{LT: &low}, answer: setAnswer, want: []string{"topics:shipping"}},
		{name: "span: labels with a span", rule: span, answer: spanAnswer, want: []string{"places:city"}},
		{name: "span: a span meeting the predicate", rule: span, predicate: &config.NumericPredicate{LTE: &low}, answer: spanAnswer, want: []string{"places:city"}},
		{name: "span: no span meets it", rule: span, predicate: &config.NumericPredicate{GTE: &strict}, answer: modelservice.Answer{Type: "span", Spans: spanAnswer.Spans[1:]}},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			test.rule.Predicate = test.predicate
			results := newSignalResults()
			if got := applyDecisionAnswer(results, test.rule, test.answer); !slices.Equal(got, test.want) {
				t.Fatalf("matched %v, want %v", got, test.want)
			}
		})
	}
	results := newSignalResults()
	applyDecisionAnswer(results, set, setAnswer)
	if results.SignalValues["decision:topics:shipping"] != 0.2 || results.SignalValues["decision:topics"] != 0.9 {
		t.Fatalf("set values = %v", results.SignalValues)
	}
	results = newSignalResults()
	applyDecisionAnswer(results, span, spanAnswer)
	if results.SignalValues["decision:places:city"] != 0.97 || results.SignalValues["decision:places:country"] != 0 || results.SignalValues["decision:places"] != 0.97 {
		t.Fatalf("span values = %v", results.SignalValues)
	}
	if _, undeclared := results.SignalValues["decision:places:street"]; undeclared {
		t.Fatal("a span of an undeclared label is not published")
	}
}

func TestUnusedDecisionSignalsAreNotAsked(t *testing.T) {
	decider := &fakeDecider{calls: map[string]int{}, requests: map[string]modelservice.Request{}, answers: map[string]modelservice.Answer{}}
	classifier := newDecisionTestClassifier(decider)
	classifier.evaluateDecisionModelSignals(context.Background(), newSignalResults(), &sync.Mutex{}, "x", "x", nil, map[string]bool{"decision:remote": true})
	if decider.calls["kai"] != 0 || decider.calls["vega"] != 1 {
		t.Fatalf("only used signals are asked: %v", decider.calls)
	}
}

type deadlineDecider struct {
	mu        sync.Mutex
	deadlines map[string]bool
	states    map[string]string
}

func (d *deadlineDecider) Decide(ctx context.Context, deployment string, request modelservice.Request) (modelservice.Response, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	_, bounded := ctx.Deadline()
	d.deadlines[deployment] = bounded
	d.states[deployment] = request.State
	return modelservice.Response{Answers: map[string]modelservice.Answer{}}, nil
}

func TestADecisionQuestionWithoutDeploymentJoinsTheDecisionModelsCall(t *testing.T) {
	decider := &deadlineDecider{deadlines: map[string]bool{}, states: map[string]string{}}
	defaults := config.DefaultGlobalConfig()
	cfg := &defaults
	cfg.DecisionRules = []config.DecisionSignalRule{
		{Name: "tools", Question: config.DecisionQuestion{Type: "noul", Instructions: "Tools?"}},
		{Name: "hard", Deployment: "kai", Question: config.DecisionQuestion{Type: "noul", Instructions: "Hard?"}},
	}
	classifier := &Classifier{Config: cfg}
	classifier.SetDecisionDecider(decider)
	used := map[string]bool{"decision:tools": true, "decision:hard": true}
	classifier.evaluateDecisionModelSignals(context.Background(), newSignalResults(), &sync.Mutex{}, "short", "the whole text", nil, used)

	if decider.states[config.DefaultDecisionDeployment] != "the whole text" || decider.states["kai"] != "the whole text" {
		t.Fatalf("every decision question reads the same authored input: %v", decider.states)
	}
	if decider.deadlines[config.DefaultDecisionDeployment] || !decider.deadlines["kai"] {
		t.Fatalf("only a declared deployment's call takes the default timeout: %v", decider.deadlines)
	}
}

type recordingDecider struct {
	mu       sync.Mutex
	requests []modelservice.Request
}

func (r *recordingDecider) Decide(_ context.Context, _ string, request modelservice.Request) (modelservice.Response, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.requests = append(r.requests, request)
	return modelservice.Response{Answers: map[string]modelservice.Answer{}}, nil
}

// A follow-up such as "now make it shorter" says nothing about its task;
// a question with prior_user_turns reads it after the turns it follows,
// and only questions that read as many turns share its call.
func TestDecisionQuestionsReadTheirPriorUserTurns(t *testing.T) {
	decider := &recordingDecider{}
	cfg := &config.RouterConfig{}
	cfg.DecisionRules = []config.DecisionSignalRule{
		{Name: "alone", Deployment: "kai", Question: config.DecisionQuestion{Type: "noul", Instructions: "?"}},
		{Name: "one", Deployment: "kai", PriorUserTurns: 1, Question: config.DecisionQuestion{Type: "noul", Instructions: "?"}},
		{Name: "many", Deployment: "kai", PriorUserTurns: 5, Question: config.DecisionQuestion{Type: "noul", Instructions: "?"}},
	}
	classifier := &Classifier{Config: cfg}
	classifier.SetDecisionDecider(decider)
	used := map[string]bool{"decision:alone": true, "decision:one": true, "decision:many": true}
	latest := strings.Repeat("ü", semanticSignalUnitLimit-10)
	prior := []string{"Write a poem about rain.", latest}

	classifier.evaluateDecisionModelSignals(context.Background(), newSignalResults(), &sync.Mutex{}, "Now make it shorter.", "Now make it shorter.", prior, used)

	states := map[string]string{}
	for _, request := range decider.requests {
		for _, question := range request.Questions {
			states[question.ID] = request.State
		}
	}
	want := map[string]string{
		"alone": "Now make it shorter.",
		"one":   latest + "\n\nNow make it shorter.",
		// five asked, two exist: oldest first, within one shared budget
		"many": "Write a po\n\n" + latest + "\n\nNow make it shorter.",
	}
	for name, state := range want {
		if states[name] != state {
			t.Errorf("%s: state %q, want %q", name, states[name], state)
		}
	}
	if len(decider.requests) != 3 {
		t.Fatalf("questions reading different turns need separate calls, got %d", len(decider.requests))
	}
}

func TestPriorUserTurnsWithoutHistoryShareTheCurrentTurnsCall(t *testing.T) {
	decider := &recordingDecider{}
	cfg := &config.RouterConfig{}
	cfg.DecisionRules = []config.DecisionSignalRule{
		{Name: "alone", Deployment: "kai", Question: config.DecisionQuestion{Type: "noul", Instructions: "?"}},
		{Name: "one", Deployment: "kai", PriorUserTurns: 1, Question: config.DecisionQuestion{Type: "noul", Instructions: "?"}},
	}
	classifier := &Classifier{Config: cfg}
	classifier.SetDecisionDecider(decider)
	used := map[string]bool{"decision:alone": true, "decision:one": true}

	classifier.evaluateDecisionModelSignals(context.Background(), newSignalResults(), &sync.Mutex{}, "hello", "hello", nil, used)

	if len(decider.requests) != 1 || decider.requests[0].State != "hello" || len(decider.requests[0].Questions) != 2 {
		t.Fatalf("a first turn has no earlier turns, so both questions share one call: %+v", decider.requests)
	}
}
