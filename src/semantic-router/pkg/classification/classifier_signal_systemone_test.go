package classification

import (
	"context"
	"slices"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

type fakeDecider struct {
	mu       sync.Mutex
	calls    map[string]int
	requests map[string]modelservice.Request
	asked    []modelservice.Request
	answers  map[string]modelservice.Answer
	failures map[string]error
}

func newFakeDecider(answers map[string]modelservice.Answer) *fakeDecider {
	return &fakeDecider{calls: map[string]int{}, requests: map[string]modelservice.Request{}, answers: answers, failures: map[string]error{}}
}

func (f *fakeDecider) Decide(_ context.Context, deployment string, request modelservice.Request) (modelservice.Response, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.calls[deployment]++
	f.requests[deployment] = request
	f.asked = append(f.asked, request)
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

func systemOneRulesForTest() []config.SystemOneRule {
	high := 0.9
	level := 1.0
	choices := []config.SystemOneOption{{Key: "code"}, {Key: "math"}}
	return []config.SystemOneRule{
		{Name: "hard", Deployment: "kai", Question: config.SystemOneQuestion{Type: "noul", Instructions: "Hard?"}},
		{Name: "strict", Deployment: "kai", Predicate: &config.NumericPredicate{GTE: &high}, Question: config.SystemOneQuestion{Type: "noul", Instructions: "Hard?"}},
		{Name: "kind", Deployment: "kai", Question: config.SystemOneQuestion{Type: "choice", Instructions: "Kind?", Choices: choices}},
		{Name: "mathy", Deployment: "kai", Labels: []string{"math"}, Question: config.SystemOneQuestion{Type: "choice", Instructions: "Kind?", Choices: choices}},
		{Name: "coder", Deployment: "kai", Labels: []string{"code"}, Question: config.SystemOneQuestion{Type: "choice", Instructions: "Kind?", Choices: choices}},
		{Name: "level", Deployment: "kai", Predicate: &config.NumericPredicate{GTE: &level}, Question: config.SystemOneQuestion{Type: "score", Instructions: "Level?", Levels: []string{"a", "b", "c"}}},
		{Name: "broken", Deployment: "kai", Question: config.SystemOneQuestion{Type: "noul", Instructions: "?"}},
		{Name: "remote", Deployment: "vega", Question: config.SystemOneQuestion{Type: "noul", Instructions: "?"}},
	}
}

func newSystemOneTestClassifier(decider modelservice.Decider, rules ...config.SystemOneRule) *Classifier {
	cfg := &config.RouterConfig{}
	cfg.SystemOneRules = systemOneRulesForTest()
	if len(rules) > 0 {
		cfg.SystemOneRules = rules
	}
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

func usedRules(rules []config.SystemOneRule) map[string]bool {
	used := map[string]bool{}
	for _, rule := range rules {
		used["systemone:"+rule.Name] = true
	}
	return used
}

func TestSystemOneSignalsBatchPerDeploymentAndMatch(t *testing.T) {
	decider := newFakeDecider(map[string]modelservice.Answer{
		"systemone:hard":   {Type: "noul", Noul: 0.7},
		"systemone:strict": {Type: "noul", Noul: 0.7},
		"systemone:kind":   {Type: "choice", Choice: "math", Probabilities: map[string]float64{"code": 0.2, "math": 0.8}},
		"systemone:mathy":  {Type: "choice", Choice: "math", Probabilities: map[string]float64{"code": 0.2, "math": 0.8}},
		"systemone:coder":  {Type: "choice", Choice: "math", Probabilities: map[string]float64{"code": 0.2, "math": 0.8}},
		"systemone:level":  {Type: "score", Score: 1.4, Probabilities: map[string]float64{"0": 0.1, "1": 0.4, "2": 0.5}},
		"systemone:broken": {Type: "noul", Error: "max_length_exceeded"},
	})
	decider.failures["vega"] = modelservice.ErrUnavailable
	classifier := newSystemOneTestClassifier(decider)
	results := newSignalResults()
	classifier.evaluateSystemOneSignals(context.Background(), results, &sync.Mutex{}, systemOneReading{request: "merge two lists"}, usedRules(systemOneRulesForTest()))

	if decider.calls["kai"] != 1 || decider.calls["vega"] != 1 {
		t.Fatalf("one call per deployment expected, got %v", decider.calls)
	}
	asked := decider.requests["kai"]
	if len(asked.Questions) != 7 || asked.State != "merge two lists" || asked.Parts != nil {
		t.Fatalf("kai request = %+v", asked)
	}
	for _, question := range asked.Questions {
		if question.ID[:len("systemone:")] != "systemone:" || !question.Truncate || question.TaskID != config.SignalTypeSystemOne {
			t.Fatalf("a rule's question goes out under the Router's ID and reads the first tokens: %+v", question)
		}
	}
	want := []string{"hard", "kind:math", "mathy:math", "mathy", "coder:math", "level"}
	if !sameMembers(results.MatchedSystemOneRules, want) {
		t.Fatalf("matched %v, want %v", results.MatchedSystemOneRules, want)
	}
	if results.SignalValues["systemone:kind:code"] != 0.2 || results.SignalValues["systemone:kind"] != 0.8 {
		t.Fatalf("choice values = %v", results.SignalValues)
	}
	if results.SignalValues["systemone:level"] != 1.4 || results.SignalConfidences["systemone:level"] != 0.5 {
		t.Fatalf("score value/confidence = %v / %v", results.SignalValues["systemone:level"], results.SignalConfidences["systemone:level"])
	}
	if results.SignalErrors["systemone:broken"] != "systemone_max_length_exceeded" {
		t.Fatalf("per-question error = %q", results.SignalErrors["systemone:broken"])
	}
	if results.SignalErrors["systemone:remote"] != "systemone_unavailable" {
		t.Fatalf("unavailable deployment must leave its signal unknown, got %q", results.SignalErrors["systemone:remote"])
	}
	if _, failed := results.SignalErrors["systemone:hard"]; failed {
		t.Fatal("answered signals carry no error")
	}
}

func sameMembers(got, want []string) bool {
	return len(got) == len(want) && !slices.ContainsFunc(want, func(name string) bool { return !slices.Contains(got, name) })
}

func TestSetAndSpanAnswersMatchCountedLabelsAndPublishEveryProbability(t *testing.T) {
	strict, low := 0.95, 0.5
	labels := []config.SystemOneOption{{Key: "billing"}, {Key: "shipping"}, {Key: "refunds"}}
	set := config.SystemOneRule{Name: "topics", Question: config.SystemOneQuestion{Type: config.SystemOneQuestionSet, Labels: labels}}
	setAnswer := modelservice.Answer{Type: "set", Probabilities: map[string]float64{"billing": 0.9, "shipping": 0.2, "refunds": 0.6}, Selected: []string{"billing", "refunds"}}
	span := config.SystemOneRule{Name: "places", Question: config.SystemOneQuestion{Type: config.SystemOneQuestionSpan, Labels: []config.SystemOneOption{{Key: "city"}, {Key: "country"}}}}
	spanAnswer := modelservice.Answer{Type: "span", Noul: 0.99, Spans: []modelservice.Span{
		{Label: "city", Text: "Paris", Probability: 0.97}, {Label: "city", Text: "Lyon", Probability: 0.4}, {Label: "street", Text: "Rue", Probability: 0.9},
	}}
	cases := []struct {
		name      string
		rule      config.SystemOneRule
		labels    []string
		ignore    []string
		predicate *config.NumericPredicate
		answer    modelservice.Answer
		want      []string
	}{
		{name: "set: the labels the model selected", rule: set, answer: setAnswer, want: []string{"topics", "topics:billing", "topics:refunds"}},
		{name: "set: labels whose probability meets the predicate", rule: set, predicate: &config.NumericPredicate{GTE: &strict}, answer: setAnswer},
		{name: "set: a predicate replaces the selection", rule: set, predicate: &config.NumericPredicate{LT: &low}, answer: setAnswer, want: []string{"topics", "topics:shipping"}},
		{name: "set: the rule counts only its labels", rule: set, labels: []string{"shipping", "refunds"}, answer: setAnswer, want: []string{"topics", "topics:refunds"}},
		{name: "set: no counted label matches", rule: set, labels: []string{"shipping"}, answer: setAnswer},
		{name: "set: ignored labels never count", rule: set, ignore: []string{"billing", "refunds"}, answer: setAnswer},
		{name: "span: labels with a span", rule: span, answer: spanAnswer, want: []string{"places", "places:city"}},
		{name: "span: a span meeting the predicate", rule: span, predicate: &config.NumericPredicate{LTE: &low}, answer: spanAnswer, want: []string{"places", "places:city"}},
		{name: "span: no span meets it", rule: span, predicate: &config.NumericPredicate{GTE: &strict}, answer: modelservice.Answer{Type: "span", Spans: spanAnswer.Spans[1:]}},
		{name: "span: an ignored label", rule: span, ignore: []string{"city"}, answer: spanAnswer},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			test.rule.Predicate, test.rule.Labels, test.rule.IgnoreLabels = test.predicate, test.labels, test.ignore
			results := newSignalResults()
			if got := applySystemOneAnswer(results, test.rule, test.answer); !sameMembers(got, test.want) {
				t.Fatalf("matched %v, want %v", got, test.want)
			}
		})
	}
	results := newSignalResults()
	applySystemOneAnswer(results, set, setAnswer)
	if results.SignalValues["systemone:topics:shipping"] != 0.2 || results.SignalValues["systemone:topics"] != 0.9 {
		t.Fatalf("set values = %v", results.SignalValues)
	}
	counted := set
	counted.IgnoreLabels = []string{"billing"}
	results = newSignalResults()
	applySystemOneAnswer(results, counted, setAnswer)
	if results.SignalValues["systemone:topics"] != 0.6 || results.SignalValues["systemone:topics:billing"] != 0.9 {
		t.Fatalf("the rule's value is the highest counted label, and every label is published: %v", results.SignalValues)
	}
	results = newSignalResults()
	applySystemOneAnswer(results, span, spanAnswer)
	if results.SignalValues["systemone:places:city"] != 0.97 || results.SignalValues["systemone:places:country"] != 0 || results.SignalValues["systemone:places"] != 0.97 {
		t.Fatalf("span values = %v", results.SignalValues)
	}
	if _, undeclared := results.SignalValues["systemone:places:street"]; undeclared {
		t.Fatal("a span of an undeclared label is not published")
	}
}

func TestUnusedSystemOneSignalsAreNotAsked(t *testing.T) {
	decider := newFakeDecider(map[string]modelservice.Answer{})
	classifier := newSystemOneTestClassifier(decider)
	classifier.evaluateSystemOneSignals(context.Background(), newSignalResults(), &sync.Mutex{}, systemOneReading{request: "x"}, map[string]bool{"systemone:remote": true})
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

func TestASystemOneRuleWithoutDeploymentJoinsTheDecisionModelsCall(t *testing.T) {
	decider := &deadlineDecider{deadlines: map[string]bool{}, states: map[string]string{}}
	defaults := config.DefaultGlobalConfig()
	cfg := &defaults
	cfg.SystemOneRules = []config.SystemOneRule{
		{Name: "tools", Question: config.SystemOneQuestion{Type: "noul", Instructions: "Tools?"}},
		{Name: "hard", Deployment: "kai", Question: config.SystemOneQuestion{Type: "noul", Instructions: "Hard?"}},
	}
	classifier := &Classifier{Config: cfg}
	classifier.SetDecisionDecider(decider)
	classifier.evaluateSystemOneSignals(context.Background(), newSignalResults(), &sync.Mutex{}, systemOneReading{request: "the whole text"}, usedRules(cfg.SystemOneRules))

	if decider.states[config.DefaultDecisionDeployment] != "the whole text" || decider.states["kai"] != "the whole text" {
		t.Fatalf("every rule reads the same authored input: %v", decider.states)
	}
	if decider.deadlines[config.DefaultDecisionDeployment] || !decider.deadlines["kai"] {
		t.Fatalf("only a declared deployment's call takes the default timeout: %v", decider.deadlines)
	}
}

// A rule that reads only the current turn asks about its text. One that reads
// more asks about named parts, and one that reads media gets the current
// turn's images or videos with its state; each other state is a further
// state of the deployment's call.
func TestSystemOneRulesReadTheirInput(t *testing.T) {
	noul := config.SystemOneQuestion{Type: "noul", Instructions: "?"}
	rules := []config.SystemOneRule{
		{Name: "plain", Deployment: "kai", Question: noul},
		{Name: "video", Deployment: "kai", Question: noul, Input: config.SystemOneInput{Video: true}},
		{Name: "context", Deployment: "kai", Question: noul, Coverage: config.SystemOneCoverageFull, Input: config.SystemOneInput{History: true, PreviousTurns: 2, ToolResults: true}},
		{Name: "chart", Deployment: "kai", Question: noul, Input: config.SystemOneInput{Image: true}},
	}
	decider := newFakeDecider(map[string]modelservice.Answer{})
	classifier := newSystemOneTestClassifier(decider, rules...)
	reading := systemOneReading{
		request: "is this chart right?", history: []string{"draw a chart", "make it blue"},
		turns: []ConversationTurn{
			{Role: "user", Text: "draw a chart"},
			{Role: "assistant", Text: "here it is"},
			{Role: "user", Text: "make it blue"},
			{Role: "assistant", Text: "done"},
		},
		toolResults: []string{"rows: 3"}, images: []string{"data:image/png;base64,AAAA"},
	}
	classifier.evaluateSystemOneSignals(context.Background(), newSignalResults(), &sync.Mutex{}, reading, usedRules(rules))
	if decider.calls["kai"] != 3 {
		t.Fatalf("three states, one call each here (the bundle fuses them): %d", decider.calls["kai"])
	}
	byQuestion := map[string]modelservice.Request{}
	for _, request := range decider.asked {
		for _, question := range request.Questions {
			byQuestion[question.ID] = request
		}
	}
	plain, video := byQuestion["systemone:plain"], byQuestion["systemone:video"]
	if plain.State != reading.request || plain.Parts != nil || len(plain.Images) != 0 || len(plain.Questions) != 2 || len(video.Videos) != 0 {
		t.Fatalf("a rule without input, and one asking for a video the request lacks, read the text: %+v", plain)
	}
	parts := byQuestion["systemone:context"]
	want := map[string]string{
		"request": reading.request, "history": "draw a chart\n\nmake it blue",
		"previous_turns": "user: make it blue\nassistant: done", "tool_results": "rows: 3",
	}
	if parts.State != "" || len(parts.Parts) != len(want) || len(parts.Images) != 0 {
		t.Fatalf("typed parts: %+v", parts)
	}
	for name, text := range want {
		if parts.Parts[name] != text {
			t.Fatalf("part %s = %q, want %q", name, parts.Parts[name], text)
		}
	}
	if parts.Questions[0].Truncate {
		t.Fatal("a full-coverage rule reads its whole input")
	}
	chart := byQuestion["systemone:chart"]
	if chart.State != reading.request || !slices.Equal(chart.Images, reading.images) {
		t.Fatalf("an image rule carries the turn's images: %+v", chart)
	}
}

// A full-coverage rule whose model could not read its whole input resolves
// as on_unscanned says; a truncating rule's failed answer is unknown.
func TestOnUnscannedResolvesAnUnreadInput(t *testing.T) {
	question := config.SystemOneQuestion{Type: "choice", Instructions: "?", Choices: []config.SystemOneOption{{Key: "a"}, {Key: "b"}}}
	rule := func(name, onUnscanned string) config.SystemOneRule {
		return config.SystemOneRule{Name: name, Deployment: "kai", Labels: []string{"a"}, Question: question, Coverage: config.SystemOneCoverageFull, OnUnscanned: onUnscanned}
	}
	rules := []config.SystemOneRule{rule("match", "match"), rule("nomatch", "no_match"), rule("unknown", ""), {Name: "truncated", Deployment: "kai", Labels: []string{"a"}, Question: question}}
	answers := map[string]modelservice.Answer{}
	for index, reason := range []string{"scan_budget_exceeded", "input_coverage_unknown", "max_length_exceeded", "scan_budget_exceeded"} {
		answers["systemone:"+rules[index].Name] = modelservice.Answer{Type: "choice", Error: reason}
	}
	classifier := newSystemOneTestClassifier(newFakeDecider(answers), rules...)
	results := newSignalResults()
	classifier.evaluateSystemOneSignals(context.Background(), results, &sync.Mutex{}, systemOneReading{request: "a long text"}, usedRules(rules))
	if !sameMembers(results.MatchedSystemOneRules, []string{"match", "match:a"}) {
		t.Fatalf("on_unscanned: match matches the rule and its labels: %v", results.MatchedSystemOneRules)
	}
	if _, failed := results.SignalErrors["systemone:nomatch"]; failed {
		t.Fatal("on_unscanned: no_match is a known non-match")
	}
	if results.SignalErrors["systemone:unknown"] != "systemone_max_length_exceeded" || results.SignalErrors["systemone:truncated"] != "systemone_scan_budget_exceeded" {
		t.Fatalf("unknown and truncating rules stay unknown: %v", results.SignalErrors)
	}
}

// A response-stage rule is asked about the model's answer, never while the
// request is routed.
func TestResponseStageRulesAreAskedAboutTheAnswer(t *testing.T) {
	rules := []config.SystemOneRule{
		{
			Name: "grounded", Deployment: "kai", Stage: config.SystemOneStageResponse, Input: config.SystemOneInput{Answer: true, Context: true},
			Question: config.SystemOneQuestion{Type: "noul", Instructions: "Is every claim supported?"},
		},
		{Name: "routing", Deployment: "kai", Question: config.SystemOneQuestion{Type: "noul", Instructions: "?"}},
	}
	decider := newFakeDecider(map[string]modelservice.Answer{"systemone:grounded": {Type: "noul", Noul: 0.2}})
	classifier := newSystemOneTestClassifier(decider, rules...)
	classifier.evaluateSystemOneSignals(context.Background(), newSignalResults(), &sync.Mutex{}, systemOneReading{request: "q"}, usedRules(rules))
	for _, question := range decider.requests["kai"].Questions {
		if question.ID == "systemone:grounded" {
			t.Fatal("the request stage does not ask a response-stage rule")
		}
	}
	signal := classifier.EvaluateResponseSystemOne(context.Background(), ResponseSystemOneInput{
		Request: "When is it open?", Answer: "On Sundays.", Context: "Open Monday to Friday.",
	}, []string{"grounded", "routing"})
	asked := decider.requests["kai"]
	if len(asked.Questions) != 1 || asked.Questions[0].ID != "systemone:grounded" || asked.Questions[0].Stage != config.SystemOneStageResponse {
		t.Fatalf("only the response-stage rule is asked: %+v", asked.Questions)
	}
	if asked.Parts["request"] != "When is it open?" || asked.Parts["answer"] != "On Sundays." || asked.Parts["context"] != "Open Monday to Friday." {
		t.Fatalf("parts = %v", asked.Parts)
	}
	if signal.Values["systemone:grounded"] != 0.2 || len(signal.MatchedRules) != 0 || len(signal.Errors) != 0 {
		t.Fatalf("response signal = %+v", signal)
	}
}
