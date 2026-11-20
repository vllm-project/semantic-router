package classification

import (
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

// A window's call succeeding and answering its question is not proof that the
// model read the window: only an answer that reports complete input coverage
// covers it. Missing, empty and partial coverage leave a window unread, as a
// missing answer and an error do.
func TestAWindowIsReadOnlyWithCompleteInputCoverage(t *testing.T) {
	const question = "preset:jailbreak"
	cases := []struct {
		name    string
		answer  modelservice.Answer
		present bool
		covered int
		reason  string
	}{
		{"complete coverage", modelservice.Answer{Type: "choice", InputCoverage: "complete"}, true, 1, ""},
		{"empty coverage", modelservice.Answer{Type: "choice"}, true, 0, "input_coverage_unknown"},
		{"truncated coverage", modelservice.Answer{Type: "choice", InputCoverage: "truncated"}, true, 0, "input_coverage_unknown"},
		{"unknown coverage", modelservice.Answer{Type: "choice", InputCoverage: "unknown"}, true, 0, "input_coverage_unknown"},
		{"missing answer", modelservice.Answer{}, false, 0, "missing_answer"},
		{"answer error", modelservice.Answer{Type: "choice", Error: "max_length_exceeded"}, true, 0, "max_length_exceeded"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			scan := &guardScan{
				cut:       1,
				windows:   []scanWindow{{first: 0, last: 0}},
				responses: make([]modelservice.Response, 1),
				errs:      make([]error, 1),
				done:      make([]bool, 1),
			}
			if tc.present {
				scan.responses[0] = modelservice.Response{Answers: map[string]modelservice.Answer{question: tc.answer}}
			}
			read, covered, reason := scan.answersTo(question)
			if covered != tc.covered || reason != tc.reason {
				t.Fatalf("covered %d/%d, reason %q: want covered %d, reason %q", covered, scan.cut, reason, tc.covered, tc.reason)
			}
			if len(read) != tc.covered {
				t.Fatalf("%d windows read: want %d", len(read), tc.covered)
			}
		})
	}
}

// One window that reports less than complete coverage leaves a scan partial,
// whatever its other windows read, so the rule resolves as on_unscanned says
// instead of on an answer that stands for only part of the input.
func TestAScanCountsAWindowReadOnlyWithCompleteInputCoverage(t *testing.T) {
	classifier, _ := refusingGuardClassifier(t, time.Minute, "          on_unscanned: match\n", 0)
	var asked askedRule
	for _, rule := range classifier.Config.ApplicationRules() {
		if rule.Type == config.SignalTypeJailbreak && rule.Rule.Name == "prompt_attack" {
			asked = applicationAsked(rule)
		}
	}
	if asked.rule.Name == "" {
		t.Fatal("no compiled jailbreak rule")
	}
	text := "Summarize the incident report and list every owner on file."
	half := len(text) / 2
	benign := func(coverage string) modelservice.Answer {
		return modelservice.Answer{Type: "choice", Choice: "benign", Probabilities: map[string]float64{"benign": 0.99, "jailbreak": 0.01}, InputCoverage: coverage}
	}
	scan := &guardScan{
		deployment: "primary",
		rules:      []askedRule{asked},
		whole:      modelservice.Request{State: text},
		text:       text,
		windows: []scanWindow{
			{start: 0, end: half, first: 0, last: 0},
			{start: half, end: len(text), runeStart: half, first: 1, last: 1},
		},
		cut:       2,
		responses: make([]modelservice.Response, 2),
		errs:      make([]error, 2),
		done:      make([]bool, 2),
	}
	scan.record(0, modelservice.Response{Answers: map[string]modelservice.Answer{asked.questionID(): benign("complete")}}, nil, 0)
	scan.record(1, modelservice.Response{Answers: map[string]modelservice.Answer{asked.questionID(): benign("")}}, nil, 0)
	results := newSignalResults()
	classifier.applyScan(results, &sync.Mutex{}, scan)

	key := signalConfidenceKey(config.SignalTypeJailbreak, asked.rule.Name)
	if got := results.ScanCoverage[key]; got != "partial 1/2" {
		t.Fatalf("scan coverage %q: a window without complete coverage is unread, want %q", got, "partial 1/2")
	}
	if !matchedOnly(results.MatchedJailbreakRules, asked.rule.Name) || len(results.SignalErrors) != 0 {
		t.Fatalf("on_unscanned match resolves the rule: matched=%v errors=%v", results.MatchedJailbreakRules, results.SignalErrors)
	}
	if !slices.Contains(results.MatchedJailbreakRules, asked.rule.Name+":jailbreak") {
		t.Fatalf("on_unscanned match counts the rule's labels: matched=%v", results.MatchedJailbreakRules)
	}

	// Every window reporting complete coverage covers the whole input.
	scan = &guardScan{
		deployment: "primary",
		rules:      []askedRule{asked},
		whole:      modelservice.Request{State: text},
		text:       text,
		windows: []scanWindow{
			{start: 0, end: half, first: 0, last: 0},
			{start: half, end: len(text), runeStart: half, first: 1, last: 1},
		},
		cut:       2,
		responses: make([]modelservice.Response, 2),
		errs:      make([]error, 2),
		done:      make([]bool, 2),
	}
	scan.record(0, modelservice.Response{Answers: map[string]modelservice.Answer{asked.questionID(): benign("complete")}}, nil, 0)
	scan.record(1, modelservice.Response{Answers: map[string]modelservice.Answer{asked.questionID(): benign("complete")}}, nil, 0)
	results = newSignalResults()
	classifier.applyScan(results, &sync.Mutex{}, scan)
	if got := results.ScanCoverage[key]; got != scanCoverageFull {
		t.Fatalf("scan coverage %q: windows read whole cover the input, want %q", got, scanCoverageFull)
	}
	if results.SignalErrorMatches[key] || len(results.SignalErrors) != 0 || len(results.MatchedJailbreakRules) != 0 {
		t.Fatalf("complete coverage resolves on the answer: matched=%v errors=%v", results.MatchedJailbreakRules, results.SignalErrors)
	}
}
