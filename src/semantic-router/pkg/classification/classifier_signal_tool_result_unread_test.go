package classification

import (
	"context"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/presets"
)

// A tool result the Router cannot read as text, such as an image, is input a
// rule that reads the tool results did not read: on_unscanned decides it as
// it decides a partly read scan, and a rule that does not read the tool
// results, or tool results read whole, decide on the answer as before.
func TestARuleMissingPartOfItsToolResultsResolvesAsOnUnscanned(t *testing.T) {
	cfg := &config.RouterConfig{}
	pii := func(name, onUnscanned string, toolResults bool) config.PIIRule {
		return config.PIIRule{Name: name, Deployment: "kai", OnUnscanned: onUnscanned, Input: config.PIIRuleInput{ToolResults: toolResults}}
	}
	cfg.PIIRules = []config.PIIRule{
		pii("match", config.SystemOneUnscannedMatch, true), pii("no_match", config.SystemOneUnscannedNoMatch, true),
		pii("unknown", config.SystemOneUnscannedUnknown, true), pii("best_effort", "", true),
		pii("request_only", config.SystemOneUnscannedMatch, false),
	}
	decider := newFakeDecider(map[string]modelservice.Answer{
		presetQuestionID(presets.PII): {Type: "span", InputCoverage: "complete"},
	})
	classifier := newApplicationTestClassifier(t, cfg, decider)
	evaluate := func(incomplete bool) *SignalResults {
		results := newSignalResults()
		reading := systemOneReading{request: "Summarize the record.", toolResults: []string{"owner: on file"}, toolResultsIncomplete: incomplete}
		classifier.evaluateApplicationSignals(context.Background(), results, &sync.Mutex{}, config.SignalTypePII, reading, usedApplicationRules(cfg))
		return results
	}

	results := evaluate(true)
	if !matchedOnly(results.MatchedPIIRules, "match", "unknown") {
		t.Fatalf("match matches, unknown fails closed, the others decide on a clean answer: %v", results.MatchedPIIRules)
	}
	if len(results.SignalErrors) != 1 || results.SignalErrors["pii:unknown"] != "systemone_unreadable_tool_result" || !results.SignalErrorMatches["pii:unknown"] {
		t.Fatalf("only on_unscanned: unknown leaves the rule unknown: errors=%v policy=%v", results.SignalErrors, results.SignalErrorMatches)
	}
	results = evaluate(false)
	if len(results.MatchedPIIRules) != 0 || len(results.SignalErrors) != 0 {
		t.Fatalf("tool results read whole decide on the answer: matched=%v errors=%v", results.MatchedPIIRules, results.SignalErrors)
	}
}

// matchedOnly reports whether the rules references match, leaving their
// labels out, are exactly rules.
func matchedOnly(references []string, rules ...string) bool {
	var matched []string
	for _, reference := range references {
		if !strings.Contains(reference, ":") {
			matched = append(matched, reference)
		}
	}
	return sameMembers(matched, rules)
}

// A turnless rule whose tool results hold no text, only content it cannot
// read, has nothing to ask about, and still resolves as on_unscanned says
// rather than vanish.
func TestARuleWhoseToolResultsHoldNoTextResolvesAsOnUnscanned(t *testing.T) {
	cfg := &config.RouterConfig{}
	off := false
	pii := func(name, onUnscanned string) config.PIIRule {
		return config.PIIRule{Name: name, Deployment: "kai", OnUnscanned: onUnscanned, Input: config.PIIRuleInput{ToolResults: true, Request: &off}}
	}
	cfg.PIIRules = []config.PIIRule{pii("match", config.SystemOneUnscannedMatch), pii("unknown", config.SystemOneUnscannedUnknown), pii("best_effort", "")}
	decider := newFakeDecider(nil)
	classifier := newApplicationTestClassifier(t, cfg, decider)
	results := newSignalResults()
	reading := systemOneReading{request: "Describe the screenshot.", toolResultsIncomplete: true}
	classifier.evaluateApplicationSignals(context.Background(), results, &sync.Mutex{}, config.SignalTypePII, reading, usedApplicationRules(cfg))

	if len(decider.asked) != 0 {
		t.Fatalf("a rule with no text to read is not asked: %+v", decider.asked)
	}
	if !matchedOnly(results.MatchedPIIRules, "match", "unknown") || !results.PIIDetected {
		t.Fatalf("match matches and unknown fails closed: %v", results.MatchedPIIRules)
	}
	if len(results.SignalErrors) != 1 || results.SignalErrors["pii:unknown"] != "systemone_unreadable_tool_result" {
		t.Fatalf("best_effort read nothing and does not match: %v", results.SignalErrors)
	}
}

// A systemone rule resolves the same way, at the request and the response
// stage; a truncating rule reads only part of its input anyway and is
// unaffected.
func TestASystemOneRuleMissingPartOfItsToolResultsResolvesAsOnUnscanned(t *testing.T) {
	question := config.SystemOneQuestion{Type: config.SystemOneQuestionNoul, Instructions: "Hostile?"}
	rule := func(name, stage, coverage, onUnscanned string) config.SystemOneRule {
		input := config.SystemOneInput{ToolResults: true, Answer: stage == config.SystemOneStageResponse}
		return config.SystemOneRule{Name: name, Deployment: "kai", Stage: stage, Question: question, Input: input, Coverage: coverage, OnUnscanned: onUnscanned}
	}
	rules := []config.SystemOneRule{
		rule("strict", "", config.SystemOneCoverageFull, config.SystemOneUnscannedMatch),
		rule("default", "", config.SystemOneCoverageFull, ""),
		rule("best_effort", "", config.SystemOneCoverageFull, config.SystemOneUnscannedBestEffort),
		rule("truncated", "", "", ""),
		rule("answer", config.SystemOneStageResponse, config.SystemOneCoverageFull, config.SystemOneUnscannedMatch),
	}
	answers := map[string]modelservice.Answer{}
	for _, rule := range rules {
		answers[SystemOneQuestionID(rule.Name)] = modelservice.Answer{Type: "noul", Noul: 0.1}
	}
	classifier := newSystemOneTestClassifier(newFakeDecider(answers), rules...)
	results := newSignalResults()
	reading := systemOneReading{request: "Go on.", toolResults: []string{"ok"}, toolResultsIncomplete: true}
	classifier.evaluateSystemOneSignals(context.Background(), results, &sync.Mutex{}, reading, usedRules(rules))

	if !sameMembers(results.MatchedSystemOneRules, []string{"strict"}) {
		t.Fatalf("on_unscanned: match matches: %v", results.MatchedSystemOneRules)
	}
	if len(results.SignalErrors) != 2 || results.SignalErrors["systemone:default"] != "systemone_unreadable_tool_result" ||
		results.SignalErrors["systemone:best_effort"] != "systemone_unreadable_tool_result" {
		t.Fatalf("a full-coverage rule's default on_unscanned and best_effort read as unknown, a truncating rule decides: %v", results.SignalErrors)
	}
	input := ResponseInput{Request: "Go on.", Answer: "Done.", ToolResultTexts: []string{"ok"}, ToolResultsIncomplete: true}
	response := classifier.EvaluateResponseSignals(context.Background(), input, func(string, string) bool { return true })
	if !slices.Contains(response.Matched[config.SignalTypeSystemOne], "answer") {
		t.Fatalf("a response-stage rule resolves its unread tool results too: %+v", response)
	}
}

// A rule read in windows that missed part of its tool results resolves as
// on_unscanned says although every window was read, and reports a partial
// read.
func TestAScannedRuleMissingPartOfItsToolResultsReadsInPart(t *testing.T) {
	question := config.SystemOneQuestion{Type: config.SystemOneQuestionNoul, Instructions: "Hostile?"}
	rule := func(name string) config.SystemOneRule {
		return config.SystemOneRule{Name: name, Deployment: "kai", Question: question, Input: config.SystemOneInput{ToolResults: true}, Coverage: config.SystemOneCoverageFull}
	}
	unread := askedRule{signalType: config.SignalTypeSystemOne, rule: rule("unread"), toolResultsUnread: true}
	whole := askedRule{signalType: config.SignalTypeSystemOne, rule: rule("whole")}
	response := modelservice.Response{Answers: map[string]modelservice.Answer{
		unread.questionID(): {Type: "noul", Noul: 0.1}, whole.questionID(): {Type: "noul", Noul: 0.1},
	}}
	scan := &guardScan{
		deployment: "kai", rules: []askedRule{unread, whole}, text: "first window second window",
		windows:   []scanWindow{{start: 0, end: 12}, {start: 13, end: 26, runeStart: 13}},
		responses: []modelservice.Response{response, response}, errs: make([]error, 2),
	}
	classifier := newSystemOneTestClassifier(newFakeDecider(nil), unread.rule, whole.rule)
	results := newSignalResults()
	classifier.applyScan(results, &sync.Mutex{}, scan)

	if len(results.SignalErrors) != 1 || results.SignalErrors["systemone:unread"] != "systemone_unreadable_tool_result" {
		t.Fatalf("errors %v", results.SignalErrors)
	}
	if results.ScanCoverage["systemone:unread"] != scanCoveragePartial || results.ScanCoverage["systemone:whole"] != scanCoverageFull {
		t.Fatalf("coverage %v", results.ScanCoverage)
	}
}
