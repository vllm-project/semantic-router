package classification

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/presets"
)

func TestReplayPromptEvidenceCoversOnlyACompleteCleanReadOfThePrompt(t *testing.T) {
	const prompt = "Summarize the meeting agenda."
	email := modelservice.Span{Label: "EMAIL_ADDRESS", Text: "alice@example.test", Probability: 0.99}
	for _, test := range []struct {
		name     string
		answer   *modelservice.Answer
		fail     error
		rules    bool
		clean    bool
		complete bool
	}{
		{name: "clean", rules: true, answer: &modelservice.Answer{Type: "span", InputCoverage: "complete"}, clean: true, complete: true},
		{name: "personal data", rules: true, answer: &modelservice.Answer{Type: "span", InputCoverage: "complete", Spans: []modelservice.Span{email}}, complete: true},
		{name: "read failed", rules: true, fail: modelservice.ErrUnavailable},
		{name: "read incomplete", rules: true, answer: &modelservice.Answer{Type: "span", Error: "max_length_exceeded"}},
		{name: "no rules"},
	} {
		t.Run(test.name, func(t *testing.T) {
			answers := map[string]modelservice.Answer{}
			if test.answer != nil {
				answers[presetQuestionID(presets.PII)] = *test.answer
			}
			decider := newFakeDecider(answers)
			if test.fail != nil {
				decider.failures["guard-model"] = test.fail
			}
			cfg := &config.RouterConfig{}
			if test.rules {
				cfg.PIIRules = []config.PIIRule{{Name: "privacy", Deployment: "guard-model"}}
			}
			classifier, err := newClassifierWithOptions(cfg)
			if err != nil {
				t.Fatal(err)
			}
			classifier.SetDecisionDecider(decider)
			evidence := classifier.ReplayPromptEvidence(t.Context(), prompt)
			if evidence.CoversClean("request", prompt) != test.clean {
				t.Fatalf("clean=%v want=%v (%+v)", evidence.CoversClean("request", prompt), test.clean, evidence)
			}
			if (evidence.Coverage == "complete") != test.complete {
				t.Fatalf("coverage=%q, want complete=%v", evidence.Coverage, test.complete)
			}
			if evidence.CoversClean("request", prompt+" ") {
				t.Fatal("evidence must cover exactly the prompt that was read")
			}
			if !test.rules && len(decider.asked) != 0 {
				t.Fatalf("a recipe without pii rules asked the model: %+v", decider.asked)
			}
		})
	}
}

// Replay's check after routing reads a prompt only when it can read it whole
// in its time. A prompt one window holds always can. A longer one cannot when
// it needs more windows than one scan reads, nor when its windows would take
// longer than scanReadShare of that time at the deployment's measured pace.
// Before a pace is measured, it may be read unless the request's own guard
// scan ran out of time on it; once there is one, that scan says nothing more:
// it may have read earlier turns or tool results, or lost windows that failed.
func TestReplayReadsAPromptOnlyWhenItCanReadItWholeInTime(t *testing.T) {
	const within = 30 * time.Second
	short := "Summarize the meeting agenda."
	const sentence = "The committee reviewed the quarterly report on regional water supplies and agreed to meet again. "
	long := strings.Repeat(sentence, 10_000)
	classifierAt := func(perWindow time.Duration) *Classifier {
		cfg := &config.RouterConfig{}
		cfg.PIIRules = []config.PIIRule{{Name: "privacy", Deployment: "guard-model"}}
		classifier, err := newClassifierWithOptions(cfg)
		if err != nil {
			t.Fatal(err)
		}
		budget := classifier.scanWindowBudget("guard-model")
		if len(long) <= budget.size {
			t.Fatalf("the long prompt fits one window of %d bytes", budget.size)
		}
		for range 5 {
			if perWindow > 0 {
				classifier.windowPaces.observe("guard-model", budget.size, perWindow)
			}
		}
		return classifier
	}
	unmeasured := classifierAt(0)
	if !unmeasured.ReplayPromptReadable(short, within, true) {
		t.Fatal("a prompt one window holds is one call, always readable")
	}
	if !unmeasured.ReplayPromptReadable(long, within, false) {
		t.Fatal("before a pace is measured, a long prompt may be read")
	}
	if unmeasured.ReplayPromptReadable(long, within, true) {
		t.Fatal("a long prompt the request's own guard scan ran out of time on was found readable")
	}
	budget := unmeasured.scanWindowBudget("guard-model")
	huge := strings.Repeat(sentence, (maxScanWindows+2)*budget.most/len(sentence))
	if cut, _ := cutScanWindows(huge, budget.size, budget.most, budget.overlap); len(cut) <= maxScanWindows {
		t.Fatalf("the huge prompt needs %d windows, want more than the %d one scan reads", len(cut), maxScanWindows)
	}
	if unmeasured.ReplayPromptReadable(huge, within, false) {
		t.Fatal("before a pace is measured, a prompt that needs more windows than one scan reads was found readable")
	}
	windows, _ := cutScanWindows(long, budget.size, budget.most, budget.overlap)
	if slow := classifierAt(within / time.Duration(len(windows))); slow.ReplayPromptReadable(long, within, false) {
		t.Fatalf("%d windows at %v each were found readable in %v", len(windows), within/time.Duration(len(windows)), within)
	}
	fast := classifierAt(within / time.Duration(10*len(windows)))
	if !fast.ReplayPromptReadable(long, within, false) {
		t.Fatalf("%d windows at %v each were found unreadable in %v", len(windows), within/time.Duration(10*len(windows)), within)
	}
	if !fast.ReplayPromptReadable(long, within, true) {
		t.Fatalf("%d windows at %v each were found unreadable in %v because a guard scan ran out of time", len(windows), within/time.Duration(10*len(windows)), within)
	}
}

// guardScanDeadline is the signals' deadline of the tests below, whose fake
// reads a 700,000-token message's windows in about 5 s, 20 ms each: a second,
// or three under -race, whose own slowness would leave no window read in one.
func guardScanDeadline() time.Duration {
	if raceDetector {
		return 3 * time.Second
	}
	return time.Second
}

// A message the request's guard scan ran out of time on is skipped without
// the guard's sign: that scan measured the deployment's pace, at which the
// check could not read the message in the same time either.
func TestReplaySkipsAMessageItsGuardScanRanOutOfTimeOnAtThePaceThatScanMeasured(t *testing.T) {
	deadline := guardScanDeadline()
	classifier, _ := refusingGuardClassifier(t, deadline, "      pii:\n        - name: personal_data\n          predicate: {gte: 0.5}\n", 20*time.Millisecond)
	if _, _, measured := classifier.windowPaces.line("primary", 4); measured {
		t.Fatal("the deployment had a pace before any scan")
	}
	text := distinctProse(700_000, 83)
	results := evaluateTurn(classifier, text)
	if coverage := results.ScanCoverage["jailbreak:prompt_attack"]; !strings.HasPrefix(coverage, scanCoveragePartial) || guarded(results) {
		t.Fatalf("the guard read %q of the message (guarded %v), want part of it and no match", coverage, guarded(results))
	}
	if classifier.ReplayPromptReadable(text, deadline, false) {
		t.Fatal("a message the guard read only in part by its deadline was found readable in that time at the pace its scan measured")
	}
}

// A guard rule that reads the earlier user turns runs out of time on a long
// conversation whose latest message, all that Replay's check reads, is three
// windows long. Its scan measured the deployment's pace, at which the check
// reads that message in time, so the check is not skipped.
func TestReplayReadsTheLatestMessageWhenAGuardScanOfTheConversationRanShort(t *testing.T) {
	classifier, _ := refusingGuardClassifier(t, guardScanDeadline(),
		"          input: {history: true}\n      pii:\n        - name: personal_data\n          predicate: {gte: 0.5}\n", 20*time.Millisecond)
	current := distinctProse(4_000, 81)
	if budget := classifier.scanWindowBudget("primary"); len(current) <= budget.size {
		t.Fatalf("the latest message fits one window of %d bytes", budget.size)
	}
	results := classifier.evaluateAllSignalsWithContext(SignalEvaluationInput{
		Text: current, CurrentUserText: current, PriorUserMessages: []string{distinctProse(700_000, 82)},
		RequestFacts: RequestFacts{Context: context.Background()},
	}, classifier.Config.Decisions, true)
	if coverage := results.ScanCoverage["jailbreak:prompt_attack"]; !strings.HasPrefix(coverage, scanCoveragePartial) || guarded(results) {
		t.Fatalf("the conversation's scan read %q of it (guarded %v), want part of it and no match", coverage, guarded(results))
	}
	if !classifier.ReplayPromptReadable(current, 30*time.Second, true) {
		t.Fatal("a latest message the deployment reads in time was found unreadable because the conversation's scan ran out of time")
	}
}
