package classification

import (
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
// the request's own guard scan of it ran out of time, nor when its windows
// would take longer than scanReadShare of that time at the deployment's
// measured pace. Before a pace is measured, it may be read.
func TestReplayReadsAPromptOnlyWhenItCanReadItWholeInTime(t *testing.T) {
	const within = 30 * time.Second
	short := "Summarize the meeting agenda."
	long := strings.Repeat("The committee reviewed the quarterly report on regional water supplies and agreed to meet again. ", 10_000)
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
	windows, _ := cutScanWindows(long, budget.size, budget.most, budget.overlap)
	if slow := classifierAt(within / time.Duration(len(windows))); slow.ReplayPromptReadable(long, within, false) {
		t.Fatalf("%d windows at %v each were found readable in %v", len(windows), within/time.Duration(len(windows)), within)
	}
	if fast := classifierAt(within / time.Duration(10*len(windows))); !fast.ReplayPromptReadable(long, within, false) {
		t.Fatalf("%d windows at %v each were found unreadable in %v", len(windows), within/time.Duration(10*len(windows)), within)
	}
}
