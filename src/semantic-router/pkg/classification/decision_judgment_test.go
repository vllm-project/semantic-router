package classification

import (
	"context"
	"errors"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

type judgmentDeciderFunc func(context.Context, string, modelservice.Request) (modelservice.Response, error)

func (f judgmentDeciderFunc) Decide(ctx context.Context, d string, r modelservice.Request) (modelservice.Response, error) {
	return f(ctx, d, r)
}

func testJudgment(t *testing.T, task string, decider modelservice.Decider) *decisionJudgment {
	t.Helper()
	definition, ok := modelservice.BuiltinTask(task)
	if !ok {
		t.Fatal(task)
	}
	card := modelservice.ModelCard{ID: "future/decision", Surfaces: []string{"decisions"}, QuestionTypes: []string{"choice", "score", "noul"}}
	plan, err := modelservice.CompileTask(definition, definition.Question, card)
	if err != nil {
		t.Fatal(err)
	}
	return &decisionJudgment{deployment: "selected", decider: decider, card: card, plan: plan}
}

func TestDecisionPIIPresenceDoesNotRequireLocations(t *testing.T) {
	for _, test := range []struct {
		name            string
		presence        float64
		partial         bool
		unproven        bool
		clipped         bool
		fail            bool
		detected, clean bool
	}{
		{name: "clean", presence: .01, clean: true},
		{name: "uncategorized personal data", presence: .95, detected: true},
		{name: "missing category is unknown", presence: .01, partial: true},
		{name: "legacy answer without proof is unknown", presence: .01, unproven: true},
		{name: "runtime cannot read the complete input", presence: .01, clipped: true},
		{name: "deadline is unknown", fail: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			calls := 0
			decider := judgmentDeciderFunc(func(_ context.Context, deployment string, r modelservice.Request) (modelservice.Response, error) {
				calls++
				if deployment != "selected" || r.State != "private input" || len(r.Questions) < 2 {
					t.Fatalf("bad batched task request: %+v", r)
				}
				if test.fail {
					return modelservice.Response{}, context.DeadlineExceeded
				}
				answers := map[string]modelservice.Answer{}
				for i, q := range r.Questions {
					if test.partial && i == len(r.Questions)-1 {
						continue
					}
					p := .01
					if i == 0 {
						p = test.presence
					}
					if !q.RequireFullInput {
						t.Fatal("privacy task did not require complete input")
					}
					coverage := "complete"
					if test.unproven || test.clipped {
						coverage = ""
					}
					answer := modelservice.Answer{Type: "noul", Noul: p, InputCoverage: coverage}
					if test.clipped {
						answer.Error = "max_length_exceeded"
					}
					answers[q.ID] = answer
				}
				return modelservice.Response{Answers: answers}, nil
			})
			judgment := testJudgment(t, "pii_presence", decider)
			categories, _ := modelservice.BuiltinTask("pii_categories")
			plan, err := modelservice.CompileTask(categories, categories.Question, judgment.card)
			if err != nil {
				t.Fatal(err)
			}
			backend := &decisionPIIBackend{judgment: judgment, categories: plan}
			cfg := &config.RouterConfig{}
			cfg.PIIRules = []config.PIIRule{{Name: "privacy", Threshold: .7}}
			cfg.PIIModel.OnError, cfg.PIIModel.OnUnscanned = config.OnErrorAllow, config.OnErrorAllow
			classifier := &Classifier{Config: cfg, piiInference: backend}
			results := &SignalResults{Metrics: &SignalMetricsCollection{}}
			classifier.evaluateDecisionPIISignal(t.Context(), results, &sync.Mutex{}, "private input", nil, backend)
			if calls != 1 || results.PIIDetected != test.detected || results.PIIContentVerified != test.clean {
				t.Fatalf("calls=%d detected=%v clean=%v", calls, results.PIIDetected, results.PIIContentVerified)
			}
			if len(results.PIIEvidence) != 1 || results.PIIEvidence[0].CoversClean("request", "private input") != test.clean {
				t.Fatalf("bad evidence: %+v", results.PIIEvidence)
			}
			if results.PIIEvidence[0].CoversClean("response", "private input") || results.PIIEvidence[0].CoversClean("request", "other input") {
				t.Fatal("evidence crossed input or stage")
			}
			if (test.partial || test.fail || test.unproven || test.clipped) && len(results.SignalErrors) == 0 {
				t.Fatal("unknown lost")
			}
			if _, err := backend.ClassifyTokens(t.Context(), "text"); err == nil {
				t.Fatal("verdict advertised invented token positions")
			}
		})
	}
}

func TestDecisionHallucinationVerdictRetainsGroundedPartsWithoutFakeSpans(t *testing.T) {
	var calls int
	j := testJudgment(t, "hallucination", judgmentDeciderFunc(func(_ context.Context, _ string, r modelservice.Request) (modelservice.Response, error) {
		calls++
		if r.State != "" || r.Parts["context"] != "library closed Sunday" || r.Parts["answer"] != "open Sunday" || r.Parts["request"] != "opening hours?" {
			t.Fatalf("lost grounded boundary: %+v", r)
		}
		return modelservice.Response{Answers: map[string]modelservice.Answer{r.Questions[0].ID: {Type: "noul", Noul: .9, InputCoverage: "complete"}}}, nil
	}))
	d := &HallucinationDetector{config: &config.HallucinationModelConfig{Threshold: .7}, judgment: j, initialized: true}
	got, err := d.Detect(t.Context(), "library closed Sunday", "opening hours?", "open Sunday")
	if err != nil || !got.HallucinationDetected || !got.ScoreAvailable || got.ScoreKind != "probability" || len(got.Spans) != 0 || len(got.UnsupportedSpans) != 0 {
		t.Fatalf("verdict lost or fake span: %+v %v", got, err)
	}
	explained, err := d.DetectWithExplanations(t.Context(), "library closed Sunday", "opening hours?", "open Sunday")
	if err != nil || !explained.HallucinationDetected || len(explained.Spans) != 0 {
		t.Fatalf("optional explanation lost verdict: %+v %v", explained, err)
	}
	if _, err = d.Detect(t.Context(), "", "question", "answer"); err == nil || calls != 2 {
		t.Fatal("missing context invoked model")
	}
	j.decider = judgmentDeciderFunc(func(_ context.Context, _ string, r modelservice.Request) (modelservice.Response, error) {
		return modelservice.Response{Answers: map[string]modelservice.Answer{r.Questions[0].ID: {Type: "noul", Noul: 0}}}, nil
	})
	if _, err := d.Detect(t.Context(), "long context", "question", "answer"); err == nil {
		t.Fatal("an unproven zero verdict certified a grounded answer")
	}
}

func TestDecisionReaskJudgesPairsButCountsConsecutiveTurnsInCode(t *testing.T) {
	var calls atomic.Int32
	current := strings.Repeat("long input ", 500) + "unresolved tail"
	j := testJudgment(t, "reask", judgmentDeciderFunc(func(_ context.Context, _ string, r modelservice.Request) (modelservice.Response, error) {
		calls.Add(1)
		if r.Parts["current"] != current || r.Questions[0].Truncate {
			t.Fatal("pair input truncated")
		}
		score := .95
		if r.Parts["prior"] == "different" {
			score = .1
		}
		return modelservice.Response{Answers: map[string]modelservice.Answer{r.Questions[0].ID: {Type: "noul", Noul: score, InputCoverage: "complete"}}}, nil
	}))
	classifier := &ReaskClassifier{rules: []config.ReaskRule{{Name: "repeat", LookbackTurns: 2, Threshold: .7}}, judgment: j}
	matches, err := classifier.ClassifyContext(t.Context(), current, []string{"older same", "different", "same", "same"})
	if err != nil || len(matches) != 1 || matches[0].MatchedTurns != 2 || calls.Load() != 3 {
		t.Fatalf("counter or pair dedupe: %+v calls=%d err=%v", matches, calls.Load(), err)
	}
	j.decider = judgmentDeciderFunc(func(context.Context, string, modelservice.Request) (modelservice.Response, error) {
		return modelservice.Response{}, errors.New("unavailable")
	})
	if got, err := classifier.ClassifyContext(t.Context(), current, []string{"same", "same"}); err == nil || len(got) != 0 {
		t.Fatal("failed pair became successful count")
	}
}

func TestDecisionReaskExactRepeatsDoNotDependOnModelConfidence(t *testing.T) {
	var calls atomic.Int32
	j := testJudgment(t, "reask", judgmentDeciderFunc(func(_ context.Context, _ string, r modelservice.Request) (modelservice.Response, error) {
		calls.Add(1)
		if r.Parts["current"] == r.Parts["prior"] {
			t.Error("an exact repeated request needs no semantic judgment")
		}
		return modelservice.Response{Answers: map[string]modelservice.Answer{r.Questions[0].ID: {Type: "noul", Noul: .1, InputCoverage: "complete"}}}, nil
	}))
	classifier := &ReaskClassifier{rules: []config.ReaskRule{{Name: "repeat", LookbackTurns: 2, Threshold: .9}}, judgment: j}
	current := "Explain vector clocks with an example."
	matches, err := classifier.ClassifyContext(t.Context(), current, []string{"different request", " " + current + " ", current})
	if err != nil || len(matches) != 1 || matches[0].MatchedTurns != 2 || matches[0].MinSimilarity != 1 || calls.Load() != 1 {
		t.Fatalf("exact repeats must retain their consecutive count: %+v calls=%d err=%v", matches, calls.Load(), err)
	}
	for _, input := range []string{"", " \n\t"} {
		matches, err = classifier.ClassifyContext(t.Context(), input, []string{input, input})
		if err != nil || len(matches) != 0 || calls.Load() != 1 {
			t.Fatalf("empty turns became repeats: %+v calls=%d err=%v", matches, calls.Load(), err)
		}
	}
	prefix := strings.Repeat("shared context ", 1000)
	matches, err = classifier.ClassifyContext(t.Context(), prefix+"new intent", []string{prefix + "old intent", prefix + "old intent"})
	if err != nil || len(matches) != 0 || calls.Load() != 2 {
		t.Fatalf("matching prefixes must not hide distinct tails: %+v calls=%d err=%v", matches, calls.Load(), err)
	}
	canceled, cancel := context.WithCancel(t.Context())
	cancel()
	if matches, err = classifier.ClassifyContext(canceled, current, []string{current, current}); !errors.Is(err, context.Canceled) || len(matches) != 0 {
		t.Fatal("an exact repeat ignored cancellation")
	}
	j.decider = judgmentDeciderFunc(func(context.Context, string, modelservice.Request) (modelservice.Response, error) {
		return modelservice.Response{}, errors.New("unavailable")
	})
	if matches, err = classifier.ClassifyContext(t.Context(), current, []string{current, "paraphrased request"}); err == nil || len(matches) != 0 {
		t.Fatal("an exact older turn hid an unknown current pair")
	}
}
