package classification

import (
	"context"
	"strings"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

type partialGuardBackend struct {
	usage              *tasks.InputUsage
	calls              int
	completeAfterFirst bool
}

func (b *partialGuardBackend) Classify(context.Context, string) (SequenceClassificationResult, error) {
	b.calls++
	if b.completeAfterFirst && b.calls > 1 {
		return SequenceClassificationResult{Probabilities: []float32{.8, .2}, Input: &tasks.InputUsage{OriginalTokens: 4, ProcessedTokens: 4}}, nil
	}
	return SequenceClassificationResult{Probabilities: []float32{.1, .9}, Input: b.usage}, nil
}

func partialGuardUsages() map[string]*tasks.InputUsage {
	return map[string]*tasks.InputUsage{
		"declared truncation":   {OriginalTokens: 4, ProcessedTokens: 4, Truncated: true},
		"short processed input": {OriginalTokens: 4, ProcessedTokens: 3},
	}
}

func TestJailbreakPartialInputCannotProduceCleanAPIResult(t *testing.T) {
	for name, usage := range partialGuardUsages() {
		t.Run(name, func(t *testing.T) {
			c := newRiskTestClassifier(&partialGuardBackend{usage: usage})
			ctx := context.Background()
			if _, err := c.ScanJailbreakRisk(ctx, "sample"); err == nil {
				t.Fatal("partial native input produced a successful scan")
			}
			if _, _, _, _, err := c.CheckForJailbreakWithRisk(ctx, "sample"); err == nil {
				t.Fatal("risk API reported a clean partial input")
			}
			if _, _, _, err := c.CheckForJailbreak(ctx, "sample"); err == nil {
				t.Fatal("legacy API reported a clean partial input")
			}
			if _, _, err := c.AnalyzeContentForJailbreak(ctx, []string{"sample"}); err == nil {
				t.Fatal("content API reported a clean partial input")
			}
		})
	}
	for name, usage := range map[string]*tasks.InputUsage{
		"legacy absent metadata": nil,
		"complete metadata":      {OriginalTokens: 4, ProcessedTokens: 4},
	} {
		t.Run(name, func(t *testing.T) {
			c := newRiskTestClassifier(&partialGuardBackend{usage: usage})
			matched, _, _, _, err := c.CheckForJailbreakWithRisk(context.Background(), "sample")
			if err != nil || matched {
				t.Fatalf("complete/legacy result changed: matched=%v err=%v", matched, err)
			}
		})
	}
}

func TestJailbreakPartialInputPreservesOnErrorPolicy(t *testing.T) {
	for _, policy := range []string{config.OnErrorAllow, config.OnErrorBlock} {
		t.Run(policy, func(t *testing.T) {
			c := newRiskTestClassifier(&partialGuardBackend{usage: &tasks.InputUsage{OriginalTokens: 5, ProcessedTokens: 4}})
			c.Config.PromptGuard.OnError = policy
			c.Config.JailbreakRules = []config.JailbreakRule{{Name: "guard", Threshold: .5}}
			results := &SignalResults{Metrics: &SignalMetricsCollection{}, SignalConfidences: map[string]float64{}}
			c.evaluateJailbreakSignal(context.Background(), results, &sync.Mutex{}, "sample", nil)
			if results.SignalErrors["jailbreak:guard"] != jailbreakEvaluationFailedCode {
				t.Fatalf("missing partial-input diagnostic: %+v", results.SignalErrors)
			}
			if results.JailbreakScoreAvailable {
				t.Fatal("partial input published a usable risk score")
			}
			wantBlock := policy == config.OnErrorBlock
			if (len(results.MatchedJailbreakRules) > 0) != wantBlock || results.SignalErrorMatches["jailbreak:guard"] != wantBlock {
				t.Fatalf("on_error contract changed: matches=%v error_matches=%v", results.MatchedJailbreakRules, results.SignalErrorMatches)
			}
		})
	}
}

func TestJailbreakCompletePositiveSurvivesPartialInput(t *testing.T) {
	backend := &partialGuardBackend{usage: &tasks.InputUsage{OriginalTokens: 5, ProcessedTokens: 4}, completeAfterFirst: true}
	c := newRiskTestClassifier(backend)
	text := strings.Repeat("sample ", 500)
	scan, err := c.ScanJailbreakRisk(context.Background(), text)
	if err != nil || scan.PartialErr == nil || scan.RiskScore != .8 || backend.calls < 2 {
		t.Fatalf("complete positive or partial diagnostic lost: scan=%+v calls=%d err=%v", scan, backend.calls, err)
	}
	signal := EvaluateResponseJailbreakSignal([]config.JailbreakRule{{Name: "matched", Threshold: .5}, {Name: "unresolved", Threshold: .95}}, &scan)
	if len(signal.MatchedRules) != 1 || signal.MatchedRules[0] != "matched" || signal.Errors["jailbreak:unresolved"] != responseJailbreakSignalFailedCode {
		t.Fatalf("response rules lost per-threshold partial semantics: %+v", signal)
	}
	backend.calls = 0
	matched, _, _, _, err := c.CheckForJailbreakWithRisk(context.Background(), text)
	if err != nil || !matched {
		t.Fatalf("complete positive should still be actionable: matched=%v err=%v", matched, err)
	}
	backend.calls = 0
	c.Config.JailbreakRules = []config.JailbreakRule{{Name: "guard", Threshold: .5}}
	results := &SignalResults{Metrics: &SignalMetricsCollection{}, SignalConfidences: map[string]float64{}}
	c.evaluateJailbreakSignal(context.Background(), results, &sync.Mutex{}, text, nil)
	if len(results.MatchedJailbreakRules) != 1 || results.SignalErrors["jailbreak:guard"] != jailbreakEvaluationFailedCode || results.SignalErrorMatches["jailbreak:guard"] {
		t.Fatalf("request signal must retain a real detection and the partial diagnostic: %+v", results)
	}
}

func TestWindowedJailbreakRejectsPartialAggregateInput(t *testing.T) {
	for name, usage := range partialGuardUsages() {
		t.Run(name, func(t *testing.T) {
			backend, model, _ := windowedGuardFixture(t)
			t.Cleanup(func() { _ = backend.Close() })
			model.usage = usage
			if _, err := backend.Classify(context.Background(), "sample"); err == nil {
				t.Fatal("window aggregate accepted declared partial input")
			}
		})
	}
}
