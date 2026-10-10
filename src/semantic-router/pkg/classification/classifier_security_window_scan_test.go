package classification

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

type securityDistributionFunc func(context.Context, string) (SequenceClassificationResult, error)

func (f securityDistributionFunc) Classify(ctx context.Context, text string) (SequenceClassificationResult, error) {
	return f(ctx, text)
}

func (securityDistributionFunc) readsWholeText() bool { return true }

func TestNativeJailbreakWindowsPreserveMiddleTailRiskAndIncompleteScans(t *testing.T) {
	text := strings.Repeat("Ordinary regional report. ", 100) + "middle attack" +
		strings.Repeat("Ordinary staffing report. ", 100) + "tail attack"
	for _, failTail := range []bool{false, true} {
		var wholeCalls atomic.Int32
		backend := securityDistributionFunc(func(_ context.Context, input string) (SequenceClassificationResult, error) {
			if input == text {
				wholeCalls.Add(1)
			}
			if input == text || failTail && strings.Contains(input, "tail attack") {
				return SequenceClassificationResult{}, binding.ErrInputLimit
			}
			risk := float32(.01)
			if strings.Contains(input, "middle attack") {
				risk = .8
			}
			if strings.Contains(input, "tail attack") {
				risk = .98
			}
			return SequenceClassificationResult{Probabilities: []float32{risk, 1 - risk}}, nil
		})
		c := newRiskTestClassifier(backend)
		// Native bindings own admission; keep their whole-input capability
		// visible, as production ownedSequenceBackend does.
		c.jailbreakInference = backend
		c.Config.JailbreakRules = []config.JailbreakRule{{Name: "attack", Threshold: .95}}
		scan, err := c.ScanJailbreakRisk(t.Context(), text)
		wantRisk := float32(.98)
		if failTail {
			wantRisk = .8
		}
		if err != nil || wholeCalls.Load() != 1 || scan.RiskScore != wantRisk || (scan.PartialErr != nil) != failTail {
			t.Fatalf("risk and coverage: %+v, %v", scan, err)
		}
		_, _, _, _, err = c.CheckForJailbreakRiskWithThreshold(t.Context(), text, .95)
		if (err != nil) != failTail {
			t.Fatalf("incomplete scan certified clean: %v", err)
		}
		// A known positive remains a positive even if another window failed.
		matched, _, _, _, err := c.CheckForJailbreakRiskWithThreshold(t.Context(), text, .7)
		if err != nil || !matched {
			t.Fatalf("lost positive middle window: %t %v", matched, err)
		}
		results := newSecuritySignalResults()
		c.evaluateJailbreakSignal(t.Context(), results, &sync.Mutex{}, text, nil)
		if len(results.MatchedJailbreakRules) != 1 || (len(results.SignalErrors) > 0) != failTail {
			t.Fatalf("routing did not retain risk or incomplete coverage: %+v", results)
		}
	}
}

func newSecuritySignalResults() *SignalResults {
	return &SignalResults{Metrics: &SignalMetricsCollection{}, SignalValues: map[string]float64{}, SignalConfidences: map[string]float64{}, SignalErrors: map[string]string{}, SignalErrorMatches: map[string]bool{}}
}

func TestNativeSafetyWindowsRequireEveryCoverageProofAndKeepMaximumRisk(t *testing.T) {
	text := strings.Repeat("Public report. ", 400) + "sensitive tail"
	for _, missingProof := range []bool{false, true} {
		decider := judgmentDeciderFunc(func(_ context.Context, _ string, request modelservice.Request) (modelservice.Response, error) {
			q := request.Questions[0]
			if !q.RequireFullInput || q.Truncate {
				t.Error("security window lost strict input coverage")
			}
			answer := modelservice.Answer{Type: "choice", Choice: "safe", InputCoverage: "complete", Probabilities: map[string]float64{"safe": .9, "unsafe": .1}}
			if request.State == text {
				answer.Error = "input_limit"
			} else if strings.Contains(request.State, "sensitive tail") {
				answer.Choice, answer.Probabilities = "unsafe", map[string]float64{"safe": .02, "unsafe": .98}
				if missingProof {
					answer.InputCoverage = ""
				}
			}
			return modelservice.Response{Answers: map[string]modelservice.Answer{q.ID: answer}}, nil
		})
		classifier := &decisionLabelClassifier{judgment: testJudgment(t, "safety", decider), labels: []string{"safe", "unsafe"}}
		result, err := classifySafetyWindows(t.Context(), classifier, text)
		if missingProof {
			if err == nil {
				t.Fatal("a window without coverage proof was accepted")
			}
			continue
		}
		if err != nil || len(result.ScoreWindows) < 2 || aggregateSafetyWindows(result, []string{"unsafe"}, selectedSafetyScore) != .98 {
			t.Fatalf("lost maximum window risk: %+v %v", result, err)
		}
	}
}

func TestNativeSecurityWindowsPreserveBudgetsErrorsAndCancellation(t *testing.T) {
	text := strings.Repeat("report ", 1000)
	for _, original := range []error{binding.ErrScanBudget, binding.ErrInvalidResult, context.DeadlineExceeded, context.Canceled} {
		var calls atomic.Int32
		windows := nativeSecurityWindows(t.Context(), text, true, func(context.Context, string) (int, error) {
			calls.Add(1)
			return 0, original
		})
		if calls.Load() != 1 || len(windows) != 1 || !errors.Is(windows[0].err, original) {
			t.Fatalf("retried a declared budget or other failure: %v", original)
		}
	}
	ctx, cancel := context.WithCancel(t.Context())
	var calls atomic.Int32
	windows := nativeSecurityWindows(ctx, text, true, func(context.Context, string) (int, error) {
		calls.Add(1)
		cancel()
		return 0, binding.ErrInputLimit
	})
	if calls.Load() != 1 || !errors.Is(windows[0].err, context.Canceled) {
		t.Fatal("cancelled request started new windows or lost cancellation")
	}
}

func TestNativeSecurityWindowsBoundConcurrentWorkAndMarkCancelledRemainder(t *testing.T) {
	var input strings.Builder
	for i := range 1000 {
		fmt.Fprintf(&input, "Record %d documents the regional staffing report.\n", i)
	}
	text := input.String()
	for _, stop := range []bool{false, true} {
		ctx, cancel := context.WithCancel(t.Context())
		var active, processed atomic.Int32
		windows := nativeSecurityWindows(ctx, text, true, func(_ context.Context, part string) (int, error) {
			if part == text {
				return 0, binding.ErrInputLimit
			}
			if active.Add(1) > 4 {
				t.Error("long input exceeded the security worker limit")
			}
			defer active.Add(-1)
			processed.Add(1)
			if stop {
				cancel()
			} else {
				time.Sleep(time.Millisecond)
			}
			return 1, nil
		})
		cancel()
		if len(windows) < 5 || (!stop && int(processed.Load()) != len(windows)) {
			t.Fatalf("window coverage lost: processed=%d total=%d", processed.Load(), len(windows))
		}
		for _, window := range windows {
			if stop && !errors.Is(window.err, context.Canceled) {
				t.Fatal("cancelled or unprocessed window was reported as safe")
			}
			if !stop && (window.err != nil || window.result != 1) {
				t.Fatal("complete window result lost")
			}
		}
	}
}
