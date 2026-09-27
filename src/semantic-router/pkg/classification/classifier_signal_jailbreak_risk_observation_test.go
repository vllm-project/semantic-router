package classification

import (
	"context"
	"errors"
	"fmt"
	"math"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

type observationBackend map[string]cachedJailbreakResult

func (b observationBackend) Classify(_ context.Context, text string) (SequenceClassificationResult, error) {
	v := b[text]
	return v.result, v.err
}

type observationCategorical struct{ label string }

func (b observationCategorical) Classify(context.Context, string) (SequenceClassificationResult, error) {
	return SequenceClassificationResult{}, tasks.ErrProbabilitiesUnavailable
}

func (b observationCategorical) Decide(context.Context, string) (tasks.LabelDecision, error) {
	return tasks.LabelDecision{Label: b.label}, nil
}

func observationDistribution(p float32) cachedJailbreakResult {
	return cachedJailbreakResult{result: SequenceClassificationResult{Probabilities: []float32{1 - p, p}}}
}

func observationClassifier(rules []config.JailbreakRule, backend SequenceClassifierBackend, onError string) *Classifier {
	cfg := &config.RouterConfig{}
	cfg.JailbreakRules = rules
	cfg.PromptGuard.Enabled = true
	cfg.PromptGuard.OnError = onError
	return &Classifier{Config: cfg, JailbreakMapping: &JailbreakMapping{LabelToIdx: map[string]int{"benign": 0, "jailbreak": 1}, IdxToLabel: map[string]string{"0": "benign", "1": "jailbreak"}}, jailbreakInference: backend}
}

func observationResults() *SignalResults {
	return &SignalResults{SignalConfidences: map[string]float64{}, SignalValues: map[string]float64{}, Metrics: &SignalMetricsCollection{}}
}

func TestGuardRiskObservationBelowThreshold(t *testing.T) {
	for _, risk := range []float32{0, .2} {
		t.Run(fmt.Sprintf("risk_%g", risk), func(t *testing.T) {
			rules := []config.JailbreakRule{{Name: "limit", Threshold: .7}}
			c := observationClassifier(rules, observationBackend{"sample": observationDistribution(risk)}, "allow")
			out := observationResults()
			c.evaluateJailbreakSignalPieces(context.Background(), out, &sync.Mutex{}, []string{"sample"}, nil)
			value, ok := out.SignalValues["jailbreak:limit"]
			if !ok || value != float64(risk) {
				t.Fatalf("risk available=%v value=%v want=%v", ok, value, risk)
			}
			if out.JailbreakDetected || len(out.MatchedJailbreakRules) != 0 || len(out.SignalConfidences) != 0 {
				t.Fatalf("observation changed matches: %+v", out)
			}
			if !out.JailbreakScoreAvailable || out.JailbreakConfidence != risk || out.Metrics.Jailbreak.ConfidenceAvailable == nil || !*out.Metrics.Jailbreak.ConfidenceAvailable {
				t.Fatalf("risk availability lost: %+v", out)
			}
		})
	}
}

func TestGuardRiskObservationScopeAndThreshold(t *testing.T) {
	rules := []config.JailbreakRule{{Name: "current", Threshold: .5}, {Name: "history-high", Threshold: .9, IncludeHistory: true}, {Name: "history-low", Threshold: .6, IncludeHistory: true}}
	backend := observationBackend{"sample": observationDistribution(.2), "prior": observationDistribution(.8)}
	c := observationClassifier(rules, backend, "allow")
	for i := 0; i < 20; i++ {
		out := observationResults()
		c.evaluateJailbreakSignalPieces(context.Background(), out, &sync.Mutex{}, []string{"sample"}, []string{"prior"})
		for name, want := range map[string]float32{"current": .2, "history-high": .8, "history-low": .8} {
			got, ok := out.SignalValues["jailbreak:"+name]
			if !ok || got != float64(want) {
				t.Fatalf("%s risk=%v available=%v", name, got, ok)
			}
		}
		if !out.JailbreakDetected || !slices.Equal(out.MatchedJailbreakRules, []string{"history-low"}) || len(out.SignalConfidences) != 1 || out.JailbreakConfidence != .8 {
			t.Fatalf("threshold aggregation drift: %+v", out)
		}
	}
}

func TestGuardRiskObservationSameScoreCannotHideDetection(t *testing.T) {
	c := observationClassifier(nil, nil, "allow")
	out := observationResults()
	out.JailbreakScoreAvailable = true
	out.JailbreakConfidence = .6
	c.recordJailbreakRuleMatch(config.JailbreakRule{Name: "low", Threshold: .5}, "jailbreak", .6, time.Now(), out, &sync.Mutex{})
	if !out.JailbreakDetected || out.JailbreakType != "jailbreak" {
		t.Fatalf("equal recorded risk hid match: %+v", out)
	}
}

func TestGuardRiskObservationInvalidAndPartial(t *testing.T) {
	partial := observationDistribution(.2)
	partial.result.Input = &tasks.InputUsage{OriginalTokens: 10, ProcessedTokens: 5, Truncated: true}
	invalid := []cachedJailbreakResult{{err: errors.New("synthetic failure")}, partial, {}, {result: SequenceClassificationResult{Probabilities: []float32{1, float32(math.NaN())}}}}
	for _, entry := range invalid {
		for _, policy := range []string{"allow", "block"} {
			c := observationClassifier([]config.JailbreakRule{{Name: "limit", Threshold: .7}}, observationBackend{"sample": entry}, policy)
			out := observationResults()
			c.evaluateJailbreakSignalPieces(context.Background(), out, &sync.Mutex{}, []string{"sample"}, nil)
			if out.JailbreakScoreAvailable || len(out.SignalValues) != 0 || len(out.SignalConfidences) != 0 || out.SignalErrors["jailbreak:limit"] == "" {
				t.Fatalf("invented invalid risk: %+v", out)
			}
			if out.JailbreakDetected != (policy == "block") || out.SignalErrorMatches["jailbreak:limit"] != (policy == "block") {
				t.Fatal("on_error behavior changed")
			}
		}
	}
	for _, risk := range []float32{0, .8} {
		for _, policy := range []string{"allow", "block"} {
			rules := []config.JailbreakRule{{Name: "limit", Threshold: .7, IncludeHistory: true}}
			c := observationClassifier(rules, observationBackend{"sample": observationDistribution(risk), "prior": {err: errors.New("synthetic failure")}}, policy)
			out := observationResults()
			c.evaluateJailbreakSignalPieces(context.Background(), out, &sync.Mutex{}, []string{"sample"}, []string{"prior"})
			got, ok := out.SignalValues["jailbreak:limit"]
			if !ok || got != float64(risk) || out.SignalErrors["jailbreak:limit"] == "" {
				t.Fatalf("valid piece or error lost: %+v", out)
			}
			wantMatch := risk >= .7 || policy == "block"
			if out.JailbreakDetected != wantMatch {
				t.Fatal("partial match behavior changed")
			}
			if out.SignalErrorMatches["jailbreak:limit"] != (risk < .7 && policy == "block") {
				t.Fatal("partial policy sentinel changed")
			}
		}
	}
}

func TestGuardRiskObservationCategoricalHasNoProbability(t *testing.T) {
	for _, label := range []string{"benign", "jailbreak"} {
		c := observationClassifier([]config.JailbreakRule{{Name: "limit", Threshold: .7}}, observationCategorical{label}, "allow")
		out := observationResults()
		c.evaluateJailbreakSignalPieces(context.Background(), out, &sync.Mutex{}, []string{"sample"}, nil)
		if out.JailbreakScoreAvailable || len(out.SignalValues) != 0 || len(out.SignalConfidences) != 0 {
			t.Fatalf("categorical risk invented: %+v", out)
		}
		if out.JailbreakDetected != (label == "jailbreak") {
			t.Fatal("categorical match changed")
		}
	}
}

// The observation change must not alter the established request match path,
// including its legacy zero-score behavior at a zero threshold.
func TestGuardRiskObservationPreservesMatchMatrix(t *testing.T) {
	for _, risk := range []float32{0, .2, .5, .8, 1} {
		for _, threshold := range []float32{0, .2, .5, .8, 1} {
			for _, failed := range []bool{false, true} {
				for _, policy := range []string{"allow", "block"} {
					backend := observationBackend{"sample": observationDistribution(risk)}
					history := []string(nil)
					if failed {
						backend["prior"] = cachedJailbreakResult{err: errors.New("synthetic failure")}
						history = []string{"prior"}
					}
					c := observationClassifier([]config.JailbreakRule{{Name: "limit", Threshold: threshold, IncludeHistory: true}}, backend, policy)
					out := observationResults()
					c.evaluateJailbreakSignalPieces(context.Background(), out, &sync.Mutex{}, []string{"sample"}, history)
					numericMatch := risk > 0 && risk >= threshold
					policyMatch := !numericMatch && failed && policy == "block"
					if out.JailbreakDetected != (numericMatch || policyMatch) || out.SignalErrorMatches["jailbreak:limit"] != policyMatch {
						t.Fatalf("match changed risk=%v threshold=%v failed=%v policy=%s", risk, threshold, failed, policy)
					}
					_, hasConfidence := out.SignalConfidences["jailbreak:limit"]
					if hasConfidence != numericMatch {
						t.Fatal("observation changed matched confidences")
					}
					if got, ok := out.SignalValues["jailbreak:limit"]; !ok || got != float64(risk) {
						t.Fatal("observed risk lost")
					}
				}
			}
		}
	}
}

func TestGuardRiskObservationMatchesSecurityAPI(t *testing.T) {
	for _, risk := range []float32{0, .2, .8} {
		c := observationClassifier([]config.JailbreakRule{{Name: "limit", Threshold: .7}}, observationBackend{"sample": observationDistribution(risk)}, "allow")
		c.Config.PromptGuard.ModelID = "synthetic"
		c.Config.PromptGuard.JailbreakMappingPath = "synthetic"
		verdict, err := c.CheckForJailbreakVerdict(context.Background(), "sample", .7)
		if err != nil {
			t.Fatal(err)
		}
		out := observationResults()
		c.evaluateJailbreakSignalPieces(context.Background(), out, &sync.Mutex{}, []string{"sample"}, nil)
		if verdict.RiskScore == nil || *verdict.RiskScore != out.JailbreakConfidence || verdict.Detected != out.JailbreakDetected {
			t.Fatal("request observation differs from security API")
		}
	}
}

func TestGuardRiskObservationCarriesTheScanWindow(t *testing.T) {
	rules := []config.JailbreakRule{{Name: "limit", Threshold: .7}}
	scanned := observationDistribution(.8)
	scanned.result.Window = &tasks.ScanWindow{Start: 255, End: 765, Count: 2}
	c := observationClassifier(rules, observationBackend{"sample": scanned}, "allow")
	out := observationResults()

	c.evaluateJailbreakSignalPieces(context.Background(), out, &sync.Mutex{}, []string{"sample"}, nil)

	for key, want := range map[string]float64{
		"jailbreak:limit":              .8,
		"jailbreak:limit:window_start": 255,
		"jailbreak:limit:window_end":   765,
		"jailbreak:limit:windows":      2,
	} {
		got, ok := out.SignalValues[key]
		if !ok || math.Abs(got-want) > 1e-6 {
			t.Fatalf("%s available=%v value=%v want=%v", key, ok, got, want)
		}
	}
}

func TestGuardRiskObservationOmitsAWindowItDoesNotHave(t *testing.T) {
	rules := []config.JailbreakRule{{Name: "limit", Threshold: .7}}
	c := observationClassifier(rules, observationBackend{"sample": observationDistribution(.8)}, "allow")
	out := observationResults()

	c.evaluateJailbreakSignalPieces(context.Background(), out, &sync.Mutex{}, []string{"sample"}, nil)

	for _, key := range []string{"jailbreak:limit:window_start", "jailbreak:limit:window_end", "jailbreak:limit:windows"} {
		if _, ok := out.SignalValues[key]; ok {
			t.Fatalf("a scan that reported no window published %s: %v", key, out.SignalValues)
		}
	}
}
