package classification

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

type categoricalGuard struct {
	label string
	fail  bool
}

func (b categoricalGuard) Classify(context.Context, string) (SequenceClassificationResult, error) {
	return SequenceClassificationResult{}, tasks.ErrProbabilitiesUnavailable
}

func (b categoricalGuard) Decide(_ context.Context, text string) (tasks.LabelDecision, error) {
	if b.fail && strings.Contains(text, "fail") {
		return tasks.LabelDecision{}, errors.New("remote unavailable")
	}
	label := b.label
	if strings.Contains(text, "attack") {
		label = "jailbreak"
	}
	return tasks.LabelDecision{Label: label, SourceLabel: "unsafe"}, nil
}

func TestCategoricalGuardPreservesPolicyWithoutProbabilities(t *testing.T) {
	classifier := newRiskTestClassifier(categoricalGuard{label: "jailbreak"})
	defer classifier.Close()
	verdict, err := classifier.CheckForJailbreakVerdict(context.Background(), "attack", 0.9999)
	if err != nil || !verdict.Detected || verdict.Label != "jailbreak" || verdict.Confidence != nil || verdict.RiskScore != nil || verdict.Decision == nil {
		t.Fatalf("verdict=%+v err=%v", verdict, err)
	}
	if _, scanErr := classifier.ScanJailbreakRisk(context.Background(), "attack"); !errors.Is(scanErr, tasks.ErrProbabilitiesUnavailable) {
		t.Fatalf("probability API error=%v", scanErr)
	}
	if _, _, _, checkErr := classifier.CheckForJailbreak(context.Background(), "attack"); !errors.Is(checkErr, tasks.ErrProbabilitiesUnavailable) {
		t.Fatalf("confidence API error=%v", checkErr)
	}
	found, detections, err := classifier.AnalyzeContentForJailbreak(context.Background(), []string{"attack"})
	if err != nil || !found || len(detections) != 1 || detections[0].Confidence != nil || detections[0].Decision == nil {
		t.Fatalf("detections=%+v err=%v", detections, err)
	}
}

func TestCategoricalGuardAdmissionAndRoutingHaveNoSyntheticScores(t *testing.T) {
	classifier := newRiskTestClassifier(admittedSequenceClassifier{backend: categoricalGuard{label: "jailbreak"}})
	defer classifier.Close()
	if jailbreakDecisionBackend(classifier.jailbreakInference) == nil {
		t.Fatal("admission hid categorical capability")
	}
	if jailbreakDecisionBackend(admittedSequenceClassifier{backend: withProbsMock([]float32{.9, .1})}) != nil {
		t.Fatal("sequence backend incorrectly became categorical")
	}
	decision := tasks.LabelDecision{Label: "jailbreak", SourceLabel: "unsafe"}
	cache := map[string][]cachedJailbreakResult{"input": {{decision: &decision}}}
	results := &SignalResults{SignalConfidences: map[string]float64{}}
	classifier.evaluateBERTJailbreakRule(config.JailbreakRule{Name: "guard", Threshold: .99}, []string{"input"}, cache, time.Now(), results, &sync.Mutex{})
	if !results.JailbreakDetected || len(results.MatchedJailbreakRules) != 1 || results.JailbreakDecision == nil {
		t.Fatalf("lost categorical routing result: %+v", results)
	}
	if len(results.SignalConfidences) != 0 {
		t.Fatalf("fabricated scores: %v", results.SignalConfidences)
	}
	scan := JailbreakScan{Type: "jailbreak", Decision: &decision, CategoricalMatch: true}
	response := EvaluateResponseJailbreakSignal([]config.JailbreakRule{{Name: "response", Threshold: .999}}, &scan)
	if len(response.MatchedRules) != 1 || len(response.Confidences) != 0 || len(response.Errors) != 0 {
		t.Fatalf("response verdict=%+v", response)
	}
}
