package classification

import (
	"context"
	"encoding/json"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestEmptyFactAndFeedbackPolicyDefaultsHaveNoScores(t *testing.T) {
	fact := &FactCheckClassifier{initialized: true}
	feedback := &FeedbackDetector{initialized: true}
	factResult, err := fact.Classify(context.Background(), "")
	if err != nil {
		t.Fatal(err)
	}
	feedbackResult, err := feedback.Classify(context.Background(), "")
	if err != nil {
		t.Fatal(err)
	}
	if factResult.NeedsFactCheck || factResult.Label != FactCheckLabelNotNeeded || feedbackResult.FeedbackType != FeedbackLabelSatisfied {
		t.Fatal("empty-input policy changed")
	}
	for _, result := range []interface{}{factResult, feedbackResult} {
		raw, marshalErr := json.Marshal(result)
		if marshalErr != nil {
			t.Fatal(marshalErr)
		}
		var decoded map[string]interface{}
		if decodeErr := json.Unmarshal(raw, &decoded); decodeErr != nil {
			t.Fatal(decodeErr)
		}
		if decoded["confidence"] != nil || decoded["confidence_available"] != false || decoded["policy_default"] != "empty_text" {
			t.Fatalf("policy default fabricated confidence: %s", raw)
		}
	}
	classifier := &Classifier{Config: &config.RouterConfig{}, factCheckClassifier: fact, feedbackDetector: feedback}
	results := &SignalResults{Metrics: &SignalMetricsCollection{}}
	classifier.evaluateFactCheckSignal(context.Background(), results, &sync.Mutex{}, "")
	classifier.evaluateUserFeedbackSignal(context.Background(), results, &sync.Mutex{}, "", true)
	for _, metric := range []SignalMetrics{results.Metrics.FactCheck, results.Metrics.UserFeedback} {
		if metric.ConfidenceAvailable == nil || *metric.ConfidenceAvailable || metric.PolicyDefault != "empty_text" {
			t.Fatalf("metric lost unavailable policy default: %+v", metric)
		}
	}
}

func TestModalityKeywordPolicyDoesNotBecomeModelConfidence(t *testing.T) {
	classifier := &Classifier{Config: &config.RouterConfig{}}
	classifier.Config.ModalityDetector.Method = config.ModalityDetectionKeyword
	classifier.Config.ModalityDetector.Keywords = []string{"draw"}
	for _, text := range []string{"", "hello", "draw a cat"} {
		result := classifier.classifyModalityWithContext(context.Background(), text, &classifier.Config.ModalityDetector.ModalityDetectionConfig)
		if result.ConfidenceAvailable {
			t.Fatal("heuristic modality became a model score")
		}
		results := &SignalResults{Metrics: &SignalMetricsCollection{}}
		classifier.evaluateModalitySignal(context.Background(), results, &sync.Mutex{}, text)
		raw, err := json.Marshal(results.Metrics.Modality)
		if err != nil {
			t.Fatal(err)
		}
		var decoded map[string]interface{}
		if decodeErr := json.Unmarshal(raw, &decoded); decodeErr != nil {
			t.Fatal(decodeErr)
		}
		if decoded["confidence"] != nil || decoded["confidence_available"] != false || decoded["method"] != result.Method {
			t.Fatalf("heuristic metric exposed a probability: %s", raw)
		}
		want := "AR"
		if text == "draw a cat" {
			want = "DIFFUSION"
		}
		if result.Modality != want {
			t.Fatalf("policy route changed: %+v", result)
		}
	}
}
