package classification

import (
	"math"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func newLanguageSignalClassifierForTest(t *testing.T, rules []config.LanguageRule) (*Classifier, *LanguageClassifier) {
	t.Helper()

	languageClassifier, err := NewLanguageClassifier(rules)
	if err != nil {
		t.Fatalf("failed to create language classifier: %v", err)
	}

	classifier := &Classifier{
		Config: &config.RouterConfig{
			IntelligentRouting: config.IntelligentRouting{
				Signals: config.Signals{LanguageRules: rules},
			},
		},
		languageClassifier: languageClassifier,
	}

	return classifier, languageClassifier
}

func TestEvaluateLanguageSignal_UsesDefaultThresholdForUnsetRules(t *testing.T) {
	rules := []config.LanguageRule{
		{Name: "ru"},
		{Name: "fr", Threshold: 0.5},
	}
	classifier, languageClassifier := newLanguageSignalClassifierForTest(t, rules)
	text := "Привет, как дела?"

	baseline, err := languageClassifier.Classify(text)
	if err != nil {
		t.Fatalf("baseline classification failed: %v", err)
	}
	if baseline.LanguageCode != "ru" {
		t.Fatalf("expected baseline classification to detect Russian, got %q", baseline.LanguageCode)
	}
	if baseline.Confidence >= 0.5 {
		t.Fatalf("expected probe text confidence below strict threshold, got %f", baseline.Confidence)
	}
	if got := lowestLanguageThreshold(rules); got != float32(defaultLanguageThreshold) {
		t.Fatalf("expected lowest effective threshold %f, got %f", defaultLanguageThreshold, got)
	}

	results := newSignalResultsForTest()
	var mu sync.Mutex
	classifier.evaluateLanguageSignal(results, &mu, text)

	if len(results.MatchedLanguageRules) != 1 || results.MatchedLanguageRules[0] != "ru" {
		t.Fatalf("expected Russian rule to match, got %v", results.MatchedLanguageRules)
	}
	if math.Abs(results.Metrics.Language.Confidence-baseline.Confidence) > 1e-9 {
		t.Fatalf("expected signal confidence %f, got %f", baseline.Confidence, results.Metrics.Language.Confidence)
	}
}

func TestEvaluateLanguageSignal_EnforcesPerRuleThresholdAfterSharedClassification(t *testing.T) {
	rules := []config.LanguageRule{
		{Name: "ru", Threshold: 0.5},
		{Name: "fr"},
	}
	classifier, languageClassifier := newLanguageSignalClassifierForTest(t, rules)
	text := "Привет, как дела?"

	baseline, err := languageClassifier.ClassifyWithThreshold(text, float32(defaultLanguageThreshold))
	if err != nil {
		t.Fatalf("baseline classification failed: %v", err)
	}
	if baseline.LanguageCode != "ru" {
		t.Fatalf("expected baseline classification to detect Russian, got %q", baseline.LanguageCode)
	}
	if baseline.Confidence >= 0.5 {
		t.Fatalf("expected probe text confidence below strict rule threshold, got %f", baseline.Confidence)
	}
	if got := lowestLanguageThreshold(rules); got != float32(defaultLanguageThreshold) {
		t.Fatalf("expected lowest effective threshold %f, got %f", defaultLanguageThreshold, got)
	}

	results := newSignalResultsForTest()
	var mu sync.Mutex
	classifier.evaluateLanguageSignal(results, &mu, text)

	if len(results.MatchedLanguageRules) != 0 {
		t.Fatalf("expected strict Russian rule to be skipped, got %v", results.MatchedLanguageRules)
	}
	if math.Abs(results.Metrics.Language.Confidence-baseline.Confidence) > 1e-9 {
		t.Fatalf("expected signal confidence %f, got %f", baseline.Confidence, results.Metrics.Language.Confidence)
	}
}
