package classification

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestCalibratedTokenCounterFallsBackBeforeCalibration(t *testing.T) {
	counter := NewCalibratedTokenCounter()
	count, err := counter.CountTokens(strings.Repeat("x", 400))
	if err != nil {
		t.Fatalf("CountTokens returned error: %v", err)
	}
	if count != 100 {
		t.Fatalf("expected char/4 fallback, got %d", count)
	}
}

func TestCalibratedTokenCounterLearnsObservedRatio(t *testing.T) {
	counter := NewCalibratedTokenCounter(WithDecay(0.9))
	for i := 0; i < 20; i++ {
		counter.Observe("code", 2000, 1000)
	}

	estimate := counter.Estimate("code", 4000)
	if estimate < 1900 || estimate > 2100 {
		t.Fatalf("expected calibrated estimate near 2000, got %d", estimate)
	}

	mean, _, samples, calibrated := counter.GetRatio("code")
	if !calibrated {
		t.Fatalf("expected category to be calibrated after enough samples")
	}
	if samples != 20 {
		t.Fatalf("expected 20 samples, got %d", samples)
	}
	if mean < 1.9 || mean > 2.1 {
		t.Fatalf("expected learned ratio near 2.0, got %.3f", mean)
	}
}

func TestCalibratedTokenCounterConservativeEstimateAvoidsUnderCount(t *testing.T) {
	mean := NewCalibratedTokenCounter(WithDecay(0.9))
	conservative := NewCalibratedTokenCounter(WithDecay(0.9), WithConservativeEstimate())
	observations := []struct {
		bytes  int
		tokens int
	}{
		{4000, 1000},
		{3000, 1000},
		{5000, 1000},
		{2800, 1000},
		{4200, 1000},
	}
	for i := 0; i < 4; i++ {
		for _, obs := range observations {
			mean.Observe("mixed", obs.bytes, obs.tokens)
			conservative.Observe("mixed", obs.bytes, obs.tokens)
		}
	}

	meanEstimate := mean.Estimate("mixed", 4000)
	conservativeEstimate := conservative.Estimate("mixed", 4000)
	if conservativeEstimate < meanEstimate {
		t.Fatalf("expected conservative estimate >= mean estimate, got %d < %d", conservativeEstimate, meanEstimate)
	}
}

func TestBuildClassifierUsesCalibratedContextCounter(t *testing.T) {
	classifier, err := BuildClassifier(&config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Signals: config.Signals{
				ContextRules: []config.ContextRule{{
					Name:      "long_context",
					MinTokens: config.TokenCount("0"),
					MaxTokens: config.TokenCount("10K"),
				}},
			},
		},
	}, nil, nil, nil)
	if err != nil {
		t.Fatalf("BuildClassifier returned error: %v", err)
	}

	for i := 0; i < 20; i++ {
		classifier.ObserveTokenUsage("", 2000, 1000)
	}

	_, _, _, calibrated := classifier.TokenCalibrationRatio("")
	if !calibrated {
		t.Fatalf("expected classifier token calibrator to be active")
	}

	_, tokenCount, err := classifier.contextClassifier.Classify(strings.Repeat("x", 4000))
	if err != nil {
		t.Fatalf("context Classify returned error: %v", err)
	}
	if tokenCount < 1900 || tokenCount > 2100 {
		t.Fatalf("expected calibrated context token count near 2000, got %d", tokenCount)
	}
}
