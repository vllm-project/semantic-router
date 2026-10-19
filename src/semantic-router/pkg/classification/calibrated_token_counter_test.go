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
		counter.Observe("code", 8000, 4000)
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
		{8000, 2000},
		{6000, 2000},
		{10000, 2000},
		{5600, 2000},
		{8400, 2000},
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
		classifier.ObserveTokenUsage("", 8000, 4000)
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

func TestCalibratedTokenCounterIgnoresTemplateDominatedShortSamples(t *testing.T) {
	counter := NewCalibratedTokenCounter(WithConservativeEstimate())
	// A one-line chat prompt: 30 content bytes, but the provider reports the
	// chat template and role tokens too.
	for i := 0; i < 50; i++ {
		counter.Observe("", len("What is the capital of France?"), 25)
	}
	if _, _, samples, calibrated := counter.GetRatio(""); samples != 0 || calibrated {
		t.Fatalf("short samples must not calibrate, got samples=%d calibrated=%v", samples, calibrated)
	}

	const proseBytes = 204_643
	if estimate := counter.Estimate("", proseBytes); estimate != (proseBytes+3)/4 {
		t.Fatalf("expected the 4 bytes/token default for long prose, got %d", estimate)
	}
}

func TestCalibratedTokenCounterLongProseStaysBelowLongContextBand(t *testing.T) {
	counter := NewCalibratedTokenCounter(WithConservativeEstimate())
	for i := 0; i < 50; i++ {
		counter.Observe("", 30, 25)
	}
	// Long English prose measured on a production backend: 204,643 bytes
	// were 41,827 prompt tokens.
	for i := 0; i < minCalibrationSamplesForUse; i++ {
		counter.Observe("", 204_643, 41_827)
	}

	// About 55K real tokens of the same prose.
	estimate := counter.Estimate("", 55_000*204_643/41_827)
	if estimate < 50_000 || estimate > 60_000 {
		t.Fatalf("expected a realistic estimate near 55K tokens, got %d", estimate)
	}
}
