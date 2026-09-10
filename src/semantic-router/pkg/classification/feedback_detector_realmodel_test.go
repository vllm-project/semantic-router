package classification

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"testing"

	candle "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// realFeedbackThreshold is the threshold config/config.yaml configures for the
// feedback detector.
const realFeedbackThreshold = 0.7

// realFeedbackModelPath resolves the on-disk mmBERT-32K feedback model, honoring
// the VLLM_SR_FEEDBACK_MODEL override and otherwise looking for the repo-root
// models/ download. It returns "" when the model is not present.
func realFeedbackModelPath() string {
	if p := os.Getenv("VLLM_SR_FEEDBACK_MODEL"); p != "" {
		if _, err := os.Stat(p); err == nil {
			return p
		}
		return ""
	}
	// go test runs with the package directory as the working directory.
	def := filepath.Join("..", "..", "..", "..", "models", "mmbert32k-feedback-detector-merged")
	if _, err := os.Stat(def); err == nil {
		return def
	}
	return ""
}

// setupRealFeedbackDetector initializes the real mmBERT-32K feedback model behind
// a detector configured the way config.yaml configures it. It skips the test when
// the model is not present (e.g. minimal-model CI).
func setupRealFeedbackDetector(t *testing.T) *FeedbackDetector {
	t.Helper()

	modelPath := realFeedbackModelPath()
	if modelPath == "" {
		t.Skip("mmBERT-32K feedback model not present; set VLLM_SR_FEEDBACK_MODEL or run `make download-models`")
	}

	detector, err := NewFeedbackDetector(&config.FeedbackDetectorConfig{
		Enabled:      true,
		ModelID:      modelPath,
		Threshold:    realFeedbackThreshold,
		UseCPU:       true,
		UseMmBERT32K: true,
	})
	if err != nil {
		t.Fatalf("build feedback detector: %v", err)
	}
	if err := detector.Initialize(); err != nil {
		t.Fatalf("initialize feedback detector from %q: %v", modelPath, err)
	}
	return detector
}

// satisfiedIndexOf reports which index the loaded mapping gives the satisfied
// class, so the test reads the probability the detector is expected to report.
func satisfiedIndexOf(t *testing.T, d *FeedbackDetector) int {
	t.Helper()
	for idx, label := range d.mapping.IdxToLabel {
		if label == FeedbackLabelSatisfied {
			parsed, err := strconv.Atoi(idx)
			if err != nil {
				t.Fatalf("mapping index %q is not a number: %v", idx, err)
			}
			return parsed
		}
	}
	t.Fatalf("the loaded mapping names no %q class", FeedbackLabelSatisfied)
	return -1
}

// TestFeedbackConfidenceIsTheSatisfiedProbabilityRealModel exercises the whole
// path against the real model - the mmBERT-32K FFI, the loaded mapping, and the
// threshold rule - and guards the #3534 contract: a below-threshold prediction is
// reported as satisfied, so the confidence beside it is the model's own
// P(satisfied), not 1 - P(argmax).
func TestFeedbackConfidenceIsTheSatisfiedProbabilityRealModel(t *testing.T) {
	detector := setupRealFeedbackDetector(t)
	satisfiedIdx := satisfiedIndexOf(t, detector)

	texts := []string{
		"looks good to me",
		"this is what I wanted",
		"that is not what I asked for",
		"can you explain what you mean by that",
		"give me a different answer",
	}

	for _, text := range texts {
		t.Run(text, func(t *testing.T) {
			result, err := detector.Classify(context.Background(), text)
			if err != nil {
				t.Fatalf("Classify(%q): %v", text, err)
			}
			probs, err := candle.ClassifyMmBert32KFeedbackWithProbs(text)
			if err != nil {
				t.Fatalf("ClassifyMmBert32KFeedbackWithProbs(%q): %v", text, err)
			}
			if len(probs.Probabilities) < 2 {
				t.Fatalf("probabilities = %v, want the full per-class distribution", probs.Probabilities)
			}
			if sum := sumFeedbackProbabilities(probs.Probabilities); math.Abs(float64(sum-1.0)) > 1e-3 {
				t.Errorf("probabilities sum = %.4f, want ~1.0 (softmax distribution)", sum)
			}
			t.Logf("text=%q argmax=%d confidence=%.6f probabilities=%v reported=%s %.6f",
				text, probs.Class, probs.Confidence, probs.Probabilities, result.FeedbackType, result.Confidence)

			if probs.Confidence >= realFeedbackThreshold {
				if result.Confidence != probs.Confidence {
					t.Errorf("above the threshold the reported confidence is %v, want the model's %v",
						result.Confidence, probs.Confidence)
				}
				return
			}
			if result.FeedbackType != FeedbackLabelSatisfied {
				t.Errorf("below the threshold the label is %q, want %q", result.FeedbackType, FeedbackLabelSatisfied)
			}
			want := probs.Probabilities[satisfiedIdx]
			if result.Confidence != want {
				t.Errorf("reported confidence is %v, want P(satisfied) %v", result.Confidence, want)
			}
			if regressed := float32(1.0) - probs.Confidence; result.Confidence == regressed && want != regressed {
				t.Errorf("reported confidence is 1 - P(argmax) (%v), the #3534 defect", regressed)
			}
		})
	}
}

// TestFeedbackConfidenceUnderAThresholdNoPredictionMeetsRealModel raises the
// threshold above what any prediction reaches, so the uncertain branch runs
// whatever the model happens to think of the text.
func TestFeedbackConfidenceUnderAThresholdNoPredictionMeetsRealModel(t *testing.T) {
	detector := setupRealFeedbackDetector(t)
	satisfiedIdx := satisfiedIndexOf(t, detector)
	detector.config.Threshold = 0.999999

	const text = "thanks, that answers it"
	result, err := detector.Classify(context.Background(), text)
	if err != nil {
		t.Fatalf("Classify(%q): %v", text, err)
	}
	probs, err := candle.ClassifyMmBert32KFeedbackWithProbs(text)
	if err != nil {
		t.Fatalf("ClassifyMmBert32KFeedbackWithProbs(%q): %v", text, err)
	}
	t.Logf("text=%q argmax=%d confidence=%.6f probabilities=%v reported=%s %.6f",
		text, probs.Class, probs.Confidence, probs.Probabilities, result.FeedbackType, result.Confidence)

	if result.FeedbackType != FeedbackLabelSatisfied {
		t.Fatalf("the label is %q, want %q", result.FeedbackType, FeedbackLabelSatisfied)
	}
	if want := probs.Probabilities[satisfiedIdx]; result.Confidence != want {
		t.Fatalf("reported confidence is %v, want P(satisfied) %v", result.Confidence, want)
	}
}

func sumFeedbackProbabilities(probs []float32) float32 {
	var sum float32
	for _, p := range probs {
		sum += p
	}
	return sum
}
