package classification

import (
	"context"
	"math"
	"strconv"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func setupRealFeedbackDetector(t *testing.T) *FeedbackDetector {
	t.Helper()
	cfg := config.DefaultGlobalConfig().FeedbackDetector
	cfg.ModelID = requireRealModel(t, "VLLM_SR_FEEDBACK_MODEL", cfg.ModelID)
	detector, err := NewFeedbackDetector(&cfg)
	if err != nil {
		t.Fatalf("build feedback detector: %v", err)
	}
	t.Cleanup(func() {
		if err := detector.Close(); err != nil {
			t.Errorf("close feedback detector: %v", err)
		}
	})
	if err := detector.Initialize(); err != nil {
		t.Fatalf("initialize feedback detector: %v", err)
	}
	assertRealModelCPU(t, detector.backend.handle.Capability())
	if _, ok := detector.mapping.LabelToIdx[FeedbackLabelNoFeedback]; !ok {
		t.Fatalf("published feedback mapping lacks the explicit %q class", FeedbackLabelNoFeedback)
	}
	return detector
}

// The published five-class model preserves its prediction when abstaining;
// uncertainty must never be reported as evidence of satisfaction.
func TestFeedbackPredictionAndAbstentionRealModel(t *testing.T) {
	detector := setupRealFeedbackDetector(t)
	texts := []string{
		"Thank you, that answers my question perfectly.",
		"That answer is incorrect; please check your calculation.",
		"Could you clarify what you mean by that?",
		"Please give me a different answer.",
		"What is the capital of France?",
	}
	for _, text := range texts {
		t.Run(text, func(t *testing.T) {
			result, err := detector.Classify(context.Background(), text)
			if err != nil {
				t.Fatalf("classify feedback: %v", err)
			}
			raw, err := detector.backend.Classify(context.Background(), text)
			if err != nil {
				t.Fatalf("read owned feedback distribution: %v", err)
			}
			assertRealModelDistribution(t, raw.Probabilities, len(detector.mapping.IdxToLabel))
			class, confidence := deriveArgmax(raw.Probabilities)
			label := detector.mapping.IdxToLabel[strconv.Itoa(class)]
			if result.FeedbackType != label || result.Class != class || math.Abs(float64(result.Confidence-confidence)) > 1e-5 {
				t.Fatalf("prediction %+v differs from model class=%d label=%s confidence=%g", result, class, label, confidence)
			}
			if !result.ConfidenceAvailable || result.PolicyDefault != "" || result.Abstained != (confidence < detector.config.Threshold) {
				t.Fatalf("incorrect feedback abstention/confidence contract: %+v", result)
			}
			t.Logf("input=%q result=%+v probabilities=%v", text, result, raw.Probabilities)
		})
	}
}

func TestFeedbackForcedAbstentionPreservesPredictionRealModel(t *testing.T) {
	detector := setupRealFeedbackDetector(t)
	const text = "What is the capital of France?"
	raw, err := detector.backend.Classify(context.Background(), text)
	if err != nil {
		t.Fatal(err)
	}
	assertRealModelDistribution(t, raw.Probabilities, len(detector.mapping.IdxToLabel))
	class, confidence := deriveArgmax(raw.Probabilities)
	// A test-only threshold above the probability range guarantees this branch,
	// including checkpoints whose fp32 softmax rounds the winning score to 1.
	detector.config.Threshold = 1.01
	result, err := detector.Classify(context.Background(), text)
	if err != nil {
		t.Fatal(err)
	}
	if !result.Abstained || !result.ConfidenceAvailable || result.Class != class ||
		result.FeedbackType != detector.mapping.IdxToLabel[strconv.Itoa(class)] ||
		math.Abs(float64(result.Confidence-confidence)) > 1e-5 {
		t.Fatalf("abstention changed the real prediction: result=%+v probabilities=%v", result, raw.Probabilities)
	}
}
