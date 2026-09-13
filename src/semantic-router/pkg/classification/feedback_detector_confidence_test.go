package classification

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

// The four classes the shipped detector declares, in the order its config.json
// gives them.
func fourClassMapping() *FeedbackMapping {
	return &FeedbackMapping{IdxToLabel: map[string]string{
		"0": FeedbackLabelSatisfied,
		"1": FeedbackLabelNeedClarification,
		"2": FeedbackLabelWrongAnswer,
		"3": FeedbackLabelWantDifferent,
	}}
}

func TestApplyThresholdReportsTheSatisfiedProbability(t *testing.T) {
	d := &FeedbackDetector{mapping: fourClassMapping()}

	// "looks good to me" through the shipped model: want_different wins at
	// 0.6395, under the configured threshold of 0.7, and P(satisfied) is
	// 3.56e-06. Reporting 1 - 0.6395 claimed 0.3605, which is the mass on
	// wrong_answer plus satisfied, not satisfied alone.
	result := tasks.ClassResultWithProbs{
		Class:         3,
		Confidence:    0.6395,
		Probabilities: []float32{0.00000356, 0.0000412, 0.36048, 0.63946},
		NumClasses:    4,
	}

	label, confidence := d.applyThreshold(FeedbackLabelWantDifferent, result, 0.7)
	if label != FeedbackLabelSatisfied {
		t.Fatalf("below the threshold the label is %q, want %q", label, FeedbackLabelSatisfied)
	}
	if confidence != result.Probabilities[0] {
		t.Fatalf("confidence is %v, want P(satisfied) %v", confidence, result.Probabilities[0])
	}
	if got := float32(1.0) - result.Confidence; confidence == got {
		t.Fatalf("confidence is still 1 - confidence (%v)", got)
	}
}

func TestApplyThresholdLeavesAConfidentPredictionAlone(t *testing.T) {
	d := &FeedbackDetector{mapping: fourClassMapping()}
	result := tasks.ClassResultWithProbs{
		Class:         2,
		Confidence:    0.9999,
		Probabilities: []float32{0.00000001, 0.00000004, 0.9999, 0.0000012},
		NumClasses:    4,
	}

	label, confidence := d.applyThreshold(FeedbackLabelWrongAnswer, result, 0.7)
	if label != FeedbackLabelWrongAnswer || confidence != result.Confidence {
		t.Fatalf("a prediction at or above the threshold changed: %q %v", label, confidence)
	}
}

func TestApplyThresholdReadsTheSatisfiedIndexFromTheMapping(t *testing.T) {
	// The satisfied class is not index 0 in every mapping, so the index the
	// mapping gives it decides which probability is read.
	d := &FeedbackDetector{mapping: &FeedbackMapping{IdxToLabel: map[string]string{
		"0": FeedbackLabelWantDifferent,
		"1": FeedbackLabelWrongAnswer,
		"2": FeedbackLabelSatisfied,
		"3": FeedbackLabelNeedClarification,
	}}}
	result := tasks.ClassResultWithProbs{
		Class:         0,
		Confidence:    0.51,
		Probabilities: []float32{0.51, 0.4, 0.08, 0.01},
		NumClasses:    4,
	}

	label, confidence := d.applyThreshold(FeedbackLabelWantDifferent, result, 0.7)
	if label != FeedbackLabelSatisfied || confidence != result.Probabilities[2] {
		t.Fatalf("applyThreshold() = %q, %v; want %q, %v",
			label, confidence, FeedbackLabelSatisfied, result.Probabilities[2])
	}
}

func TestApplyThresholdKeepsThePredictionWhenSatisfiedIsUnreadable(t *testing.T) {
	// A mapping that names no satisfied class, and a model that returned no
	// probabilities, are both cases where P(satisfied) does not exist.
	for name, d := range map[string]*FeedbackDetector{
		"no satisfied label": {mapping: &FeedbackMapping{
			IdxToLabel: map[string]string{"0": FeedbackLabelWrongAnswer},
		}},
		"no probabilities": {mapping: fourClassMapping()},
	} {
		t.Run(name, func(t *testing.T) {
			result := tasks.ClassResultWithProbs{Class: 3, Confidence: 0.42}
			if name == "no satisfied label" {
				result.Probabilities = []float32{0.42}
				result.NumClasses = 1
			}
			label, confidence := d.applyThreshold(FeedbackLabelWantDifferent, result, 0.7)
			if label != FeedbackLabelWantDifferent || confidence != result.Confidence {
				t.Fatalf("the prediction was replaced: %q %v", label, confidence)
			}
		})
	}
}
