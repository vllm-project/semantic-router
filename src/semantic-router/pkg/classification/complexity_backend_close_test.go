package classification

import (
	"context"
	"testing"
)

type closableScorer struct {
	closed bool
}

func (c *closableScorer) Score(ctx context.Context, text string) (float64, error) { return 0, nil }

func (c *closableScorer) Close() error {
	c.closed = true
	return nil
}

// A classifier is rebuilt per recipe and again on every dynamic-config reload,
// so a backend whose connector is never released leaks an idle HTTP transport
// each time.
func TestClassifierCloseReleasesTheComplexityBackends(t *testing.T) {
	scorer := &closableScorer{}
	labels := &closableScorer{}

	classifier := &Classifier{
		complexityScoreBackend: scorer,
		// The label path holds a SequenceClassifierBackend, but Close is
		// discovered by type assertion, so the same recorder serves here.
		complexityLabelBackend: &closableSequenceBackend{closableScorer: labels},
	}

	if err := classifier.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	if !scorer.closed {
		t.Error("the score.v1 backend's connector was not released")
	}
	if !labels.closed {
		t.Error("the label_distribution.v1 backend's connector was not released")
	}
}

type closableSequenceBackend struct {
	*closableScorer
}

func (c *closableSequenceBackend) Classify(ctx context.Context, text string) (SequenceClassificationResult, error) {
	return SequenceClassificationResult{}, nil
}
