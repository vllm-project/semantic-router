package classification

import (
	"context"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type closingSequenceBackend struct {
	closed bool
}

func (b *closingSequenceBackend) Classify(context.Context, string) (SequenceClassificationResult, error) {
	return SequenceClassificationResult{}, nil
}

func (b *closingSequenceBackend) Close() error {
	b.closed = true
	return nil
}

type closingLabelClassifier struct {
	closed bool
}

func (c *closingLabelClassifier) Classify(context.Context, string) (labelClassification, error) {
	return labelClassification{}, nil
}

func (c *closingLabelClassifier) Close() error {
	c.closed = true
	return nil
}

func TestClassifierCloseClosesRemoteClassifierResources(t *testing.T) {
	jailbreak := &closingSequenceBackend{}
	generic := &closingLabelClassifier{}
	classifier, err := newClassifierWithOptions(
		&config.RouterConfig{},
		withJailbreak(nil, nil, jailbreak),
		withGenericClassifiers(map[string]labelClassifier{"remote": generic}),
	)
	if err != nil {
		t.Fatalf("newClassifierWithOptions() error = %v", err)
	}

	if err := classifier.Close(); err != nil {
		t.Fatalf("Classifier.Close() error = %v", err)
	}
	if !jailbreak.closed || !generic.closed {
		t.Fatalf("closed resources = jailbreak:%t generic:%t", jailbreak.closed, generic.closed)
	}
}
