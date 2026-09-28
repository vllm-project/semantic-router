package classification

import (
	"context"
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

// categoryHTTPBackend adapts the shared complete-distribution sequence backend
// to category's historical inference interface. Keeping this adapter at the
// family seam means all existing category evaluation, fallback, metrics,
// MatchedDomainRules, and routing-prior consumers remain unchanged.
type categoryHTTPBackend struct {
	backend SequenceClassifierBackend
}

func (c *categoryHTTPBackend) Close() error {
	if closer, ok := c.backend.(interface{ Close() error }); ok {
		return closer.Close()
	}
	return nil
}

func (c *categoryHTTPBackend) ownsAdmission() bool { return ownsModelAdmission(c.backend) }

// fallbackToTop1OnProbabilityError reports whether the historical evaluator
// fallback is meaningful for this implementation. The remote endpoint already
// returns the complete distribution for both interface methods, so retrying it
// through Classify would duplicate a failed request without changing its
// semantics.
func (*categoryHTTPBackend) fallbackToTop1OnProbabilityError() bool { return false }

func newCategoryHTTPBackend(
	external *config.ExternalModelConfig,
	mapping *CategoryMapping,
	deadline time.Duration,
) (CategoryInference, error) {
	if external == nil {
		return nil, fmt.Errorf("category backend external model is required")
	}
	backend, err := newHTTPClassifierInference(external, mapping, deadline)
	if err != nil {
		return nil, fmt.Errorf("failed to create category http_classify backend: %w", err)
	}
	return &categoryHTTPBackend{backend: backend}, nil
}

func (c *categoryHTTPBackend) classify(ctx context.Context, text string) (SequenceClassificationResult, error) {
	return c.backend.Classify(ctx, text)
}

func (c *categoryHTTPBackend) Classify(ctx context.Context, text string) (tasks.ClassResult, error) {
	result, err := c.classify(ctx, text)
	if err != nil {
		return tasks.ClassResult{}, err
	}
	class, confidence := deriveArgmax(result.Probabilities)
	if class < 0 {
		return tasks.ClassResult{}, fmt.Errorf("category backend returned an empty probability distribution")
	}
	return tasks.ClassResult{Class: class, Confidence: confidence}, nil
}

func (c *categoryHTTPBackend) ClassifyWithProbabilities(ctx context.Context, text string) (tasks.ClassResultWithProbs, error) {
	result, err := c.classify(ctx, text)
	if err != nil {
		return tasks.ClassResultWithProbs{}, err
	}
	class, confidence := deriveArgmax(result.Probabilities)
	if class < 0 {
		return tasks.ClassResultWithProbs{}, fmt.Errorf("category backend returned an empty probability distribution")
	}
	return tasks.ClassResultWithProbs{
		Class:         class,
		Confidence:    confidence,
		Probabilities: append([]float32(nil), result.Probabilities...),
	}, nil
}
