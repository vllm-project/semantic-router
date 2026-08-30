package classification

import (
	"context"
	"fmt"
	"time"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// categoryHTTPBackend adapts the shared complete-distribution sequence backend
// to category's historical inference interface. Keeping this adapter at the
// family seam means all existing category evaluation, fallback, metrics,
// MatchedDomainRules, and routing-prior consumers remain unchanged.
type categoryHTTPBackend struct {
	backend SequenceClassifierBackend
}

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

func (c *categoryHTTPBackend) Classify(ctx context.Context, text string) (candle_binding.ClassResult, error) {
	result, err := c.classify(ctx, text)
	if err != nil {
		return candle_binding.ClassResult{}, err
	}
	class, confidence := deriveArgmax(result.Probabilities)
	if class < 0 {
		return candle_binding.ClassResult{}, fmt.Errorf("category backend returned an empty probability distribution")
	}
	return candle_binding.ClassResult{Class: class, Confidence: confidence}, nil
}

func (c *categoryHTTPBackend) ClassifyWithProbabilities(ctx context.Context, text string) (candle_binding.ClassResultWithProbs, error) {
	result, err := c.classify(ctx, text)
	if err != nil {
		return candle_binding.ClassResultWithProbs{}, err
	}
	class, confidence := deriveArgmax(result.Probabilities)
	if class < 0 {
		return candle_binding.ClassResultWithProbs{}, fmt.Errorf("category backend returned an empty probability distribution")
	}
	return candle_binding.ClassResultWithProbs{
		Class:         class,
		Confidence:    confidence,
		Probabilities: append([]float32(nil), result.Probabilities...),
	}, nil
}
