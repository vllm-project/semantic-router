package classification

import (
	"context"
	"fmt"
	"sync"

	candle "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/admission"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Model ownership stays within the recipe. Construction records the contract;
// runtime initialization loads weights, and Close drains active predictions.
type nativeSafetyClassifier struct {
	mu         sync.RWMutex
	options    candle.SequenceModelOptions
	model      *candle.SequenceModel
	closed     bool
	gate       admission.Admissioner
	deployment string
}

func newNativeSafetyClassifier(model config.SequenceHeadModelConfig, labels []string, multiLabel bool) (labelClassifier, error) {
	if model.ModelID == "" || model.InputLimit() <= 0 {
		return nil, fmt.Errorf("safety head requires a model path and a positive context limit")
	}
	deployment := "safety"
	if multiLabel {
		deployment = "hazard"
	}
	return &nativeSafetyClassifier{deployment: deployment, options: candle.SequenceModelOptions{
		ModelPath: config.ResolveModelPath(model.ModelID), UseCPU: model.UseCPU,
		MaxSequenceLength: model.InputLimit(), Labels: append([]string(nil), labels...), MultiLabel: multiLabel,
	}}, nil
}

func (c *nativeSafetyClassifier) Initialize() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return fmt.Errorf("safety head is closed")
	}
	if c.model != nil {
		return nil
	}
	model, err := candle.OpenSequenceModel(c.options)
	if err != nil {
		return fmt.Errorf("prepare safety head: %w", err)
	}
	c.model = model
	return nil
}

func (c *nativeSafetyClassifier) Classify(ctx context.Context, input string) (labelClassification, error) {
	return admitModelInference(ctx, c.gate, c.deployment, func() (labelClassification, error) {
		return c.classify(ctx, input)
	})
}

func (c *nativeSafetyClassifier) classify(ctx context.Context, input string) (labelClassification, error) {
	if err := ctx.Err(); err != nil {
		return labelClassification{}, err
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	if c.closed || c.model == nil {
		return labelClassification{}, fmt.Errorf("safety head is not initialized")
	}
	scores, err := c.model.Classify(input)
	if err != nil {
		return labelClassification{}, err
	}
	if err := ctx.Err(); err != nil {
		return labelClassification{}, err
	}
	result := labelClassification{Scores: make(map[string]float64, len(c.options.Labels))}
	for i, label := range c.options.Labels {
		result.Scores[label] = float64(scores[i])
	}
	return result, nil
}

func (c *Classifier) applySafetyAdmissionGates(registry *admission.Registry) {
	for _, detector := range c.safetyClassifiers {
		for _, head := range []labelClassifier{detector.binary, detector.hazard} {
			if shared, ok := head.(*sharedSafetyHead); ok {
				if native, ok := shared.labelClassifier.(*nativeSafetyClassifier); ok {
					native.gate = registry.For(native.deployment)
				}
			}
		}
	}
}

func (c *nativeSafetyClassifier) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.closed = true
	return c.model.Close()
}
