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
	window     *candle.SequenceWindowOptions
}

func newNativeSafetyClassifier(model config.SequenceHeadModelConfig, labels []string, multiLabel bool) (labelClassifier, error) {
	if model.ModelID == "" || model.InputLimit() <= 0 {
		return nil, fmt.Errorf("safety head requires a model path and a positive context limit")
	}
	if err := model.ValidateWindow(); err != nil {
		return nil, err
	}
	var window *candle.SequenceWindowOptions
	if model.Window != nil {
		window = &candle.SequenceWindowOptions{Size: model.Window.Size, Overlap: model.Window.Overlap}
	}
	deployment := "safety"
	if multiLabel {
		deployment = "hazard"
	}
	return &nativeSafetyClassifier{deployment: deployment, window: window, options: candle.SequenceModelOptions{
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
	if c.window != nil {
		windows, err := c.model.ClassifyWindows(input, *c.window)
		if err != nil {
			return labelClassification{}, err
		}
		if err := ctx.Err(); err != nil {
			return labelClassification{}, err
		}
		result := labelClassification{ScoreWindows: make([]map[string]float64, len(windows))}
		for index, window := range windows {
			scores := make(map[string]float64, len(c.options.Labels))
			for i, label := range c.options.Labels {
				scores[label] = float64(window.Scores[i])
			}
			result.ScoreWindows[index] = scores
		}
		return result, nil
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
