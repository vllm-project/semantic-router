package classification

import (
	"context"
	"fmt"
	"math"
	"sync"

	candle "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type jailbreakWindowModel interface {
	ClassifyWindows(string, candle.SequenceWindowOptions) ([]candle.SequenceWindowScores, error)
	Close() error
}

// Each recipe owns its handle; native tokenization preserves IDs and coverage.
type windowedJailbreakBackend struct {
	mu       sync.RWMutex
	model    jailbreakWindowModel
	closed   bool
	labels   []string
	positive []int
	maxInput int
	window   candle.SequenceWindowOptions
	open     func(candle.SequenceModelOptions) (jailbreakWindowModel, error)
}

func newWindowedJailbreakBackend(cfg config.PromptGuardConfig, mapping *JailbreakMapping) (*windowedJailbreakBackend, error) {
	if err := cfg.ValidateWindow(); err != nil {
		return nil, err
	}
	if cfg.Window == nil || mapping == nil || mapping.LabelCount() < 2 {
		return nil, fmt.Errorf("windowed jailbreak model requires a window and complete class mapping")
	}
	labels := make([]string, mapping.LabelCount())
	for index := range labels {
		label, ok := mapping.LabelFromIndex(index)
		if !ok {
			return nil, fmt.Errorf("windowed jailbreak mapping has no class at index %d", index)
		}
		labels[index] = label
	}
	var positive []int
	for _, label := range resolvePositiveLabels(cfg.PositiveLabels) {
		index, ok := mapping.IndexForLabel(label)
		if !ok || index < 0 || index >= len(labels) {
			return nil, fmt.Errorf("windowed jailbreak positive label %q is absent from the model mapping", label)
		}
		positive = append(positive, index)
	}
	return &windowedJailbreakBackend{
		labels: labels, positive: positive,
		maxInput: (config.SequenceHeadModelConfig{MaxSequenceLength: cfg.MaxSequenceLength}).InputLimit(),
		window:   candle.SequenceWindowOptions{Size: cfg.Window.Size, Overlap: cfg.Window.Overlap},
		open: func(options candle.SequenceModelOptions) (jailbreakWindowModel, error) {
			return candle.OpenSequenceModel(options)
		},
	}, nil
}

func (c *windowedJailbreakBackend) Init(modelID string, useCPU bool, classes ...int) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return fmt.Errorf("windowed jailbreak model is closed")
	}
	if modelID == "" || len(classes) != 1 || classes[0] != len(c.labels) {
		return fmt.Errorf("windowed jailbreak initialization differs from its model contract")
	}
	if c.model != nil {
		return nil
	}
	model, err := c.open(candle.SequenceModelOptions{
		ModelPath: config.ResolveModelPath(modelID), UseCPU: useCPU,
		MaxSequenceLength: c.maxInput, Labels: append([]string(nil), c.labels...),
	})
	if err != nil {
		return fmt.Errorf("load windowed jailbreak model: %w", err)
	}
	c.model = model
	return nil
}

func (c *windowedJailbreakBackend) Classify(ctx context.Context, text string) (SequenceClassificationResult, error) {
	if err := ctx.Err(); err != nil {
		return SequenceClassificationResult{}, err
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	if c.closed || c.model == nil {
		return SequenceClassificationResult{}, fmt.Errorf("windowed jailbreak model is not initialized")
	}
	windows, err := c.model.ClassifyWindows(text, c.window)
	if err != nil {
		return SequenceClassificationResult{}, err
	}
	if err := ctx.Err(); err != nil {
		return SequenceClassificationResult{}, err
	}
	return c.riskiestWindow(windows)
}

func validateJailbreakWindowScores(scores []float32, classes int) error {
	if len(scores) != classes {
		return fmt.Errorf("windowed jailbreak result has incorrect class count")
	}
	var sum float64
	for _, score := range scores {
		if math.IsNaN(float64(score)) || math.IsInf(float64(score), 0) || score < 0 || score > 1 {
			return fmt.Errorf("windowed jailbreak result contains invalid probabilities")
		}
		sum += float64(score)
	}
	if math.Abs(sum-1) > 1e-3 {
		return fmt.Errorf("windowed jailbreak probabilities do not sum to one")
	}
	return nil
}

func (c *windowedJailbreakBackend) riskiestWindow(windows []candle.SequenceWindowScores) (SequenceClassificationResult, error) {
	var selected []float32
	best := float32(-1)
	for _, window := range windows {
		if err := validateJailbreakWindowScores(window.Scores, len(c.labels)); err != nil {
			return SequenceClassificationResult{}, err
		}
		var risk float32
		for _, index := range c.positive {
			risk += window.Scores[index]
		}
		if risk > best {
			best, selected = risk, window.Scores
		}
	}
	if selected == nil {
		return SequenceClassificationResult{}, fmt.Errorf("windowed jailbreak model returned no windows")
	}
	// Keep a real distribution: per-class maxima would mix different windows
	// and invent probability mass when a policy combines positive labels.
	return SequenceClassificationResult{Probabilities: append([]float32(nil), selected...)}, nil
}

func (c *windowedJailbreakBackend) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return nil
	}
	c.closed = true
	if c.model == nil {
		return nil
	}
	return c.model.Close()
}
