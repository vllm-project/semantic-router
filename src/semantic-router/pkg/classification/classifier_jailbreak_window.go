package classification

import (
	"context"
	"fmt"
	"math"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

// The backend owns a typed task handle; the runtime owns shared native resources.
type windowedJailbreakBackend struct {
	mu       sync.RWMutex
	closed   bool
	labels   []string
	positive []int
	spec     config.ResolvedModelBinding
	window   tasks.TextWindowsRequest
	handle   *binding.Resolved[tasks.TextWindowsRequest, tasks.WindowedLabelDistribution]
	prepare  func(context.Context) (*binding.Resolved[tasks.TextWindowsRequest, tasks.WindowedLabelDistribution], error)
}

func newWindowedJailbreakBackend(cfg config.PromptGuardConfig, mapping *JailbreakMapping, models ...*classifierModelRuntime) (*windowedJailbreakBackend, error) {
	runtime := consumerModelRuntime(models)
	spec := runtime.localSpec("prompt_guard", cfg.ModelID, "modernbert", config.RemoteClassifierContractLabelDistribution, cfg.UseCPU, cfg.MaxSequenceLength)
	var windowErr error
	if _, bound := runtime.plan.Lookup(runtime.recipe, "prompt_guard"); bound {
		windowErr = cfg.ValidateBoundWindow(spec.Deployment)
	} else {
		windowErr = cfg.ValidateWindow()
	}
	if windowErr != nil {
		return nil, windowErr
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
	window := tasks.TextWindowsRequest{Size: cfg.Window.Size, Overlap: cfg.Window.Overlap}
	return &windowedJailbreakBackend{
		labels: labels, positive: positive, spec: spec, window: window,
		prepare: func(ctx context.Context) (*binding.Resolved[tasks.TextWindowsRequest, tasks.WindowedLabelDistribution], error) {
			return runtime.runtime.SequenceWindows(ctx, spec, window)
		},
	}, nil
}

func (c *windowedJailbreakBackend) Init(_ string, _ bool, classes ...int) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return binding.ErrClosed
	}
	if len(classes) != 1 || classes[0] != len(c.labels) {
		return fmt.Errorf("windowed jailbreak initialization differs from its mapping")
	}
	if c.handle != nil {
		return nil
	}
	handle, err := c.prepare(context.Background())
	if err != nil {
		return err
	}
	limits := handle.Capability().Limits
	if limits.DocumentTokens < c.spec.Deployment.Input.MaxTokens || limits.ForwardTokens() < c.window.Size {
		_ = handle.Close()
		return fmt.Errorf("%w: Guard document budget %d or window %d exceeds prepared scan/forward capacity (%d/%d)", binding.ErrCapability, c.spec.Deployment.Input.MaxTokens, c.window.Size, limits.DocumentTokens, limits.ForwardTokens())
	}
	if err := validateNativeLabelOrder(handle.Capability().Labels, c.labels, nil); err != nil {
		_ = handle.Close()
		return err
	}
	c.handle = handle
	return nil
}

func (c *windowedJailbreakBackend) Classify(ctx context.Context, text string) (SequenceClassificationResult, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	if c.closed || c.handle == nil {
		return SequenceClassificationResult{}, binding.ErrClosed
	}
	input := c.window
	input.Text = text
	output, err := c.handle.Call(ctx, string(c.spec.Recipe), input)
	if err != nil {
		return SequenceClassificationResult{}, err
	}
	if coverageErr := validateJailbreakInputCoverage(output.Input); coverageErr != nil {
		return SequenceClassificationResult{}, coverageErr
	}
	result, err := c.riskiestWindow(output.Windows)
	result.Input = output.Input
	return result, err
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

func (c *windowedJailbreakBackend) riskiestWindow(windows []tasks.LabelDistributionWindow) (SequenceClassificationResult, error) {
	var selected []float32
	var at tasks.LabelDistributionWindow
	best := float32(-1)
	for _, window := range windows {
		if err := validateJailbreakWindowScores(window.Probabilities, len(c.labels)); err != nil {
			return SequenceClassificationResult{}, err
		}
		var risk float32
		for _, index := range c.positive {
			risk += window.Probabilities[index]
		}
		if risk > best {
			best, selected, at = risk, window.Probabilities, window
		}
	}
	if selected == nil {
		return SequenceClassificationResult{}, fmt.Errorf("windowed jailbreak model returned no windows")
	}
	// Keep a real distribution: per-class maxima would mix different windows
	// and invent probability mass when a policy combines positive labels. The
	// window the distribution came from travels with it, since one window of
	// many decided the score a threshold then reads.
	return SequenceClassificationResult{
		Probabilities: append([]float32(nil), selected...),
		Window:        &tasks.ScanWindow{Start: at.Start, End: at.End, Count: len(windows)},
	}, nil
}

func (c *windowedJailbreakBackend) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.closed = true
	if c.handle == nil {
		return nil
	}
	return c.handle.Close()
}
