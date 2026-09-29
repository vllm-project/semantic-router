package classification

import (
	"context"
	"fmt"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

// The prepared token task owns one complete input, including all overlap work.
type windowedPIIBackend struct {
	mu      sync.RWMutex
	closed  bool
	spec    config.ResolvedModelBinding
	labels  []string
	window  tasks.TextWindowsRequest
	handle  *binding.Resolved[tasks.TextWindowsRequest, tasks.WindowedTokenClassification]
	prepare func(context.Context) (*binding.Resolved[tasks.TextWindowsRequest, tasks.WindowedTokenClassification], error)
}

func newWindowedPIIBackend(cfg config.PIIModel, mapping *PIIMapping, models ...*classifierModelRuntime) (*windowedPIIBackend, error) {
	runtime := consumerModelRuntime(models)
	spec := runtime.localSpec("pii_classifier", cfg.ModelID, "mmbert32k", config.RemoteClassifierContractTokenSpans, cfg.UseCPU, cfg.MaxSequenceLength)
	if cfg.Window == nil {
		return nil, fmt.Errorf("PII token windows require geometry")
	}
	var err error
	if _, bound := runtime.plan.Lookup(runtime.recipe, "pii_classifier"); bound {
		err = cfg.ValidateBoundWindow(spec.Deployment)
	} else {
		err = cfg.ValidateWindow()
		spec.Deployment.Input.Overflow = "window"
		if spec.Deployment.Input.MaxTokens == 0 {
			spec.Deployment.Input.MaxTokens = 512
		}
	}
	if err != nil {
		return nil, err
	}
	var labels []string
	if mapping != nil {
		labels = indexedNativeLabels(mapping.IdxToLabel)
	}
	window := tasks.TextWindowsRequest{Size: cfg.Window.Size, Overlap: cfg.Window.Overlap}
	return &windowedPIIBackend{spec: spec, labels: labels, window: window, prepare: func(ctx context.Context) (*binding.Resolved[tasks.TextWindowsRequest, tasks.WindowedTokenClassification], error) {
		return runtime.runtime.TokenWindows(ctx, spec, window)
	}}, nil
}

func (b *windowedPIIBackend) Init(_ string, _ bool, _ int) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.closed {
		return binding.ErrClosed
	}
	if b.handle != nil {
		return nil
	}
	handle, err := b.prepare(context.Background())
	if err != nil {
		return err
	}
	if err := validateNativeLabelOrder(handle.Capability().Labels, b.labels, nil); err != nil {
		_ = handle.Close()
		return err
	}
	b.handle = handle
	return nil
}

func (b *windowedPIIBackend) ClassifyTokens(ctx context.Context, text string) (tasks.TokenClassificationResult, error) {
	b.mu.RLock()
	defer b.mu.RUnlock()
	if b.closed || b.handle == nil {
		return tasks.TokenClassificationResult{}, binding.ErrClosed
	}
	input := b.window
	input.Text = text
	result, err := b.handle.Call(ctx, string(b.spec.Recipe), input)
	return result.Result, err
}

func (b *windowedPIIBackend) Close() error {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.closed = true
	if b.handle == nil {
		return nil
	}
	return b.handle.Close()
}
