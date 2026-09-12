package native

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"golang.org/x/sync/errgroup"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

// LoRABatch composes the maintained three merged classifiers. It retains one
// typed resource per task; it does not imply one joint or vectorized forward.
type LoRABatch struct {
	mu       sync.RWMutex
	closed   bool
	intent   *binding.Resolved[string, tasks.LabelDistribution]
	pii      *binding.Resolved[string, tasks.TokenClassificationResult]
	security *binding.Resolved[string, tasks.LabelDistribution]
}

type LoRABatchOutput = tasks.ClassificationBatch

func (r *Runtime) LoRABatch(ctx context.Context, intent, pii, security config.ResolvedModelBinding) (_ *LoRABatch, err error) {
	for _, spec := range []config.ResolvedModelBinding{intent, pii, security} {
		if spec.Deployment.Provider != "candle" {
			return nil, fmt.Errorf("%w: merged LoRA task requires Candle", binding.ErrCapability)
		}
	}
	m := &LoRABatch{}
	defer func() {
		if err != nil {
			_ = m.Close()
		}
	}()
	if m.intent, err = r.Sequence(ctx, intent); err != nil {
		return nil, err
	}
	if m.pii, err = r.Tokens(ctx, pii); err != nil {
		return nil, err
	}
	if m.security, err = r.Sequence(ctx, security); err != nil {
		return nil, err
	}
	return m, nil
}

func (m *LoRABatch) Labels() (intent, pii, security []string) {
	return m.intent.Capability().Labels, m.pii.Capability().Labels, m.security.Capability().Labels
}

func (m *LoRABatch) ClassifyBatch(ctx context.Context, recipe string, texts []string) (LoRABatchOutput, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.closed {
		return LoRABatchOutput{}, binding.ErrClosed
	}
	if len(texts) == 0 {
		return LoRABatchOutput{}, fmt.Errorf("empty text batch")
	}
	result := LoRABatchOutput{Intent: make([]tasks.LabelDistribution, len(texts)), PII: make([]tasks.TokenClassificationResult, len(texts)), Security: make([]tasks.LabelDistribution, len(texts))}
	group, ctx := errgroup.WithContext(ctx)
	group.Go(func() error {
		for i, text := range texts {
			value, err := m.intent.Call(ctx, recipe, text)
			if err != nil {
				return err
			}
			result.Intent[i] = value
		}
		return nil
	})
	group.Go(func() error {
		for i, text := range texts {
			value, err := m.pii.Call(ctx, recipe, text)
			if err != nil {
				return err
			}
			result.PII[i] = value
		}
		return nil
	})
	group.Go(func() error {
		for i, text := range texts {
			value, err := m.security.Call(ctx, recipe, text)
			if err != nil {
				return err
			}
			result.Security[i] = value
		}
		return nil
	})
	if err := group.Wait(); err != nil {
		return LoRABatchOutput{}, err
	}
	return result, nil
}

func (m *LoRABatch) Close() error {
	if m == nil {
		return nil
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return nil
	}
	m.closed = true
	var errs []error
	if m.intent != nil {
		errs = append(errs, m.intent.Close())
	}
	if m.pii != nil {
		errs = append(errs, m.pii.Close())
	}
	if m.security != nil {
		errs = append(errs, m.security.Close())
	}
	return errors.Join(errs...)
}
