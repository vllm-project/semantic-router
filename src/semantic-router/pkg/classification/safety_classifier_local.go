package classification

import (
	"context"
	"fmt"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

// ownedLabelTask adapts a typed prepared task to safety policy. All inference,
// admission and cleanup remain on the generation's normal model resource pool.
type ownedLabelTask[I, O any] struct {
	mu      sync.RWMutex
	closed  bool
	labels  []string
	recipe  string
	prepare func(context.Context) (*binding.Resolved[I, O], error)
	input   func(string) I
	convert func(O) (labelClassification, error)
	handle  *binding.Resolved[I, O]
}

func (c *ownedLabelTask[I, O]) Initialize() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return binding.ErrClosed
	}
	if c.handle != nil {
		return nil
	}
	handle, err := c.prepare(context.Background())
	if err != nil {
		return err
	}
	if err := validateNativeLabelOrder(handle.Capability().Labels, c.labels, nil); err != nil {
		_ = handle.Close()
		return err
	}
	c.handle = handle
	return nil
}

func (c *ownedLabelTask[I, O]) Classify(ctx context.Context, text string) (labelClassification, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	if c.closed || c.handle == nil {
		return labelClassification{}, binding.ErrClosed
	}
	result, err := c.handle.Call(ctx, c.recipe, c.input(text))
	if err != nil {
		return labelClassification{}, err
	}
	return c.convert(result)
}

func (c *ownedLabelTask[I, O]) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.closed = true
	if c.handle == nil {
		return nil
	}
	return c.handle.Close()
}
func (*ownedLabelTask[I, O]) ownsAdmission() bool { return true }

func namedLabelScores(labels []string, scores []float32) (map[string]float64, error) {
	if len(labels) != len(scores) {
		return nil, fmt.Errorf("head returned %d scores for %d labels", len(scores), len(labels))
	}
	result := make(map[string]float64, len(labels))
	for i, label := range labels {
		result[label] = float64(scores[i])
	}
	return result, nil
}

func newOwnedSafetyClassifier(models *classifierModelRuntime, spec config.ResolvedModelBinding, labels []string, multiLabel bool, window *config.SequenceHeadWindowConfig) labelClassifier {
	labels = append([]string(nil), labels...)
	if window == nil {
		if multiLabel {
			return &ownedLabelTask[string, tasks.LabelScores]{
				labels: labels, recipe: string(spec.Recipe),
				prepare: func(ctx context.Context) (*binding.Resolved[string, tasks.LabelScores], error) {
					return models.runtime.Scores(ctx, spec)
				},
				input: func(text string) string { return text }, convert: func(out tasks.LabelScores) (labelClassification, error) {
					scores, err := namedLabelScores(labels, out.Scores)
					return labelClassification{Scores: scores}, err
				},
			}
		}
		return &ownedLabelTask[string, tasks.LabelDistribution]{
			labels: labels, recipe: string(spec.Recipe),
			prepare: func(ctx context.Context) (*binding.Resolved[string, tasks.LabelDistribution], error) {
				return models.runtime.Sequence(ctx, spec)
			},
			input: func(text string) string { return text }, convert: func(out tasks.LabelDistribution) (labelClassification, error) {
				scores, err := namedLabelScores(labels, out.Probabilities)
				return labelClassification{Scores: scores}, err
			},
		}
	}
	options := tasks.TextWindowsRequest{Size: window.Size, Overlap: window.Overlap}
	input := func(text string) tasks.TextWindowsRequest { r := options; r.Text = text; return r }
	if multiLabel {
		return &ownedLabelTask[tasks.TextWindowsRequest, tasks.WindowedLabelScores]{
			labels: labels, recipe: string(spec.Recipe),
			prepare: func(ctx context.Context) (*binding.Resolved[tasks.TextWindowsRequest, tasks.WindowedLabelScores], error) {
				return models.runtime.ScoreWindows(ctx, spec, options)
			}, input: input,
			convert: func(out tasks.WindowedLabelScores) (labelClassification, error) {
				result := labelClassification{ScoreWindows: make([]map[string]float64, len(out.Windows))}
				for i, w := range out.Windows {
					scores, err := namedLabelScores(labels, w.Scores)
					if err != nil {
						return labelClassification{}, err
					}
					result.ScoreWindows[i] = scores
				}
				return result, nil
			},
		}
	}
	return &ownedLabelTask[tasks.TextWindowsRequest, tasks.WindowedLabelDistribution]{
		labels: labels, recipe: string(spec.Recipe),
		prepare: func(ctx context.Context) (*binding.Resolved[tasks.TextWindowsRequest, tasks.WindowedLabelDistribution], error) {
			return models.runtime.SequenceWindows(ctx, spec, options)
		}, input: input,
		convert: func(out tasks.WindowedLabelDistribution) (labelClassification, error) {
			result := labelClassification{ScoreWindows: make([]map[string]float64, len(out.Windows))}
			for i, w := range out.Windows {
				scores, err := namedLabelScores(labels, w.Probabilities)
				if err != nil {
					return labelClassification{}, err
				}
				result.ScoreWindows[i] = scores
			}
			return result, nil
		},
	}
}
