package native

import (
	"context"
	"fmt"
	"io"

	candle "github.com/vllm-project/semantic-router/candle-binding"
	ort "github.com/vllm-project/semantic-router/onnx-binding/instance"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/operatingpoint"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

// SequenceWindows retains every full window result; routing policy chooses its aggregate.
func (r *Runtime) SequenceWindows(ctx context.Context, spec config.ResolvedModelBinding, window tasks.TextWindowsRequest) (_ *binding.Resolved[tasks.TextWindowsRequest, tasks.WindowedLabelDistribution], callErr error) {
	defer func() { observePreparationFailure(spec, callErr) }()
	if err := tasks.ValidateTextWindows(window); err != nil {
		return nil, err
	}
	if spec.Deployment.Provider == "ort" {
		return r.ortSequenceWindows(ctx, spec, window)
	}
	if spec.Deployment.Provider != "candle" {
		return nil, fmt.Errorf("%w: window provider unavailable", binding.ErrCapability)
	}
	resource, model, err := r.prepareCandleSequence(ctx, windowLoadSpec(spec))
	if err != nil {
		return nil, err
	}
	info, err := model.Info()
	if err != nil {
		_ = resource.Close()
		return nil, err
	}
	capability := candleCapability(spec, info)
	return finishWindowTask(ctx, spec, r.sequenceWindows, capability, resource, window,
		func(_ context.Context, _ io.Closer, input tasks.TextWindowsRequest) (tasks.WindowedLabelDistribution, error) {
			result, err := model.ClassifyWindows(input.Text, candle.SequenceWindowOptions{Size: input.Size, Overlap: input.Overlap})
			if err != nil {
				return tasks.WindowedLabelDistribution{}, nativeError(err)
			}
			output := tasks.WindowedLabelDistribution{Input: candleInputUsage(result.Input), ContentTokens: result.ContentTokens, Windows: make([]tasks.LabelDistributionWindow, len(result.Windows))}
			for i, item := range result.Windows {
				if err := checkResultLabels(result.Labels, info.Labels, len(item.Probabilities)); err != nil {
					return tasks.WindowedLabelDistribution{}, err
				}
				output.Windows[i] = tasks.LabelDistributionWindow{Start: item.Start, End: item.End, Probabilities: item.Probabilities}
			}
			return output, nil
		})
}

func (r *Runtime) ortSequenceWindows(ctx context.Context, spec config.ResolvedModelBinding, window tasks.TextWindowsRequest) (*binding.Resolved[tasks.TextWindowsRequest, tasks.WindowedLabelDistribution], error) {
	resource, err := r.ortResourceWithExecutionLimit(ctx, windowLoadSpec(spec), "sequence", window.Size, func(options ort.Options) (io.Closer, error) { return ort.LoadSequenceClassifier(options) })
	if err != nil {
		return nil, err
	}
	var info ort.Info
	err = resource.Use(ctx, func(value io.Closer) error { var e error; info, e = value.(*ort.SequenceClassifier).Info(); return e })
	var capability binding.Capability
	if err == nil {
		capability, err = ortCapability(spec, info)
	}
	if err != nil {
		_ = resource.Close()
		return nil, err
	}
	return finishWindowTask(ctx, spec, r.sequenceWindows, capability, resource, window,
		func(_ context.Context, value io.Closer, input tasks.TextWindowsRequest) (tasks.WindowedLabelDistribution, error) {
			result, err := value.(*ort.SequenceClassifier).ClassifyWindows(input.Text, ort.SequenceWindowOptions{Size: input.Size, Overlap: input.Overlap})
			if err != nil {
				return tasks.WindowedLabelDistribution{}, ortError(err)
			}
			output := tasks.WindowedLabelDistribution{Input: ortInputUsage(result.Input), ContentTokens: result.ContentTokens, Windows: make([]tasks.LabelDistributionWindow, len(result.Windows))}
			for i, item := range result.Windows {
				if err := checkResultLabels(result.Labels, info.Labels, len(item.Probabilities)); err != nil {
					return tasks.WindowedLabelDistribution{}, err
				}
				output.Windows[i] = tasks.LabelDistributionWindow{Start: item.Start, End: item.End, Probabilities: item.Probabilities}
			}
			return output, nil
		})
}

// ScoreWindows retains every full window result; routing policy chooses its aggregate.
func (r *Runtime) ScoreWindows(ctx context.Context, spec config.ResolvedModelBinding, window tasks.TextWindowsRequest) (_ *binding.Resolved[tasks.TextWindowsRequest, tasks.WindowedLabelScores], callErr error) {
	defer func() { observePreparationFailure(spec, callErr) }()
	if err := tasks.ValidateTextWindows(window); err != nil {
		return nil, err
	}
	if spec.Deployment.Provider == "ort" {
		return r.ortScoreWindows(ctx, spec, window)
	}
	if spec.Deployment.Provider != "candle" {
		return nil, fmt.Errorf("%w: window provider unavailable", binding.ErrCapability)
	}
	resource, model, err := r.prepareCandleScores(ctx, windowLoadSpec(spec))
	if err != nil {
		return nil, err
	}
	info, err := model.Info()
	if err != nil {
		_ = resource.Close()
		return nil, err
	}
	capability := candleCapability(spec, info)
	return finishWindowTask(ctx, spec, r.scoreWindows, capability, resource, window,
		func(_ context.Context, _ io.Closer, input tasks.TextWindowsRequest) (tasks.WindowedLabelScores, error) {
			result, err := model.ScoreWindows(input.Text, candle.SequenceWindowOptions{Size: input.Size, Overlap: input.Overlap})
			if err != nil {
				return tasks.WindowedLabelScores{}, nativeError(err)
			}
			output := tasks.WindowedLabelScores{Input: candleInputUsage(result.Input), ContentTokens: result.ContentTokens, Windows: make([]tasks.LabelScoresWindow, len(result.Windows))}
			for i, item := range result.Windows {
				if err := checkResultLabels(result.Labels, info.Labels, len(item.Scores)); err != nil {
					return tasks.WindowedLabelScores{}, err
				}
				output.Windows[i] = tasks.LabelScoresWindow{Start: item.Start, End: item.End, Scores: item.Scores}
			}
			return output, nil
		})
}

func (r *Runtime) ortScoreWindows(ctx context.Context, spec config.ResolvedModelBinding, window tasks.TextWindowsRequest) (*binding.Resolved[tasks.TextWindowsRequest, tasks.WindowedLabelScores], error) {
	return r.ortScoreWindowsWithPolicy(ctx, spec, window, nil)
}

func (r *Runtime) ortScoreWindowsWithPolicy(ctx context.Context, spec config.ResolvedModelBinding, window tasks.TextWindowsRequest, policy *operatingpoint.Policy) (*binding.Resolved[tasks.TextWindowsRequest, tasks.WindowedLabelScores], error) {
	resource, err := r.ortResourceWithExecutionLimit(ctx, windowLoadSpec(spec), "label_scores", window.Size, func(options ort.Options) (io.Closer, error) { return ort.LoadLabelScorer(options) })
	if err != nil {
		return nil, err
	}
	var info ort.Info
	err = resource.Use(ctx, func(value io.Closer) error { var e error; info, e = value.(*ort.LabelScorer).Info(); return e })
	var capability binding.Capability
	if err == nil {
		capability, err = ortCapability(spec, info)
	}
	if err != nil {
		_ = resource.Close()
		return nil, err
	}
	if policy != nil {
		if err = validateOperatingPointSession(policy, spec, info); err != nil {
			_ = resource.Close()
			return nil, err
		}
	}
	return finishWindowTask(ctx, spec, r.scoreWindows, capability, resource, window,
		func(_ context.Context, value io.Closer, input tasks.TextWindowsRequest) (tasks.WindowedLabelScores, error) {
			result, err := value.(*ort.LabelScorer).ScoreWindows(input.Text, ort.SequenceWindowOptions{Size: input.Size, Overlap: input.Overlap})
			if err != nil {
				return tasks.WindowedLabelScores{}, ortError(err)
			}
			output := tasks.WindowedLabelScores{Input: ortInputUsage(result.Input), ContentTokens: result.ContentTokens, Windows: make([]tasks.LabelScoresWindow, len(result.Windows))}
			for i, item := range result.Windows {
				if err := checkResultLabels(result.Labels, info.Labels, len(item.Scores)); err != nil {
					return tasks.WindowedLabelScores{}, err
				}
				output.Windows[i] = tasks.LabelScoresWindow{Start: item.Start, End: item.End, Scores: item.Scores}
			}
			return output, nil
		})
}

// finishWindowTask owns the shared window contract across providers and head
// types. Providers execute complete windows; routing policy reduces the output.
func finishWindowTask[O any](ctx context.Context, spec config.ResolvedModelBinding, task *binding.Task[tasks.TextWindowsRequest, O], capability binding.Capability, resource *binding.Resource, window tasks.TextWindowsRequest, infer func(context.Context, io.Closer, tasks.TextWindowsRequest) (O, error)) (*binding.Resolved[tasks.TextWindowsRequest, O], error) {
	if window.Size > capability.Limits.EffectiveTokens() {
		_ = resource.Close()
		return nil, fmt.Errorf("%w: window exceeds effective token budget", binding.ErrCapability)
	}
	capability.Limits.Overflow = "window"
	capability.Window = &binding.WindowCapability{Size: window.Size, Overlap: window.Overlap}
	warmup := window
	warmup.Text = "warmup"
	return finishNativeTask(ctx, spec, task, capability, resource,
		func(ctx context.Context, value io.Closer, input tasks.TextWindowsRequest) (O, error) {
			if input.Size != window.Size || input.Overlap != window.Overlap {
				var zero O
				return zero, fmt.Errorf("%w: window settings differ from the prepared consumer", binding.ErrInvalidInput)
			}
			return infer(ctx, value, input)
		}, warmup)
}

// The window operation owns scanning. Individual windows must fit; the native
// loader's single-input overflow option never truncates a windowed request.
func windowLoadSpec(spec config.ResolvedModelBinding) config.ResolvedModelBinding {
	if spec.Deployment.Input.Overflow == "window" {
		spec.Deployment.Input.Overflow = "reject"
	}
	return spec
}
