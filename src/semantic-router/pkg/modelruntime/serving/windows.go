package serving

import (
	"context"
	"fmt"
	"io"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

// SequenceWindows binds a categorical head over overlapping windows of the
// whole document. Every window's distribution is returned; routing policy
// chooses the aggregate.
func (r *Runtime) SequenceWindows(ctx context.Context, spec config.ResolvedModelBinding, window tasks.TextWindowsRequest) (_ *binding.Resolved[tasks.TextWindowsRequest, tasks.WindowedLabelDistribution], callErr error) {
	defer func() { observePreparationFailure(spec, callErr) }()
	ctx, cancel := preparationContext(ctx)
	defer cancel()
	if err := r.rejectQuestionWindows(ctx, spec); err != nil {
		return nil, err
	}
	t, capability, options, err := r.prepareWindows(ctx, spec, kindSequence, window)
	if err != nil {
		return nil, err
	}
	return publish(ctx, r.sequenceWindows, t, capability, func(ctx context.Context, _ io.Closer, input tasks.TextWindowsRequest) (tasks.WindowedLabelDistribution, error) {
		result, err := r.classifyWindows(ctx, t, window, input, options)
		if err == nil {
			err = requireWindowValues(result)
		}
		if err != nil {
			return tasks.WindowedLabelDistribution{}, err
		}
		out := tasks.WindowedLabelDistribution{Input: inputUsage(result.Input), Windows: make([]tasks.LabelDistributionWindow, len(result.Windows))}
		for i, item := range result.Windows {
			out.Windows[i] = tasks.LabelDistributionWindow{Start: item.Start, End: item.End, Probabilities: float32s(item.Probabilities)}
		}
		out.ContentTokens = contentTokens(result.Windows)
		return out, nil
	}, windowWarmup(window))
}

// ScoreWindows binds an independent-label head over overlapping windows.
func (r *Runtime) ScoreWindows(ctx context.Context, spec config.ResolvedModelBinding, window tasks.TextWindowsRequest) (_ *binding.Resolved[tasks.TextWindowsRequest, tasks.WindowedLabelScores], callErr error) {
	defer func() { observePreparationFailure(spec, callErr) }()
	ctx, cancel := preparationContext(ctx)
	defer cancel()
	return r.scoreWindowsTask(ctx, spec, window)
}

func (r *Runtime) scoreWindowsTask(ctx context.Context, spec config.ResolvedModelBinding, window tasks.TextWindowsRequest) (*binding.Resolved[tasks.TextWindowsRequest, tasks.WindowedLabelScores], error) {
	t, capability, options, err := r.prepareWindows(ctx, spec, kindScores, window)
	if err != nil {
		return nil, err
	}
	return publish(ctx, r.scoreWindows, t, capability, func(ctx context.Context, _ io.Closer, input tasks.TextWindowsRequest) (tasks.WindowedLabelScores, error) {
		result, err := r.classifyWindows(ctx, t, window, input, options)
		if err == nil {
			err = requireWindowValues(result)
		}
		if err != nil {
			return tasks.WindowedLabelScores{}, err
		}
		out := tasks.WindowedLabelScores{Input: inputUsage(result.Input), Windows: make([]tasks.LabelScoresWindow, len(result.Windows))}
		for i, item := range result.Windows {
			out.Windows[i] = tasks.LabelScoresWindow{Start: item.Start, End: item.End, Scores: float32s(item.Scores)}
		}
		out.ContentTokens = contentTokens(result.Windows)
		return out, nil
	}, windowWarmup(window))
}

// TokenWindows binds a token head over overlapping windows and returns one
// globally decoded span set with the exact window coverage.
func (r *Runtime) TokenWindows(ctx context.Context, spec config.ResolvedModelBinding, window tasks.TextWindowsRequest) (_ *binding.Resolved[tasks.TextWindowsRequest, tasks.WindowedTokenClassification], callErr error) {
	defer func() { observePreparationFailure(spec, callErr) }()
	if spec.Deployment.Input.Overflow != "window" {
		return nil, fmt.Errorf("%w: token windows require explicit window overflow", binding.ErrCapability)
	}
	ctx, cancel := preparationContext(ctx)
	defer cancel()
	t, capability, options, err := r.prepareWindows(ctx, spec, kindToken, window)
	if err != nil {
		return nil, err
	}
	return publish(ctx, r.tokenWindows, t, capability, func(ctx context.Context, _ io.Closer, input tasks.TextWindowsRequest) (tasks.WindowedTokenClassification, error) {
		result, err := r.classifyWindows(ctx, t, window, input, options)
		if err != nil {
			return tasks.WindowedTokenClassification{}, err
		}
		if len(result.Windows) == 0 && (result.Input == nil || result.Input.Windows == nil || *result.Input.Windows < 1) {
			return tasks.WindowedTokenClassification{}, fmt.Errorf("%w: windowed result reports no windows", binding.ErrInvalidResult)
		}
		spans, err := tokenResult(input.Text, result)
		if err != nil {
			return tasks.WindowedTokenClassification{}, err
		}
		out := tasks.WindowedTokenClassification{Result: spans, ContentTokens: contentTokens(result.Windows)}
		for _, item := range result.Windows {
			out.Windows = append(out.Windows, [2]int{item.Start, item.End})
		}
		return out, nil
	}, windowWarmup(window))
}

// prepareWindows checks the geometry against the card: one window must fit
// a forward, and the deployment's document budget must fit what the runtime
// scans. The runtime owns tokenization, offsets and coverage.
func (r *Runtime) prepareWindows(ctx context.Context, spec config.ResolvedModelBinding, kind string, window tasks.TextWindowsRequest) (*target, binding.Capability, classifyOptions, error) {
	if err := tasks.ValidateTextWindows(window); err != nil {
		return nil, binding.Capability{}, classifyOptions{}, err
	}
	budget := spec.Deployment.Input.MaxTokens
	if budget <= 0 {
		return nil, binding.Capability{}, classifyOptions{}, fmt.Errorf("%w: a window task needs a positive document budget (input.max_tokens)", binding.ErrCapability)
	}
	if window.Size > budget {
		return nil, binding.Capability{}, classifyOptions{}, fmt.Errorf("%w: window exceeds the document budget", binding.ErrCapability)
	}
	t, capability, err := r.prepareHead(ctx, spec, kind, inputText)
	if err != nil {
		return nil, binding.Capability{}, classifyOptions{}, err
	}
	capability.Limits.DocumentTokens = t.card.MaxInputTokens
	capability.Limits.Overflow = "window"
	capability.Window = &binding.WindowCapability{Size: window.Size, Overlap: window.Overlap}
	if t.card.MaxInputTokens <= 0 || capability.Limits.ForwardTokens() < window.Size || capability.Limits.DocumentTokens < budget {
		_ = t.resource.Close()
		return nil, binding.Capability{}, classifyOptions{}, fmt.Errorf("%w: deployment %q cannot scan %d tokens in %d-token windows", binding.ErrCapability, spec.Binding.Deployment, budget, window.Size)
	}
	options := classifyOptions{overflow: "window", maxTokens: budget, window: &modelservice.Window{Tokens: window.Size, Overlap: window.Overlap}}
	return t, capability, options, nil
}

// classifyWindows runs one document; the request must keep the prepared
// geometry.
func (r *Runtime) classifyWindows(ctx context.Context, t *target, prepared, input tasks.TextWindowsRequest, options classifyOptions) (modelservice.ClassifyResult, error) {
	if input.Size != prepared.Size || input.Overlap != prepared.Overlap {
		return modelservice.ClassifyResult{}, fmt.Errorf("%w: window settings differ from the prepared consumer", binding.ErrInvalidInput)
	}
	return r.classify(ctx, t, modelservice.ClassifyInput{Text: input.Text}, options)
}

// requireWindowValues checks that a sequence or score head returned every
// window's values; routing policy reduces them. Token heads merge their spans
// in the runtime and report only the window count.
func requireWindowValues(result modelservice.ClassifyResult) error {
	if len(result.Windows) == 0 {
		return fmt.Errorf("%w: windowed result reports no windows", binding.ErrInvalidResult)
	}
	return nil
}

func contentTokens(windows []modelservice.ClassifyWindow) int {
	if len(windows) == 0 {
		return 0
	}
	return windows[len(windows)-1].End
}

func windowWarmup(window tasks.TextWindowsRequest) tasks.TextWindowsRequest {
	window.Text = "warmup"
	return window
}
