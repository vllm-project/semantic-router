package native

import (
	"context"
	"fmt"
	"io"

	candle "github.com/vllm-project/semantic-router/candle-binding"
	ort "github.com/vllm-project/semantic-router/onnx-binding/instance"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

func validateWindowTokens(input tasks.TextWindowsRequest, output tasks.WindowedTokenClassification) error {
	if output.Result.TruncatedAt != nil || !output.Result.HasScores() {
		return fmt.Errorf("token windows returned partial or unscored spans")
	}
	if err := tasks.ValidateWindowCoverage(output.ContentTokens, output.Windows, output.Result.Input); err != nil {
		return err
	}
	return validateSpans(input.Text, output.Result)
}

// TokenWindows scans exact token IDs and returns a single globally decoded BIO
// result. Window geometry is consumer policy; the native provider owns offsets.
func (r *Runtime) TokenWindows(ctx context.Context, spec config.ResolvedModelBinding, window tasks.TextWindowsRequest) (_ *binding.Resolved[tasks.TextWindowsRequest, tasks.WindowedTokenClassification], callErr error) {
	defer func() { observePreparationFailure(spec, callErr) }()
	if err := tasks.ValidateTextWindows(window); err != nil {
		return nil, err
	}
	if spec.Deployment.Input.Overflow != "window" {
		return nil, fmt.Errorf("%w: token windows require explicit window overflow", binding.ErrCapability)
	}
	if spec.Deployment.Provider == "ort" {
		return r.ortTokenWindows(ctx, spec, window)
	}
	if spec.Deployment.Provider != "candle" {
		return nil, fmt.Errorf("%w: token window provider unavailable", binding.ErrCapability)
	}
	resource, model, info, err := r.prepareCandleTokens(ctx, windowLoadSpec(spec), window.Size)
	if err != nil {
		return nil, err
	}
	capability := candleCapability(spec, info)
	capability.Limits.DocumentTokens = info.DocumentMaxInputTokens
	if err := validateTokenWindowBudget(spec, capability); err != nil {
		_ = resource.Close()
		return nil, err
	}
	return finishWindowTask(ctx, spec, r.tokenWindows, capability, resource, window,
		func(_ context.Context, _ io.Closer, input tasks.TextWindowsRequest) (tasks.WindowedTokenClassification, error) {
			result, err := model.ClassifyWindows(input.Text, candle.SequenceWindowOptions{Size: input.Size, Overlap: input.Overlap})
			if err != nil {
				return tasks.WindowedTokenClassification{}, nativeError(err)
			}
			available := true
			out := tasks.TokenClassificationResult{Input: candleInputUsage(result.Input), ScoresAvailable: &available}
			for _, span := range result.Spans {
				out.Entities = append(out.Entities, tasks.TokenEntity{EntityType: span.Label, Start: span.Start, End: span.End, Text: span.Text, Confidence: span.Confidence})
			}
			return tasks.WindowedTokenClassification{Result: out, ContentTokens: result.ContentTokens, Windows: result.Windows}, nil
		})
}

func validateTokenWindowBudget(spec config.ResolvedModelBinding, capability binding.Capability) error {
	limit := spec.Deployment.Input.MaxTokens
	if limit <= 0 || capability.Limits.DocumentTokens < limit {
		return fmt.Errorf("%w: native token task cannot cover the document budget", binding.ErrCapability)
	}
	return nil
}

func (r *Runtime) ortTokenWindows(ctx context.Context, spec config.ResolvedModelBinding, window tasks.TextWindowsRequest) (*binding.Resolved[tasks.TextWindowsRequest, tasks.WindowedTokenClassification], error) {
	resource, err := r.ortResourceWithExecutionLimit(ctx, windowLoadSpec(spec), "token", window.Size, func(options ort.Options) (io.Closer, error) { return ort.LoadTokenClassifier(options) })
	if err != nil {
		return nil, err
	}
	var info ort.Info
	err = resource.Use(ctx, func(value io.Closer) error { var e error; info, e = value.(*ort.TokenClassifier).Info(); return e })
	var capability binding.Capability
	if err == nil {
		capability, err = ortCapability(spec, info)
		capability.Limits.DocumentTokens = info.DocumentMaxInputTokens
	}
	if err == nil {
		err = validateTokenWindowBudget(spec, capability)
	}
	if err != nil {
		_ = resource.Close()
		return nil, err
	}
	return finishWindowTask(ctx, spec, r.tokenWindows, capability, resource, window,
		func(_ context.Context, value io.Closer, input tasks.TextWindowsRequest) (tasks.WindowedTokenClassification, error) {
			result, err := value.(*ort.TokenClassifier).DetectWindows(input.Text, ort.SequenceWindowOptions{Size: input.Size, Overlap: input.Overlap})
			if err != nil {
				return tasks.WindowedTokenClassification{}, ortError(err)
			}
			available := true
			out := tasks.TokenClassificationResult{Input: ortInputUsage(result.Input), ScoresAvailable: &available}
			for _, span := range result.Spans {
				out.Entities = append(out.Entities, tasks.TokenEntity{EntityType: span.EntityType, Start: span.Start, End: span.End, Text: span.Text, Confidence: span.Confidence})
			}
			return tasks.WindowedTokenClassification{Result: out, ContentTokens: result.ContentTokens, Windows: result.Windows}, nil
		})
}
