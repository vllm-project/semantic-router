package native

import (
	"context"
	"fmt"
	"io"
	"slices"

	candle "github.com/vllm-project/semantic-router/candle-binding"
	ort "github.com/vllm-project/semantic-router/onnx-binding/instance"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

func checkResultLabels(got, want []string, size int) error {
	if size != len(want) || !slices.Equal(got, want) {
		return fmt.Errorf("%w: result labels differ from the prepared head", binding.ErrInvalidResult)
	}
	return nil
}

func (r *Runtime) prepareCandleScores(ctx context.Context, spec config.ResolvedModelBinding, windowSize int) (*binding.Resource, *candle.LabelScorer, error) {
	resource, err := r.candleResourceWithWindow(ctx, spec, false, windowSize)
	if err != nil {
		return nil, nil, err
	}
	var model *candle.LabelScorer
	err = resource.Use(ctx, func(value io.Closer) error {
		backbone := value.(*candleBackbone)
		head := spec.Binding.Head
		if head == "" {
			head = spec.Deployment.Artifact
		}
		if backbone.encoder != nil {
			model, err = backbone.encoder.BindLabelScoreHead(head)
		} else if backbone.sequence != nil {
			model, err = backbone.sequence.BindLabelScoreHead(head)
		} else {
			model, err = backbone.tokens.BindLabelScoreHead(head)
		}
		return err
	})
	if err == nil {
		err = resource.Own(model)
	}
	if err != nil {
		if model != nil {
			_ = model.Close()
		}
		_ = resource.Close()
		return nil, nil, err
	}
	return resource, model, nil
}

// Scores binds a multi-label head separately from categorical classification.
// Both use the same resource pool and may share a modern encoder backbone.
func (r *Runtime) Scores(ctx context.Context, spec config.ResolvedModelBinding) (_ *binding.Resolved[string, tasks.LabelScores], callErr error) {
	defer func() { observePreparationFailure(spec, callErr) }()
	if spec.Deployment.Provider == "ort" {
		return r.ortScores(ctx, spec)
	}
	if spec.Deployment.Provider != "candle" {
		return nil, fmt.Errorf("%w: label score provider unavailable", binding.ErrCapability)
	}
	resource, model, err := r.prepareCandleScores(ctx, spec, 0)
	if err != nil {
		return nil, err
	}
	info, err := model.Info()
	if err != nil {
		_ = resource.Close()
		return nil, err
	}
	return finishNativeTask(ctx, spec, r.scores, candleCapability(spec, info), resource,
		func(_ context.Context, _ io.Closer, text string) (tasks.LabelScores, error) {
			result, inferErr := model.Score(text)
			if inferErr != nil {
				return tasks.LabelScores{}, nativeError(inferErr)
			}
			if err := checkResultLabels(result.Labels, info.Labels, len(result.Scores)); err != nil {
				return tasks.LabelScores{}, err
			}
			return tasks.LabelScores{Scores: result.Scores, Input: candleInputUsage(result.Input)}, nil
		}, "warmup")
}

func (r *Runtime) ortScores(ctx context.Context, spec config.ResolvedModelBinding) (*binding.Resolved[string, tasks.LabelScores], error) {
	resource, err := r.ortResource(ctx, spec, "label_scores", func(options ort.Options) (io.Closer, error) { return ort.LoadLabelScorer(options) })
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
	return finishNativeTask(ctx, spec, r.scores, capability, resource,
		func(_ context.Context, value io.Closer, text string) (tasks.LabelScores, error) {
			result, inferErr := value.(*ort.LabelScorer).Score(text)
			if inferErr != nil {
				return tasks.LabelScores{}, ortError(inferErr)
			}
			if err := checkResultLabels(result.Labels, info.Labels, len(result.Scores)); err != nil {
				return tasks.LabelScores{}, err
			}
			return tasks.LabelScores{Scores: result.Scores, Input: ortInputUsage(result.Input)}, nil
		}, "warmup")
}
