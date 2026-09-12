package classification

import (
	"context"
	"fmt"
	"io"
	"math"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

type remoteSequenceBinding struct {
	handle *binding.Resolved[string, tasks.LabelDistribution]
	recipe string
}

func (b *remoteSequenceBinding) Classify(ctx context.Context, text string) (tasks.LabelDistribution, error) {
	return b.handle.Call(ctx, b.recipe, text)
}
func (b *remoteSequenceBinding) Close() error      { return b.handle.Close() }
func (*remoteSequenceBinding) ownsAdmission() bool { return true }

func prepareRemoteSequence(models *classifierModelRuntime, spec config.ResolvedModelBinding, external *config.ExternalModelConfig, backend SequenceClassifierBackend) (*remoteSequenceBinding, error) {
	closer, ok := backend.(io.Closer)
	if !ok {
		return nil, fmt.Errorf("remote sequence backend has no lifecycle")
	}
	handle, err := remoteTaskBinding(context.Background(), models, spec, external, closer, backend.Classify, func(_ string, out tasks.LabelDistribution) error {
		if len(out.Probabilities) == 0 {
			return fmt.Errorf("missing label distribution")
		}
		for _, probability := range out.Probabilities {
			if math.IsNaN(float64(probability)) || math.IsInf(float64(probability), 0) || probability < 0 || probability > 1 {
				return fmt.Errorf("invalid label probability")
			}
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	return &remoteSequenceBinding{handle: handle, recipe: string(spec.Recipe)}, nil
}

type remoteTokenBinding struct {
	handle *binding.Resolved[string, tasks.TokenClassificationResult]
	recipe string
}

func (b *remoteTokenBinding) ClassifyTokens(ctx context.Context, text string) (tasks.TokenClassificationResult, error) {
	return b.handle.Call(ctx, b.recipe, text)
}
func (b *remoteTokenBinding) Close() error      { return b.handle.Close() }
func (*remoteTokenBinding) ownsAdmission() bool { return true }

func prepareRemoteTokens(models *classifierModelRuntime, spec config.ResolvedModelBinding, external *config.ExternalModelConfig, backend *HTTPTokenClassifierInference) (*remoteTokenBinding, error) {
	handle, err := remoteTaskBinding(context.Background(), models, spec, external, backend, backend.ClassifyTokens, func(text string, out tasks.TokenClassificationResult) error {
		for _, span := range out.Entities {
			if span.Start < 0 || span.End > len(text) || span.Start >= span.End || text[span.Start:span.End] != span.Text {
				return fmt.Errorf("invalid token span")
			}
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	return &remoteTokenBinding{handle: handle, recipe: string(spec.Recipe)}, nil
}

type remoteScoreBinding struct {
	handle *binding.Resolved[string, tasks.ScoreResult]
	recipe string
}

func (b *remoteScoreBinding) Score(ctx context.Context, text string) (float64, error) {
	result, err := b.handle.Call(ctx, b.recipe, text)
	return result.Value, err
}
func (b *remoteScoreBinding) Close() error { return b.handle.Close() }

func prepareRemoteScore(models *classifierModelRuntime, spec config.ResolvedModelBinding, external *config.ExternalModelConfig, backend ScoringBackend) (*remoteScoreBinding, error) {
	closer, ok := backend.(io.Closer)
	if !ok {
		return nil, fmt.Errorf("remote score backend has no lifecycle")
	}
	handle, err := remoteTaskBinding(context.Background(), models, spec, external, closer, func(ctx context.Context, text string) (tasks.ScoreResult, error) {
		score, err := backend.Score(ctx, text)
		return tasks.ScoreResult{Value: score}, err
	}, func(_ string, result tasks.ScoreResult) error {
		return (tasks.ScoreSemantics{Unit: "model_score"}).Validate(result)
	})
	if err != nil {
		return nil, err
	}
	return &remoteScoreBinding{handle: handle, recipe: string(spec.Recipe)}, nil
}
