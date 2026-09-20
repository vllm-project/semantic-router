//go:build openvino && !windows && cgo

package native

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"path/filepath"

	ov "github.com/vllm-project/semantic-router/openvino-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

func openvinoError(err error) error {
	if errors.Is(err, ov.ErrInputTooLong) {
		return fmt.Errorf("%w: %w", binding.ErrInputLimit, err)
	}
	if errors.Is(err, ov.ErrClosed) {
		return fmt.Errorf("%w: %w", binding.ErrClosed, err)
	}
	return err
}

func openvinoUsage(input ov.InputUsage) *tasks.InputUsage {
	return &tasks.InputUsage{OriginalTokens: input.OriginalTokens, ProcessedTokens: input.ProcessedTokens, Truncated: input.Truncated}
}

func (a openvinoArtifact) options() ov.ModelOptions {
	return ov.ModelOptions{ModelPath: a.Graph, Device: a.Device, MaxTokens: a.MaxTokens, Overflow: a.Overflow, EndTokenIDs: a.EndTokenIDs, PadTokenID: a.PadTokenID}
}

func (a openvinoArtifact) capability(spec config.ResolvedModelBinding) binding.Capability {
	return binding.Capability{
		Contract: spec.Binding.Contract, Provider: "openvino", Device: a.Device, Precision: "native", Labels: a.Labels,
		Limits: binding.Limits{ModelTokens: a.ModelTokens, TaskTokens: a.ModelTokens, DeploymentTokens: a.MaxTokens, Overflow: a.Overflow},
	}
}

// Each IR already contains its task head. Distinct graphs/tasks must retain
// independent sessions; compatible recipe references acquire the same handle.
func (r *Runtime) openvinoResource(ctx context.Context, spec config.ResolvedModelBinding, a openvinoArtifact, load func() (io.Closer, error)) (*binding.Resource, binding.ResourceIdentity, error) {
	revision, err := r.artifactRevision(ctx, a.Root)
	if err != nil {
		return nil, binding.ResourceIdentity{}, err
	}
	graphRevision, err := r.artifactRevision(ctx, filepath.Dir(a.Graph))
	if err != nil {
		return nil, binding.ResourceIdentity{}, err
	}
	execution, _ := json.Marshal(struct {
		Options                 ov.ModelOptions
		Contract, GraphRevision string
		Classes                 int
	}{a.options(), spec.Binding.Contract, graphRevision, len(a.Labels)})
	identity := binding.ResourceIdentity{Artifact: a.Root, Revision: spec.Deployment.Revision + ":" + revision, Provider: "openvino", Device: a.Device, Precision: "native", Execution: string(execution)}
	budget, gate := resourceAdmission(spec)
	resource, err := r.Pool.Acquire(ctx, identity, budget, gate, func(context.Context) (io.Closer, error) { return load() })
	return resource, identity, err
}

type openvinoEmbeddingEngine struct{ model *ov.EmbeddingModel }

func (e *openvinoEmbeddingEngine) Close() error { return e.model.Close() }
func (e *openvinoEmbeddingEngine) embed(text string, options embedding.Options) (tasks.EmbeddingResult, error) {
	if options.Layer != 0 {
		return tasks.EmbeddingResult{}, fmt.Errorf("%w: OpenVINO IR has no layer view", binding.ErrCapability)
	}
	out, err := e.model.Embed(text)
	if err != nil {
		return tasks.EmbeddingResult{}, openvinoError(err)
	}
	if options.Dimension != 0 && options.Dimension != len(out.Values) {
		return tasks.EmbeddingResult{}, fmt.Errorf("%w: OpenVINO IR has no dimension crop", binding.ErrCapability)
	}
	return tasks.EmbeddingResult{Embedding: out.Values, Input: openvinoUsage(out.Input)}, nil
}

func (r *Runtime) openvinoEmbedding(ctx context.Context, spec config.ResolvedModelBinding, view embedding.Options) (*preparedEmbedding, error) {
	if view.Layer != 0 {
		return nil, fmt.Errorf("%w: OpenVINO IR has no layer view", binding.ErrCapability)
	}
	a, err := readOpenVINOArtifact(spec)
	if err != nil {
		return nil, err
	}
	// Embedding label metadata does not change graph execution.
	a.Labels = nil
	resource, identity, err := r.openvinoResource(ctx, spec, a, func() (io.Closer, error) {
		model, loadErr := ov.LoadEmbeddingModel(a.options())
		if loadErr != nil {
			return nil, openvinoError(loadErr)
		}
		return &embeddingEngine{openvino: &openvinoEmbeddingEngine{model: model}}, nil
	})
	if err != nil {
		return nil, err
	}
	prepared := &preparedEmbedding{resource: resource, identity: identity, capability: a.capability(spec)}
	err = resource.Use(ctx, func(value io.Closer) error {
		dimension, warmErr := warmEmbeddingModel(value.(*embeddingEngine), view, nil)
		prepared.capability.Embedding = &binding.EmbeddingCapability{Dimension: dimension, Pooling: "mean_or_exported", Normalization: "none", Modalities: []string{"text"}}
		prepared.dimensionContract = nativeOnlyDimensionContract(dimension)
		return warmErr
	})
	if err != nil {
		_ = resource.Close()
		return nil, err
	}
	return prepared, nil
}

func (r *Runtime) openvinoSequence(ctx context.Context, spec config.ResolvedModelBinding) (*binding.Resolved[string, tasks.LabelDistribution], error) {
	a, err := readOpenVINOArtifact(spec)
	if err != nil {
		return nil, err
	}
	if len(a.Labels) == 0 {
		return nil, fmt.Errorf("%w: OpenVINO classifier requires export id2label", binding.ErrCapability)
	}
	resource, _, err := r.openvinoResource(ctx, spec, a, func() (io.Closer, error) {
		model, loadErr := ov.LoadClassifierModel(a.options(), len(a.Labels))
		if loadErr != nil {
			return nil, openvinoError(loadErr)
		}
		return model, nil
	})
	if err != nil {
		return nil, err
	}
	return finishNativeTask(ctx, spec, r.sequence, a.capability(spec), resource, func(_ context.Context, value io.Closer, text string) (tasks.LabelDistribution, error) {
		out, err := value.(*ov.ClassifierModel).Classify(text)
		if err == nil && len(out.Probabilities) != len(a.Labels) {
			return tasks.LabelDistribution{}, fmt.Errorf("%w: OpenVINO graph labels differ from export metadata", binding.ErrInvalidResult)
		}
		return tasks.LabelDistribution{Probabilities: out.Probabilities, Input: openvinoUsage(out.Input)}, openvinoError(err)
	}, "warmup")
}
