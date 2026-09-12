package native

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"

	candle "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

type candleBackbone struct {
	encoder  *candle.Backbone
	sequence *candle.SequenceClassifier
	tokens   *candle.TokenClassifier
}

func (b *candleBackbone) Close() error {
	if b.encoder != nil {
		return b.encoder.Close()
	}
	if b.sequence != nil {
		return b.sequence.Close()
	}
	return b.tokens.Close()
}

func candleOptions(spec config.ResolvedModelBinding) candle.InstanceOptions {
	d := spec.Deployment.WithDefaults()
	precision := d.Precision
	if precision == "native" || precision == "fp32" {
		precision = "float32"
	}
	adapter := spec.Binding.Adapter
	if adapter == "auto" {
		adapter = ""
	}
	return candle.InstanceOptions{ModelPath: d.Artifact, ModelType: adapter, Device: d.Device, Precision: precision, MaxInputTokens: d.Input.MaxTokens, Overflow: d.Input.Overflow}
}

func candleSharesBackbone(adapter string) bool {
	switch adapter {
	case "modernbert", "mmbert", "mmbert32k", "mmbert-32k":
		return true
	default:
		return false
	}
}

func (r *Runtime) candleResource(ctx context.Context, spec config.ResolvedModelBinding, tokenTask bool) (*binding.Resource, error) {
	options := candleOptions(spec)
	revision, err := r.artifactRevision(ctx, options.ModelPath)
	if err != nil {
		return nil, err
	}
	if options.ModelType == "" {
		data, readErr := os.ReadFile(filepath.Join(options.ModelPath, "config.json"))
		if readErr != nil {
			return nil, readErr
		}
		var metadata struct {
			ModelType string `json:"model_type"`
		}
		if decodeErr := json.Unmarshal(data, &metadata); decodeErr != nil {
			return nil, decodeErr
		}
		options.ModelType = metadata.ModelType
		if options.ModelType == "" {
			options.ModelType = "modernbert"
		}
	}
	// Current modern sequence/token adapters share their immutable backbone.
	// Other families are keyed by task as well, until their provider explicitly
	// implements cross-head sharing.
	execution, _ := json.Marshal(struct {
		Options   candle.InstanceOptions
		TokenTask bool
	}{options, tokenTask && !candleSharesBackbone(options.ModelType)})
	identity := binding.ResourceIdentity{Artifact: options.ModelPath, Revision: spec.Deployment.Revision + ":" + revision, Provider: "candle", Device: options.Device, Precision: options.Precision, Execution: string(execution)}
	budget, gate := resourceAdmission(spec)
	return r.Pool.Acquire(ctx, identity, budget, gate, func(context.Context) (io.Closer, error) {
		if candleSharesBackbone(options.ModelType) {
			encoder, loadErr := candle.LoadBackbone(options)
			if loadErr != nil {
				return nil, nativeError(loadErr)
			}
			return &candleBackbone{encoder: encoder}, nil
		}
		if tokenTask {
			model, err := candle.LoadTokenClassifier(options)
			if err != nil {
				return nil, err
			}
			return &candleBackbone{tokens: model}, nil
		}
		model, err := candle.LoadSequenceClassifier(options)
		if err != nil {
			return nil, err
		}
		return &candleBackbone{sequence: model}, nil
	})
}

func (r *Runtime) Sequence(ctx context.Context, spec config.ResolvedModelBinding) (_ *binding.Resolved[string, tasks.LabelDistribution], callErr error) {
	defer func() { observePreparationFailure(spec, callErr) }()
	if spec.Deployment.Provider == "ort" {
		return r.ortSequence(ctx, spec)
	}
	if spec.Deployment.Provider != "candle" {
		return nil, fmt.Errorf("%w: sequence provider %q is unavailable", binding.ErrCapability, spec.Deployment.Provider)
	}
	resource, err := r.candleResource(ctx, spec, false)
	if err != nil {
		return nil, err
	}
	var model *candle.SequenceClassifier
	err = resource.Use(ctx, func(value io.Closer) error {
		backbone := value.(*candleBackbone)
		head := spec.Binding.Head
		if head == "" && backbone.sequence != nil {
			model, err = backbone.sequence.Clone()
			return err
		}
		if head == "" {
			head = spec.Deployment.Artifact
		}
		if backbone.encoder != nil {
			model, err = backbone.encoder.BindSequenceHead(head)
		} else if backbone.sequence != nil {
			model, err = backbone.sequence.BindSequenceHead(head)
		} else {
			model, err = backbone.tokens.BindSequenceHead(head)
		}
		return err
	})
	if err != nil {
		// Use may observe cancellation after Clone/BindHead completed.
		// Ownership has not yet transferred to the resource in this branch.
		if model != nil {
			_ = model.Close()
		}
		_ = resource.Close()
		return nil, err
	}
	if err = resource.Own(model); err != nil {
		_ = model.Close()
		_ = resource.Close()
		return nil, err
	}
	info, err := model.Info()
	if err != nil {
		_ = resource.Close()
		return nil, err
	}
	capability := candleCapability(spec, info)
	bound, err := r.sequence.Resolve(taskIdentity(spec), capability, resource, func(_ context.Context, _ io.Closer, text string) (tasks.LabelDistribution, error) {
		result, inferErr := model.Classify(text)
		return tasks.LabelDistribution{Probabilities: result.Probabilities, Input: candleInputUsage(result.Input)}, nativeError(inferErr)
	})
	if err == nil {
		_, err = bound.Call(ctx, string(spec.Recipe), "warmup")
	}
	if err != nil {
		_ = resource.Close()
		return nil, err
	}
	bound.Ready()
	return bound, nil
}

func (r *Runtime) Tokens(ctx context.Context, spec config.ResolvedModelBinding) (_ *binding.Resolved[string, tasks.TokenClassificationResult], callErr error) {
	defer func() { observePreparationFailure(spec, callErr) }()
	if spec.Deployment.Provider == "ort" {
		return r.ortTokens(ctx, spec)
	}
	if spec.Deployment.Provider != "candle" {
		return nil, fmt.Errorf("%w: token provider %q is unavailable", binding.ErrCapability, spec.Deployment.Provider)
	}
	resource, err := r.candleResource(ctx, spec, true)
	if err != nil {
		return nil, err
	}
	var model *candle.TokenClassifier
	err = resource.Use(ctx, func(value io.Closer) error {
		backbone := value.(*candleBackbone)
		head := spec.Binding.Head
		if head == "" && backbone.tokens != nil {
			model, err = backbone.tokens.Clone()
			return err
		}
		if head == "" {
			head = spec.Deployment.Artifact
		}
		if backbone.encoder != nil {
			model, err = backbone.encoder.BindTokenHead(head)
		} else if backbone.tokens != nil {
			model, err = backbone.tokens.BindTokenHead(head)
		} else {
			model, err = backbone.sequence.BindTokenHead(head)
		}
		return err
	})
	if err != nil {
		// Use may observe cancellation after Clone/BindHead completed.
		// Ownership has not yet transferred to the resource in this branch.
		if model != nil {
			_ = model.Close()
		}
		_ = resource.Close()
		return nil, err
	}
	if err = resource.Own(model); err != nil {
		_ = model.Close()
		_ = resource.Close()
		return nil, err
	}
	info, err := model.Info()
	if err != nil {
		_ = resource.Close()
		return nil, err
	}
	bound, err := r.tokens.Resolve(taskIdentity(spec), candleCapability(spec, info), resource, func(_ context.Context, _ io.Closer, text string) (tasks.TokenClassificationResult, error) {
		result, inferErr := model.ClassifyTokens(text)
		available := true
		out := tasks.TokenClassificationResult{Input: candleInputUsage(result.Input), ScoresAvailable: &available, Entities: make([]tasks.TokenEntity, len(result.Spans))}
		for i, span := range result.Spans {
			out.Entities[i] = tasks.TokenEntity{EntityType: span.Label, Start: span.Start, End: span.End, Text: span.Text, Confidence: span.Confidence}
		}
		if result.Input.Truncated && inferErr == nil {
			inferErr = tasks.ErrTokenSpansTruncated
		}
		return out, nativeError(inferErr)
	})
	if err == nil {
		_, err = bound.Call(ctx, string(spec.Recipe), "warmup")
	}
	if err != nil {
		_ = resource.Close()
		return nil, err
	}
	bound.Ready()
	return bound, nil
}

func candleCapability(spec config.ResolvedModelBinding, info candle.InstanceInfo) binding.Capability {
	return binding.Capability{Contract: spec.Binding.Contract, Provider: "candle", Device: info.Device, Precision: info.Precision, Labels: info.Labels, Limits: binding.Limits{ModelTokens: info.ArchitecturalMaxTokens, TaskTokens: 512, DeploymentTokens: spec.Deployment.Input.MaxTokens, Overflow: info.Overflow}}
}

func nativeError(err error) error {
	if err == nil {
		return nil
	}
	var native *candle.InstanceError
	if errors.As(err, &native) {
		switch native.Code {
		case "capability", "configuration":
			return fmt.Errorf("%w: %w", binding.ErrCapability, err)
		case "input_limit":
			return fmt.Errorf("%w: %w", binding.ErrInputLimit, err)
		case "result_invalid":
			return fmt.Errorf("%w: %w", binding.ErrInvalidResult, err)
		}
	}
	return err
}

func candleInputUsage(input candle.InputMetadata) *tasks.InputUsage {
	if input.InputTokens == 0 && input.ProcessedTokens == 0 {
		return nil
	}
	return &tasks.InputUsage{OriginalTokens: input.InputTokens, ProcessedTokens: input.ProcessedTokens, Truncated: input.Truncated}
}
