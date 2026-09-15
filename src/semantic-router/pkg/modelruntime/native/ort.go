package native

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"path/filepath"
	"strconv"
	"strings"

	ort "github.com/vllm-project/semantic-router/onnx-binding/instance"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

func ortOptions(spec config.ResolvedModelBinding) (ort.Options, error) {
	d := spec.Deployment.WithDefaults()
	options := ort.Options{ModelPath: d.Artifact, ModelFile: spec.Binding.Head, Precision: d.Precision, MaxInputTokens: d.Input.MaxTokens, Overflow: d.Input.Overflow}
	switch {
	case d.Device == "cpu":
		options.Provider = "cpu"
	case strings.HasPrefix(d.Device, "migraphx:"):
		index, err := strconv.Atoi(strings.TrimPrefix(d.Device, "migraphx:"))
		if err != nil || index < 0 {
			return options, fmt.Errorf("%w: invalid MIGraphX device index", binding.ErrCapability)
		}
		options.Provider, options.DeviceID = "migraphx", index
	default:
		return options, fmt.Errorf("%w: ORT requires cpu or migraphx:index", binding.ErrCapability)
	}
	if options.Precision != "native" && (options.Provider != "migraphx" || options.Precision != "fp16") {
		return options, fmt.Errorf("%w: ORT supports native graph precision or explicit MIGraphX fp16 conversion", binding.ErrCapability)
	}
	switch options.Overflow {
	case "reject":
	case "truncate":
		options.Overflow = "truncate_right"
	default:
		return options, fmt.Errorf("%w: ORT adapter supports reject or truncate input policy", binding.ErrCapability)
	}
	if options.ModelFile != "" && filepath.Ext(options.ModelFile) != ".onnx" {
		return options, fmt.Errorf("%w: ORT head must identify a complete ONNX graph", binding.ErrCapability)
	}
	return options, nil
}

func (r *Runtime) ortResource(ctx context.Context, spec config.ResolvedModelBinding, task string, load func(ort.Options) (io.Closer, error)) (*binding.Resource, error) {
	options, err := ortOptions(spec)
	if err != nil {
		return nil, err
	}
	revision, err := r.artifactRevision(ctx, options.ModelPath)
	if err != nil {
		return nil, err
	}
	headRevision := ""
	if options.ModelFile != "" {
		path := options.ModelFile
		if !filepath.IsAbs(path) {
			path = filepath.Join(options.ModelPath, path)
		}
		// External ONNX tensors resolve relative to the graph. Include that
		// directory so replacing a .data file also changes physical identity.
		headRevision, err = r.artifactRevision(ctx, filepath.Dir(path))
		if err != nil {
			return nil, err
		}
	}
	// Current ONNX exports include their head in the graph. Sharing is valid for
	// the entire graph and adapter, never for different heads at the same path.
	execution, err := json.Marshal(struct {
		Options                     ort.Options
		Task, Adapter, HeadRevision string
	}{options, task, spec.Binding.Adapter, headRevision})
	if err != nil {
		return nil, err
	}
	d := spec.Deployment.WithDefaults()
	identity := binding.ResourceIdentity{Artifact: options.ModelPath, Revision: d.Revision + ":" + revision, Provider: "ort", Device: d.Device, Precision: d.Precision, Execution: string(execution)}
	budget, gate := resourceAdmission(spec)
	return r.Pool.Acquire(ctx, identity, budget, gate, func(context.Context) (io.Closer, error) {
		model, err := load(options)
		if err != nil {
			// Do not pass a typed nil model through io.Closer to Pool cleanup.
			return nil, ortError(err)
		}
		return model, nil
	})
}

func ortCapability(spec config.ResolvedModelBinding, info ort.Info) (binding.Capability, error) {
	if len(info.Sessions) == 0 {
		return binding.Capability{}, fmt.Errorf("%w: ORT returned no loaded session evidence", binding.ErrCapability)
	}
	d := spec.Deployment.WithDefaults()
	for _, session := range info.Sessions {
		device := "cpu"
		switch session.Provider {
		case "CPUExecutionProvider":
		case "MIGraphXExecutionProvider":
			if !session.CPUFallbackDisabled {
				return binding.Capability{}, fmt.Errorf("%w: MIGraphX session permits CPU fallback", binding.ErrCapability)
			}
			device = fmt.Sprintf("migraphx:%d", session.DeviceID)
		default:
			return binding.Capability{}, fmt.Errorf("%w: unrecognized ORT execution provider", binding.ErrCapability)
		}
		if device != d.Device || session.Precision != d.Precision {
			return binding.Capability{}, fmt.Errorf("%w: actual ORT execution differs from deployment", binding.ErrCapability)
		}
	}
	return binding.Capability{Contract: spec.Binding.Contract, Provider: "ort", Device: d.Device, Precision: d.Precision, Labels: info.Labels, Limits: binding.Limits{ModelTokens: info.ModelLimit, TaskTokens: info.TaskLimit, DeploymentTokens: d.Input.MaxTokens, Overflow: d.Input.Overflow}}, nil
}

func (r *Runtime) ortSequence(ctx context.Context, spec config.ResolvedModelBinding) (*binding.Resolved[string, tasks.LabelDistribution], error) {
	resource, err := r.ortResource(ctx, spec, "sequence", func(options ort.Options) (io.Closer, error) { return ort.LoadSequenceClassifier(options) })
	if err != nil {
		return nil, err
	}
	var info ort.Info
	err = resource.Use(ctx, func(value io.Closer) error {
		var infoErr error
		info, infoErr = value.(*ort.SequenceClassifier).Info()
		return infoErr
	})
	var capability binding.Capability
	if err == nil {
		capability, err = ortCapability(spec, info)
	}
	var bound *binding.Resolved[string, tasks.LabelDistribution]
	if err == nil {
		bound, err = r.sequence.Resolve(taskIdentity(spec), capability, resource, func(_ context.Context, value io.Closer, text string) (tasks.LabelDistribution, error) {
			result, inferErr := value.(*ort.SequenceClassifier).Classify(text)
			return tasks.LabelDistribution{Probabilities: result.Probabilities, Input: ortInputUsage(result.Input)}, ortError(inferErr)
		})
	}
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

func (r *Runtime) ortTokens(ctx context.Context, spec config.ResolvedModelBinding) (*binding.Resolved[string, tasks.TokenClassificationResult], error) {
	resource, err := r.ortResource(ctx, spec, "token", func(options ort.Options) (io.Closer, error) { return ort.LoadTokenClassifier(options) })
	if err != nil {
		return nil, err
	}
	var info ort.Info
	err = resource.Use(ctx, func(value io.Closer) error {
		var infoErr error
		info, infoErr = value.(*ort.TokenClassifier).Info()
		return infoErr
	})
	var capability binding.Capability
	if err == nil {
		capability, err = ortCapability(spec, info)
	}
	var bound *binding.Resolved[string, tasks.TokenClassificationResult]
	if err == nil {
		bound, err = r.tokens.Resolve(taskIdentity(spec), capability, resource, func(_ context.Context, value io.Closer, text string) (tasks.TokenClassificationResult, error) {
			result, inferErr := value.(*ort.TokenClassifier).Detect(text)
			available := true
			output := tasks.TokenClassificationResult{Input: ortInputUsage(result.Input), Entities: make([]tasks.TokenEntity, len(result.Spans)), ScoresAvailable: &available}
			for i, span := range result.Spans {
				output.Entities[i] = tasks.TokenEntity{EntityType: span.EntityType, Start: span.Start, End: span.End, Text: span.Text, Confidence: span.Confidence}
			}
			if inferErr == nil && result.Input.Truncated {
				inferErr = tasks.ErrTokenSpansTruncated
			}
			return output, ortError(inferErr)
		})
	}
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

func ortError(err error) error {
	if err == nil {
		return nil
	}
	var native *ort.Error
	if errors.As(err, &native) {
		switch native.Kind {
		case "capability", "configuration", "invalid_input":
			return fmt.Errorf("%w: %w", binding.ErrCapability, err)
		case "input_limit":
			return fmt.Errorf("%w: %w", binding.ErrInputLimit, err)
		case "invalid_output":
			return fmt.Errorf("%w: %w", binding.ErrInvalidResult, err)
		case "closed":
			return fmt.Errorf("%w: %w", binding.ErrClosed, err)
		}
	}
	return err
}

func ortInputUsage(input ort.InputUsage) *tasks.InputUsage {
	if input.OriginalTokens == 0 && input.ProcessedTokens == 0 {
		return nil
	}
	return &tasks.InputUsage{OriginalTokens: input.OriginalTokens, ProcessedTokens: input.ProcessedTokens, Truncated: input.Truncated}
}
