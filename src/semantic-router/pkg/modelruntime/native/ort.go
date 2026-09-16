package native

import (
	"context"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
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
	options := ort.Options{ModelPath: d.Artifact, ModelFile: spec.Binding.Head, Precision: d.Precision, MaxInputTokens: d.Input.MaxTokens, Overflow: d.Input.Overflow, CustomOpsProfile: d.CustomOpsProfile, CompilationCacheDir: d.CompilationCacheDir}
	if err := d.ValidateCompilationCache(); err != nil {
		return options, fmt.Errorf("%w: %w", binding.ErrCapability, err)
	}
	switch {
	case d.Device == "cpu":
		options.Provider = "cpu"
	case strings.HasPrefix(d.Device, "rocm:"):
		index, err := strconv.Atoi(strings.TrimPrefix(d.Device, "rocm:"))
		if err != nil || index < 0 {
			return options, fmt.Errorf("%w: invalid ROCm device index", binding.ErrCapability)
		}
		options.Provider, options.DeviceID = "rocm", index
	case strings.HasPrefix(d.Device, "migraphx:"):
		index, err := strconv.Atoi(strings.TrimPrefix(d.Device, "migraphx:"))
		if err != nil || index < 0 {
			return options, fmt.Errorf("%w: invalid MIGraphX device index", binding.ErrCapability)
		}
		options.Provider, options.DeviceID = "migraphx", index
	default:
		return options, fmt.Errorf("%w: ORT requires cpu, rocm:index or migraphx:index", binding.ErrCapability)
	}
	if options.CustomOpsProfile != "" && (options.Provider != "rocm" || options.CustomOpsProfile != "ck_flash_attention") {
		return options, fmt.Errorf("%w: custom ops require the trusted ROCm CK profile", binding.ErrCapability)
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
	if options.CompilationCacheDir != "" {
		if err := validateORTCacheLocation(options); err != nil {
			return options, fmt.Errorf("%w: %w", binding.ErrCapability, err)
		}
	}
	return options, nil
}

// Resolve existing parents without creating a cache directory. This also
// prevents a symlinked cache parent from adding mutable files to an artifact.
func resolveCachePath(path string) (string, error) {
	resolved, err := filepath.EvalSymlinks(path)
	if err == nil {
		return resolved, nil
	}
	if !os.IsNotExist(err) {
		return "", err
	}
	parent := filepath.Dir(path)
	if parent == path {
		return "", err
	}
	resolved, err = resolveCachePath(parent)
	if err != nil {
		return "", err
	}
	return filepath.Join(resolved, filepath.Base(path)), nil
}

func validateORTCacheLocation(options ort.Options) error {
	cache, err := resolveCachePath(filepath.Clean(options.CompilationCacheDir))
	if err != nil {
		return err
	}
	paths := []string{options.ModelPath}
	if options.ModelFile != "" {
		head := options.ModelFile
		if !filepath.IsAbs(head) {
			head = filepath.Join(options.ModelPath, head)
		}
		paths = append(paths, filepath.Dir(head))
	}
	for _, path := range paths {
		absolute, err := filepath.Abs(path)
		if err != nil {
			return err
		}
		artifact, err := resolveCachePath(absolute)
		if err != nil {
			return err
		}
		if info, statErr := os.Stat(artifact); statErr == nil && !info.IsDir() {
			artifact = filepath.Dir(artifact)
		}
		relative, err := filepath.Rel(artifact, cache)
		if err != nil || (relative != ".." && !strings.HasPrefix(relative, ".."+string(filepath.Separator))) {
			return fmt.Errorf("compilation_cache_dir must be outside model and graph artifact directories")
		}
	}
	return nil
}

func (r *Runtime) ortResource(ctx context.Context, spec config.ResolvedModelBinding, task string, load func(ort.Options) (io.Closer, error)) (*binding.Resource, error) {
	return r.ortResourceWithExecutionLimit(ctx, spec, task, 0, load)
}

func (r *Runtime) ortResourceWithExecutionLimit(ctx context.Context, spec config.ResolvedModelBinding, task string, executionLimit int, load func(ort.Options) (io.Closer, error)) (*binding.Resource, error) {
	options, err := ortOptions(spec)
	if err != nil {
		return nil, err
	}
	if executionLimit < 0 || (options.MaxInputTokens > 0 && executionLimit > options.MaxInputTokens) {
		return nil, fmt.Errorf("%w: execution window exceeds document budget", binding.ErrCapability)
	}
	// Include physical execution geometry in the existing resource pool identity.
	options.ExecutionMaxInputTokens = executionLimit
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
		case "ROCMExecutionProvider":
			if !session.CPUFallbackDisabled {
				return binding.Capability{}, fmt.Errorf("%w: ROCm session permits CPU fallback", binding.ErrCapability)
			}
			device = fmt.Sprintf("rocm:%d", session.DeviceID)
		case "MIGraphXExecutionProvider":
			if !session.CPUFallbackDisabled {
				return binding.Capability{}, fmt.Errorf("%w: MIGraphX session permits CPU fallback", binding.ErrCapability)
			}
			device = fmt.Sprintf("migraphx:%d", session.DeviceID)
		default:
			return binding.Capability{}, fmt.Errorf("%w: unrecognized ORT execution provider", binding.ErrCapability)
		}
		profile := session.CustomOpsProfile
		if profile == "none" {
			profile = ""
		}
		if profile != d.CustomOpsProfile {
			return binding.Capability{}, fmt.Errorf("%w: actual ORT custom ops profile differs from deployment", binding.ErrCapability)
		}
		if profile == "ck_flash_attention" {
			digest, err := hex.DecodeString(session.CustomOpsSHA256)
			if err != nil || len(digest) != 32 || session.CustomOpsLibrary != "/usr/local/lib/libort_ck_flash_attn.so.1" {
				return binding.Capability{}, fmt.Errorf("%w: CK session lacks trusted library content evidence", binding.ErrCapability)
			}
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
