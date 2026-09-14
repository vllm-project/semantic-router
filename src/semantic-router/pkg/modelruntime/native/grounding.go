package native

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	candle "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

func (r *Runtime) candleTaskResource(ctx context.Context, spec config.ResolvedModelBinding, task string, load func(candle.InstanceOptions) (io.Closer, error)) (*binding.Resource, error) {
	if spec.Deployment.Provider != "candle" {
		return nil, fmt.Errorf("%w: %s is unavailable on provider %q", binding.ErrCapability, task, spec.Deployment.Provider)
	}
	if spec.Binding.Head != "" {
		return nil, fmt.Errorf("%w: %s does not support separate heads", binding.ErrCapability, task)
	}
	options := candleOptions(spec)
	revision, err := r.artifactRevision(ctx, options.ModelPath)
	if err != nil {
		return nil, err
	}
	execution, _ := json.Marshal(struct {
		Task    string
		Options candle.InstanceOptions
	}{task, options})
	identity := binding.ResourceIdentity{Artifact: options.ModelPath, Revision: spec.Deployment.Revision + ":" + revision, Provider: "candle", Device: options.Device, Precision: options.Precision, Execution: string(execution)}
	budget, gate := resourceAdmission(spec)
	return r.Pool.Acquire(ctx, identity, budget, gate, func(context.Context) (io.Closer, error) {
		resource, err := load(options)
		if err != nil {
			return nil, nativeError(err)
		}
		return resource, nil
	})
}

func (r *Runtime) Grounded(ctx context.Context, spec config.ResolvedModelBinding, threshold float32) (_ *binding.Resolved[tasks.GroundedTextRequest, tasks.TokenClassificationResult], callErr error) {
	defer func() { observePreparationFailure(spec, callErr) }()
	resource, err := r.candleTaskResource(ctx, spec, "grounded", func(options candle.InstanceOptions) (io.Closer, error) {
		return candle.LoadHallucinationDetector(options)
	})
	if err != nil {
		return nil, err
	}
	var info candle.InstanceInfo
	err = resource.Use(ctx, func(value io.Closer) error {
		var infoErr error
		info, infoErr = value.(*candle.HallucinationDetector).Info()
		return infoErr
	})
	if err != nil {
		_ = resource.Close()
		return nil, err
	}
	bound, err := r.grounded.Resolve(taskIdentity(spec), candleCapability(spec, info), resource, func(_ context.Context, value io.Closer, input tasks.GroundedTextRequest) (tasks.TokenClassificationResult, error) {
		result, inferErr := value.(*candle.HallucinationDetector).Detect(input.Context, input.Question, input.Answer, threshold)
		if inferErr != nil {
			return tasks.TokenClassificationResult{}, nativeError(inferErr)
		}
		available := true
		out := tasks.TokenClassificationResult{
			Input:           candleInputUsage(result.Input),
			ScoresAvailable: &available,
			Entities:        make([]tasks.TokenEntity, len(result.Spans)),
		}
		// The legacy native API uses 1 for an empty set. That sentinel is not
		// model evidence and must not become a probability of a clean answer.
		if len(result.Spans) > 0 {
			out.Summary = &tasks.ScoreResult{Value: float64(result.Confidence)}
			out.SummarySemantics = &tasks.ScoreSemantics{Unit: "max_hallucinated_token_score", Direction: tasks.HigherIsPositive, Calibrated: false}
		}
		for i, span := range result.Spans {
			out.Entities[i] = tasks.TokenEntity{Text: span.Text, Start: span.Start, End: span.End, EntityType: span.Label, Confidence: span.Confidence}
		}
		return out, nil
	})
	if err == nil {
		_, err = bound.Call(ctx, string(spec.Recipe), tasks.GroundedTextRequest{Context: "A warmup sentence.", Question: "What is this?", Answer: "A sentence."})
	}
	if err != nil {
		_ = resource.Close()
		return nil, err
	}
	bound.Ready()
	return bound, nil
}

func (r *Runtime) TextPair(ctx context.Context, spec config.ResolvedModelBinding) (_ *binding.Resolved[tasks.TextPairRequest, tasks.LabelDistribution], callErr error) {
	defer func() { observePreparationFailure(spec, callErr) }()
	resource, err := r.candleTaskResource(ctx, spec, "text_pair", func(options candle.InstanceOptions) (io.Closer, error) { return candle.LoadNLIClassifier(options) })
	if err != nil {
		return nil, err
	}
	var info candle.InstanceInfo
	err = resource.Use(ctx, func(value io.Closer) error {
		var infoErr error
		info, infoErr = value.(*candle.NLIClassifier).Info()
		return infoErr
	})
	if err != nil {
		_ = resource.Close()
		return nil, err
	}
	order, err := nliLabelOrder(info.Labels)
	if err != nil {
		_ = resource.Close()
		return nil, err
	}
	capability := candleCapability(spec, info)
	capability.Labels = []string{"entailment", "neutral", "contradiction"}
	bound, err := r.pair.Resolve(taskIdentity(spec), capability, resource, func(_ context.Context, value io.Closer, input tasks.TextPairRequest) (tasks.LabelDistribution, error) {
		result, inferErr := value.(*candle.NLIClassifier).Classify(input.Premise, input.Hypothesis)
		if inferErr != nil {
			return tasks.LabelDistribution{}, nativeError(inferErr)
		}
		if len(result.Probabilities) != 3 {
			return tasks.LabelDistribution{}, fmt.Errorf("%w: NLI requires three probabilities", binding.ErrInvalidResult)
		}
		return tasks.LabelDistribution{Input: candleInputUsage(result.Input), Probabilities: []float32{result.Probabilities[order[0]], result.Probabilities[order[1]], result.Probabilities[order[2]]}}, nil
	})
	if err == nil {
		_, err = bound.Call(ctx, string(spec.Recipe), tasks.TextPairRequest{Premise: "A warmup sentence.", Hypothesis: "A sentence."})
	}
	if err != nil {
		_ = resource.Close()
		return nil, err
	}
	bound.Ready()
	return bound, nil
}

func nliLabelOrder(labels []string) ([3]int, error) {
	order := [3]int{-1, -1, -1}
	for i, label := range labels {
		switch strings.ToLower(label) {
		case "entailment":
			order[0] = i
		case "neutral":
			order[1] = i
		case "contradiction":
			order[2] = i
		default:
			return order, fmt.Errorf("%w: NLI label %q has no declared meaning", binding.ErrCapability, label)
		}
	}
	for _, index := range order {
		if index < 0 {
			return order, fmt.Errorf("%w: NLI must declare entailment, neutral, contradiction labels", binding.ErrCapability)
		}
	}
	return order, nil
}
