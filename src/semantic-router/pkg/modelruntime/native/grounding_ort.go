package native

import (
	"context"
	"fmt"
	"io"

	ort "github.com/vllm-project/semantic-router/onnx-binding/instance"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

func (r *Runtime) ortGrounded(ctx context.Context, spec config.ResolvedModelBinding) (*binding.Resolved[tasks.GroundedTextRequest, tasks.TokenClassificationResult], error) {
	if spec.Binding.Adapter != "vela_halu" {
		return nil, fmt.Errorf("%w: ORT grounding requires the vela_halu pair adapter", binding.ErrCapability)
	}
	resource, err := r.ortResourcePrepared(ctx, spec, "grounded", 0, prepareFixedGPUExecution, func(options ort.Options) (io.Closer, error) { return ort.LoadGroundedClassifier(options) })
	if err != nil {
		return nil, err
	}
	var info ort.Info
	err = resource.Use(ctx, func(value io.Closer) error {
		var infoErr error
		info, infoErr = value.(*ort.GroundedClassifier).Info()
		return infoErr
	})
	var capability binding.Capability
	if err == nil {
		capability, err = ortCapability(spec, info)
	}
	var bound *binding.Resolved[tasks.GroundedTextRequest, tasks.TokenClassificationResult]
	if err == nil {
		bound, err = r.grounded.Resolve(taskIdentity(spec), capability, resource, func(_ context.Context, value io.Closer, input tasks.GroundedTextRequest) (tasks.TokenClassificationResult, error) {
			result, inferErr := value.(*ort.GroundedClassifier).Detect(input.Context, input.Question, input.Answer)
			if inferErr != nil {
				return tasks.TokenClassificationResult{}, ortError(inferErr)
			}
			available := true
			out := tasks.TokenClassificationResult{Input: ortInputUsage(result.Input), ScoresAvailable: &available, Entities: make([]tasks.TokenEntity, len(result.Spans))}
			maximum := float32(0)
			for i, span := range result.Spans {
				out.Entities[i] = tasks.TokenEntity{Text: span.Text, Start: span.Start, End: span.End, EntityType: span.EntityType, Confidence: span.Confidence}
				maximum = max(maximum, span.Confidence)
			}
			if len(result.Spans) > 0 {
				out.Summary = &tasks.ScoreResult{Value: float64(maximum)}
				out.SummarySemantics = &tasks.ScoreSemantics{Unit: "max_hallucinated_token_score", Direction: tasks.HigherIsPositive, Calibrated: false}
			}
			return out, nil
		})
	}
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
