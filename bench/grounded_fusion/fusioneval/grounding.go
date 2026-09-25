package main

import (
	"context"
	"io"
	"runtime"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/native"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

// prepareNLI keeps this evaluator's explicitly selected Candle model independent
// of other runs and of the image's implicit classification provider. TextPair
// validates NLI labels and returns entailment/neutral/contradiction in that order.
func prepareNLI(opt options) (looper.NLIClassifyFunc, io.Closer, error) {
	device := "cpu"
	if !opt.useCPU {
		device = "cuda:0"
		if runtime.GOOS == "darwin" {
			device = "metal:0"
		}
	}
	spec := config.ResolvedModelBinding{
		Recipe: "fusioneval",
		Name:   "hallucination_explainer",
		Binding: config.ModelBinding{
			Deployment: "fusioneval-nli",
			Contract:   "text_pair_distribution.v1",
			Adapter:    "auto",
		},
		Deployment: config.ModelDeployment{
			Artifact:  opt.nliModel,
			Provider:  "candle",
			Device:    device,
			Precision: "native",
			Input:     config.ModelInputBudget{Overflow: "truncate"},
		},
	}
	handle, err := native.New(nil).TextPair(context.Background(), spec)
	if err != nil {
		return nil, nil, err
	}
	classify := func(ctx context.Context, premise, hypothesis string) (float32, float32, error) {
		result, classifyErr := handle.Call(ctx, string(spec.Recipe), tasks.TextPairRequest{
			Premise: premise, Hypothesis: hypothesis,
		})
		if classifyErr != nil {
			return 0, 0, classifyErr
		}
		return result.Probabilities[0], result.Probabilities[2], nil
	}
	return classify, handle, nil
}
