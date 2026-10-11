package looper

import (
	"context"

	"github.com/openai/openai-go"
)

// ModelSampling fixes benchmark sampling at the final dispatch boundary,
// including algorithm-generated verifier and synthesis requests.
type ModelSampling struct {
	Temperature float64
	TopP        float64
}

type modelSamplingContextKey struct{}

// WithModelSampling copies the model policy so concurrent calls cannot mutate it.
// Callers without this optional policy retain normal algorithm sampling.
func WithModelSampling(ctx context.Context, models map[string]ModelSampling) context.Context {
	policy := make(map[string]ModelSampling, len(models))
	for model, sampling := range models {
		policy[model] = sampling
	}
	return context.WithValue(ctx, modelSamplingContextKey{}, policy)
}

func applyModelSampling(ctx context.Context, req *openai.ChatCompletionNewParams, model string) *openai.ChatCompletionNewParams {
	policy, _ := ctx.Value(modelSamplingContextKey{}).(map[string]ModelSampling)
	sampling, ok := policy[model]
	if !ok {
		return req
	}
	cloned := cloneRequest(req)
	cloned.Temperature = openai.Float(sampling.Temperature)
	cloned.TopP = openai.Float(sampling.TopP)
	return cloned
}
