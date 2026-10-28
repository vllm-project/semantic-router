package selection

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestMultiFactorOutputForecastIsNotNativeCapacity(t *testing.T) {
	forecast := 4096
	cfg := DefaultMultiFactorConfig()
	cfg.ExpectedOutputTokens = &forecast
	cfg.Weights = MultiFactorWeights{Cost: 1}
	selector := buildMFSelector(cfg, map[string]config.ModelParams{
		"small": {Pricing: config.ModelPricing{PromptPer1M: 1, CompletionPer1M: 2}},
		"large": {Pricing: config.ModelPricing{PromptPer1M: 1, CompletionPer1M: 2}},
	}, nil, nil, nil)
	ctx := &SelectionContext{CandidateModels: candidates("small", "large"), CandidateDemands: map[string]CandidateDemand{
		"small": {InputTokens: 100, MaxOutputTokens: llmprotocol.Int64(262044)},
		"large": {InputTokens: 100, MaxOutputTokens: llmprotocol.Int64(1048476)},
	}}
	signals := selector.gatherSignals(ctx.CandidateModels, ctx)
	require.Equal(t, signals[0].cost, signals[1].cost)
	require.InDelta(t, .008292, signals[0].cost, 1e-10)
	require.Equal(t, 4096, selector.costContext("large", ctx).ExpectedOutputTokens)
	ctx.ExpectedOutputTokens = 20
	require.Equal(t, 20, selector.costContext("large", ctx).ExpectedOutputTokens)
	ctx.ExpectedOutputTokens = 0
	ctx.CandidateDemands["small"] = CandidateDemand{InputTokens: 100, MaxOutputTokens: llmprotocol.Int64(7)}
	require.Equal(t, 7, selector.costContext("small", ctx).ExpectedOutputTokens)
	require.Equal(t, 0, ctx.ExpectedOutputTokens, "forecast must not mutate caller budget")
	cfg.ExpectedOutputTokens = nil
	missing := buildMFSelector(cfg, nil, nil, nil, nil)
	_, err := missing.Select(context.Background(), ctx)
	require.ErrorContains(t, err, "expected_output_tokens")
}
