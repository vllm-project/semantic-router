package extproc

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"

func (r *OpenAIRouter) buildReplayUsageCost(ctx *RequestContext, usage responseUsageMetrics) routerreplay.UsageCost {
	totalTokens := usage.promptTokens + usage.completionTokens
	if totalTokens == 0 {
		return routerreplay.UsageCost{}
	}

	snapshot := routerreplay.UsageCost{
		PromptTokens:     replayIntPtr(usage.promptTokens),
		CompletionTokens: replayIntPtr(usage.completionTokens),
		TotalTokens:      replayIntPtr(totalTokens),
	}

	if r == nil || r.Config == nil || ctx == nil || ctx.RequestModel == "" {
		return snapshot
	}

	selectedPromptRate, selectedCompletionRate, selectedCurrency, ok := r.Config.GetModelPricing(ctx.RequestModel)
	if !ok {
		return snapshot
	}

	baselineModel, baselinePromptRate, baselineCompletionRate, baselineCurrency, ok := r.Config.GetMostExpensivePricedModel()
	if !ok {
		return snapshot
	}

	currency := selectedCurrency
	if currency == "" {
		currency = baselineCurrency
	}
	if currency == "" {
		currency = "USD"
	}

	actualCost := (float64(usage.promptTokens)*selectedPromptRate +
		float64(usage.completionTokens)*selectedCompletionRate) / 1_000_000.0
	baselineCost := (float64(usage.promptTokens)*baselinePromptRate +
		float64(usage.completionTokens)*baselineCompletionRate) / 1_000_000.0
	costSavings := baselineCost - actualCost

	snapshot.ActualCost = replayFloat64Ptr(actualCost)
	snapshot.BaselineCost = replayFloat64Ptr(baselineCost)
	snapshot.CostSavings = replayFloat64Ptr(costSavings)
	snapshot.Currency = replayStringPtr(currency)
	snapshot.BaselineModel = replayStringPtr(baselineModel)

	return snapshot
}

func replayIntPtr(value int) *int {
	return &value
}

func replayFloat64Ptr(value float64) *float64 {
	return &value
}

func replayStringPtr(value string) *string {
	return &value
}
