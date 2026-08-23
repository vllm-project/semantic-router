package extproc

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"

func (r *OpenAIRouter) buildReplayUsageCost(ctx *RequestContext, usage responseUsageMetrics) routerreplay.UsageCost {
	totalTokens := responseUsageTotal(usage)
	if totalTokens == 0 {
		return routerreplay.UsageCost{}
	}

	snapshot := routerreplay.UsageCost{
		TotalTokens: replayIntPtr(totalTokens),
	}
	if usage.promptTokens > 0 || usage.completionTokens > 0 {
		snapshot.PromptTokens = replayIntPtr(usage.promptTokens)
		snapshot.CachedPromptTokens = replayIntPtr(usage.cachedPromptTokens)
		snapshot.CacheWriteTokens = replayIntPtr(usage.cacheWriteTokens)
		snapshot.CompletionTokens = replayIntPtr(usage.completionTokens)
	}

	if !responseUsageHasPricableBreakdown(usage) || r == nil || r.Config == nil || ctx == nil || ctx.RequestModel == "" {
		return snapshot
	}

	selectedPricing, ok := r.Config.GetFullModelPricing(ctx.RequestModel)
	if !ok {
		return snapshot
	}

	baselineModel, baselinePricing, ok := r.Config.GetMostExpensiveFullModelPricing()
	if !ok {
		return snapshot
	}

	currency := selectedPricing.Currency
	if currency == "" {
		currency = baselinePricing.Currency
	}
	if currency == "" {
		currency = "USD"
	}

	actualCost := costForResponseUsage(usage, selectedPricing)
	baselineCost := costForResponseUsage(usage, baselinePricing)
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
