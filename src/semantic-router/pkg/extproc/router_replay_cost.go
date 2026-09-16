package extproc

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
)

func (r *OpenAIRouter) buildReplayUsageCost(ctx *RequestContext, usage responseUsageMetrics) routerreplay.UsageCost {
	totalTokens := usage.promptTokens + usage.completionTokens
	if totalTokens == 0 {
		return routerreplay.UsageCost{}
	}

	snapshot := routerreplay.UsageCost{
		PromptTokens:       replayIntPtr(usage.promptTokens),
		CachedPromptTokens: replayIntPtr(usage.cachedPromptTokens),
		CacheWriteTokens:   replayIntPtr(usage.cacheWriteTokens),
		CompletionTokens:   replayIntPtr(usage.completionTokens),
		TotalTokens:        replayIntPtr(totalTokens),
	}

	if r == nil || r.Config == nil || ctx == nil || ctx.RequestModel == "" {
		return snapshot
	}

	selectedPricing, ok := r.Config.GetFullModelPricing(ctx.RequestModel)
	if !ok {
		return snapshot
	}

	actualCost := costForResponseUsage(usage, selectedPricing)
	currency := normalizeReplayCurrency(selectedPricing.Currency)
	baselineModel, baselineCost := r.replayBaselineCost(ctx, usage, currency, actualCost)
	costSavings := baselineCost - actualCost

	snapshot.ActualCost = replayFloat64Ptr(actualCost)
	snapshot.BaselineCost = replayFloat64Ptr(baselineCost)
	snapshot.CostSavings = replayFloat64Ptr(costSavings)
	snapshot.Currency = replayStringPtr(currency)
	snapshot.BaselineModel = replayStringPtr(baselineModel)

	return snapshot
}

// replayBaselineCost compares this request's recorded usage at the configured
// rates of its decision candidates. Other recipes and currencies cannot inflate
// savings. A passthrough request has only its selected model as a baseline.
func (r *OpenAIRouter) replayBaselineCost(
	ctx *RequestContext,
	usage responseUsageMetrics,
	currency string,
	selectedCost float64,
) (string, float64) {
	model, cost := ctx.RequestModel, selectedCost
	if ctx.VSRSelectedDecision == nil {
		return model, cost
	}
	for _, candidate := range ctx.VSRSelectedDecision.ModelRefs {
		pricing, ok := r.Config.GetFullModelPricing(candidate.Model)
		if !ok || normalizeReplayCurrency(pricing.Currency) != currency {
			continue
		}
		candidateCost := costForResponseUsage(usage, pricing)
		if candidateCost > cost || (candidateCost == cost && candidate.Model < model) {
			model, cost = candidate.Model, candidateCost
		}
	}
	return model, cost
}

func normalizeReplayCurrency(currency string) string {
	return strings.ToUpper(strings.TrimSpace(currency))
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
