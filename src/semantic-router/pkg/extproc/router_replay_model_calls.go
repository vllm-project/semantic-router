package extproc

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

func (r *OpenAIRouter) replayModelCalls(ctx *RequestContext) []store.RouterCallCost {
	if ctx == nil {
		return nil
	}
	var calls []store.RouterCallCost
	for _, measured := range ctx.ModelCallMeter.Calls() {
		calls = append(calls, r.priceModelCall(measured))
	}
	return calls
}

func (r *OpenAIRouter) priceModelCall(measured modelservice.ModelCall) store.RouterCallCost {
	call := store.RouterCallCost{
		Kind: measured.Kind, Model: measured.Model,
		PromptTokens: int64(measured.InputTokens), CompletionTokens: int64(measured.OutputTokens),
	}
	pricing, priced := r.Config.DeploymentPricing(measured.Deployment)
	switch {
	// a reply that reported one token counter and not the other
	case measured.InputTokens < 0 || measured.OutputTokens < 0,
		!measured.HasUsage && measured.InputTokens+measured.OutputTokens > 0:
		addUnpricedRouterCalls(&call, store.UnpricedUsageIncomplete, 1)
	case !measured.HasUsage:
		addUnpricedRouterCalls(&call, store.UnpricedNoUsage, 1)
	case !priced:
		addUnpricedRouterCalls(&call, store.UnpricedPricingNotConfigured, 1)
	default:
		call.Calls = 1
		call.Currency = normalizeReplayCurrency(pricing.Currency)
		call.Cost = replayFloat64Ptr((float64(measured.InputTokens)*pricing.PromptPer1M + float64(measured.OutputTokens)*pricing.CompletionPer1M) / 1e6)
	}
	return call
}

// replaceReplayModelCalls retains independently recorded Looper costs while
// refreshing the request meter, without counting the same snapshot twice.
func (r *OpenAIRouter) replaceReplayModelCalls(ctx *RequestContext, accounting *store.CostAccounting) *store.CostAccounting {
	measured := r.replayModelCalls(ctx)
	if len(measured) == 0 {
		return accounting
	}
	if accounting == nil {
		accounting = &store.CostAccounting{}
	}
	var retained []store.RouterCallCost
	for _, call := range accounting.RouterCalls {
		if call.Kind == store.RouterCallKindLooper {
			retained = append(retained, call)
		}
	}
	retained = append(retained, measured...)
	accounting.RouterCalls = retained
	return accounting
}

func (r *OpenAIRouter) refreshReplayModelCalls(ctx *RequestContext, recorder *routerreplay.Recorder) {
	if len(ctx.ModelCallMeter.Calls()) == 0 {
		return
	}
	record, ok := recorder.GetRecord(ctx.RouterReplayID)
	if !ok {
		return
	}
	r.writeRouterReplayUsageCost(ctx, recordUsageCost(record))
}

// recordUsageCost is the usage and cost a record holds, to rewrite it with
// one part changed.
func recordUsageCost(record routerreplay.RoutingRecord) routerreplay.UsageCost {
	return routerreplay.UsageCost{
		ProviderCache: record.ProviderCache, PromptTokens: record.PromptTokens,
		CachedPromptTokens: record.CachedPromptTokens, CacheWriteTokens: record.CacheWriteTokens,
		CompletionTokens: record.CompletionTokens, TotalTokens: record.TotalTokens,
		ActualCost: record.ActualCost, BaselineCost: record.BaselineCost,
		CostSavings: record.CostSavings, Currency: record.Currency, BaselineModel: record.BaselineModel,
		CostAccounting: record.CostAccounting,
	}
}
