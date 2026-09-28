package extproc

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
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
	snapshot.ActualCost = replayFloat64Ptr(actualCost)
	snapshot.Currency = replayStringPtr(currency)
	baselineModel, baselineCost := r.replayBaselineCost(ctx, usage, currency, actualCost)
	if baselineModel == "" {
		return snapshot
	}
	costSavings := baselineCost - actualCost

	snapshot.BaselineCost = replayFloat64Ptr(baselineCost)
	snapshot.CostSavings = replayFloat64Ptr(costSavings)
	snapshot.BaselineModel = replayStringPtr(baselineModel)

	return snapshot
}

// replayBaselineCost compares this request's recorded usage at the configured
// rates of the selected recipe's complete model pool across all decisions.
// Other recipes and currencies cannot inflate savings. A passthrough request
// has only its selected model as a baseline.
func (r *OpenAIRouter) replayBaselineCost(
	ctx *RequestContext,
	usage responseUsageMetrics,
	currency string,
	selectedCost float64,
) (string, float64) {
	recipe := ctx.Routing.SelectedRecipe()
	if recipe == nil {
		return ctx.RequestModel, selectedCost
	}
	model, cost := "", 0.0
	for candidate := range replayRecipeModelPool(recipe, r.Config.DefaultModel) {
		pricing, ok := r.Config.GetFullModelPricing(candidate)
		if !ok || normalizeReplayCurrency(pricing.Currency) != currency {
			continue
		}
		candidateCost := costForResponseUsage(usage, pricing)
		if model == "" || candidateCost > cost || (candidateCost == cost && candidate < model) {
			model, cost = candidate, candidateCost
		}
	}
	return model, cost
}

func replayRecipeModelPool(recipe *config.RoutingRecipe, defaultModel string) map[string]struct{} {
	models := make(map[string]struct{})
	add := func(model string) {
		if model = strings.TrimSpace(model); model != "" {
			models[model] = struct{}{}
		}
	}
	for _, decision := range recipe.Profile.Decisions {
		for _, ref := range decision.ModelRefs {
			add(ref.Model)
		}
		for _, iteration := range decision.CandidateIterations {
			if iteration.Source == "models" {
				for _, ref := range iteration.Models {
					add(ref.Model)
				}
			}
		}
		if decision.Action != nil && decision.Action.Type == config.DecisionActionRoute {
			add(decision.Action.Destination)
			continue
		}
		// Only an empty candidate decision can make the router default part of
		// this recipe's declared generation pool. Strict requirements disable
		// that fallback; fast responses do not invoke a model at all.
		if len(decision.ModelRefs) == 0 && decision.GetFastResponseConfig() == nil &&
			(decision.Algorithm == nil || decision.Algorithm.MinimumCandidates == 0) &&
			!selection.CandidateRequirementsEnabled(recipe.Profile.CandidateRequirements) {
			add(defaultModel)
		}
	}
	// Planner and judge models are auxiliary calls, not generation candidates.
	return models
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
