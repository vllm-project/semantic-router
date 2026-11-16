package extproc

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

func (r *OpenAIRouter) buildReplayUsageCost(ctx *RequestContext, usage responseUsageMetrics) routerreplay.UsageCost {
	var selectedPricing *config.ModelPricing
	if r != nil && r.Config != nil && ctx != nil && ctx.RequestModel != "" {
		if pricing, ok := r.Config.GetFullModelPricing(ctx.RequestModel); ok {
			selectedPricing = &pricing
		}
	}
	accounting := r.replayPricingAccounting(ctx)
	providerCache := buildProviderCacheAccounting(ctx, selectedPricing)
	totalTokens := usage.promptTokens + usage.completionTokens
	if totalTokens == 0 && !responseUsageHasPricableBreakdown(usage) {
		accounting.UnpricedReason = store.UnpricedNoUsage
		snapshot := routerreplay.UsageCost{ProviderCache: providerCache, CostAccounting: accounting}
		if selectedPricing != nil {
			snapshot.Currency = replayStringPtr(normalizeReplayCurrency(selectedPricing.Currency))
		}
		return snapshot
	}

	snapshot := routerreplay.UsageCost{
		ProviderCache:      providerCache,
		PromptTokens:       replayIntPtr(usage.promptTokens),
		CachedPromptTokens: replayIntPtr(usage.cachedPromptTokens),
		CacheWriteTokens:   replayIntPtr(usage.cacheWriteTokens),
		CompletionTokens:   replayIntPtr(usage.completionTokens),
		TotalTokens:        replayIntPtr(totalTokens),
		CostAccounting:     accounting,
	}

	if selectedPricing == nil {
		accounting.UnpricedReason = store.UnpricedPricingNotConfigured
		return snapshot
	}

	actualCost := costForResponseUsage(usage, *selectedPricing)
	currency := normalizeReplayCurrency(selectedPricing.Currency)
	snapshot.ActualCost = replayFloat64Ptr(actualCost)
	snapshot.Currency = replayStringPtr(currency)
	baseline := r.replayBaselineCost(ctx, usage, currency, actualCost)
	accounting.BaselineSource, accounting.BaselineUnavailable = baseline.source, baseline.unavailable
	if baseline.model == "" {
		return snapshot
	}
	costSavings := baseline.cost - actualCost

	snapshot.BaselineCost = replayFloat64Ptr(baseline.cost)
	snapshot.CostSavings = replayFloat64Ptr(costSavings)
	snapshot.BaselineModel = replayStringPtr(baseline.model)

	return snapshot
}

// replayPricingAccounting labels where the selected model's rates come from:
// the operator's pricing or the catalog's list price, and whether the model
// runs on the client's own login, whose plan the rates do not describe.
func (r *OpenAIRouter) replayPricingAccounting(ctx *RequestContext) *store.CostAccounting {
	accounting := &store.CostAccounting{}
	if r == nil || r.Config == nil || ctx == nil || ctx.RequestModel == "" {
		return accounting
	}
	provenance, priced := r.Config.ModelPricingProvenance(ctx.RequestModel)
	accounting.CallerLogin = provenance.CallerLogin
	if priced {
		accounting.PricingSource = store.PricingSourceConfigured
		if provenance.CatalogListPrice {
			accounting.PricingSource = store.PricingSourceCatalogList
		}
	}
	return accounting
}

// replayBaseline is the model a record's savings compare with, how it was
// chosen, and why there is none when model is empty.
type replayBaseline struct {
	model       string
	cost        float64
	source      string
	unavailable string
}

// replayBaselineCost prices this request's recorded usage at the baseline's
// rates. The operator's cost_baseline_model is the baseline when set, and a
// request routed to a dearer model then saves a negative amount. Otherwise it
// is the most expensive model of the selected recipe's complete pool across
// all decisions; other recipes and currencies cannot inflate savings, and a
// passthrough request has only its selected model as a baseline.
func (r *OpenAIRouter) replayBaselineCost(
	ctx *RequestContext,
	usage responseUsageMetrics,
	currency string,
	selectedCost float64,
) replayBaseline {
	if model := configuredReplayBaselineModel(ctx); model != "" {
		return r.configuredReplayBaseline(model, usage, currency)
	}
	recipe := ctx.Routing.SelectedRecipe()
	if recipe == nil {
		return replayBaseline{model: ctx.RequestModel, cost: selectedCost, source: store.BaselineSourceSelectedModel}
	}
	return r.recipePoolReplayBaseline(recipe, usage, currency)
}

// configuredReplayBaseline never falls back to another model: a baseline it
// cannot price leaves the record without one, and says why.
func (r *OpenAIRouter) configuredReplayBaseline(model string, usage responseUsageMetrics, currency string) replayBaseline {
	baseline := replayBaseline{source: store.BaselineSourceConfigured}
	pricing, ok := r.Config.GetFullModelPricing(model)
	switch {
	case !ok:
		baseline.unavailable = store.UnpricedPricingNotConfigured
	case normalizeReplayCurrency(pricing.Currency) != currency:
		baseline.unavailable = store.UnpricedCurrencyMismatch
	default:
		baseline.model, baseline.cost = model, costForResponseUsage(usage, pricing)
	}
	return baseline
}

func (r *OpenAIRouter) recipePoolReplayBaseline(
	recipe *config.RoutingRecipe,
	usage responseUsageMetrics,
	currency string,
) replayBaseline {
	baseline := replayBaseline{source: store.BaselineSourceRecipePool, unavailable: store.UnpricedPricingNotConfigured}
	for candidate := range replayRecipeModelPool(recipe, r.Config.DefaultModel) {
		pricing, ok := r.Config.GetFullModelPricing(candidate)
		if !ok {
			continue
		}
		if normalizeReplayCurrency(pricing.Currency) != currency {
			if baseline.model == "" {
				baseline.unavailable = store.UnpricedCurrencyMismatch
			}
			continue
		}
		candidateCost := costForResponseUsage(usage, pricing)
		if baseline.model == "" || candidateCost > baseline.cost || (candidateCost == baseline.cost && candidate < baseline.model) {
			baseline.model, baseline.cost, baseline.unavailable = candidate, candidateCost, ""
		}
	}
	return baseline
}

func configuredReplayBaselineModel(ctx *RequestContext) string {
	if ctx == nil || ctx.RouterReplayPluginConfig == nil {
		return ""
	}
	return strings.TrimSpace(ctx.RouterReplayPluginConfig.CostBaselineModel)
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
