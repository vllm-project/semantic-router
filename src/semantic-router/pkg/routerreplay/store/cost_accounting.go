package store

import "maps"

// UsageCost captures token usage and pricing-derived cost details for a record.
type UsageCost struct {
	ProviderCache      *ProviderCacheAccounting `json:"provider_cache,omitempty"`
	PromptTokens       *int                     `json:"prompt_tokens,omitempty"`
	CachedPromptTokens *int                     `json:"cached_prompt_tokens,omitempty"`
	CacheWriteTokens   *int                     `json:"cache_write_tokens,omitempty"`
	CompletionTokens   *int                     `json:"completion_tokens,omitempty"`
	TotalTokens        *int                     `json:"total_tokens,omitempty"`
	ActualCost         *float64                 `json:"actual_cost,omitempty"`
	BaselineCost       *float64                 `json:"baseline_cost,omitempty"`
	CostSavings        *float64                 `json:"cost_savings,omitempty"`
	Currency           *string                  `json:"currency,omitempty"`
	BaselineModel      *string                  `json:"baseline_model,omitempty"`
	CostAccounting     *CostAccounting          `json:"cost_accounting,omitempty"`
}

// Where the selected model's rates came from.
const (
	// PricingSourceConfigured is the operator's providers.models[].pricing.
	PricingSourceConfigured = "configured"
	// PricingSourceCatalogList is the built-in catalog's list price for the
	// model's provider, used when the operator set no pricing: an estimate.
	PricingSourceCatalogList = "catalog_list_price"
)

// How a record's baseline model was chosen.
const (
	// BaselineSourceConfigured is router_replay's cost_baseline_model.
	BaselineSourceConfigured = "configured_model"
	// BaselineSourceRecipePool is the default: the most expensive estimate in
	// the selected recipe's model pool for the same recorded usage.
	BaselineSourceRecipePool = "recipe_most_expensive"
	// BaselineSourceSelectedModel is a request that routed through no
	// recipe, compared with its own model.
	BaselineSourceSelectedModel = "selected_model"
)

// Why a cost figure is missing. Each one counts as unpriced, never as free.
const (
	UnpricedNoUsage              = "no_usage"
	UnpricedUsageIncomplete      = "usage_breakdown_unavailable"
	UnpricedPricingNotConfigured = "pricing_not_configured"
	UnpricedCurrencyMismatch     = "currency_mismatch"
	// UnpricedAttemptNotTraced is a Looper call whose price no attempt trace
	// kept: past the trace bound, or made by an algorithm that traces none.
	// Its tokens are in the request's.
	UnpricedAttemptNotTraced = "attempt_not_traced"
	// UnpricedNotRecorded is a record that carries no cost accounting, such
	// as one written before the Router recorded why a price is missing.
	UnpricedNotRecorded = "price_not_recorded"
	// UnpricedNotCompleted is a request that has not completed.
	UnpricedNotCompleted = "not_completed"
	// UnpricedCountedInParent is the record of a Looper hop, whose call the
	// record of the request that made it counts.
	UnpricedCountedInParent = "counted_in_parent"
)

// RouterCallKindLooper is a Looper attempt: a model call the Router made
// itself while it served the request.
const RouterCallKindLooper = "looper"

// CostAccounting says where a record's cost figures came from and what they
// leave out, so a reader can tell a configured price from a list-price
// estimate, the chosen baseline from the default one, and the selected
// model's cost from the calls the Router made on its own behalf.
type CostAccounting struct {
	// PricingSource is where the selected model's rates came from, empty
	// when the model has none.
	PricingSource string `json:"pricing_source,omitempty"`
	// CallerLogin marks a model served on the client's own provider login
	// (auth.credential: caller): its cost is what the same tokens cost at API
	// prices, not what the client's plan charges.
	CallerLogin bool `json:"caller_login,omitempty"`
	// UnpricedReason says why ActualCost is missing.
	UnpricedReason string `json:"unpriced_reason,omitempty"`
	// BaselineSource is how the baseline was chosen. BaselineUnavailable says
	// why a priced record has no baseline.
	BaselineSource      string `json:"baseline_source,omitempty"`
	BaselineUnavailable string `json:"baseline_unavailable,omitempty"`
	// RouterCalls are the model calls the Router made itself to serve the
	// request, costed apart from ActualCost. Runtime tasks include their model.
	RouterCalls []RouterCallCost `json:"router_calls,omitempty"`
}

// RouterCallCost describes model calls the Router made itself. Cost sums
// the priced calls in Currency; the others are UnpricedCalls, by reason.
type RouterCallCost struct {
	Model            string         `json:"model,omitempty"`
	Kind             string         `json:"kind"`
	Calls            int            `json:"calls"`
	PromptTokens     int64          `json:"prompt_tokens,omitempty"`
	CompletionTokens int64          `json:"completion_tokens,omitempty"`
	Cost             *float64       `json:"cost,omitempty"`
	Currency         string         `json:"currency,omitempty"`
	UnpricedCalls    int            `json:"unpriced_calls,omitempty"`
	UnpricedReasons  map[string]int `json:"unpriced_reasons,omitempty"`
}

func cloneCostAccounting(value *CostAccounting) *CostAccounting {
	if value == nil {
		return nil
	}
	result := *value
	if value.RouterCalls != nil {
		result.RouterCalls = make([]RouterCallCost, len(value.RouterCalls))
		for i, call := range value.RouterCalls {
			call.Cost = cloneFloat64Ptr(call.Cost)
			call.UnpricedReasons = maps.Clone(call.UnpricedReasons)
			result.RouterCalls[i] = call
		}
	}
	return &result
}
