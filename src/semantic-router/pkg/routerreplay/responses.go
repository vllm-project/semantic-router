// Package-level response contracts are shared by runtime serialization and
// management OpenAPI generation.
package routerreplay

import "time"

type ListResponse struct {
	Object     string          `json:"object"`
	Count      int             `json:"count"`
	Total      int             `json:"total"`
	Limit      int             `json:"limit"`
	Offset     int             `json:"offset"`
	HasMore    bool            `json:"has_more"`
	NextOffset *int            `json:"next_offset,omitempty"`
	Data       []RoutingRecord `json:"data"`
}

type AggregateResponse struct {
	Object               string                        `json:"object"`
	RecordCount          int                           `json:"record_count"`
	Lifecycle            LifecycleSummary              `json:"lifecycle"`
	Summary              AggregateCostSummary          `json:"summary"`
	ProviderCache        AggregateProviderCacheSummary `json:"provider_cache"`
	ModelSelection       []AggregateValue              `json:"model_selection"`
	DecisionDistribution []AggregateValue              `json:"decision_distribution"`
	SignalDistribution   []AggregateValue              `json:"signal_distribution"`
	TokenVolume          AggregateTokenVolume          `json:"token_volume"`
	TokenBreakdown       AggregateTokenBuckets         `json:"token_breakdown"`
	AvailableRecipes     []string                      `json:"available_recipes"`
	AvailableDecisions   []string                      `json:"available_decisions"`
	AvailableModels      []string                      `json:"available_models"`
	// Reviews is present when the filtered records include approval reviews.
	Reviews *ReviewAggregate `json:"reviews,omitempty"`
	// AvailableReviewModels lists the models that answered any review.
	AvailableReviewModels []string `json:"available_review_models,omitempty"`
	// Traffic is present when the query asked for it with an interval.
	Traffic *TrafficSeries `json:"traffic,omitempty"`
}

type LifecycleSummary struct {
	Completed  int `json:"completed"`
	Failed     int `json:"failed"`
	Aborted    int `json:"aborted"`
	InProgress int `json:"in_progress"`
	Unknown    int `json:"unknown"`
}

// AggregateCostSummary compares the records with a baseline over
// CostRecordCount completed records: GrossSaved is BaselineSpend minus
// ActualSpend, the selected models' cost, and TotalSaved is GrossSaved minus
// NettedRouterCallSpend, the calls the Router made itself for those records.
// It also totals every priced cost of every record, failed ones included
// (TotalSpend), the calls the Router made itself included apart as
// RouterCallSpend.
type AggregateCostSummary struct {
	TotalSaved            float64 `json:"total_saved"`
	GrossSaved            float64 `json:"gross_saved"`
	NettedRouterCallSpend float64 `json:"netted_router_call_spend"`
	BaselineSpend         float64 `json:"baseline_spend"`
	ActualSpend           float64 `json:"actual_spend"`
	RouterCallSpend       float64 `json:"router_call_spend"`
	TotalSpend            float64 `json:"total_spend"`
	Currency              string  `json:"currency,omitempty"`
	CostRecordCount       int     `json:"cost_record_count"`
	ExcludedRecordCount   int     `json:"excluded_record_count"`
	// ByCurrency retains every complete estimate without converting or summing
	// currencies. Flat amounts are populated only when exactly one currency exists.
	ByCurrency []CurrencyCostSummary `json:"by_currency,omitempty"`
	// Coverage says how much of the filtered traffic the amounts leave out.
	Coverage AggregateCostCoverage `json:"coverage"`
}

type CurrencyCostSummary struct {
	Currency              string  `json:"currency"`
	TotalSaved            float64 `json:"total_saved"`
	GrossSaved            float64 `json:"gross_saved"`
	NettedRouterCallSpend float64 `json:"netted_router_call_spend"`
	BaselineSpend         float64 `json:"baseline_spend"`
	ActualSpend           float64 `json:"actual_spend"`
	RouterCallSpend       float64 `json:"router_call_spend"`
	TotalSpend            float64 `json:"total_spend"`
	CostRecordCount       int     `json:"cost_record_count"`
}

// AggregateCostCoverage counts the filtered records whose every cost has a
// price, those with some, and those with none; a record that has not
// completed is at most partly priced. UnpricedReasons counts the partly
// priced and unpriced records by the reason of their first missing price. A
// Looper hop's record is in none of them but in LooperHopRecords: the record
// of the request that made the hop counts its call.
// ListPriceRecords are priced at a catalog list price, an estimate.
// CallerLoginRecords ran on the client's own provider login, priced or not:
// a price shown for one is what its tokens cost at API prices, not what the
// client's plan charges. BaselineSources counts the records compared with a
// baseline by how it was chosen; BaselineUnavailable, the priced completed
// records without one, by why. NettedRouterCalls counts the Router calls
// netted out of savings by kind. NetUnavailable counts the records with a
// baseline left out of savings because a Router call they made has no price
// in their currency, by the reason of the first such call: an unknown cost is
// never netted as zero.
type AggregateCostCoverage struct {
	PricedRecords       int            `json:"priced_records"`
	PartlyPricedRecords int            `json:"partly_priced_records"`
	UnpricedRecords     int            `json:"unpriced_records"`
	UnpricedReasons     map[string]int `json:"unpriced_reasons,omitempty"`
	LooperHopRecords    int            `json:"looper_hop_records"`
	ListPriceRecords    int            `json:"list_price_records"`
	CallerLoginRecords  int            `json:"caller_login_records"`
	RouterCalls         int            `json:"router_calls"`
	UnpricedRouterCalls int            `json:"unpriced_router_calls"`
	BaselineSources     map[string]int `json:"baseline_sources,omitempty"`
	BaselineUnavailable map[string]int `json:"baseline_unavailable,omitempty"`
	NettedRouterCalls   map[string]int `json:"netted_router_calls,omitempty"`
	NetUnavailable      map[string]int `json:"net_unavailable,omitempty"`
}

// AggregateProviderCacheSummary totals the same-model provider prompt-cache
// counterfactual over settled records: completed, not served from the Router
// cache, with complete usage and captured prices. NetSavings is NoCacheSpend
// minus ActualSpend, so requests whose cache writes cost more than their
// reads saved reduce it; no request is clamped at zero. Flat amounts are
// populated only when exactly one currency exists.
type AggregateProviderCacheSummary struct {
	NoCacheSpend    float64                      `json:"no_cache_spend"`
	ActualSpend     float64                      `json:"actual_spend"`
	NetSavings      float64                      `json:"net_savings"`
	Currency        string                       `json:"currency,omitempty"`
	PricedRecords   int                          `json:"priced_records"`
	ExcludedRecords int                          `json:"excluded_records"`
	ByCurrency      []ProviderCacheCurrencyTotal `json:"by_currency,omitempty"`
	ByDecision      []ProviderCacheSpendEntry    `json:"by_decision"`
	BySelectedModel []ProviderCacheSpendEntry    `json:"by_selected_model"`
}

type ProviderCacheCurrencyTotal struct {
	Currency      string  `json:"currency"`
	NoCacheSpend  float64 `json:"no_cache_spend"`
	ActualSpend   float64 `json:"actual_spend"`
	NetSavings    float64 `json:"net_savings"`
	PricedRecords int     `json:"priced_records"`
}

// ProviderCacheSpendEntry is one decision or selected model in one currency.
type ProviderCacheSpendEntry struct {
	Name          string  `json:"name"`
	Currency      string  `json:"currency"`
	NoCacheSpend  float64 `json:"no_cache_spend"`
	ActualSpend   float64 `json:"actual_spend"`
	NetSavings    float64 `json:"net_savings"`
	PricedRecords int     `json:"priced_records"`
}

type AggregateValue struct {
	Name  string `json:"name"`
	Value int    `json:"value"`
}

type AggregateTokenVolume struct {
	InputTokens         int `json:"input_tokens"`
	OutputTokens        int `json:"output_tokens"`
	TotalTokens         int `json:"total_tokens"`
	ExcludedRecordCount int `json:"excluded_record_count"`
}

type AggregateTokenBuckets struct {
	ByDecision      []AggregateTokenEntry `json:"by_decision"`
	BySelectedModel []AggregateTokenEntry `json:"by_selected_model"`
}

type AggregateTokenEntry struct {
	Name         string `json:"name"`
	InputTokens  int    `json:"input_tokens"`
	OutputTokens int    `json:"output_tokens"`
	TotalTokens  int    `json:"total_tokens"`
}

type TrajectoryFunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type TrajectoryToolCall struct {
	ID       string                 `json:"id"`
	Type     string                 `json:"type"`
	Function TrajectoryFunctionCall `json:"function"`
}

type TrajectoryMessage struct {
	ConversationID string               `json:"conversation_id,omitempty"`
	Role           string               `json:"role"`
	Content        string               `json:"content,omitempty"`
	ToolCalls      []TrajectoryToolCall `json:"tool_calls,omitempty"`
	ToolCallID     string               `json:"tool_call_id,omitempty"`
	ToolName       string               `json:"tool_name,omitempty"`
	// ContentTruncated marks message text or a tool result cut at max_tool_trace_bytes.
	ContentTruncated bool `json:"content_truncated,omitempty"`
	TurnIndex        int  `json:"turn_index"`
}

type TrajectoryResponse struct {
	Object      string              `json:"object"`
	SessionID   string              `json:"session_id"`
	Recipe      string              `json:"recipe"`
	RecordCount int                 `json:"record_count"`
	TurnCount   int                 `json:"turn_count"`
	Messages    []TrajectoryMessage `json:"messages"`
	Routes      []TrajectoryRoute   `json:"routes"`
}

// TrajectoryRoute is one routed call of a session. Agent is the short hash of
// the agent identity header a harness sent with a subagent's call, empty for
// the main conversation. TurnID and ParentTurnID are the record's agent turn
// ids, so a session's calls can be read by the Codex turn that sent or
// started them.
type TrajectoryRoute struct {
	ConversationID       string    `json:"conversation_id,omitempty"`
	RecordID             string    `json:"record_id"`
	Timestamp            time.Time `json:"timestamp"`
	TurnIndex            int       `json:"turn_index"`
	Decision             string    `json:"decision,omitempty"`
	SelectedModel        string    `json:"selected_model,omitempty"`
	PreviousModel        string    `json:"previous_model,omitempty"`
	SelectionMethod      string    `json:"selection_method,omitempty"`
	SelectionReasoning   string    `json:"selection_reasoning,omitempty"`
	SessionPolicyApplied bool      `json:"session_policy_applied"`
	SessionAction        string    `json:"session_action,omitempty"`
	SessionReason        string    `json:"session_reason,omitempty"`
	Agent                string    `json:"agent,omitempty"`
	TurnID               string    `json:"turn_id,omitempty"`
	ParentTurnID         string    `json:"parent_turn_id,omitempty"`
	LifecycleState       string    `json:"lifecycle_state"`
	ResponseStatus       int       `json:"response_status,omitempty"`
	DurationMS           int64     `json:"duration_ms"`
}
