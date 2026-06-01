package sessiontelemetry

import (
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/consts"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

const ttl = 1 * time.Hour

type turnState struct {
	cumulativePrompt                int64
	cumulativeCached                int64
	cumulativeEstimatedCached       int64
	cumulativeCompletion            int64
	cumulativeCost                  float64 // in the currency supplied by TurnPricing.Currency
	cumulativeEstimatedCacheSavings float64
	lastCacheAccountingSource       string
	lastSeen                        time.Time
}

var (
	mu    sync.Mutex
	store = make(map[string]*turnState)
	nowFn = time.Now // overridden in tests
)

// ResetForTesting clears in-memory session state (tests only).
func ResetForTesting() {
	mu.Lock()
	defer mu.Unlock()
	store = make(map[string]*turnState)
	ResetRouterSessionMemoryForTesting()
}

func evictLocked(t time.Time) {
	for k, v := range store {
		if t.Sub(v.lastSeen) > ttl {
			delete(store, k)
		}
	}
}

// ResponseAPIInput identifies a Response API (/v1/responses) conversation.
type ResponseAPIInput struct {
	ConversationID string
	HistoryLen     int
}

// ChatInput identifies a Chat Completions session when user id and messages are available.
type ChatInput struct {
	UserID   string
	Messages []ChatMessage
}

// TurnPricing carries the active per-1M token prices stamped onto a dispatch log entry.
// All rates are in Currency (default "USD").
//
// isConfigured() returns true when at least one rate is non-zero or Currency is explicitly
// set (even with zero rates) — the latter supports free/self-hosted models that
// still need cost=0 and savings attribution.
type TurnPricing struct {
	Currency         string
	PromptPer1M      float64
	CompletionPer1M  float64
	CachedInputPer1M float64
}

// isConfigured reports whether pricing is active: at least one billable rate is set, or
// Currency is explicit (covers zero-rate free models that still produce cost=0 telemetry).
func (p TurnPricing) isConfigured() bool {
	return p.PromptPer1M != 0 || p.CompletionPer1M != 0 || p.CachedInputPer1M != 0 || p.Currency != ""
}

// TurnParams carries usage and routing context for one completed LLM round-trip.
type TurnParams struct {
	RequestID string
	Model     string
	Domain    string // VSR category; empty -> "unknown"

	PromptTokens                int
	CachedPromptTokens          int
	EstimatedCachedPromptTokens int
	CompletionTokens            int
	EstimatedCacheSavings       float64
	CacheAccountingSource       string
	CacheAccountingConfidence   float64

	// Pricing holds the active per-1M token rates for this dispatch.
	// When zero, cost fields are omitted from the log event and no cost
	// histogram observation is recorded.
	Pricing TurnPricing

	ResponseAPI *ResponseAPIInput
	Chat        *ChatInput
}

type turnCumulativeState struct {
	prompt     int64
	cached     int64
	completion int64
	cost       float64
}

func normalizeDomain(d string) string {
	if d == "" {
		return consts.UnknownLabel
	}
	return d
}

func resolve(p TurnParams) (storeKey string, turn int, apiKind string, publicSessionID string, ok bool) {
	if p.ResponseAPI != nil {
		if p.ResponseAPI.ConversationID == "" {
			return "", 0, "", "", false
		}
		cid := p.ResponseAPI.ConversationID
		turn = p.ResponseAPI.HistoryLen + 1
		return "respapi:" + cid, turn, "responses", cid, true
	}
	if p.Chat != nil && p.Chat.UserID != "" && len(p.Chat.Messages) > 0 {
		sid := DeriveChatCompletionsSessionID(p.Chat.Messages, p.Chat.UserID)
		turn = ChatTurnNumber(p.Chat.Messages)
		return "chat:" + sid, turn, "chat_completions", sid, true
	}
	return "", 0, "", "", false
}

// RecordTurn updates cumulative session token/cost state, Prometheus histograms, and emits a structured log event.
func RecordTurn(p TurnParams) {
	if p.PromptTokens+p.CompletionTokens <= 0 {
		return
	}
	key, turn, apiKind, publicSessionID, ok := resolve(p)
	if !ok {
		return
	}
	model := p.Model
	if model == "" {
		model = consts.UnknownLabel
	}
	domain := normalizeDomain(p.Domain)

	costThisTurn := computeCost(p.PromptTokens, p.CachedPromptTokens, p.CompletionTokens, p.Pricing)

	t := nowFn()
	cumulative := recordTurnState(key, p, costThisTurn, t)
	recordRouterSessionUsage(publicSessionID, model, p, costThisTurn, t)

	metrics.RecordSessionTurnTokens(model, domain, float64(p.PromptTokens), float64(p.CompletionTokens))
	currency := pricingCurrency(p.Pricing)
	if p.Pricing.isConfigured() {
		metrics.RecordSessionTurnCost(model, domain, currency, costThisTurn)
	}
	logSessionTurn(p, publicSessionID, turn, apiKind, model, domain, currency, costThisTurn, cumulative, t)
}

func recordTurnState(key string, p TurnParams, costThisTurn float64, t time.Time) turnCumulativeState {
	mu.Lock()
	defer mu.Unlock()

	evictLocked(t)
	st := store[key]
	if st == nil {
		st = &turnState{}
		store[key] = st
	}
	st.cumulativePrompt += int64(p.PromptTokens)
	st.cumulativeCached += int64(clampCachedPromptTokens(p.PromptTokens, p.CachedPromptTokens))
	st.cumulativeEstimatedCached += int64(clampCachedPromptTokens(p.PromptTokens, p.EstimatedCachedPromptTokens))
	st.cumulativeCompletion += int64(p.CompletionTokens)
	st.cumulativeCost += costThisTurn
	if p.EstimatedCacheSavings > 0 {
		st.cumulativeEstimatedCacheSavings += p.EstimatedCacheSavings
	}
	if p.CacheAccountingSource != "" {
		st.lastCacheAccountingSource = p.CacheAccountingSource
	}
	st.lastSeen = t
	return turnCumulativeState{
		prompt:     st.cumulativePrompt,
		cached:     st.cumulativeCached,
		completion: st.cumulativeCompletion,
		cost:       st.cumulativeCost,
	}
}

func recordRouterSessionUsage(publicSessionID string, model string, p TurnParams, costThisTurn float64, t time.Time) {
	RecordSessionUsage(SessionUsageParams{
		SessionID:                   publicSessionID,
		Model:                       model,
		PromptTokens:                p.PromptTokens,
		CachedPromptTokens:          p.CachedPromptTokens,
		EstimatedCachedPromptTokens: p.EstimatedCachedPromptTokens,
		CompletionTokens:            p.CompletionTokens,
		Cost:                        costThisTurn,
		EstimatedCacheSavings:       p.EstimatedCacheSavings,
		CacheAccountingSource:       p.CacheAccountingSource,
		Timestamp:                   t,
	})
}

func pricingCurrency(pricing TurnPricing) string {
	currency := pricing.Currency
	if currency == "" {
		currency = "USD"
	}
	return currency
}

func logSessionTurn(
	p TurnParams,
	publicSessionID string,
	turn int,
	apiKind string,
	model string,
	domain string,
	currency string,
	costThisTurn float64,
	cumulative turnCumulativeState,
	t time.Time,
) {
	fields := map[string]interface{}{
		"request_id":                      p.RequestID,
		"session_id":                      publicSessionID,
		"turn_number":                     turn,
		"api":                             apiKind,
		"model":                           model,
		"domain":                          domain,
		"prompt_tokens":                   p.PromptTokens,
		"cached_prompt_tokens":            clampCachedPromptTokens(p.PromptTokens, p.CachedPromptTokens),
		"completion_tokens":               p.CompletionTokens,
		"cumulative_prompt_tokens":        cumulative.prompt,
		"cumulative_cached_prompt_tokens": cumulative.cached,
		"cumulative_completion_tokens":    cumulative.completion,
		"timestamp":                       t.UTC().Format(time.RFC3339Nano),
	}
	if p.CacheAccountingSource != "" {
		fields["router_cache_accounting_source"] = p.CacheAccountingSource
		fields["router_estimated_cached_prompt_tokens"] = clampCachedPromptTokens(p.PromptTokens, p.EstimatedCachedPromptTokens)
		fields["router_cache_accounting_confidence"] = p.CacheAccountingConfidence
		if p.EstimatedCacheSavings > 0 {
			fields["router_estimated_cache_savings"] = p.EstimatedCacheSavings
		}
	}
	if p.Pricing.isConfigured() {
		fields["pricing_prompt_per_1m"] = p.Pricing.PromptPer1M
		fields["pricing_completion_per_1m"] = p.Pricing.CompletionPer1M
		fields["pricing_cached_input_per_1m"] = p.Pricing.CachedInputPer1M
		fields["pricing_currency"] = currency
		fields["cost_this_turn"] = costThisTurn
		fields["cumulative_cost"] = cumulative.cost
	}
	logging.LogEvent("session_turn_tokens", fields)
}

// computeCost returns the turn cost in the pricing currency given token counts
// and rates. Cached prompt tokens use CachedInputPer1M when configured.
func computeCost(promptTokens, cachedPromptTokens, completionTokens int, pricing TurnPricing) float64 {
	if !pricing.isConfigured() {
		return 0
	}
	cached := clampCachedPromptTokens(promptTokens, cachedPromptTokens)
	uncachedPrompt := promptTokens - cached
	return (float64(uncachedPrompt)*pricing.PromptPer1M +
		float64(cached)*pricing.CachedInputPer1M +
		float64(completionTokens)*pricing.CompletionPer1M) / 1_000_000.0
}

func clampCachedPromptTokens(promptTokens, cachedPromptTokens int) int {
	if cachedPromptTokens < 0 {
		return 0
	}
	if cachedPromptTokens > promptTokens {
		return promptTokens
	}
	return cachedPromptTokens
}
