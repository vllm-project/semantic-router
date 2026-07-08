package sessiontelemetry

import (
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/consts"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

const (
	// ttl is how long an idle session's cumulative state is retained before it
	// becomes eligible for eviction.
	ttl = 1 * time.Hour

	// evictInterval bounds how often the full-store TTL sweep runs. TTL eviction is
	// a soft memory reclaim (an expired entry only costs memory until it is removed),
	// so the sweep is throttled to at most once per interval. This keeps the per-turn
	// hot path (recordTurnState holds mu for every completed LLM round-trip) O(1)
	// amortized instead of O(active sessions) on every call. Worst-case retention
	// becomes ttl + evictInterval, which is negligible relative to ttl.
	evictInterval = 1 * time.Minute
)

// maxTelemetrySessions caps the number of sessions tracked by the cumulative
// turn store so memory stays bounded under high session cardinality even within
// the TTL window. Mirrors last_model.go's maxLastModelSessions. It is a var (not
// a const) so tests can exercise the eviction path without inserting tens of
// thousands of entries.
var maxTelemetrySessions = 50_000

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
	mu        sync.Mutex
	store     = make(map[string]*turnState)
	lastEvict time.Time  // last time the full-store TTL sweep ran; throttles evictLocked
	nowFn     = time.Now // overridden in tests
)

// ResetForTesting clears in-memory session state (tests only).
func ResetForTesting() {
	mu.Lock()
	defer mu.Unlock()
	store = make(map[string]*turnState)
	lastEvict = time.Time{}
	ResetRouterSessionMemoryForTesting()
}

// evictLocked removes sessions whose lastSeen is older than ttl. The full-store
// scan is throttled to at most once per evictInterval so the per-turn hot path
// (recordTurnState, called under mu for every completed LLM round-trip) stays
// O(1) amortized rather than O(active sessions) on every call. Callers must hold mu.
func evictLocked(t time.Time) {
	if !lastEvict.IsZero() && t.Sub(lastEvict) < evictInterval {
		return
	}
	lastEvict = t
	for k, v := range store {
		if t.Sub(v.lastSeen) > ttl {
			delete(store, k)
		}
	}
}

// evictOldestLocked evicts the oldest session among a bounded random sample
// (approximate LRU — see evictionSampleSize). It is a best-effort safety valve
// for the size cap when the throttled TTL sweep has not freed room. Callers must
// hold mu.
func evictOldestLocked() {
	var oldestKey string
	var oldestSeen time.Time
	sampled := 0
	for k, v := range store {
		if sampled == 0 || v.lastSeen.Before(oldestSeen) {
			oldestKey, oldestSeen = k, v.lastSeen
		}
		sampled++
		if sampled >= evictionSampleSize {
			break
		}
	}
	if sampled > 0 {
		delete(store, oldestKey)
	}
}

// telemetrySessionCount returns the number of tracked sessions (tests only).
func telemetrySessionCount() int {
	mu.Lock()
	defer mu.Unlock()
	return len(store)
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
	if st != nil && t.Sub(st.lastSeen) > ttl {
		st = nil
	}
	if st == nil {
		if len(store) >= maxTelemetrySessions {
			evictOldestLocked()
		}
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
