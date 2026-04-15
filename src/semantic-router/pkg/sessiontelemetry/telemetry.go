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
	cumulativePrompt     int64
	cumulativeCompletion int64
	cumulativeCost       float64 // in the currency supplied by TurnPricing.Currency
	lastSeen             time.Time
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
// set (even with zero rates) — the latter supports free/self-hosted models that still need
// cost=0 and savings attribution. CachedInputPer1M is stamped as log metadata but is not
// included in the configured check or cost formula until cached token counts are tracked.
type TurnPricing struct {
	Currency         string
	PromptPer1M      float64
	CompletionPer1M  float64
	CachedInputPer1M float64 // stamped in log metadata; not applied to cost (cached tokens not yet tracked)
}

// isConfigured reports whether pricing is active: at least one billable rate is set, or
// Currency is explicit (covers zero-rate free models that still produce cost=0 telemetry).
func (p TurnPricing) isConfigured() bool {
	return p.PromptPer1M != 0 || p.CompletionPer1M != 0 || p.Currency != ""
}

// TurnParams carries usage and routing context for one completed LLM round-trip.
type TurnParams struct {
	RequestID string
	Model     string
	Domain    string // VSR category; empty -> "unknown"

	PromptTokens     int
	CompletionTokens int

	// Pricing holds the active per-1M token rates for this dispatch.
	// When zero, cost fields are omitted from the log event and no cost
	// histogram observation is recorded.
	Pricing TurnPricing

	ResponseAPI *ResponseAPIInput
	Chat        *ChatInput
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

	costThisTurn := computeCost(p.PromptTokens, p.CompletionTokens, p.Pricing)

	t := nowFn()
	mu.Lock()
	evictLocked(t)
	st := store[key]
	if st == nil {
		st = &turnState{}
		store[key] = st
	}
	st.cumulativePrompt += int64(p.PromptTokens)
	st.cumulativeCompletion += int64(p.CompletionTokens)
	st.cumulativeCost += costThisTurn
	st.lastSeen = t
	cumP := st.cumulativePrompt
	cumC := st.cumulativeCompletion
	cumCost := st.cumulativeCost
	mu.Unlock()

	metrics.RecordSessionTurnTokens(model, domain, float64(p.PromptTokens), float64(p.CompletionTokens))

	currency := p.Pricing.Currency
	if currency == "" {
		currency = "USD"
	}
	if p.Pricing.isConfigured() {
		metrics.RecordSessionTurnCost(model, domain, currency, costThisTurn)
	}

	fields := map[string]interface{}{
		"request_id":                   p.RequestID,
		"session_id":                   publicSessionID,
		"turn_number":                  turn,
		"api":                          apiKind,
		"model":                        model,
		"domain":                       domain,
		"prompt_tokens":                p.PromptTokens,
		"completion_tokens":            p.CompletionTokens,
		"cumulative_prompt_tokens":     cumP,
		"cumulative_completion_tokens": cumC,
		"timestamp":                    t.UTC().Format(time.RFC3339Nano),
	}
	if p.Pricing.isConfigured() {
		fields["pricing_prompt_per_1m"] = p.Pricing.PromptPer1M
		fields["pricing_completion_per_1m"] = p.Pricing.CompletionPer1M
		fields["pricing_cached_input_per_1m"] = p.Pricing.CachedInputPer1M
		fields["pricing_currency"] = currency
		fields["cost_this_turn"] = costThisTurn
		fields["cumulative_cost"] = cumCost
	}
	logging.LogEvent("session_turn_tokens", fields)
}

// computeCost returns the turn cost in the pricing currency given token counts and rates.
// Only prompt and completion tokens are costed; cached-input tokens are not tracked yet.
// Returns 0 when pricing is not configured.
func computeCost(promptTokens, completionTokens int, pricing TurnPricing) float64 {
	if !pricing.isConfigured() {
		return 0
	}
	return (float64(promptTokens)*pricing.PromptPer1M +
		float64(completionTokens)*pricing.CompletionPer1M) / 1_000_000.0
}
