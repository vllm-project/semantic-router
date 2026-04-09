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
	lastSeen             time.Time
}

var (
	mu     sync.Mutex
	store  = make(map[string]*turnState)
	nowFn  = time.Now // overridden in tests
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

// TurnParams carries usage and routing context for one completed LLM round-trip.
type TurnParams struct {
	RequestID string
	Model     string
	Domain    string // VSR category; empty -> "unknown"

	PromptTokens     int
	CompletionTokens int

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

// RecordTurn updates cumulative session token state, Prometheus histograms, and emits a structured log event.
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
	st.lastSeen = t
	cumP := st.cumulativePrompt
	cumC := st.cumulativeCompletion
	mu.Unlock()

	metrics.RecordSessionTurnTokens(model, domain, float64(p.PromptTokens), float64(p.CompletionTokens))

	logging.LogEvent("session_turn_tokens", map[string]interface{}{
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
	})
}
