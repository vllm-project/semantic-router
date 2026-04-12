package sessiontelemetry

import (
	"sync"
	"time"
)

// ModelTransitionEvent records a model switch within a session.
type ModelTransitionEvent struct {
	SessionID           string    `json:"session_id"`
	TurnIndex           int       `json:"turn_index"`
	FromModel           string    `json:"from_model"`
	ToModel             string    `json:"to_model"`
	TTFTMs              float64   `json:"ttft_ms"`
	CacheWarmthEstimate float64   `json:"cache_warmth_estimate"`
	PreviousResponseID  string    `json:"previous_response_id,omitempty"`
	Timestamp           time.Time `json:"timestamp"`
}

// MaxTransitionEventsPerSession limits stored events per session.
const MaxTransitionEventsPerSession = 200

// transitionStore holds in-memory transition events by session ID (test and debug use).
type transitionStore struct {
	mu     sync.RWMutex
	events map[string][]ModelTransitionEvent
}

var globalTransitionStore = &transitionStore{
	events: make(map[string][]ModelTransitionEvent),
}

// RecordTransition appends evt to the in-memory store and drops the oldest
// entries once the per-session cap is reached.
func RecordTransition(evt ModelTransitionEvent) {
	globalTransitionStore.mu.Lock()
	defer globalTransitionStore.mu.Unlock()
	list := globalTransitionStore.events[evt.SessionID]
	list = append(list, evt)
	if len(list) > MaxTransitionEventsPerSession {
		trimmed := make([]ModelTransitionEvent, MaxTransitionEventsPerSession)
		copy(trimmed, list[len(list)-MaxTransitionEventsPerSession:])
		list = trimmed
	}
	globalTransitionStore.events[evt.SessionID] = list
}

// GetTransitions returns a copy of the transition events for sessionID.
func GetTransitions(sessionID string) []ModelTransitionEvent {
	globalTransitionStore.mu.RLock()
	defer globalTransitionStore.mu.RUnlock()
	src := globalTransitionStore.events[sessionID]
	out := make([]ModelTransitionEvent, len(src))
	copy(out, src)
	return out
}

// ResetTransitionsForTesting clears the in-memory transition store.
func ResetTransitionsForTesting() {
	globalTransitionStore.mu.Lock()
	defer globalTransitionStore.mu.Unlock()
	globalTransitionStore.events = make(map[string][]ModelTransitionEvent)
}
