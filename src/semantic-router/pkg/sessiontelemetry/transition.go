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

// maxTransitionSessions caps the number of sessions tracked by the transition
// store. The per-session slice was already bounded, but the session map itself
// had no bound (and, unlike the other session stores, no TTL either), so it grew
// one key per distinct session for the lifetime of the process. Mirrors
// last_model.go's maxLastModelSessions. A var (not a const) so tests can
// exercise the eviction path without inserting tens of thousands of entries.
var maxTransitionSessions = 50_000

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
	if evt.SessionID == "" {
		return
	}
	globalTransitionStore.mu.Lock()
	defer globalTransitionStore.mu.Unlock()
	list, exists := globalTransitionStore.events[evt.SessionID]
	if !exists && len(globalTransitionStore.events) >= maxTransitionSessions {
		globalTransitionStore.evictOldestLocked()
	}
	list = append(list, evt)
	if len(list) > MaxTransitionEventsPerSession {
		trimmed := make([]ModelTransitionEvent, MaxTransitionEventsPerSession)
		copy(trimmed, list[len(list)-MaxTransitionEventsPerSession:])
		list = trimmed
	}
	globalTransitionStore.events[evt.SessionID] = list
}

// evictOldestLocked evicts the session whose most recent transition event is the
// oldest among a bounded random sample (approximate LRU — see
// evictionSampleSize), bounding the session map. Best-effort safety valve for the
// size cap. Callers must hold mu.
func (s *transitionStore) evictOldestLocked() {
	var oldestKey string
	var oldestSeen time.Time
	sampled := 0
	for k, list := range s.events {
		// RecordTransition always stores at least one event per key.
		last := list[len(list)-1].Timestamp
		if sampled == 0 || last.Before(oldestSeen) {
			oldestKey, oldestSeen = k, last
		}
		sampled++
		if sampled >= evictionSampleSize {
			break
		}
	}
	if sampled > 0 {
		delete(s.events, oldestKey)
	}
}

// transitionSessionCount returns the number of tracked sessions (tests only).
func transitionSessionCount() int {
	globalTransitionStore.mu.RLock()
	defer globalTransitionStore.mu.RUnlock()
	return len(globalTransitionStore.events)
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
