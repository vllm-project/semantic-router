package latency

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

// TransitionLog stores transition events by session ID.
type TransitionLog struct {
	mu     sync.RWMutex
	events map[string][]ModelTransitionEvent
}

var globalTransitionLog = &TransitionLog{
	events: make(map[string][]ModelTransitionEvent),
}

// RecordTransition records evt and drops the oldest entries after the per-session cap.
func RecordTransition(evt ModelTransitionEvent) {
	globalTransitionLog.mu.Lock()
	defer globalTransitionLog.mu.Unlock()
	list := globalTransitionLog.events[evt.SessionID]
	list = append(list, evt)
	if len(list) > MaxTransitionEventsPerSession {
		trimmed := make([]ModelTransitionEvent, MaxTransitionEventsPerSession)
		copy(trimmed, list[len(list)-MaxTransitionEventsPerSession:])
		list = trimmed
	}
	globalTransitionLog.events[evt.SessionID] = list
}

// GetTransitions returns a copy of the transition events for sessionID.
func GetTransitions(sessionID string) []ModelTransitionEvent {
	globalTransitionLog.mu.RLock()
	defer globalTransitionLog.mu.RUnlock()
	src := globalTransitionLog.events[sessionID]
	out := make([]ModelTransitionEvent, len(src))
	copy(out, src)
	return out
}

// ResetTransitionLog clears the transition log.
func ResetTransitionLog() {
	globalTransitionLog.mu.Lock()
	defer globalTransitionLog.mu.Unlock()
	globalTransitionLog.events = make(map[string][]ModelTransitionEvent)
}
