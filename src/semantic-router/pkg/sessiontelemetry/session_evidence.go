package sessiontelemetry

import "time"

// TurnOutcomeCategory classifies a session-turn outcome. Provider and tool
// failures are environment noise, never model regressions.
type TurnOutcomeCategory string

const (
	TurnProgress      TurnOutcomeCategory = "progress"
	TurnNoProgress    TurnOutcomeCategory = "no_progress"
	TurnRegression    TurnOutcomeCategory = "regression"
	TurnProviderError TurnOutcomeCategory = "provider_error"
	TurnToolError     TurnOutcomeCategory = "tool_error"
	TurnMissing       TurnOutcomeCategory = "missing"
)

// Turn outcome provenance.
const (
	TurnSourceRouterObserved = "router_observed" // derived from the response path
	TurnSourceOutcomeIngest  = "outcome_ingest"  // derived from the learning outcome ingest
)

// Recent-window package defaults; PL-0041 TASK-07 wires these to config.
const (
	defaultRecentWindowSize = 8
	defaultRecentWindowTTL  = 15 * time.Minute
)

// categoryAttributable is the single source of truth for failure attribution.
func categoryAttributable(category TurnOutcomeCategory) bool {
	switch category {
	case TurnProgress, TurnNoProgress, TurnRegression:
		return true
	case TurnProviderError, TurnToolError, TurnMissing:
		return false
	default:
		return false
	}
}

// TurnOutcome is a content-minimal typed fact about one session turn: enums
// and scalars only, never prompt or response text. ModelAttributable is
// derived from Category, not caller-supplied.
type TurnOutcome struct {
	TurnIndex         int                 `json:"turn_index"`
	Timestamp         int64               `json:"timestamp_unix_ms"` // unix milliseconds
	Model             string              `json:"model"`
	Category          TurnOutcomeCategory `json:"category"`
	ModelAttributable bool                `json:"model_attributable"`
	Confidence        float64             `json:"confidence,omitempty"`
	OutputTokens      int64               `json:"output_tokens,omitempty"`
	LatencyMs         int64               `json:"latency_ms,omitempty"`
	Source            string              `json:"source,omitempty"`
}

// Time returns the outcome timestamp as time.Time (zero when unset).
func (o TurnOutcome) Time() time.Time {
	if o.Timestamp <= 0 {
		return time.Time{}
	}
	return time.UnixMilli(o.Timestamp)
}

// RecordTurnOutcome appends one typed turn outcome to the session's bounded
// recent window. Callers pass the event time (when the outcome occurred);
// outcome ingest may deliver an older event after newer ones arrived, so
// insertion is by event time. Outcomes older than the window TTL are rejected.
func RecordTurnOutcome(sessionID string, outcome TurnOutcome, timestamp time.Time) {
	if sessionID == "" {
		return
	}
	ts := timestamp
	if ts.IsZero() {
		ts = outcome.Time()
	}
	if ts.IsZero() {
		return
	}
	if outcome.Category == "" {
		outcome.Category = TurnMissing
	}
	outcome.ModelAttributable = categoryAttributable(outcome.Category)
	outcome.Timestamp = ts.UnixMilli()

	s := globalRouterSessionMemory
	s.mu.Lock()
	if s.nowFn().Sub(ts) > defaultRecentWindowTTL {
		s.mu.Unlock()
		return
	}
	st := s.sessionLocked(sessionID)
	if st.lastSeen.IsZero() || ts.After(st.lastSeen) {
		st.lastSeen = ts
	}
	st.recentOutcomes = appendTurnOutcome(st.recentOutcomes, outcome, ts)
	s.mu.Unlock()

	persistRouterSessionState(sessionID)
}

// RecentTurnOutcomes returns a pruned, cloned copy of the recent window ordered
// oldest → newest; unknown or expired sessions return an empty window (cold
// start). A zero now falls back to the store clock.
func RecentTurnOutcomes(sessionID string, now time.Time) []TurnOutcome {
	if sessionID == "" {
		return nil
	}
	s := globalRouterSessionMemory
	s.mu.Lock()
	if now.IsZero() {
		now = s.nowFn()
	}
	st := s.sessions[sessionID]
	if st == nil {
		s.mu.Unlock()
		return sharedRecentTurnOutcomes(sessionID, now)
	}
	if now.Sub(st.lastSeen) > routerMemoryTTL {
		s.mu.Unlock()
		return nil
	}
	window := cloneTurnOutcomes(st.recentOutcomes)
	s.mu.Unlock()
	return pruneTurnOutcomes(window, defaultRecentWindowTTL, now)
}

// sharedRecentTurnOutcomes recovers the window from the shared store on a
// local miss.
func sharedRecentTurnOutcomes(sessionID string, now time.Time) []TurnOutcome {
	snapshot, ok := loadSharedRouterSessionSnapshot(sessionID, now)
	if !ok {
		return nil
	}
	return pruneTurnOutcomes(snapshot.RecentOutcomes, defaultRecentWindowTTL, now)
}

// appendTurnOutcome prunes by TTL, inserts by event time (capture and ingest
// are independent writers, so arrival order cannot be assumed), then trims to
// capacity. Callers must hold the store lock.
func appendTurnOutcome(outcomes []TurnOutcome, outcome TurnOutcome, now time.Time) []TurnOutcome {
	outcomes = pruneTurnOutcomes(outcomes, defaultRecentWindowTTL, now)
	// A full window cannot accept an outcome older than everything in it.
	if len(outcomes) >= defaultRecentWindowSize && outcome.Timestamp < outcomes[0].Timestamp {
		return outcomes
	}
	i := len(outcomes)
	for i > 0 && outcomes[i-1].Timestamp > outcome.Timestamp {
		i--
	}
	outcomes = append(outcomes, TurnOutcome{})
	copy(outcomes[i+1:], outcomes[i:])
	outcomes[i] = outcome
	if len(outcomes) > defaultRecentWindowSize {
		outcomes = outcomes[len(outcomes)-defaultRecentWindowSize:]
	}
	return outcomes
}

// pruneTurnOutcomes returns the entries newer than ttl, without assuming
// ordering and without mutating the input. A zero now returns an empty window
// rather than stale evidence; entries without a usable timestamp are kept
// defensively.
func pruneTurnOutcomes(outcomes []TurnOutcome, ttl time.Duration, now time.Time) []TurnOutcome {
	if len(outcomes) == 0 || ttl <= 0 {
		return outcomes
	}
	if now.IsZero() {
		return nil
	}
	cutoff := now.Add(-ttl)
	kept := make([]TurnOutcome, 0, len(outcomes))
	for _, o := range outcomes {
		ts := o.Time()
		if ts.IsZero() || !ts.Before(cutoff) {
			kept = append(kept, o)
		}
	}
	return kept
}

// cloneTurnOutcomes returns a deep copy so readers cannot mutate store state.
func cloneTurnOutcomes(in []TurnOutcome) []TurnOutcome {
	if len(in) == 0 {
		return nil
	}
	out := make([]TurnOutcome, len(in))
	copy(out, in)
	return out
}
