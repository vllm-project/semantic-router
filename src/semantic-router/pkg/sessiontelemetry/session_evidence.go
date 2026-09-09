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

// Recent-window bounds used until the gate configures a session.
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
// derived from Category, not caller-supplied. RequestID ties the response
// capture and the outcome ingest of the same turn together so the window can
// hold one fact per turn.
type TurnOutcome struct {
	RequestID         string              `json:"request_id,omitempty"`
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

// ConfigureTurnOutcomeWindow applies the active progress-gate window policy to
// one session. The gate calls this on the request path before reading, so both
// writers bound the window the way the operator configured it.
func ConfigureTurnOutcomeWindow(sessionID string, size int, ttl time.Duration, now time.Time) {
	if sessionID == "" {
		return
	}
	size, ttl = normalizeWindowPolicy(size, ttl)

	s := globalRouterSessionMemory
	s.mu.Lock()
	_, known := s.sessions[sessionID]
	if now.IsZero() {
		now = s.nowFn()
	}
	s.mu.Unlock()
	// Recover shared state first: creating an empty local session here would
	// otherwise shadow a window that only exists in the shared store.
	if !known {
		_, _ = loadSharedRouterSessionSnapshot(sessionID, now)
	}

	s.mu.Lock()
	st := s.sessionLocked(sessionID)
	st.outcomeWindowSize = size
	st.outcomeWindowTTL = ttl
	st.recentOutcomes = trimTurnOutcomes(pruneTurnOutcomes(st.recentOutcomes, ttl, now), size)
	if st.lastSeen.IsZero() {
		st.lastSeen = now
	}
	s.mu.Unlock()
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
	size, ttl := s.sessions[sessionID].windowPolicy()
	if s.nowFn().Sub(ts) > ttl {
		s.mu.Unlock()
		return
	}
	st := s.sessionLocked(sessionID)
	if st.lastSeen.IsZero() || ts.After(st.lastSeen) {
		st.lastSeen = ts
	}
	st.recentOutcomes = appendTurnOutcome(st.recentOutcomes, outcome, ts, size, ttl)
	s.mu.Unlock()

	persistRouterSessionState(sessionID)
}

// RecentTurnOutcomes returns the window under the packaged default policy, for
// callers that do not own a progress-gate config.
func RecentTurnOutcomes(sessionID string, now time.Time) []TurnOutcome {
	return RecentTurnOutcomesWithPolicy(sessionID, now, defaultRecentWindowSize, defaultRecentWindowTTL)
}

// RecentTurnOutcomesWithPolicy returns a pruned, cloned copy of the recent
// window ordered oldest → newest under the caller's policy; unknown or expired
// sessions return an empty window (cold start). A zero now falls back to the
// store clock.
func RecentTurnOutcomesWithPolicy(sessionID string, now time.Time, size int, ttl time.Duration) []TurnOutcome {
	if sessionID == "" {
		return nil
	}
	size, ttl = normalizeWindowPolicy(size, ttl)
	s := globalRouterSessionMemory
	s.mu.Lock()
	if now.IsZero() {
		now = s.nowFn()
	}
	st := s.sessions[sessionID]
	if st == nil {
		s.mu.Unlock()
		return sharedRecentTurnOutcomes(sessionID, now, size, ttl)
	}
	if now.Sub(st.lastSeen) > routerMemoryTTL {
		s.mu.Unlock()
		return nil
	}
	window := cloneTurnOutcomes(st.recentOutcomes)
	s.mu.Unlock()
	return trimTurnOutcomes(pruneTurnOutcomes(window, ttl, now), size)
}

// sharedRecentTurnOutcomes recovers the window from the shared store on a
// local miss.
func sharedRecentTurnOutcomes(sessionID string, now time.Time, size int, ttl time.Duration) []TurnOutcome {
	snapshot, ok := loadSharedRouterSessionSnapshot(sessionID, now)
	if !ok {
		return nil
	}
	return trimTurnOutcomes(pruneTurnOutcomes(snapshot.RecentOutcomes, ttl, now), size)
}

// windowPolicy returns the session's configured bounds, falling back to the
// package defaults for sessions the gate has not configured yet.
func (st *routerSessionState) windowPolicy() (int, time.Duration) {
	if st == nil {
		return defaultRecentWindowSize, defaultRecentWindowTTL
	}
	return normalizeWindowPolicy(st.outcomeWindowSize, st.outcomeWindowTTL)
}

func normalizeWindowPolicy(size int, ttl time.Duration) (int, time.Duration) {
	if size <= 0 {
		size = defaultRecentWindowSize
	}
	if ttl <= 0 {
		ttl = defaultRecentWindowTTL
	}
	return size, ttl
}

func trimTurnOutcomes(outcomes []TurnOutcome, size int) []TurnOutcome {
	if size <= 0 || len(outcomes) <= size {
		return outcomes
	}
	return outcomes[len(outcomes)-size:]
}

// appendTurnOutcome prunes by TTL, merges a second writer's view of the same
// turn, inserts by event time (capture and ingest are independent writers, so
// arrival order cannot be assumed), then trims to capacity. Callers must hold
// the store lock.
func appendTurnOutcome(outcomes []TurnOutcome, outcome TurnOutcome, now time.Time, size int, ttl time.Duration) []TurnOutcome {
	outcomes = pruneTurnOutcomes(outcomes, ttl, now)
	for i := range outcomes {
		// Only the two writers' views of one turn merge: response capture and
		// outcome ingest. Two captures are always distinct turns.
		if (outcome.Source == TurnSourceOutcomeIngest || outcomes[i].Source == TurnSourceOutcomeIngest) &&
			sameTurn(outcomes[i], outcome) {
			outcomes[i] = mergeTurnOutcome(outcomes[i], outcome)
			return outcomes
		}
	}
	// A full window cannot accept an outcome older than everything in it.
	if len(outcomes) >= size && outcome.Timestamp < outcomes[0].Timestamp {
		return outcomes
	}
	i := len(outcomes)
	for i > 0 && outcomes[i-1].Timestamp > outcome.Timestamp {
		i--
	}
	outcomes = append(outcomes, TurnOutcome{})
	copy(outcomes[i+1:], outcomes[i:])
	outcomes[i] = outcome
	return trimTurnOutcomes(outcomes, size)
}

// sameTurn reports whether two facts describe one turn: the same request when
// both carry a request ID, the same turn index and model when neither does.
// A mixed pair cannot be proven identical and stays separate.
func sameTurn(a, b TurnOutcome) bool {
	if a.RequestID != "" || b.RequestID != "" {
		return a.RequestID != "" && b.RequestID != "" && a.RequestID == b.RequestID
	}
	return a.TurnIndex == b.TurnIndex && a.Model == b.Model
}

// mergeTurnOutcome combines the two writers' views of one turn: the ingest
// verdict owns the semantic category, response capture owns the usage
// measurements, and the first writer's event time keeps the window ordered.
func mergeTurnOutcome(existing, incoming TurnOutcome) TurnOutcome {
	merged := existing
	switch incoming.Source {
	case TurnSourceOutcomeIngest:
		merged.Category = incoming.Category
		merged.Confidence = incoming.Confidence
		merged.Source = TurnSourceOutcomeIngest
	case TurnSourceRouterObserved:
		merged.OutputTokens = incoming.OutputTokens
		merged.LatencyMs = incoming.LatencyMs
		if existing.Source != TurnSourceOutcomeIngest {
			merged.Category = incoming.Category
			merged.Source = TurnSourceRouterObserved
		}
	}
	if merged.RequestID == "" {
		merged.RequestID = incoming.RequestID
	}
	merged.ModelAttributable = categoryAttributable(merged.Category)
	return merged
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
