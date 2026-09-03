package sessiontelemetry

import "time"

// TurnOutcomeCategory classifies one observed session-turn outcome for the
// evidence-calibrated switch gate (PL-0041 / issue #3377). Categories carry
// failure attribution: provider and tool failures are infrastructure noise
// and must never be counted as model regressions.
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

// Recent-window bounds for the evidence-calibrated switch gate. These are
// package defaults; PL-0041 TASK-07 will wire them to progress_gate config.
const (
	defaultRecentWindowSize = 8
	defaultRecentWindowTTL  = 15 * time.Minute
)

// categoryAttributable is the single source of truth for failure attribution:
// only categories that describe the model's own generated output count as model
// evidence. Provider, tool, and missing outcomes are environment noise and must
// never be read as model regressions.
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

// TurnOutcome is a content-minimal, typed fact about one session turn.
// It stores only enums and scalars — never prompt or response text — so the
// window can gate switches without retaining private conversation content.
//
// ModelAttributable is always derived from Category by RecordTurnOutcome;
// callers cannot set an attribution that contradicts the category.
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
// recent window. It leaves turn/switch/cost counters alone — those stay owned
// by RecordSessionDecision/RecordSessionUsage.
//

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

// RecentTurnOutcomes returns a pruned, cloned copy of the session's recent
// outcome window ordered oldest → newest. Unknown, expired, or empty sessions
// return an empty window, which the gate reads as cold start.
//
// This reads the window directly instead of going through
// GetRouterSessionSnapshot: the gate runs on every turn and does not need the
// full snapshot clone (counters, model turns, policy map). A zero `now` falls
// back to the store clock so TTL pruning can never be silently skipped.
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

// sharedRecentTurnOutcomes recovers a window from the shared store on a local
// miss. It reuses the snapshot loader so hydration and TTL semantics stay in
// one place, and stays fail-open when no shared store is configured.
func sharedRecentTurnOutcomes(sessionID string, now time.Time) []TurnOutcome {
	snapshot, ok := loadSharedRouterSessionSnapshot(sessionID, now)
	if !ok {
		return nil
	}
	return pruneTurnOutcomes(snapshot.RecentOutcomes, defaultRecentWindowTTL, now)
}

// appendTurnOutcome applies both window bounds in order: TTL prune, append,
// capacity trim. Callers must hold the store lock.
func appendTurnOutcome(outcomes []TurnOutcome, outcome TurnOutcome, now time.Time) []TurnOutcome {
	outcomes = pruneTurnOutcomes(outcomes, defaultRecentWindowTTL, now)
	outcomes = append(outcomes, outcome)
	if len(outcomes) > defaultRecentWindowSize {
		outcomes = outcomes[len(outcomes)-defaultRecentWindowSize:]
	}
	return outcomes
}

// pruneTurnOutcomes drops entries older than ttl. Entries without a usable
// timestamp are kept (defensive: capture always sets one). A zero `now` is
// treated as "no reference clock available" and returns an empty window rather
// than an unpruned one: callers must never see stale evidence as fresh.
// The input slice is assumed to be owned by the caller.
func pruneTurnOutcomes(outcomes []TurnOutcome, ttl time.Duration, now time.Time) []TurnOutcome {
	if len(outcomes) == 0 || ttl <= 0 {
		return outcomes
	}
	if now.IsZero() {
		return nil
	}
	cutoff := now.Add(-ttl)
	keep := 0
	for keep < len(outcomes) {
		ts := outcomes[keep].Time()
		if !ts.IsZero() && ts.Before(cutoff) {
			keep++
			continue
		}
		break
	}
	if keep == 0 {
		return outcomes
	}
	return outcomes[keep:]
}

// cloneTurnOutcomes returns a deep copy so snapshot readers cannot mutate
// store-internal state.
func cloneTurnOutcomes(in []TurnOutcome) []TurnOutcome {
	if len(in) == 0 {
		return nil
	}
	out := make([]TurnOutcome, len(in))
	copy(out, in)
	return out
}
