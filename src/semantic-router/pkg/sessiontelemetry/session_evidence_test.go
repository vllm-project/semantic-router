package sessiontelemetry

import (
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"
)

// The relative timelines stay in the past; future evidence is rejected.
var fixedBase = time.Now().Add(-12 * time.Minute).Truncate(time.Minute)

func outcomeAt(base time.Time, offsetMinutes int, turn int, category TurnOutcomeCategory, attributable bool) TurnOutcome {
	return TurnOutcome{
		TurnIndex:         turn,
		Timestamp:         base.Add(time.Duration(offsetMinutes) * time.Minute).UnixMilli(),
		Model:             "model-a",
		Category:          category,
		ModelAttributable: attributable,
		Confidence:        0.5,
		OutputTokens:      128,
		LatencyMs:         900,
		Source:            TurnSourceRouterObserved,
	}
}

func mustWindow(t *testing.T, sessionID string, now time.Time) []TurnOutcome {
	t.Helper()
	window := RecentTurnOutcomes(sessionID, now)
	if window == nil {
		window = []TurnOutcome{}
	}
	return window
}

// 1. Capacity boundary: writing 9 outcomes keeps the newest 8 in order.
func TestRecordTurnOutcomeWindowCapacity(t *testing.T) {
	ResetRouterSessionMemoryForTesting()
	const sessionID = "cap-window"

	for i := 0; i < 9; i++ {
		outcome := outcomeAt(fixedBase, i, i, TurnNoProgress, true)
		RecordTurnOutcome(sessionID, outcome, fixedBase.Add(time.Duration(i)*time.Minute))
	}

	window := mustWindow(t, sessionID, fixedBase.Add(9*time.Minute))
	if len(window) != defaultRecentWindowSize {
		t.Fatalf("window size = %d, want %d", len(window), defaultRecentWindowSize)
	}
	for i, o := range window {
		wantTurn := i + 1 // oldest evicted was turn 0
		if o.TurnIndex != wantTurn {
			t.Fatalf("window[%d].TurnIndex = %d, want %d", i, o.TurnIndex, wantTurn)
		}
	}
}

// 2a. TTL pruning: stale entries disappear, fresh ones survive.
func TestRecentTurnOutcomesTTLPruning(t *testing.T) {
	ResetRouterSessionMemoryForTesting()
	const sessionID = "ttl-prune"

	// t=0 stale by read time, t=10m fresh (TTL=15m).
	RecordTurnOutcome(sessionID, outcomeAt(fixedBase, 0, 1, TurnNoProgress, true), fixedBase)
	RecordTurnOutcome(sessionID, outcomeAt(fixedBase, 10, 2, TurnNoProgress, true), fixedBase.Add(10*time.Minute))

	window := mustWindow(t, sessionID, fixedBase.Add(20*time.Minute))
	if len(window) != 1 || window[0].TurnIndex != 2 {
		t.Fatalf("window = %+v, want only turn 2", window)
	}
}

// 2b. Fully expired window reads as cold start.
func TestRecentTurnOutcomesFullyExpiredColdStart(t *testing.T) {
	ResetRouterSessionMemoryForTesting()
	const sessionID = "ttl-cold"

	RecordTurnOutcome(sessionID, outcomeAt(fixedBase, 0, 1, TurnRegression, true), fixedBase)

	window := RecentTurnOutcomes(sessionID, fixedBase.Add(defaultRecentWindowTTL+time.Minute))
	if len(window) != 0 {
		t.Fatalf("window = %+v, want empty (cold start)", window)
	}
}

// 3. Unknown session is a cold start, not an error.
func TestRecentTurnOutcomesColdStartUnknownSession(t *testing.T) {
	ResetRouterSessionMemoryForTesting()
	if window := RecentTurnOutcomes("no-such-session", fixedBase); len(window) != 0 {
		t.Fatalf("window = %+v, want empty", window)
	}
	if window := RecentTurnOutcomes("", fixedBase); len(window) != 0 {
		t.Fatalf("empty session id window = %+v, want empty", window)
	}
}

// 4. Combined bounds: TTL prune runs before capacity trim on every write.
func TestAppendTurnOutcomeCombinedBounds(t *testing.T) {
	base := fixedBase
	outcomes := []TurnOutcome{
		outcomeAt(base, 0, 1, TurnNoProgress, true),  // stale by t=20m
		outcomeAt(base, 4, 2, TurnNoProgress, true),  // stale by t=20m
		outcomeAt(base, 12, 3, TurnNoProgress, true), // fresh
	}
	outcomes = appendTurnOutcome(outcomes, outcomeAt(base, 20, 4, TurnNoProgress, true), base.Add(20*time.Minute),
		defaultRecentWindowSize, defaultRecentWindowTTL)

	if len(outcomes) != 2 {
		t.Fatalf("len = %d, want 2 (stale pruned, then appended)", len(outcomes))
	}
	if outcomes[0].TurnIndex != 3 || outcomes[1].TurnIndex != 4 {
		t.Fatalf("order = %d,%d, want 3,4", outcomes[0].TurnIndex, outcomes[1].TurnIndex)
	}
}

// 5. Concurrent writers on one session stay race-free and bounded.
func TestRecordTurnOutcomeConcurrent(t *testing.T) {
	ResetRouterSessionMemoryForTesting()
	const sessionID = "concurrent"
	const writers = 16

	var wg sync.WaitGroup
	for w := 0; w < writers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for i := 0; i < 5; i++ {
				ts := fixedBase.Add(time.Duration(w*5+i) * time.Second)
				RecordTurnOutcome(sessionID, outcomeAt(fixedBase, 0, w*5+i, TurnProgress, true), ts)
			}
		}(w)
	}
	wg.Wait()

	window := mustWindow(t, sessionID, fixedBase.Add(2*time.Minute))
	if len(window) != defaultRecentWindowSize {
		t.Fatalf("window size = %d, want %d", len(window), defaultRecentWindowSize)
	}
}

// 6. Persistence round-trip: outcome window survives save/load through a
// shared store and the hydrate path.
type fakeSessionStateStore struct {
	mu       sync.Mutex
	payloads map[string][]byte
}

func newFakeSessionStateStore() *fakeSessionStateStore {
	return &fakeSessionStateStore{payloads: map[string][]byte{}}
}

func (f *fakeSessionStateStore) Load(sessionID string) (RouterSessionSnapshot, bool, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	payload, ok := f.payloads[sessionID]
	if !ok {
		return RouterSessionSnapshot{}, false, nil
	}
	var snapshot RouterSessionSnapshot
	if err := json.Unmarshal(payload, &snapshot); err != nil {
		return RouterSessionSnapshot{}, false, err
	}
	return snapshot, true, nil
}

func (f *fakeSessionStateStore) Save(snapshot RouterSessionSnapshot, ttl time.Duration) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	payload, err := json.Marshal(snapshot)
	if err != nil {
		return err
	}
	f.payloads[snapshot.SessionID] = payload
	return nil
}

func (f *fakeSessionStateStore) Close() error { return nil }

func TestRecentOutcomesPersistRoundTrip(t *testing.T) {
	ResetRouterSessionMemoryForTesting()
	store := newFakeSessionStateStore()
	SetRouterSessionStateStore(store)
	defer SetRouterSessionStateStore(nil)

	const sessionID = "round-trip"
	RecordTurnOutcome(sessionID, outcomeAt(fixedBase, 0, 1, TurnNoProgress, true), fixedBase)
	RecordTurnOutcome(sessionID, outcomeAt(fixedBase, 1, 2, TurnRegression, true), fixedBase.Add(time.Minute))

	// Drop the in-memory copy and read back through the shared store.
	ResetRouterSessionMemoryForTesting()
	window := mustWindow(t, sessionID, fixedBase.Add(2*time.Minute))
	if len(window) != 2 || window[0].TurnIndex != 1 || window[1].TurnIndex != 2 {
		t.Fatalf("window after round-trip = %+v", window)
	}
	if !window[1].ModelAttributable || window[1].Category != TurnRegression {
		t.Fatalf("round-trip lost attribution facts: %+v", window[1])
	}
}

// 7. Legacy shared-store payloads without the window field hydrate cleanly.
func TestLegacySnapshotWithoutWindowField(t *testing.T) {
	legacy := `{"session_id":"legacy","current_model":"model-a","last_seen":"2026-09-03T12:00:00Z"}`
	var snapshot RouterSessionSnapshot
	if err := json.Unmarshal([]byte(legacy), &snapshot); err != nil {
		t.Fatalf("unmarshal legacy payload: %v", err)
	}
	if len(snapshot.RecentOutcomes) != 0 {
		t.Fatalf("legacy window = %+v, want empty", snapshot.RecentOutcomes)
	}
}

// 8. Content minimization: the persisted fact carries no free-text fields.
func TestTurnOutcomeContentMinimization(t *testing.T) {
	payload, err := json.Marshal(outcomeAt(fixedBase, 0, 1, TurnNoProgress, true))
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	encoded := strings.ToLower(string(payload))
	for _, forbidden := range []string{"prompt", "content", "message", "text", "response_body"} {
		if strings.Contains(encoded, forbidden) {
			t.Fatalf("marshal contains forbidden free-text key %q: %s", forbidden, encoded)
		}
	}
}

// 9. Decision/usage bookkeeping and the outcome window do not clobber
// each other.
func TestRecordSessionDecisionPreservesWindow(t *testing.T) {
	ResetRouterSessionMemoryForTesting()
	const sessionID = "both-writers"

	RecordTurnOutcome(sessionID, outcomeAt(fixedBase, 0, 1, TurnNoProgress, true), fixedBase)
	RecordSessionDecision(SessionDecisionParams{
		SessionID:     sessionID,
		PreviousModel: "model-a",
		SelectedModel: "model-a",
		TurnIndex:     1,
		Timestamp:     fixedBase.Add(time.Minute),
	})

	snapshot, ok := GetRouterSessionSnapshot(sessionID, fixedBase.Add(2*time.Minute))
	if !ok {
		t.Fatalf("snapshot missing")
	}
	if snapshot.CurrentModel != "model-a" || snapshot.TurnCount != 2 {
		t.Fatalf("decision bookkeeping changed: %+v", snapshot)
	}
	if len(snapshot.RecentOutcomes) != 1 || snapshot.RecentOutcomes[0].TurnIndex != 1 {
		t.Fatalf("decision clobbered the window: %+v", snapshot.RecentOutcomes)
	}
}

// 10. Capture-bug degradation: empty category and missing timestamps.
func TestRecordTurnOutcomeNormalization(t *testing.T) {
	ResetRouterSessionMemoryForTesting()
	const sessionID = "normalize"

	// No timestamp anywhere: skipped entirely.
	RecordTurnOutcome(sessionID, TurnOutcome{TurnIndex: 0, Category: TurnProgress}, time.Time{})
	if window := RecentTurnOutcomes(sessionID, fixedBase); len(window) != 0 {
		t.Fatalf("timestamp-less record kept: %+v", window)
	}

	// Empty category degrades to missing, attribution cleared.
	RecordTurnOutcome(sessionID, TurnOutcome{TurnIndex: 1, Model: "model-a"}, fixedBase)
	window := mustWindow(t, sessionID, fixedBase.Add(time.Minute))
	if len(window) != 1 {
		t.Fatalf("window = %+v", window)
	}
	if window[0].Category != TurnMissing || window[0].ModelAttributable {
		t.Fatalf("empty category not normalized: %+v", window[0])
	}
}

// 12. Outcomes older than the window TTL can never become evidence, so they are
// rejected at the door instead of creating session state nobody can use.
func TestRecordTurnOutcomeRejectsStaleOutcome(t *testing.T) {
	ResetRouterSessionMemoryForTesting()
	store := newFakeSessionStateStore()
	SetRouterSessionStateStore(store)
	defer SetRouterSessionStateStore(nil)

	stale := time.Now().Add(-defaultRecentWindowTTL - time.Minute)
	const sessionID = "stale"

	RecordTurnOutcome(sessionID, TurnOutcome{
		TurnIndex: 1,
		Model:     "model-a",
		Category:  TurnNoProgress,
	}, stale)

	if window := RecentTurnOutcomes(sessionID, stale.Add(time.Second)); len(window) != 0 {
		t.Fatalf("stale outcome was recorded: %+v", window)
	}
	if _, ok := store.payloads[sessionID]; ok {
		t.Fatalf("stale outcome created a persisted session")
	}
	if _, ok := GetRouterSessionSnapshot(sessionID, time.Now()); ok {
		t.Fatalf("stale outcome created a live session")
	}
}

// 12b. An outcome that trails wall clock but is still inside the window TTL is
// accepted, and later ages out of the window normally.
func TestRecordTurnOutcomeAcceptsRecentTrailingOutcome(t *testing.T) {
	ResetRouterSessionMemoryForTesting()
	const sessionID = "trailing"

	trailing := time.Now().Add(-time.Minute)
	RecordTurnOutcome(sessionID, TurnOutcome{
		TurnIndex: 1,
		Model:     "model-a",
		Category:  TurnRegression,
	}, trailing)

	window := RecentTurnOutcomes(sessionID, trailing.Add(time.Minute))
	if len(window) != 1 || window[0].TurnIndex != 1 {
		t.Fatalf("trailing outcome rejected: %+v", window)
	}
	if aged := RecentTurnOutcomes(sessionID, trailing.Add(defaultRecentWindowTTL+time.Minute)); len(aged) != 0 {
		t.Fatalf("window = %+v, want empty once past the window TTL", aged)
	}
}

// 13. Attribution is derived from the category and cannot be contradicted by
// the caller.
func TestRecordTurnOutcomeAttributionIsDerived(t *testing.T) {
	cases := []struct {
		category TurnOutcomeCategory
		supplied bool
		want     bool
	}{
		{TurnProgress, false, true},
		{TurnNoProgress, false, true},
		{TurnRegression, false, true},
		{TurnProviderError, true, false}, // caller lies: says attributable
		{TurnToolError, true, false},     // caller lies: says attributable
		{TurnMissing, true, false},
	}

	for _, tc := range cases {
		ResetRouterSessionMemoryForTesting()
		sessionID := "derive-" + string(tc.category)
		RecordTurnOutcome(sessionID, TurnOutcome{
			TurnIndex:         1,
			Model:             "model-a",
			Category:          tc.category,
			ModelAttributable: tc.supplied,
		}, fixedBase)

		window := mustWindow(t, sessionID, fixedBase.Add(time.Minute))
		if len(window) != 1 {
			t.Fatalf("%s: window = %+v", tc.category, window)
		}
		if window[0].ModelAttributable != tc.want {
			t.Fatalf("%s: attributable = %v, want %v",
				tc.category, window[0].ModelAttributable, tc.want)
		}
	}
}

// 14. A zero reference clock must never surface stale evidence as fresh.
func TestPruneTurnOutcomesZeroClockIsConservative(t *testing.T) {
	outcomes := []TurnOutcome{outcomeAt(fixedBase, 0, 1, TurnNoProgress, true)}
	if got := pruneTurnOutcomes(outcomes, defaultRecentWindowTTL, time.Time{}); len(got) != 0 {
		t.Fatalf("zero clock returned %d entries, want 0", len(got))
	}
}

// 15. A zero `now` on the read path falls back to the store clock instead of
// skipping TTL pruning.
func TestRecentTurnOutcomesZeroNowUsesStoreClock(t *testing.T) {
	ResetRouterSessionMemoryForTesting()
	const sessionID = "zero-now"

	RecordTurnOutcome(sessionID, outcomeAt(fixedBase, 0, 1, TurnNoProgress, true), fixedBase)
	setRouterSessionMemoryNowForTesting(func() time.Time {
		return fixedBase.Add(defaultRecentWindowTTL + time.Minute)
	})
	defer setRouterSessionMemoryNowForTesting(nil)

	if window := RecentTurnOutcomes(sessionID, time.Time{}); len(window) != 0 {
		t.Fatalf("window = %+v, want empty (store clock says expired)", window)
	}
}

// 16. Window recovery from the shared store does not need the caller to read a
// full snapshot first.
func TestRecentTurnOutcomesRecoversFromSharedStore(t *testing.T) {
	ResetRouterSessionMemoryForTesting()
	store := newFakeSessionStateStore()
	SetRouterSessionStateStore(store)
	defer SetRouterSessionStateStore(nil)

	const sessionID = "shared-recover"
	RecordTurnOutcome(sessionID, outcomeAt(fixedBase, 0, 1, TurnRegression, true), fixedBase)

	ResetRouterSessionMemoryForTesting() // local miss forces shared-store path
	window := RecentTurnOutcomes(sessionID, fixedBase.Add(time.Minute))
	if len(window) != 1 || window[0].Category != TurnRegression {
		t.Fatalf("shared-store recovery failed: %+v", window)
	}
}

// 11b. Delayed arrival regression: an older verdict arriving after a newer one
// (the outcome-ingest writer) must land at its event-time position and still
// expire — it can neither corrupt the timeline nor survive behind a newer
// entry past the window TTL.
func TestRecordTurnOutcomeDelayedArrivalStaysOrderedAndExpires(t *testing.T) {
	ResetRouterSessionMemoryForTesting()
	const sessionID = "delayed-arrival"

	// The newer turn completes and is captured first.
	RecordTurnOutcome(sessionID, outcomeAt(fixedBase, 10, 2, TurnProgress, true), fixedBase.Add(10*time.Minute))
	// The older verdict arrives later through the ingest path, carrying its
	// original event time.
	RecordTurnOutcome(sessionID, outcomeAt(fixedBase, 2, 1, TurnNoProgress, true), fixedBase.Add(2*time.Minute))

	// Ordered by event time despite arrival order: turn 1 before turn 2.
	window := mustWindow(t, sessionID, fixedBase.Add(12*time.Minute))
	if len(window) != 2 || window[0].TurnIndex != 1 || window[1].TurnIndex != 2 {
		t.Fatalf("window not ordered by event time: %+v", window)
	}

	// At read time +18min the delayed turn 1 is 16 minutes old (expired) while
	// turn 2 is only 8 minutes old: the delayed entry must be gone.
	window = mustWindow(t, sessionID, fixedBase.Add(18*time.Minute))
	if len(window) != 1 || window[0].TurnIndex != 2 {
		t.Fatalf("expired delayed outcome survived: %+v", window)
	}
}

// 12. The two writers' views of one turn merge into a single fact: the ingest
// verdict owns the category, response capture owns the usage measurements.
func TestRecordTurnOutcomeMergesSameTurn(t *testing.T) {
	for _, tc := range []struct {
		name    string
		capture bool
	}{
		{"capture_then_ingest", false},
		{"ingest_then_capture", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ResetRouterSessionMemoryForTesting()
			const sessionID = "merge-turn"
			capture := TurnOutcome{
				RequestID:    "req-1",
				TurnIndex:    5,
				Timestamp:    fixedBase.UnixMilli(),
				Model:        "model-a",
				Category:     TurnProgress,
				OutputTokens: 42,
				LatencyMs:    700,
				Source:       TurnSourceRouterObserved,
			}
			ingest := TurnOutcome{
				RequestID:  "req-1",
				TurnIndex:  5,
				Timestamp:  fixedBase.Add(time.Second).UnixMilli(),
				Model:      "model-a",
				Category:   TurnRegression,
				Confidence: 0.9,
				Source:     TurnSourceOutcomeIngest,
			}
			if tc.capture {
				RecordTurnOutcome(sessionID, ingest, fixedBase.Add(time.Second))
				RecordTurnOutcome(sessionID, capture, fixedBase)
			} else {
				RecordTurnOutcome(sessionID, capture, fixedBase)
				RecordTurnOutcome(sessionID, ingest, fixedBase.Add(time.Second))
			}

			window := RecentTurnOutcomes(sessionID, fixedBase.Add(2*time.Second))
			if len(window) != 1 {
				t.Fatalf("same turn produced %d facts: %+v", len(window), window)
			}
			got := window[0]
			if got.Category != TurnRegression || got.Source != TurnSourceOutcomeIngest || got.Confidence != 0.9 {
				t.Fatalf("ingest verdict must own the semantics: %+v", got)
			}
			if got.OutputTokens != 42 || got.LatencyMs != 700 || got.RequestID != "req-1" {
				t.Fatalf("capture measurements must survive the merge: %+v", got)
			}
			if !got.ModelAttributable {
				t.Fatalf("merged fact must be attributable via the ingest category: %+v", got)
			}
		})
	}
}

// 13. Distinct turns never merge, even from the same model and writer.
func TestRecordTurnOutcomeKeepsDistinctTurns(t *testing.T) {
	ResetRouterSessionMemoryForTesting()
	const sessionID = "distinct-turns"

	for i := 0; i < 3; i++ {
		RecordTurnOutcome(sessionID, TurnOutcome{
			RequestID: fmt.Sprintf("req-%d", i),
			TurnIndex: 0, // stateless clients repeat the index every turn
			Model:     "model-a",
			Category:  TurnNoProgress,
			Source:    TurnSourceRouterObserved,
		}, fixedBase.Add(time.Duration(i)*time.Second))
	}

	window := RecentTurnOutcomes(sessionID, fixedBase.Add(10*time.Second))
	if len(window) != 3 {
		t.Fatalf("distinct requests must stay separate facts: %+v", window)
	}
}

// 11. A configured window policy bounds both writes and reads.
func TestConfiguredWindowPolicyBoundsWindow(t *testing.T) {
	ResetRouterSessionMemoryForTesting()
	const sessionID = "configured-window"
	setRouterSessionMemoryNowForTesting(func() time.Time { return fixedBase.Add(3 * time.Minute) })
	defer setRouterSessionMemoryNowForTesting(nil)

	// Capacity 2 must evict on write even though the package default is 8.
	ConfigureTurnOutcomeWindow(sessionID, 2, 5*time.Minute, fixedBase)
	for i := 0; i < 3; i++ {
		RecordTurnOutcome(sessionID, outcomeAt(fixedBase, i, i, TurnNoProgress, true),
			fixedBase.Add(time.Duration(i)*time.Minute))
	}

	window := RecentTurnOutcomesWithPolicy(sessionID, fixedBase.Add(3*time.Minute), 2, 5*time.Minute)
	if len(window) != 2 || window[0].TurnIndex != 1 || window[1].TurnIndex != 2 {
		t.Fatalf("configured capacity not applied: %+v", window)
	}

	// A TTL shorter than the package default must expire evidence early:
	// at t+2m only the outcome recorded at t+2m is within 45s.
	if aged := RecentTurnOutcomesWithPolicy(sessionID, fixedBase.Add(2*time.Minute), 2, 45*time.Second); len(aged) != 1 {
		t.Fatalf("configured read TTL not applied: %+v", aged)
	}
}

// 12. Reads return clones: mutating a read must not leak into the store.
func TestRecentTurnOutcomesReturnClones(t *testing.T) {
	ResetRouterSessionMemoryForTesting()
	const sessionID = "clone"

	RecordTurnOutcome(sessionID, outcomeAt(fixedBase, 0, 1, TurnNoProgress, true), fixedBase)

	window := mustWindow(t, sessionID, fixedBase.Add(time.Minute))
	window[0].Category = TurnProgress
	window[0].ModelAttributable = false

	again := mustWindow(t, sessionID, fixedBase.Add(2*time.Minute))
	if again[0].Category != TurnNoProgress || !again[0].ModelAttributable {
		t.Fatalf("mutation leaked into store: %+v", again[0])
	}
}
