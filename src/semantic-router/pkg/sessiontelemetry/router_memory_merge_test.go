package sessiontelemetry

import (
	"encoding/json"
	"fmt"
	"sync"
	"testing"
	"time"
)

// casSessionStateStore models one shared slot written by several router
// replicas. Merge performs a real read-modify-write under the lock, which is
// the contract the Redis store gets from WATCH/MULTI.
type casSessionStateStore struct {
	mu     sync.Mutex
	stored map[string][]byte
	merges int
	saves  int
}

func newCASSessionStateStore() *casSessionStateStore {
	return &casSessionStateStore{stored: map[string][]byte{}}
}

func (s *casSessionStateStore) Load(sessionID string) (RouterSessionSnapshot, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	payload, ok := s.stored[sessionID]
	if !ok {
		return RouterSessionSnapshot{}, false, nil
	}
	var snapshot RouterSessionSnapshot
	if err := json.Unmarshal(payload, &snapshot); err != nil {
		return RouterSessionSnapshot{}, false, err
	}
	return snapshot, true, nil
}

func (s *casSessionStateStore) Save(snapshot RouterSessionSnapshot, _ time.Duration) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	payload, err := json.Marshal(snapshot)
	if err != nil {
		return err
	}
	s.stored[snapshot.SessionID] = payload
	s.saves++
	return nil
}

func (s *casSessionStateStore) Merge(local RouterSessionSnapshot, _ time.Duration) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	merged := local
	if payload, ok := s.stored[local.SessionID]; ok {
		var remote RouterSessionSnapshot
		if err := json.Unmarshal(payload, &remote); err != nil {
			return err
		}
		merged = mergeRouterSessionSnapshots(remote, local)
	}
	payload, err := json.Marshal(merged)
	if err != nil {
		return err
	}
	s.stored[local.SessionID] = payload
	s.merges++
	return nil
}

func (s *casSessionStateStore) Close() error { return nil }

func (s *casSessionStateStore) snapshot(t *testing.T, sessionID string) RouterSessionSnapshot {
	t.Helper()
	snapshot, ok, err := s.Load(sessionID)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if !ok {
		t.Fatalf("session %q was never persisted", sessionID)
	}
	return snapshot
}

func mergeTestOutcome(requestID string, at time.Time) TurnOutcome {
	return TurnOutcome{
		RequestID:         requestID,
		Model:             "cheap",
		Category:          TurnRegression,
		ModelAttributable: true,
		Timestamp:         at.UnixMilli(),
	}
}

// TestConcurrentReplicaPersistenceKeepsEveryFact is the multi-writer
// regression: replicas that loaded the same session each append their own
// outcome and switch timestamp, then persist. A whole-snapshot save would leave
// only the last writer's facts.
func TestConcurrentReplicaPersistenceKeepsEveryFact(t *testing.T) {
	const (
		sessionID = "shared-replicas"
		replicas  = 8
	)
	store := newCASSessionStateStore()
	base := time.Now().Add(-time.Minute)
	windowTTL := time.Hour

	// Every replica starts from the same stored state.
	seed := RouterSessionSnapshot{
		SessionID:               sessionID,
		LastSeen:                base,
		OutcomeWindowSize:       32,
		OutcomeWindowTTLSeconds: int(windowTTL / time.Second),
	}
	if err := store.Save(seed, routerMemoryTTL); err != nil {
		t.Fatalf("seed: %v", err)
	}

	var wg sync.WaitGroup
	for replica := 0; replica < replicas; replica++ {
		wg.Add(1)
		go func(replica int) {
			defer wg.Done()
			loaded := store.snapshot(t, sessionID)
			at := base.Add(time.Duration(replica) * time.Second)
			loaded.RecentOutcomes = append(loaded.RecentOutcomes, mergeTestOutcome(fmt.Sprintf("replica-%d", replica), at))
			loaded.SwitchTimestamps = append(loaded.SwitchTimestamps, at.UnixMilli())
			loaded.LastSeen = at
			if err := store.Merge(loaded, routerMemoryTTL); err != nil {
				t.Errorf("replica %d merge: %v", replica, err)
			}
		}(replica)
	}
	wg.Wait()

	final := store.snapshot(t, sessionID)
	gotOutcomes := map[string]bool{}
	for _, outcome := range final.RecentOutcomes {
		gotOutcomes[outcome.RequestID] = true
	}
	for replica := 0; replica < replicas; replica++ {
		if !gotOutcomes[fmt.Sprintf("replica-%d", replica)] {
			t.Fatalf("replica %d lost its outcome: %+v", replica, final.RecentOutcomes)
		}
	}
	if len(final.SwitchTimestamps) != replicas {
		t.Fatalf("switch history lost entries: %+v", final.SwitchTimestamps)
	}
}

// TestPersistencePrefersMergeOverWholeSnapshotSave locks the wiring: a store
// that can merge must not be written through the clobbering Save path.
func TestPersistencePrefersMergeOverWholeSnapshotSave(t *testing.T) {
	ResetRouterSessionMemoryForTesting()
	store := newCASSessionStateStore()
	SetRouterSessionStateStore(store)
	t.Cleanup(func() {
		SetRouterSessionStateStore(nil)
		ResetRouterSessionMemoryForTesting()
	})

	RecordTurnOutcome("merge-preferred", mergeTestOutcome("turn-1", time.Now()), time.Now())

	if store.merges == 0 {
		t.Fatal("persistence did not use the atomic merge")
	}
	if store.saves != 0 {
		t.Fatalf("persistence fell back to whole-snapshot saves: %d", store.saves)
	}
}

// TestMergeRouterSessionSnapshotsIsMonotonic covers the merge rules directly,
// including the ordering the gate reads: evidence, switch history and policy.
func TestMergeRouterSessionSnapshotsIsMonotonic(t *testing.T) {
	now := time.Now()
	remote := RouterSessionSnapshot{
		SessionID:               "merge-rules",
		CurrentModel:            "cheap",
		LastSeen:                now.Add(-time.Second),
		LastSwitchAt:            now.Add(-time.Minute),
		SwitchTimestamps:        []int64{now.Add(-time.Minute).UnixMilli()},
		OutcomeWindowSize:       8,
		OutcomeWindowTTLSeconds: 900,
		TurnCount:               3,
		SwitchCount:             1,
		ModelTurns:              map[string]int{"cheap": 3},
		CumulativeCost:          1,
		RecentOutcomes:          []TurnOutcome{mergeTestOutcome("remote-turn", now.Add(-30*time.Second))},
	}
	local := RouterSessionSnapshot{
		SessionID:               "merge-rules",
		CurrentModel:            "frontier",
		LastSeen:                now,
		LastSwitchAt:            now,
		SwitchTimestamps:        []int64{now.UnixMilli()},
		OutcomeWindowSize:       16,
		OutcomeWindowTTLSeconds: 3600,
		TurnCount:               2,
		SwitchCount:             4,
		ModelTurns:              map[string]int{"cheap": 1, "frontier": 1},
		CumulativeCost:          5,
		RecentOutcomes:          []TurnOutcome{mergeTestOutcome("local-turn", now)},
	}

	merged := mergeRouterSessionSnapshots(remote, local)

	if merged.CurrentModel != "frontier" {
		t.Fatalf("current model = %q, want the newer replica's view", merged.CurrentModel)
	}
	if len(merged.RecentOutcomes) != 2 {
		t.Fatalf("evidence window lost a writer's fact: %+v", merged.RecentOutcomes)
	}
	if merged.RecentOutcomes[0].RequestID != "remote-turn" || merged.RecentOutcomes[1].RequestID != "local-turn" {
		t.Fatalf("evidence not ordered by event time: %+v", merged.RecentOutcomes)
	}
	if len(merged.SwitchTimestamps) != 2 {
		t.Fatalf("switch history lost a timestamp: %+v", merged.SwitchTimestamps)
	}
	if !merged.LastSwitchAt.Equal(now) {
		t.Fatalf("last switch = %v, want the later timestamp", merged.LastSwitchAt)
	}
	if merged.TurnCount != 3 || merged.SwitchCount != 4 {
		t.Fatalf("counters regressed: turns=%d switches=%d", merged.TurnCount, merged.SwitchCount)
	}
	if merged.ModelTurns["cheap"] != 3 || merged.ModelTurns["frontier"] != 1 {
		t.Fatalf("per-model turns regressed: %+v", merged.ModelTurns)
	}
	if merged.CumulativeCost != 5 {
		t.Fatalf("cumulative cost = %v, want the higher observation", merged.CumulativeCost)
	}
	if merged.OutcomeWindowSize != 16 || merged.OutcomeWindowTTLSeconds != 3600 {
		t.Fatalf("window policy narrowed: size=%d ttl=%d", merged.OutcomeWindowSize, merged.OutcomeWindowTTLSeconds)
	}
}

// TestMergeRouterSessionSnapshotsDeduplicatesSharedFacts makes sure a fact both
// replicas observed is not counted twice: the oscillation guard reads the
// switch series, and the window is bounded by turn identity.
func TestMergeRouterSessionSnapshotsDeduplicatesSharedFacts(t *testing.T) {
	now := time.Now()
	shared := mergeTestOutcome("same-turn", now)
	switchAt := now.Add(-time.Minute).UnixMilli()
	remote := RouterSessionSnapshot{
		SessionID:        "dedup",
		LastSeen:         now,
		SwitchTimestamps: []int64{switchAt},
		RecentOutcomes:   []TurnOutcome{shared},
	}
	local := RouterSessionSnapshot{
		SessionID:        "dedup",
		LastSeen:         now,
		SwitchTimestamps: []int64{switchAt},
		RecentOutcomes:   []TurnOutcome{shared},
	}

	merged := mergeRouterSessionSnapshots(remote, local)

	if len(merged.RecentOutcomes) != 1 {
		t.Fatalf("the same turn was recorded twice: %+v", merged.RecentOutcomes)
	}
	if len(merged.SwitchTimestamps) != 1 {
		t.Fatalf("the same switch was recorded twice: %+v", merged.SwitchTimestamps)
	}
}

// TestMergeRouterSessionSnapshotsIgnoresExpiredRemote keeps an idle-expired
// session from being resurrected by a stale replica.
func TestMergeRouterSessionSnapshotsIgnoresExpiredRemote(t *testing.T) {
	now := time.Now()
	remote := RouterSessionSnapshot{
		SessionID:      "expired",
		CurrentModel:   "cheap",
		LastSeen:       now.Add(-2 * routerMemoryTTL),
		RecentOutcomes: []TurnOutcome{mergeTestOutcome("stale-turn", now.Add(-2*routerMemoryTTL))},
	}
	local := RouterSessionSnapshot{
		SessionID:    "expired",
		CurrentModel: "frontier",
		LastSeen:     now,
	}

	merged := mergeRouterSessionSnapshots(remote, local)

	if merged.CurrentModel != "frontier" || len(merged.RecentOutcomes) != 0 {
		t.Fatalf("expired remote state was resurrected: %+v", merged)
	}
}

// A payload the merge cannot read still holds another writer's outcomes and
// switch history, so it must abort rather than be overwritten by the local
// view.
func TestMergeStoredSnapshotRefusesUnreadablePayload(t *testing.T) {
	local := RouterSessionSnapshot{
		SessionID:    "corrupt",
		CurrentModel: "frontier",
		LastSeen:     time.Now(),
	}

	if _, err := mergeStoredSnapshot([]byte("{not json"), local); err == nil {
		t.Fatal("unreadable payload merged without an error, so a Set would overwrite it")
	}
}

func TestMergeStoredSnapshotFoldsReadablePayload(t *testing.T) {
	now := time.Now()
	stored, err := json.Marshal(RouterSessionSnapshot{
		SessionID:      "readable",
		LastSeen:       now.Add(-time.Minute),
		RecentOutcomes: []TurnOutcome{mergeTestOutcome("remote-turn", now.Add(-time.Minute))},
	})
	if err != nil {
		t.Fatalf("marshal stored snapshot: %v", err)
	}
	local := RouterSessionSnapshot{
		SessionID:      "readable",
		LastSeen:       now,
		RecentOutcomes: []TurnOutcome{mergeTestOutcome("local-turn", now)},
	}

	merged, err := mergeStoredSnapshot(stored, local)
	if err != nil {
		t.Fatalf("readable payload failed to merge: %v", err)
	}
	if len(merged.RecentOutcomes) != 2 {
		t.Fatalf("merge dropped a writer's outcome: %+v", merged.RecentOutcomes)
	}
}
