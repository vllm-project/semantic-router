package sessiontelemetry

import (
	"testing"
	"time"
)

func TestRecordSessionDecisionCountsUserTurns(t *testing.T) {
	ResetRouterSessionMemoryForTesting()
	t.Cleanup(ResetRouterSessionMemoryForTesting)
	for _, p := range []SessionDecisionParams{
		{SessionID: "k", SelectedModel: "a", UserTurn: true},
		{SessionID: "k", SelectedModel: "a"},
		{SessionID: "k", SelectedModel: "b", UserTurn: true, Released: true},
	} {
		RecordSessionDecision(p)
	}
	snapshot, ok := GetRouterSessionSnapshot("k", time.Now())
	if !ok {
		t.Fatal("session snapshot missing")
	}
	if snapshot.UserTurnCount != 2 || snapshot.LastReleaseUserTurn != 2 {
		t.Fatalf("user turns = %d, last release = %d; want 2, 2", snapshot.UserTurnCount, snapshot.LastReleaseUserTurn)
	}
}

func TestMergeRouterSessionSnapshotKeepsReleaseCounters(t *testing.T) {
	now := time.Now()
	remote := RouterSessionSnapshot{SessionID: "k", LastSeen: now, UserTurnCount: 5, LastReleaseUserTurn: 4}
	local := RouterSessionSnapshot{SessionID: "k", LastSeen: now, UserTurnCount: 3, LastReleaseUserTurn: 1}
	merged := mergeRouterSessionSnapshots(remote, local)
	if merged.UserTurnCount != 5 || merged.LastReleaseUserTurn != 4 {
		t.Fatalf("merged = %d/%d, want 5/4", merged.UserTurnCount, merged.LastReleaseUserTurn)
	}
}

func TestRouterSessionStateStoreRestoresReleaseCounters(t *testing.T) {
	ResetRouterSessionMemoryForTesting()
	store := &fakeRouterSessionStateStore{}
	SetRouterSessionStateStore(store)
	t.Cleanup(func() {
		SetRouterSessionStateStore(nil)
		ResetRouterSessionMemoryForTesting()
	})
	RecordSessionDecision(SessionDecisionParams{SessionID: "k", SelectedModel: "a", UserTurn: true, Released: true, Timestamp: time.Now()})

	ResetRouterSessionMemoryForTesting()
	snapshot, ok := GetRouterSessionSnapshot("k", time.Now())
	if !ok {
		t.Fatal("shared session snapshot was not restored")
	}
	if snapshot.UserTurnCount != 1 || snapshot.LastReleaseUserTurn != 1 {
		t.Fatalf("restored = %d/%d, want 1/1", snapshot.UserTurnCount, snapshot.LastReleaseUserTurn)
	}
}
