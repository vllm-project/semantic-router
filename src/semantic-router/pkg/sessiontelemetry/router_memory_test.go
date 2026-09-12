package sessiontelemetry

import (
	"math"
	"testing"
	"time"
)

func TestRouterSessionMemoryRecordsDecisionAndUsage(t *testing.T) {
	ResetRouterSessionMemoryForTesting()
	base := time.Date(2026, time.May, 30, 10, 0, 0, 0, time.UTC)
	setRouterSessionMemoryNowForTesting(func() time.Time { return base })
	defer setRouterSessionMemoryNowForTesting(nil)

	RecordSessionDecision(SessionDecisionParams{
		SessionID:     "sess-agent",
		UserID:        "user-a",
		SelectedModel: "small",
		DecisionName:  "coding",
		TurnIndex:     0,
		Policy: map[string]interface{}{
			"decision_reason": "first_turn",
		},
		Timestamp: base,
	})
	RecordSessionUsage(SessionUsageParams{
		SessionID:                   "sess-agent",
		Model:                       "small",
		PromptTokens:                1000,
		CachedPromptTokens:          250,
		CacheWriteTokens:            125,
		EstimatedCachedPromptTokens: 500,
		CompletionTokens:            300,
		Cost:                        0.002,
		EstimatedCacheSavings:       0.0045,
		CacheAccountingSource:       "router_estimated",
		Timestamp:                   base.Add(2 * time.Second),
	})
	RecordSessionDecision(SessionDecisionParams{
		SessionID:     "sess-agent",
		PreviousModel: "small",
		SelectedModel: "frontier",
		DecisionName:  "math",
		TurnIndex:     1,
		Timestamp:     base.Add(10 * time.Second),
	})

	snapshot, ok := GetRouterSessionSnapshot("sess-agent", base.Add(20*time.Second))
	if !ok {
		t.Fatal("expected session memory snapshot")
	}
	if snapshot.CurrentModel != "frontier" {
		t.Fatalf("current model = %q, want frontier", snapshot.CurrentModel)
	}
	if snapshot.SwitchCount != 1 {
		t.Fatalf("switch count = %d, want 1", snapshot.SwitchCount)
	}
	if snapshot.CumulativeCachedTokens != 250 {
		t.Fatalf("cached tokens = %d, want 250", snapshot.CumulativeCachedTokens)
	}
	if snapshot.CumulativeCacheWriteTokens != 125 {
		t.Fatalf("cache-write tokens = %d, want 125", snapshot.CumulativeCacheWriteTokens)
	}
	if snapshot.CumulativeEstimatedCachedTokens != 500 {
		t.Fatalf("estimated cached tokens = %d, want 500", snapshot.CumulativeEstimatedCachedTokens)
	}
	if math.Abs(snapshot.CumulativeEstimatedCacheSavings-0.0045) > 1e-9 {
		t.Fatalf("estimated cache savings = %f, want 0.0045", snapshot.CumulativeEstimatedCacheSavings)
	}
	if snapshot.LastCacheAccountingSource != "router_estimated" {
		t.Fatalf("last cache accounting source = %q, want router_estimated", snapshot.LastCacheAccountingSource)
	}
	if snapshot.LastDecisionReason != "first_turn" {
		t.Fatalf("last decision reason = %q, want first_turn", snapshot.LastDecisionReason)
	}
	if snapshot.LastDecisionName != "math" {
		t.Fatalf("last decision name = %q, want math", snapshot.LastDecisionName)
	}
	if snapshot.IdleFor != 10*time.Second {
		t.Fatalf("idle = %s, want 10s", snapshot.IdleFor)
	}
}

func TestRouterSessionSnapshotSwitchState(t *testing.T) {
	ResetRouterSessionMemoryForTesting()
	base := time.Now().Truncate(time.Second)
	setRouterSessionMemoryNowForTesting(func() time.Time { return base.Add(6 * time.Minute) })
	defer setRouterSessionMemoryNowForTesting(nil)

	const sessionID = "switch-state"
	RecordSessionDecision(SessionDecisionParams{
		SessionID:     sessionID,
		SelectedModel: "model-a",
		Timestamp:     base,
	})
	RecordSessionDecision(SessionDecisionParams{
		SessionID:     sessionID,
		PreviousModel: "model-a",
		SelectedModel: "model-b",
		Timestamp:     base.Add(10 * time.Second),
	})
	// Regular activity after the switch must refresh LastSeen but not
	// LastSwitchAt: cooldown measures the switch, not the conversation.
	RecordSessionUsage(SessionUsageParams{
		SessionID:        sessionID,
		Model:            "model-b",
		CompletionTokens: 1,
		Timestamp:        base.Add(5 * time.Minute),
	})

	snapshot, ok := GetRouterSessionSnapshot(sessionID, base.Add(5*time.Minute+time.Second))
	if !ok {
		t.Fatal("snapshot missing")
	}
	if !snapshot.LastSwitchAt.Equal(base.Add(10 * time.Second)) {
		t.Fatalf("last switch = %v, want the t+10s switch, not the t+5m activity", snapshot.LastSwitchAt)
	}
	if !snapshot.LastSeen.Equal(base.Add(5 * time.Minute)) {
		t.Fatalf("last seen = %v, want t+5m", snapshot.LastSeen)
	}

	// A session that never switched reports a zero switch time.
	RecordSessionDecision(SessionDecisionParams{
		SessionID:     "no-switch",
		SelectedModel: "model-a",
		Timestamp:     base,
	})
	fresh, _ := GetRouterSessionSnapshot("no-switch", base.Add(time.Minute))
	if !fresh.LastSwitchAt.IsZero() {
		t.Fatalf("never-switched session has LastSwitchAt = %v", fresh.LastSwitchAt)
	}

	// Persistence round-trip must keep the switch time.
	store := newFakeSessionStateStore()
	SetRouterSessionStateStore(store)
	defer SetRouterSessionStateStore(nil)
	ResetRouterSessionMemoryForTesting()
	RecordSessionDecision(SessionDecisionParams{
		SessionID: sessionID, SelectedModel: "model-a", Timestamp: base,
	})
	RecordSessionDecision(SessionDecisionParams{
		SessionID: sessionID, PreviousModel: "model-a", SelectedModel: "model-b",
		Timestamp: base.Add(10 * time.Second),
	})
	ResetRouterSessionMemoryForTesting()
	recovered, ok := GetRouterSessionSnapshot(sessionID, base.Add(20*time.Second))
	if !ok {
		t.Fatal("snapshot did not recover from the shared store")
	}
	if !recovered.LastSwitchAt.Equal(base.Add(10 * time.Second)) {
		t.Fatalf("last switch lost across restart: %v", recovered.LastSwitchAt)
	}
}

func TestSwitchTimestampsWindowCount(t *testing.T) {
	ResetRouterSessionMemoryForTesting()
	base := time.Now().Truncate(time.Second)
	setRouterSessionMemoryNowForTesting(func() time.Time { return base.Add(30 * time.Minute) })
	defer setRouterSessionMemoryNowForTesting(nil)

	const sessionID = "switch-window"
	ConfigureTurnOutcomeWindow(sessionID, 8, 15*time.Minute, base)
	switchAt := func(previous, next string, at time.Time) {
		RecordSessionDecision(SessionDecisionParams{
			SessionID: sessionID, PreviousModel: previous, SelectedModel: next, Timestamp: at,
		})
	}
	switchAt("", "model-a", base)
	switchAt("model-a", "model-b", base.Add(time.Minute))
	switchAt("model-b", "model-a", base.Add(2*time.Minute))

	snapshot, ok := GetRouterSessionSnapshot(sessionID, base.Add(10*time.Minute))
	if !ok {
		t.Fatal("snapshot missing")
	}
	if snapshot.SwitchCount != 2 {
		t.Fatalf("lifetime switch count = %d, want 2", snapshot.SwitchCount)
	}
	if got := CountRecentSwitches(snapshot.SwitchTimestamps, 15*time.Minute, base.Add(10*time.Minute)); got != 2 {
		t.Fatalf("window count = %d, want 2 inside 15m", got)
	}
	if got := CountRecentSwitches(snapshot.SwitchTimestamps, 30*time.Second, base.Add(10*time.Minute)); got != 0 {
		t.Fatalf("window count = %d, want 0 inside 30s", got)
	}

	// A later switch prunes the series by the session's window TTL.
	switchAt("model-a", "model-b", base.Add(20*time.Minute))
	snapshot, _ = GetRouterSessionSnapshot(sessionID, base.Add(21*time.Minute))
	if len(snapshot.SwitchTimestamps) != 1 || snapshot.SwitchTimestamps[0] != base.Add(20*time.Minute).UnixMilli() {
		t.Fatalf("stale switch timestamps not pruned: %v", snapshot.SwitchTimestamps)
	}

	// Persistence round-trip keeps the series.
	store := newFakeSessionStateStore()
	SetRouterSessionStateStore(store)
	defer SetRouterSessionStateStore(nil)
	ResetRouterSessionMemoryForTesting()
	switchAt("", "model-a", base)
	switchAt("model-a", "model-b", base.Add(time.Minute))
	ResetRouterSessionMemoryForTesting()
	recovered, ok := GetRouterSessionSnapshot(sessionID, base.Add(2*time.Minute))
	if !ok || len(recovered.SwitchTimestamps) != 1 {
		t.Fatalf("switch timestamps lost across restart: ok=%v %+v", ok, recovered)
	}
}
