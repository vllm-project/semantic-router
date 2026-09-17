package sessiontelemetry

import (
	"math"
	"testing"
	"time"
)

func TestSessionUsageObservationYieldsToDispatchOwnership(t *testing.T) {
	ResetRouterSessionMemoryForTesting()
	t.Cleanup(ResetRouterSessionMemoryForTesting)
	RecordSessionUsage(SessionUsageParams{SessionID: "usage-only", Model: "small", PromptTokens: 10})
	snapshot, ok := GetRouterSessionSnapshot("usage-only", time.Now())
	if !ok || snapshot.CurrentModel != "small" || snapshot.TurnCount != 0 || snapshot.CumulativePromptTokens != 10 {
		t.Fatalf("usage-only flow must retain its observed model: %+v, found=%t", snapshot, ok)
	}
	RecordSessionDecision(SessionDecisionParams{SessionID: "usage-only", SelectedModel: "frontier", DecisionName: "reasoning"})
	RecordSessionUsage(SessionUsageParams{SessionID: "usage-only", Model: "small", PromptTokens: 15})
	snapshot, ok = GetRouterSessionSnapshot("usage-only", time.Now())
	if !ok || snapshot.CurrentModel != "frontier" || snapshot.TurnCount != 1 || snapshot.CumulativePromptTokens != 25 {
		t.Fatalf("usage must defer to the dispatch owner without losing tokens: %+v, found=%t", snapshot, ok)
	}
}

func TestLateSessionUsagePreservesLatestDecision(t *testing.T) {
	ResetRouterSessionMemoryForTesting()
	t.Cleanup(ResetRouterSessionMemoryForTesting)
	RecordSessionDecision(SessionDecisionParams{
		SessionID: "overlap", SelectedModel: "small", DecisionName: "simple",
	})
	RecordSessionDecision(SessionDecisionParams{
		SessionID: "overlap", SelectedModel: "frontier", DecisionName: "reasoning", TurnIndex: 1,
		ActiveToolLoop: true, Policy: map[string]interface{}{"decision_reason": "switch_allowed"},
	})
	RecordSessionUsage(SessionUsageParams{SessionID: "overlap", Model: "frontier", PromptTokens: 20, Cost: .02})
	RecordSessionUsage(SessionUsageParams{SessionID: "overlap", Model: "small", PromptTokens: 10, Cost: .01})
	snapshot, ok := GetRouterSessionSnapshot("overlap", time.Now())
	if !ok || snapshot.CurrentModel != "frontier" || snapshot.LastDecisionName != "reasoning" ||
		!snapshot.ActiveToolLoop || snapshot.LastDecisionReason != "switch_allowed" ||
		snapshot.TurnCount != 2 || snapshot.SwitchCount != 1 {
		t.Fatalf("late usage changed the latest decision: %+v, found=%t", snapshot, ok)
	}
	if snapshot.CumulativePromptTokens != 30 || math.Abs(snapshot.CumulativeCost-.03) > 1e-9 {
		t.Fatalf("late usage must still be billed: %+v", snapshot)
	}
}

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
