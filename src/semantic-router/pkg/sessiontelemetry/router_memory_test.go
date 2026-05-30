package sessiontelemetry

import (
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
		TurnIndex:     0,
		Policy: map[string]interface{}{
			"decision_reason": "first_turn",
		},
		Timestamp: base,
	})
	RecordSessionUsage(SessionUsageParams{
		SessionID:          "sess-agent",
		Model:              "small",
		PromptTokens:       1000,
		CachedPromptTokens: 250,
		CompletionTokens:   300,
		Cost:               0.002,
		Timestamp:          base.Add(2 * time.Second),
	})
	RecordSessionDecision(SessionDecisionParams{
		SessionID:     "sess-agent",
		PreviousModel: "small",
		SelectedModel: "frontier",
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
	if snapshot.LastDecisionReason != "first_turn" {
		t.Fatalf("last decision reason = %q, want first_turn", snapshot.LastDecisionReason)
	}
	if snapshot.IdleFor != 10*time.Second {
		t.Fatalf("idle = %s, want 10s", snapshot.IdleFor)
	}
}
