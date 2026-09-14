package sessiontelemetry

import (
	"encoding/json"
	"math"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestRouterSessionMemoryPreservesExactCandidate(t *testing.T) {
	ResetRouterSessionMemoryForTesting()
	t.Cleanup(ResetRouterSessionMemoryForTesting)
	now := time.Now()
	enabled := true
	candidate := config.ModelRef{Model: "model", LoRAName: "adapter", ModelReasoningControl: config.ModelReasoningControl{
		UseReasoning: &enabled, ReasoningEffort: "high",
	}}
	RecordSessionDecision(SessionDecisionParams{SessionID: "exact", SelectedModel: "model", SelectedCandidate: &candidate, Timestamp: now})
	enabled = false
	candidate.ReasoningEffort = "low"
	snapshot, ok := GetRouterSessionSnapshot("exact", now)
	if !ok || snapshot.CurrentCandidate == nil || snapshot.CurrentCandidate.ReasoningEffort != "high" || !*snapshot.CurrentCandidate.UseReasoning {
		t.Fatalf("recorded candidate was mutated: %+v", snapshot.CurrentCandidate)
	}
	encoded, err := json.Marshal(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	var restored RouterSessionSnapshot
	if err := json.Unmarshal(encoded, &restored); err != nil {
		t.Fatal(err)
	}
	hydrateRouterSessionSnapshot(restored)
	*restored.CurrentCandidate.UseReasoning = false
	restored.CurrentCandidate.ReasoningEffort = "low"
	snapshot, _ = GetRouterSessionSnapshot("exact", now)
	if snapshot.CurrentCandidate == nil || snapshot.CurrentCandidate.ReasoningEffort != "high" || !*snapshot.CurrentCandidate.UseReasoning || snapshot.CurrentCandidate.LoRAName != "adapter" {
		t.Fatalf("restored candidate lost identity: %+v", snapshot.CurrentCandidate)
	}
	*snapshot.CurrentCandidate.UseReasoning = false
	snapshot, _ = GetRouterSessionSnapshot("exact", now)
	if !*snapshot.CurrentCandidate.UseReasoning {
		t.Fatal("snapshot mutation changed stored candidate")
	}
	RecordSessionUsage(SessionUsageParams{SessionID: "exact", Model: "adapter", Timestamp: now})
	snapshot, _ = GetRouterSessionSnapshot("exact", now)
	if snapshot.CurrentCandidate == nil || snapshot.CurrentModel != "model" {
		t.Fatal("LoRA usage discarded the selected candidate")
	}
	RecordSessionUsage(SessionUsageParams{SessionID: "exact", Model: "rerouted", Timestamp: now})
	snapshot, _ = GetRouterSessionSnapshot("exact", now)
	if snapshot.CurrentCandidate != nil {
		t.Fatal("rerouted usage retained a stale candidate")
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
