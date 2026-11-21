package sessiontelemetry

import (
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// The fallback candidate that answered takes the session over from the model
// the request was routed to, unless a newer dispatch has taken it since.
func TestServedModelTakesOverTheRoutedModel(t *testing.T) {
	ResetRouterSessionMemoryForTesting()
	t.Cleanup(ResetRouterSessionMemoryForTesting)
	now := time.Now()
	RecordSessionDecision(SessionDecisionParams{SessionID: "s", SelectedModel: "fail-model", Timestamp: now})
	RecordSessionServedModel("s", "fail-model", "cheap-model", &config.ModelRef{Model: "cheap-model"})
	snapshot, _ := GetRouterSessionSnapshot("s", now)
	if snapshot.CurrentModel != "cheap-model" || snapshot.CurrentCandidate == nil || snapshot.CurrentCandidate.Model != "cheap-model" ||
		snapshot.ModelTurns["cheap-model"] != 1 || snapshot.ModelTurns["fail-model"] != 0 {
		t.Fatalf("snapshot after the fallback = %+v", snapshot)
	}
	if model, _ := GetLastModel("s"); model != "cheap-model" {
		t.Fatalf("last model = %q", model)
	}

	RecordSessionDecision(SessionDecisionParams{SessionID: "s", SelectedModel: "main-model", TurnIndex: 1, Timestamp: now})
	RecordSessionServedModel("s", "cheap-model", "other-model", nil)
	if snapshot, _ = GetRouterSessionSnapshot("s", now); snapshot.CurrentModel != "main-model" {
		t.Fatalf("a late fallback replaced a newer dispatch: %q", snapshot.CurrentModel)
	}
}

// A request held apart from the session, a subagent's or a side request's,
// counts in the session's usage without naming its model or its cache.
func TestApartUsageLeavesTheSessionModelAndReceipt(t *testing.T) {
	ResetRouterSessionMemoryForTesting()
	t.Cleanup(ResetRouterSessionMemoryForTesting)
	now := time.Now()
	RecordSessionUsage(SessionUsageParams{SessionID: "s", Model: "side-model", PromptTokens: 100, CacheReported: true, Apart: true, Timestamp: now})
	snapshot, _ := GetRouterSessionSnapshot("s", now)
	if snapshot.CurrentModel != "" || snapshot.LastCacheReceipt != nil || snapshot.CumulativePromptTokens != 100 {
		t.Fatalf("snapshot after apart usage = %+v", snapshot)
	}
}
