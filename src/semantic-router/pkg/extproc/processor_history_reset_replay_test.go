package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/historyreset"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

// A fail-closed rejection is exactly the outcome that needs an audit trail, so
// it must produce a finalized Replay record like any other router-side refusal
// rather than returning before capture starts.
func TestFailClosedResetFinalizesReplayAsFailed(t *testing.T) {
	recorder := routerreplay.NewRecorder(store.NewMemoryStore(10, 0))
	router := &OpenAIRouter{ReplayRecorder: recorder}

	replayConfig := config.DefaultRouterReplayPluginConfig()
	replayConfig.Enabled = true
	replayConfig.CaptureResponseBody = true
	ctx := &RequestContext{
		RequestID:                "reset-rejected",
		RequestModel:             "entrypoint-model",
		VSRSelectedModel:         "backend-model",
		VSRSelectedDecisionName:  "reset",
		SourceFormat:             llmprotocol.OpenAIChatV1,
		SemanticRequest:          testNeutralRequest("entrypoint-model", "hello"),
		RouterReplayPluginConfig: &replayConfig,
		HistoryResetDiagnostics: &historyreset.Diagnostics{
			Signal:       "topic_boundary",
			Scope:        historyreset.ScopeEligibleHistory,
			TriggerClass: historyreset.TriggerUnknown,
			Outcome:      historyreset.OutcomeFailed,
			Reason:       historyreset.ReasonEvidenceUnknown,
		},
	}

	resp := router.respondContextTransformationRejected(ctx, "entrypoint-model")
	if resp.GetImmediateResponse() == nil {
		t.Fatal("expected an immediate rejection response")
	}
	if ctx.RouterReplayID == "" {
		t.Fatal("the rejection did not create a replay record")
	}
	record, found := recorder.GetRecord(ctx.RouterReplayID)
	if !found {
		t.Fatalf("replay record %q not found", ctx.RouterReplayID)
	}
	if record.LifecycleState != routerreplay.LifecycleFailed {
		t.Fatalf("lifecycle state = %q, want %q", record.LifecycleState, routerreplay.LifecycleFailed)
	}
	if record.TerminalReason != "history_reset_"+historyreset.ReasonEvidenceUnknown {
		t.Fatalf("terminal reason = %q", record.TerminalReason)
	}
	if record.ResponseStatus != 503 {
		t.Fatalf("response status = %d, want 503", record.ResponseStatus)
	}
}

// The reset diagnostic reaches Replay with its scope and without content.
func TestResetReplayDiagnosticsCarryTheEligibleScope(t *testing.T) {
	ctx := &RequestContext{
		HistoryResetDiagnostics: &historyreset.Diagnostics{
			Signal:           "topic_boundary",
			Scope:            historyreset.ScopeEligibleHistory,
			TriggerClass:     historyreset.TriggerChange,
			Version:          "v1",
			Outcome:          historyreset.OutcomeApplied,
			Reason:           historyreset.ReasonApplied,
			ExaminedMessages: 5,
			RetainedMessages: 1,
			RemovedMessages:  4,
			RemovedTurns:     2,
		},
	}
	record := historyResetReplayDiagnostics(ctx)
	if record == nil || record.Scope != historyreset.ScopeEligibleHistory {
		t.Fatalf("replay record must carry the eligible scope: %+v", record)
	}
	if record.RemovedTurns != 2 || record.RetainedMessages != 1 {
		t.Fatalf("unexpected replay counts %+v", record)
	}
}
