package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

// A selection rejection on a decision that mounts only a per-decision
// router_replay plugin (no global replay config, hence no global recorder)
// must still land a failed replay record under that decision. The matched
// decision is known at rejection time, so the recorder must be resolved by
// its name instead of silently falling back to a missing global recorder.
func TestRespondSelectionRejectedFinalizesPerDecisionReplay(t *testing.T) {
	decisionName := "simple_general"
	recorder := routerreplay.NewRecorder(store.NewMemoryStore(10, 0))
	router := &OpenAIRouter{
		ReplayRecorder:  nil, // no global router_replay config in this deployment
		ReplayRecorders: map[string]*routerreplay.Recorder{config.RoutingDecisionKey(config.DefaultRecipeName, decisionName): recorder},
	}

	replayConfig := config.DefaultRouterReplayPluginConfig()
	replayConfig.Enabled = true
	replayConfig.CaptureResponseBody = true
	ctx := &RequestContext{
		RequestID:                "selection-rejected-request",
		SourceFormat:             llmprotocol.OpenAIChatV1,
		SemanticRequest:          testNeutralRequest("entrypoint-model", "hello"),
		RouterReplayPluginConfig: &replayConfig,
		VSRSelectedDecision:      &config.Decision{Name: decisionName},
	}

	resp := router.respondSelectionRejected(ctx, "entrypoint-model", selection.ErrNoEligibleCandidates)

	if ctx.RouterReplayID == "" {
		t.Fatal("expected the 503 path to create a replay record for the matched decision")
	}
	record, found := recorder.GetRecord(ctx.RouterReplayID)
	if !found {
		t.Fatalf("replay record %q not found in the per-decision recorder", ctx.RouterReplayID)
	}
	if record.LifecycleState != routerreplay.LifecycleFailed {
		t.Fatalf("lifecycle state = %q, want %q", record.LifecycleState, routerreplay.LifecycleFailed)
	}
	if record.TerminalReason != "selection_rejected" {
		t.Fatalf("terminal reason = %q, want %q", record.TerminalReason, "selection_rejected")
	}
	if record.Decision != decisionName {
		t.Fatalf("record decision = %q, want %q", record.Decision, decisionName)
	}
	if immediate := resp.GetImmediateResponse(); immediate == nil {
		t.Fatal("expected an immediate 503 response")
	}
}
