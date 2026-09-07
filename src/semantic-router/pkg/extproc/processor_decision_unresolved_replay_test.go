package extproc

import (
	"fmt"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

func TestRespondDecisionUnresolvedFinalizesReplayAsFailed(t *testing.T) {
	recorder := routerreplay.NewRecorder(store.NewMemoryStore(10, 0))
	router := &OpenAIRouter{ReplayRecorder: recorder}

	replayConfig := config.DefaultRouterReplayPluginConfig()
	replayConfig.Enabled = true
	replayConfig.CaptureResponseBody = true
	ctx := &RequestContext{
		RequestID:                "unresolved-request",
		SourceFormat:             llmprotocol.OpenAIChatV1,
		SemanticRequest:          testNeutralRequest("entrypoint-model", "hello"),
		RouterReplayPluginConfig: &replayConfig,
		VSRDecisionDiagnostics: decision.EvaluationDiagnostics{
			AppliedUnknownPolicies: map[string]string{"guarded": "fail_request"},
		},
	}

	decisionErr := fmt.Errorf(
		"decision evaluation failed: %w",
		decision.ErrDecisionUnresolved,
	)
	resp := router.respondDecisionUnresolved(ctx, "entrypoint-model", decisionErr)

	if ctx.RouterReplayID == "" {
		t.Fatal("expected the 503 path to create a replay record")
	}
	record, found := recorder.GetRecord(ctx.RouterReplayID)
	if !found {
		t.Fatalf("replay record %q not found", ctx.RouterReplayID)
	}
	if record.LifecycleState != routerreplay.LifecycleFailed {
		t.Fatalf("lifecycle state = %q, want %q", record.LifecycleState, routerreplay.LifecycleFailed)
	}
	if record.TerminalReason != "decision_unresolved" {
		t.Fatalf("terminal reason = %q, want %q", record.TerminalReason, "decision_unresolved")
	}
	if record.RouteDiagnostics == nil ||
		record.RouteDiagnostics.AppliedUnknownPolicies["guarded"] != "fail_request" {
		t.Fatalf("route diagnostics = %+v, want applied unknown policy guarded=fail_request", record.RouteDiagnostics)
	}

	immediate := resp.GetImmediateResponse()
	if immediate == nil {
		t.Fatal("expected an immediate 503 response")
	}
	var replayHeader string
	for _, option := range immediate.GetHeaders().GetSetHeaders() {
		if option.GetHeader().GetKey() == headers.RouterReplayID {
			replayHeader = string(option.GetHeader().GetRawValue())
		}
	}
	if replayHeader != ctx.RouterReplayID {
		t.Fatalf("replay header = %q, want %q", replayHeader, ctx.RouterReplayID)
	}
}

func TestRespondDecisionUnresolvedKeepsErrorResponseShape(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := &RequestContext{RequestID: "no-replay"}

	resp := router.respondDecisionUnresolved(ctx, "model", decision.ErrDecisionUnresolved)

	immediate := resp.GetImmediateResponse()
	if immediate == nil {
		t.Fatal("expected an immediate response")
	}
	body := string(immediate.GetBody())
	if body == "" || ctx.RouterReplayID != "" {
		t.Fatalf("body = %q, replay id = %q; want body without a replay record", body, ctx.RouterReplayID)
	}
}
