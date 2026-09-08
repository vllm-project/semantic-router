package extproc

import (
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

// requiredGateReplayFields must appear on every gate verdict so replay can
// answer "why did this switch happen (or not)" without the router's memory.
var requiredGateReplayFields = []string{
	"evidence_version",
	"decision",
	"switch_origin",
	"mode",
	"window_count",
}

func gateReplayConfig(mode string) config.RouterLearningProtectionConfig {
	return config.RouterLearningProtectionConfig{
		Tuning: config.RouterLearningProtectionTuning{
			ProgressGate: &config.ProgressGateTuning{
				Enabled: gateBoolPtr(true),
				Mode:    mode,
			},
		},
	}
}

func gateReplayContext(t *testing.T, sessionID string) (*OpenAIRouter, *RequestContext, *selection.SelectionContext) {
	t.Helper()
	sessiontelemetry.ResetRouterSessionMemoryForTesting()
	t.Cleanup(sessiontelemetry.ResetRouterSessionMemoryForTesting)

	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	ctx := &RequestContext{SessionID: sessionID}
	learningCtx := &selection.SelectionContext{
		SessionID: sessionID,
		CandidateModels: []config.ModelRef{
			{Model: "cheap"},
			{Model: "frontier"},
		},
		AgenticSession: &selection.AgenticSessionContext{PreviousModel: "cheap"},
	}
	return router, ctx, learningCtx
}

// proposedSwitchResult mirrors what the session-aware selector hands to the
// gate: a switch proposal that already carries a session policy trace.
func proposedSwitchResult() *selection.SelectionResult {
	return &selection.SelectionResult{
		SelectedModel: "frontier",
		SessionPolicy: &selection.SessionPolicyTrace{
			CurrentModel:   "cheap",
			SelectedModel:  "frontier",
			DecisionReason: "switch_allowed",
		},
	}
}

func gateSectionFromReplayMap(t *testing.T, policy map[string]interface{}) map[string]interface{} {
	t.Helper()
	gate, ok := policy["switch_gate"].(map[string]interface{})
	if !ok {
		t.Fatalf("switch_gate missing from replay policy: %#v", policy)
	}
	for _, field := range requiredGateReplayFields {
		if _, exists := gate[field]; !exists {
			t.Fatalf("replay gate section missing %q: %#v", field, gate)
		}
	}
	return gate
}

// replayPolicyForResult serialises a gated result the way the response path
// does, so the assertions cover the learning-policy branch that production
// actually takes rather than the raw trace.
func replayPolicyForResult(
	ctx *RequestContext,
	result *selection.SelectionResult,
	cfg config.RouterLearningProtectionConfig,
) map[string]interface{} {
	policy := protectionPolicyFromSelectionResult(
		result,
		routerLearningIdentity{scope: config.RouterLearningScopeSession, memoryKey: ctx.SessionID},
		config.DecisionAdaptationModeApply,
		"cheap",
		"frontier",
		selectedModelName(result),
		cfg,
	)
	ctx.VSRLearningPolicies.Set(policy)
	return sessionPolicyMapForTelemetry(ctx, result)
}

func TestProgressGateVerdictReachesReplayThroughLearningPolicy(t *testing.T) {
	router, ctx, learningCtx := gateReplayContext(t, "gate-replay-chain")
	cfg := gateReplayConfig(selection.GateModeEnforce)
	selector := selection.NewSessionAwareSelector(selection.DefaultSessionAwareConfig())
	result := proposedSwitchResult()

	router.applySwitchGateToResult(cfg, ctx, learningCtx, selector, result, selection.SwitchOriginEscalation)

	gate := gateSectionFromReplayMap(t, replayPolicyForResult(ctx, result, cfg))
	if gate["decision"] != selection.GateDecisionSuppress {
		t.Fatalf("empty window must suppress, got %#v", gate)
	}
	if gate["suppression_reason"] != selection.GateReasonColdStart {
		t.Fatalf("suppression_reason = %v, want cold start", gate["suppression_reason"])
	}
	if gate["evidence_version"] != selection.ProgressEvidenceVersion {
		t.Fatalf("evidence_version = %v", gate["evidence_version"])
	}
	if gate["enforced"] != true {
		t.Fatalf("enforce mode must mark the verdict enforced: %#v", gate)
	}
}

func TestProgressGateReplayOmittedWhenGateDisabled(t *testing.T) {
	router, ctx, learningCtx := gateReplayContext(t, "gate-replay-disabled")
	cfg := config.RouterLearningProtectionConfig{}
	selector := selection.NewSessionAwareSelector(selection.DefaultSessionAwareConfig())
	result := proposedSwitchResult()

	router.applySwitchGateToResult(cfg, ctx, learningCtx, selector, result, selection.SwitchOriginEscalation)

	if _, exists := replayPolicyForResult(ctx, result, cfg)["switch_gate"]; exists {
		t.Fatalf("a disabled gate must not emit a replay section")
	}
	if result.SelectedModel != "frontier" {
		t.Fatalf("a disabled gate must not touch the proposal, got %q", result.SelectedModel)
	}
}

// TestProgressGateMultiTurnReplayRecordsEveryVerdict walks a session from cold
// start to a justified escalation and asserts that every turn — suppressed or
// allowed — lands a complete gate section in replay.
func TestProgressGateMultiTurnReplayRecordsEveryVerdict(t *testing.T) {
	router, ctx, learningCtx := gateReplayContext(t, "gate-replay-multiturn")
	cfg := gateReplayConfig(selection.GateModeEnforce)
	selector := selection.NewSessionAwareSelector(selection.DefaultSessionAwareConfig())
	sessionKey := routingSessionStateKey(ctx)
	base := time.Now()

	turns := []struct {
		record        sessiontelemetry.TurnOutcomeCategory
		wantModel     string
		wantDecision  string
		wantReason    string
		wantWindowLen int
	}{
		// Nothing observed yet: the gate cannot read a trajectory.
		{"", "cheap", selection.GateDecisionSuppress, selection.GateReasonColdStart, 0},
		// One and two attributable outcomes stay below MinWindowOutcomes.
		{sessiontelemetry.TurnProgress, "cheap", selection.GateDecisionSuppress, selection.GateReasonInsufficientEvidence, 1},
		{sessiontelemetry.TurnNoProgress, "cheap", selection.GateDecisionSuppress, selection.GateReasonInsufficientEvidence, 2},
		// Two consecutive attributable regressions over a full window clear it.
		{sessiontelemetry.TurnNoProgress, "frontier", selection.GateDecisionSwitch, "", 3},
	}

	for i, turn := range turns {
		if turn.record != "" {
			sessiontelemetry.RecordTurnOutcome(sessionKey, sessiontelemetry.TurnOutcome{
				TurnIndex: i,
				Model:     "cheap",
				Category:  turn.record,
			}, base.Add(time.Duration(i)*time.Second))
		}

		result := proposedSwitchResult()
		router.applySwitchGateToResult(cfg, ctx, learningCtx, selector, result, selection.SwitchOriginEscalation)

		gate := gateSectionFromReplayMap(t, replayPolicyForResult(ctx, result, cfg))
		if gate["decision"] != turn.wantDecision {
			t.Fatalf("turn %d decision = %v, want %v", i, gate["decision"], turn.wantDecision)
		}
		if gate["suppression_reason"] != turn.wantReason {
			t.Fatalf("turn %d suppression_reason = %v, want %q", i, gate["suppression_reason"], turn.wantReason)
		}
		if gate["window_count"] != turn.wantWindowLen {
			t.Fatalf("turn %d window_count = %v, want %d", i, gate["window_count"], turn.wantWindowLen)
		}
		if result.SelectedModel != turn.wantModel {
			t.Fatalf("turn %d selected %q, want %q", i, result.SelectedModel, turn.wantModel)
		}
		// A suppressed switch must say so in the session policy, not only in the
		// gate section, so existing replay consumers see the hold.
		wantPolicyReason := "progress_gate_suppressed"
		if turn.wantDecision == selection.GateDecisionSwitch {
			wantPolicyReason = "switch_allowed"
		}
		if result.SessionPolicy.DecisionReason != wantPolicyReason {
			t.Fatalf("turn %d decision_reason = %q, want %q", i, result.SessionPolicy.DecisionReason, wantPolicyReason)
		}
	}
}

// TestProgressGateObserveModeReplayRecordsWithoutHolding is the rollout
// guarantee: observe reasons about the same evidence and records the same
// section, but the proposal still wins.
func TestProgressGateObserveModeReplayRecordsWithoutHolding(t *testing.T) {
	router, ctx, learningCtx := gateReplayContext(t, "gate-replay-observe")
	cfg := gateReplayConfig(selection.GateModeObserve)
	selector := selection.NewSessionAwareSelector(selection.DefaultSessionAwareConfig())
	result := proposedSwitchResult()

	router.applySwitchGateToResult(cfg, ctx, learningCtx, selector, result, selection.SwitchOriginEscalation)

	gate := gateSectionFromReplayMap(t, replayPolicyForResult(ctx, result, cfg))
	if gate["mode"] != selection.GateModeObserve || gate["enforced"] != false {
		t.Fatalf("observe mode must record an unenforced verdict: %#v", gate)
	}
	if gate["decision"] != selection.GateDecisionSuppress || gate["suppression_reason"] == "" {
		t.Fatalf("observe mode must still reason about the evidence: %#v", gate)
	}
	if result.SelectedModel != "frontier" {
		t.Fatalf("observe mode must not hold the current model, got %q", result.SelectedModel)
	}
	if result.SessionPolicy.DecisionReason != "switch_allowed" {
		t.Fatalf("observe mode must leave the decision reason alone, got %q", result.SessionPolicy.DecisionReason)
	}
}

// TestProgressGateSuppressionRequiresEligibleCurrentModel locks the rule that a
// suppression only declines the proposal: it may not hold a model that is no
// longer a candidate for this request.
func TestProgressGateSuppressionRequiresEligibleCurrentModel(t *testing.T) {
	router, ctx, learningCtx := gateReplayContext(t, "gate-replay-ineligible")
	// The session owns a model the request can no longer route to.
	learningCtx.CandidateModels = []config.ModelRef{{Model: "frontier"}}
	cfg := gateReplayConfig(selection.GateModeEnforce)
	selector := selection.NewSessionAwareSelector(selection.DefaultSessionAwareConfig())
	result := proposedSwitchResult()

	router.applySwitchGateToResult(cfg, ctx, learningCtx, selector, result, selection.SwitchOriginEscalation)

	gate := gateSectionFromReplayMap(t, replayPolicyForResult(ctx, result, cfg))
	if gate["decision"] != selection.GateDecisionSuppress {
		t.Fatalf("the verdict itself must still be recorded: %#v", gate)
	}
	if result.SelectedModel != "frontier" {
		t.Fatalf("an ineligible current model must not be forced back, got %q", result.SelectedModel)
	}
	if result.SessionPolicy.DecisionReason != "switch_allowed" {
		t.Fatalf("a declined hold must not claim suppression, got %q", result.SessionPolicy.DecisionReason)
	}
}
