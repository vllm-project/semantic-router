package extproc

import (
	"fmt"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

func TestProgressWindowConfiguredBeforeFirstSwitch(t *testing.T) {
	sessiontelemetry.ResetRouterSessionMemoryForTesting()
	t.Cleanup(sessiontelemetry.ResetRouterSessionMemoryForTesting)
	router := &OpenAIRouter{Config: routerLearningProtectionOnlyTestConfig(config.RouterLearningScopeConversation)}
	router.Config.RouterLearning.Protection.Tuning.ProgressGate = &config.ProgressGateTuning{
		Enabled: extprocBoolPtr(true), WindowSize: extprocIntPtr(16), WindowTTLSeconds: extprocIntPtr(3600),
	}
	selCtx := &selection.SelectionContext{CandidateModels: []config.ModelRef{{Model: "cheap"}}}
	for i := 0; i < 12; i++ {
		ctx := routerLearningRequestContext("pre-switch", "conv")
		ctx.RequestID, ctx.RequestModel = fmt.Sprint(i), "cheap"
		router.applyProtectionPreflight(routerLearningInput{ctx: ctx, selCtx: selCtx, baseResult: &selection.SelectionResult{SelectedModel: "cheap"}, selectedModelRef: &selCtx.CandidateModels[0]})
		recordSessionTurn(ctx, responseUsageMetrics{completionTokensReported: true, completionTokens: 1}, sessiontelemetry.TurnPricing{})
	}
	key := config.RoutingNamespaceKey("", "pre-switch")
	if got := sessiontelemetry.RecentTurnOutcomesWithPolicy(key, time.Now(), 16, time.Hour); len(got) != 12 {
		t.Fatalf("first switch would have lost configured history: %+v", got)
	}
}

func TestProgressTerminalCaptureWithoutBillingUsage(t *testing.T) {
	for _, tc := range []struct {
		name   string
		status int
		usage  responseUsageMetrics
		want   sessiontelemetry.TurnOutcomeCategory
	}{
		{"provider", 503, responseUsageMetrics{}, sessiontelemetry.TurnProviderError},
		{"missing", 200, responseUsageMetrics{}, sessiontelemetry.TurnMissing},
		{"empty", 200, responseUsageMetrics{completionTokensReported: true}, sessiontelemetry.TurnNoProgress},
		{"invalid", 200, responseUsageMetrics{invalid: true}, sessiontelemetry.TurnMissing},
	} {
		t.Run(tc.name, func(t *testing.T) {
			sessiontelemetry.ResetRouterSessionMemoryForTesting()
			ctx := &RequestContext{SessionID: tc.name, RequestID: "request", RequestModel: "cheap", UpstreamStatusCode: tc.status}
			cfg := selection.DefaultProgressGateConfig()
			cfg.Enabled = true
			configureProgressEvidence(ctx, cfg, time.Now())
			recordSessionTurn(ctx, tc.usage, sessiontelemetry.TurnPricing{})
			recordSessionTurn(ctx, tc.usage, sessiontelemetry.TurnPricing{})
			window := sessiontelemetry.RecentTurnOutcomes(progressEvidenceStateKey(ctx), time.Now())
			if len(window) != 1 || window[0].Category != tc.want {
				t.Fatalf("terminal capture missing or duplicated: %+v", window)
			}
		})
	}
	sessiontelemetry.ResetRouterSessionMemoryForTesting()
	ctx := &RequestContext{SessionID: "disabled", RequestModel: "cheap"}
	recordSessionTurn(ctx, responseUsageMetrics{}, sessiontelemetry.TurnPricing{})
	if got := sessiontelemetry.RecentTurnOutcomes(routingSessionStateKey(ctx), time.Now()); len(got) != 0 {
		t.Fatal("disabled gate recorded evidence")
	}
}

func TestProgressHoldCannotRestoreContextExcludedModel(t *testing.T) {
	router, ctx, learningCtx := gateReplayContext(t, "excluded-hold")
	params := router.Config.ModelConfig["cheap"]
	params.ContextWindowSize = 32
	router.Config.ModelConfig["cheap"] = params
	ctx.VSRContextTokenCount = 128
	result := proposedSwitchResult()
	router.applySwitchGateToResult(gateReplayConfig(selection.GateModeEnforce), ctx, learningCtx, selection.NewSessionAwareSelector(nil), result)
	g := result.SessionPolicy.SwitchGate
	if result.SelectedModel != "frontier" || g == nil || g.Applied || g.ApplicationReason != "current_ineligible:context_limit" {
		t.Fatalf("illegal hold or dishonest trace: result=%+v trace=%+v", result, g)
	}
}

func TestProgressHoldRechecksBudget(t *testing.T) {
	router, ctx, learningCtx := gateReplayContext(t, "budget-hold")
	params := router.Config.ModelConfig["cheap"]
	params.Pricing = config.ModelPricing{PromptPer1M: 20, CompletionPer1M: 20}
	router.Config.ModelConfig["cheap"] = params
	ctx.VSRSelectedDecision = &config.Decision{Algorithm: &config.AlgorithmConfig{
		Type: "multi_factor", MultiFactor: &config.MultiFactorSelectionConfig{
			SLO: &config.MultiFactorSLOConfig{MaxCostPer1M: 5},
		},
	}}
	result := proposedSwitchResult()
	router.applySwitchGateToResult(gateReplayConfig(selection.GateModeEnforce), ctx, learningCtx, selection.NewSessionAwareSelector(nil), result)
	g := result.SessionPolicy.SwitchGate
	if result.SelectedModel != "frontier" || g == nil || g.Applied || g.ApplicationReason != "current_ineligible:slo_or_quality" {
		t.Fatalf("budget-excluded model restored: result=%+v trace=%+v", result, g)
	}
}

func TestProgressHardLockWinsInObserve(t *testing.T) {
	router, ctx, learningCtx := gateReplayContext(t, "observe-hardlock")
	learningCtx.AgenticSession.ActiveToolLoop = true
	result := proposedSwitchResult()
	router.applySwitchGateToResult(gateReplayConfig(selection.GateModeObserve), ctx, learningCtx, selection.NewSessionAwareSelector(nil), result)
	g := result.SessionPolicy.SwitchGate
	if result.SelectedModel != "cheap" || g == nil || !g.Applied || g.Reason != "active_tool_loop" {
		t.Fatalf("observe bypassed a hard lock: %+v / %+v", result, g)
	}
}

func TestProgressRejectedRescueSurvivesFinalPolicy(t *testing.T) {
	router, ctx, learningCtx := gateReplayContext(t, "rescue-trace")
	ctx.VSRSelectedDecision = &config.Decision{Name: "rescue"}
	learningCtx.DecisionName = "rescue"
	for i := 0; i < 2; i++ {
		router.routerLearningRuntimeState().recordModelExperience("rescue", 0, "cheap", routerLearningOutcomeUnderpowered, 0.1)
	}
	cfg := gateReplayConfig(selection.GateModeEnforce)
	proposal := proposedSwitchResult()
	proposal.Score = 1
	rescue, ok := router.protectionRescueDecision(routerLearningInput{ctx: ctx, baseResult: proposal}, learningCtx, routerLearningProtectionPreflight{config: cfg, mode: config.DecisionAdaptationModeApply}, routerLearningDecision{selectionResult: proposal})
	if ok || rescue.selectionResult == nil {
		t.Fatalf("want a rejected rescue verdict, got %+v/%t", rescue, ok)
	}
	final := &selection.SelectionResult{SelectedModel: "cheap"}
	attachRejectedRescue(final, rescue.selectionResult)
	if _, ok := final.SessionPolicy.ToMap()["rescue_switch_gate"]; !ok {
		t.Fatal("rejected rescue disappeared from final replay")
	}
}

func TestProgressEvidenceKeyUsesResponseTrackingIdentity(t *testing.T) {
	ctx := &RequestContext{SessionID: "request-id", ResponseObjectState: &ResponseObjectState{SessionTrackingID: "stable-session"}}
	if key := progressEvidenceStateKey(ctx); key != config.RoutingNamespaceKey("", "stable-session") {
		t.Fatalf("unstable Responses evidence key: %s", key)
	}
}
