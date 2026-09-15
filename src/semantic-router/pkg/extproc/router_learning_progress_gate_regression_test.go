package extproc

import (
	"context"
	"errors"
	"fmt"
	"io"
	"reflect"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

// TestProgressStreamAbortBeforeEOSRecordsTerminalOutcome covers the three ways
// an ExtProc stream can end without reaching EOS. Each must leave exactly one
// non-attributable fact so the count-bounded window stays current.
func TestProgressStreamAbortBeforeEOSRecordsTerminalOutcome(t *testing.T) {
	for _, tc := range []struct {
		name string
		err  error
	}{
		{"cancel", context.Canceled},
		{"eof", io.EOF},
		{"error", errors.New("stream reset by peer")},
	} {
		t.Run(tc.name, func(t *testing.T) {
			sessiontelemetry.ResetRouterSessionMemoryForTesting()
			t.Cleanup(sessiontelemetry.ResetRouterSessionMemoryForTesting)

			router := &OpenAIRouter{}
			ctx := &RequestContext{
				SessionID:           tc.name,
				RequestID:           "request",
				RequestModel:        "cheap",
				IsStreamingResponse: true,
			}
			cfg := selection.DefaultProgressGateConfig()
			cfg.Enabled = true
			configureProgressEvidence(ctx, cfg, time.Now())

			// The receive loop can surface more than one error for a single
			// stream, so the capture must stay idempotent.
			_ = router.handleProcessReceiveError(ctx, tc.err)
			_ = router.handleProcessReceiveError(ctx, tc.err)

			if !ctx.StreamingAborted {
				t.Fatal("pre-EOS termination must mark the stream aborted")
			}
			window := sessiontelemetry.RecentTurnOutcomes(progressEvidenceStateKey(ctx), time.Now())
			if len(window) != 1 {
				t.Fatalf("pre-EOS termination must record exactly one outcome, got %+v", window)
			}
			if window[0].ModelAttributable {
				t.Fatalf("an aborted stream must stay non-attributable: %+v", window[0])
			}
		})
	}
}

// TestProgressCompletedStreamDoesNotGainAbortOutcome guards the other side: a
// stream that already recorded its EOS turn must not be double counted.
func TestProgressCompletedStreamDoesNotGainAbortOutcome(t *testing.T) {
	sessiontelemetry.ResetRouterSessionMemoryForTesting()
	t.Cleanup(sessiontelemetry.ResetRouterSessionMemoryForTesting)

	router := &OpenAIRouter{}
	ctx := &RequestContext{
		SessionID:           "completed",
		RequestID:           "request",
		RequestModel:        "cheap",
		IsStreamingResponse: true,
	}
	cfg := selection.DefaultProgressGateConfig()
	cfg.Enabled = true
	configureProgressEvidence(ctx, cfg, time.Now())

	// The EOS branch records the turn before the stream closes.
	recordSessionTurn(ctx, responseUsageMetrics{completionTokens: 7, completionTokensReported: true}, sessiontelemetry.TurnPricing{})
	ctx.StreamingComplete = true

	_ = router.handleProcessReceiveError(ctx, io.EOF)

	window := sessiontelemetry.RecentTurnOutcomes(progressEvidenceStateKey(ctx), time.Now())
	if len(window) != 1 {
		t.Fatalf("completed stream gained an extra outcome: %+v", window)
	}
	if window[0].Category != sessiontelemetry.TurnProgress {
		t.Fatalf("abort path overwrote the completed outcome: %+v", window[0])
	}
	if ctx.StreamingAborted {
		t.Fatal("a completed stream must not be marked aborted")
	}
}

// gateRejectedRouter enables an enforcing progress gate whose hard filters
// reject every candidate for the request, so any selector path that runs the
// learning stage ends up with a fail-closed rejection on the context.
func gateRejectedRouter(t *testing.T, sessionID string) (*OpenAIRouter, *RequestContext) {
	t.Helper()
	sessiontelemetry.ResetRouterSessionMemoryForTesting()
	t.Cleanup(sessiontelemetry.ResetRouterSessionMemoryForTesting)

	router := &OpenAIRouter{Config: routerLearningProtectionOnlyTestConfig(config.RouterLearningScopeConversation)}
	router.Config.RouterLearning.Protection.Tuning.ProgressGate = &config.ProgressGateTuning{
		Enabled: extprocBoolPtr(true),
		Mode:    selection.GateModeEnforce,
	}
	// The gate re-checks the request's hard filters, so a model that cannot
	// satisfy the context is dropped before any switch can be proposed.
	for _, model := range []string{"cheap", "frontier"} {
		params := router.Config.ModelConfig[model]
		params.ContextWindowSize = 32
		router.Config.ModelConfig[model] = params
	}

	ctx := routerLearningRequestContext(sessionID, sessionID)
	ctx.VSRContextTokenCount = 128
	return router, ctx
}

// TestSingleCandidatePathPropagatesGateHardRejection locks the fail-closed rule
// on the path that skips the selector entirely.
func TestSingleCandidatePathPropagatesGateHardRejection(t *testing.T) {
	router, ctx := gateRejectedRouter(t, "single-candidate-rejection")
	selCtx := &selection.SelectionContext{
		SessionID:       "single-candidate-rejection",
		DecisionName:    "only-choice",
		CandidateModels: []config.ModelRef{{Model: "cheap"}},
	}

	selected, _, err := router.selectModelFromCandidates(selCtx, nil, ctx)

	if !errors.Is(err, selection.ErrNoEligibleCandidates) {
		t.Fatalf("error = %v, want ErrNoEligibleCandidates", err)
	}
	if selected != nil {
		t.Fatalf("a gate-rejected single candidate was dispatched: %#v", selected)
	}
}

// TestFallbackPathPropagatesGateHardRejection covers the diagnostic fallback,
// which has no error return of its own and previously dropped the rejection.
func TestFallbackPathPropagatesGateHardRejection(t *testing.T) {
	router, ctx := gateRejectedRouter(t, "fallback-rejection")
	registry := selection.NewRegistry()
	registry.Register(
		selection.MethodStatic,
		selectionResultSelector{err: errors.New("selector unavailable")},
	)
	router.ModelSelector = registry
	selCtx := &selection.SelectionContext{
		SessionID:       "fallback-rejection",
		DecisionName:    "two-candidates",
		CandidateModels: []config.ModelRef{{Model: "cheap"}, {Model: "frontier"}},
	}

	selected, _, err := router.selectModelFromCandidates(selCtx, nil, ctx)

	if !errors.Is(err, selection.ErrNoEligibleCandidates) {
		t.Fatalf("error = %v, want ErrNoEligibleCandidates", err)
	}
	if selected != nil {
		t.Fatalf("a gate-rejected fallback was dispatched: %#v", selected)
	}
}

// TestSelectorPathsKeepDefaultBehaviorWhenGateDisabled guards the blast radius:
// with no gate configured both paths keep returning the selected model.
func TestSelectorPathsKeepDefaultBehaviorWhenGateDisabled(t *testing.T) {
	sessiontelemetry.ResetRouterSessionMemoryForTesting()
	t.Cleanup(sessiontelemetry.ResetRouterSessionMemoryForTesting)

	router := &OpenAIRouter{Config: routerLearningProtectionOnlyTestConfig(config.RouterLearningScopeConversation)}
	ctx := routerLearningRequestContext("gate-disabled", "gate-disabled")
	selCtx := &selection.SelectionContext{
		SessionID:       "gate-disabled",
		DecisionName:    "only-choice",
		CandidateModels: []config.ModelRef{{Model: "cheap"}},
	}

	selected, _, err := router.selectModelFromCandidates(selCtx, nil, ctx)
	if err != nil {
		t.Fatalf("disabled gate changed selection: %v", err)
	}
	if selected == nil || selected.Model != "cheap" {
		t.Fatalf("disabled gate must return the candidate, got %#v", selected)
	}
}

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

// sessionDecisionState is the part of a session snapshot a recorded decision
// owns. Preflight still installs the evidence window on a rejected request, so
// only these fields are required to stay untouched.
type sessionDecisionState struct {
	CurrentModel     string
	SwitchCount      int
	LastSwitchAt     time.Time
	SwitchTimestamps []int64
	TurnCount        int
	ModelTurns       map[string]int
	LastDecisionName string
}

func captureSessionDecisionState(t *testing.T, sessionID string) (sessionDecisionState, bool) {
	t.Helper()
	snapshot, ok := sessiontelemetry.GetRouterSessionSnapshot(sessionID, time.Now())
	if !ok {
		return sessionDecisionState{}, false
	}
	return sessionDecisionState{
		CurrentModel:     snapshot.CurrentModel,
		SwitchCount:      snapshot.SwitchCount,
		LastSwitchAt:     snapshot.LastSwitchAt,
		SwitchTimestamps: snapshot.SwitchTimestamps,
		TurnCount:        snapshot.TurnCount,
		ModelTurns:       snapshot.ModelTurns,
		LastDecisionName: snapshot.LastDecisionName,
	}, true
}

// seedSessionDecision records a known decision so a rejected request can be
// shown to leave it alone.
func seedSessionDecision(t *testing.T, sessionID string) sessionDecisionState {
	t.Helper()
	sessiontelemetry.RecordSessionDecision(sessiontelemetry.SessionDecisionParams{
		SessionID:     sessionID,
		SelectedModel: "frontier",
		DecisionName:  "seeded",
		TurnIndex:     2,
	})
	before, ok := captureSessionDecisionState(t, sessionID)
	if !ok || before.CurrentModel != "frontier" {
		t.Fatalf("seed did not take for %q: %+v (ok=%t)", sessionID, before, ok)
	}
	return before
}

func requireSessionDecisionUnchanged(t *testing.T, sessionID string, before sessionDecisionState) {
	t.Helper()
	after, ok := captureSessionDecisionState(t, sessionID)
	if !ok {
		t.Fatalf("the rejected request dropped session %q", sessionID)
	}
	if !reflect.DeepEqual(before, after) {
		t.Fatalf("a rejected request rewrote the session decision:\nbefore %+v\nafter  %+v", before, after)
	}
}

// TestRejectedSelectionLeavesSessionDecisionUnchanged covers all three selector
// paths. A gate-rejected request returns no model, so it must not become the
// session's recorded decision: that write would move the current model, append
// a switch timestamp and restart the cooldown later turns route on.
func TestRejectedSelectionLeavesSessionDecisionUnchanged(t *testing.T) {
	for _, tc := range []struct {
		name string
		// candidates picks the path: a sole candidate skips the selector, and
		// two keep the selector in play.
		candidates []config.ModelRef
		session    string
		// prepare registers the selector the path needs; without a registry the
		// lookup fails into the diagnostic fallback.
		prepare func(*testing.T, *OpenAIRouter)
	}{
		{
			name:       "single candidate",
			candidates: []config.ModelRef{{Model: "cheap"}},
			session:    "reject-state-single",
		},
		{
			name:       "selector fallback",
			candidates: []config.ModelRef{{Model: "cheap"}, {Model: "frontier"}},
			session:    "reject-state-fallback",
			prepare: func(_ *testing.T, router *OpenAIRouter) {
				registry := selection.NewRegistry()
				registry.Register(
					selection.MethodStatic,
					selectionResultSelector{err: errors.New("selector unavailable")},
				)
				router.ModelSelector = registry
			},
		},
		{
			name:       "selector result",
			candidates: []config.ModelRef{{Model: "cheap"}, {Model: "frontier"}},
			session:    "reject-state-selector",
			prepare: func(_ *testing.T, router *OpenAIRouter) {
				registry := selection.NewRegistry()
				registry.Register(selection.MethodStatic, selectionResultSelector{
					result: &selection.SelectionResult{
						SelectedModel: "frontier",
						Method:        selection.MethodStatic,
					},
				})
				router.ModelSelector = registry
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			router, ctx := gateRejectedRouter(t, tc.session)
			if tc.prepare != nil {
				tc.prepare(t, router)
			}
			before := seedSessionDecision(t, tc.session)
			selCtx := &selection.SelectionContext{
				SessionID:       tc.session,
				DecisionName:    "gate-rejected",
				CandidateModels: tc.candidates,
			}

			selected, _, err := router.selectModelFromCandidates(selCtx, nil, ctx)

			if !errors.Is(err, selection.ErrNoEligibleCandidates) {
				t.Fatalf("error = %v, want ErrNoEligibleCandidates", err)
			}
			if selected != nil {
				t.Fatalf("a gate-rejected selection was dispatched: %#v", selected)
			}
			requireSessionDecisionUnchanged(t, tc.session, before)
		})
	}
}
