package extproc

import (
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

const releaseTestHeader = "x-vsr-protection-release"

type releaseRequest struct {
	agent    string
	proposal string
	release  bool
	toolLoop bool
	decision *config.Decision
}

type releaseHarness struct {
	t        *testing.T
	router   *OpenAIRouter
	registry *selection.Registry
}

func newReleaseHarness(t *testing.T, cfg *config.RouterConfig) *releaseHarness {
	t.Helper()
	sessiontelemetry.ResetRouterSessionMemoryForTesting()
	t.Cleanup(sessiontelemetry.ResetRouterSessionMemoryForTesting)
	registry := selection.NewRegistry()
	return &releaseHarness{t: t, router: &OpenAIRouter{Config: cfg, ModelSelector: registry}, registry: registry}
}

func (h *releaseHarness) send(req releaseRequest) (string, *RequestContext) {
	h.t.Helper()
	model, ctx, err := h.stage(req)
	if err != nil || model == "" {
		h.t.Fatalf("selection failed for %+v: model=%q err=%v", req, model, err)
	}
	if err := commitAgenticSessionDecision(ctx); err != nil {
		h.t.Fatal(err)
	}
	return model, ctx
}

func (h *releaseHarness) stage(req releaseRequest) (string, *RequestContext, error) {
	h.t.Helper()
	ctx := routerLearningRequestContext("run-1", "conv-1")
	if req.agent != "" {
		ctx.Headers["x-claude-code-agent-id"] = req.agent
	}
	if req.release {
		ctx.Headers[releaseTestHeader] = "1"
	}
	ctx.VSRSelectedDecision = req.decision
	if ctx.VSRSelectedDecision == nil {
		ctx.VSRSelectedDecision = &config.Decision{Name: "choice"}
	}
	ctx.VSRConversationFacts = classification.ConversationFacts{LastMessageToolResult: req.toolLoop}
	h.registry.Register(selection.MethodStatic, selectionResultSelector{result: &selection.SelectionResult{
		SelectedModel: req.proposal, Score: 1, Confidence: 1, Method: selection.MethodStatic,
		AllScores: map[string]float64{req.proposal: 1},
	}})
	both := []config.ModelRef{{Model: "cheap"}, {Model: "frontier"}}
	selCtx := h.router.buildSelectionContext(both, ctx.VSRSelectedDecision.Name, "", nil, "", nil, ctx)
	selected, _, err := h.router.selectModelFromCandidates(selCtx, nil, ctx)
	if selected == nil {
		return "", ctx, err
	}
	return selected.Model, ctx, err
}

func releaseBlock(t *testing.T, ctx *RequestContext) map[string]interface{} {
	t.Helper()
	block, _ := ctx.VSRLearningPolicy.ToMap()["release"].(map[string]interface{})
	return block
}

func releaseWarnings(ctx *RequestContext) []string {
	var reasons []string
	for _, diagnostic := range ctx.ProtocolDiagnostics {
		if diagnostic.Field == releaseTestHeader {
			reasons = append(reasons, diagnostic.Reason)
		}
	}
	return reasons
}

func TestProtectionReleaseSwitchesAndHoldsNewModel(t *testing.T) {
	h := newReleaseHarness(t, routerLearningProtectionOnlyTestConfig(config.RouterLearningScopeSession))
	if model, _ := h.send(releaseRequest{proposal: "frontier"}); model != "frontier" {
		t.Fatalf("first request got %q", model)
	}
	if model, _ := h.send(releaseRequest{proposal: "cheap"}); model != "frontier" {
		t.Fatalf("hold broken without release: %q", model)
	}
	model, ctx := h.send(releaseRequest{proposal: "cheap", release: true})
	if model != "cheap" {
		t.Fatalf("release did not reselect: %q", model)
	}
	if ctx.VSRLearningPolicy.Action != routerLearningActionReleaseSwitch || ctx.VSRLearningPolicy.Reason != "explicit_release" {
		t.Fatalf("policy = %s/%s", ctx.VSRLearningPolicy.Action, ctx.VSRLearningPolicy.Reason)
	}
	block := releaseBlock(t, ctx)
	if block["honored"] != true || block["previous_model"] != "frontier" || block["selected_model"] != "cheap" {
		t.Fatalf("release block = %#v", block)
	}
	if replay := ctx.VSRLearningPolicy.toReplayProtection(); replay.Release == nil || !replay.Release.Honored || replay.Release.PreviousModel != "frontier" {
		t.Fatalf("replay release = %#v", replay.Release)
	}
	if warnings := releaseWarnings(ctx); len(warnings) != 0 {
		t.Fatalf("honored release warned: %v", warnings)
	}
	if model, _ := h.send(releaseRequest{proposal: "frontier"}); model != "cheap" {
		t.Fatalf("new model not held after release: %q", model)
	}
}

func TestProtectionReleaseScopedToAgent(t *testing.T) {
	cfg := routerLearningProtectionOnlyTestConfig(config.RouterLearningScopeSession)
	cfg.RouterLearning.Protection.Identity.Headers.Agent = stringPtr("x-claude-code-agent-id")
	h := newReleaseHarness(t, cfg)
	h.send(releaseRequest{proposal: "frontier"})
	h.send(releaseRequest{agent: "agent-a", proposal: "frontier"})
	h.send(releaseRequest{agent: "agent-a", proposal: "frontier"})
	if model, _ := h.send(releaseRequest{agent: "agent-a", proposal: "cheap", release: true}); model != "cheap" {
		t.Fatalf("agent release did not reselect: %q", model)
	}
	if model, _ := h.send(releaseRequest{proposal: "cheap"}); model != "frontier" {
		t.Fatalf("main agent lost its hold after a subagent release: %q", model)
	}
}

func TestProtectionReleaseRejectedInToolLoop(t *testing.T) {
	h := newReleaseHarness(t, routerLearningProtectionOnlyTestConfig(config.RouterLearningScopeSession))
	h.send(releaseRequest{proposal: "frontier"})
	model, ctx := h.send(releaseRequest{proposal: "cheap", release: true, toolLoop: true})
	if model != "frontier" {
		t.Fatalf("release honored inside a tool loop: %q", model)
	}
	if warnings := releaseWarnings(ctx); len(warnings) != 1 || warnings[0] != "release_not_at_turn_boundary" {
		t.Fatalf("warnings = %v", warnings)
	}
	if block := releaseBlock(t, ctx); block["honored"] != false || block["reason"] != "release_not_at_turn_boundary" {
		t.Fatalf("release block = %#v", block)
	}
	if model, _ := h.send(releaseRequest{proposal: "cheap", release: true}); model != "cheap" {
		t.Fatalf("rejected release consumed the interval: %q", model)
	}
}

func TestProtectionReleaseIntervalCountsUserTurns(t *testing.T) {
	h := newReleaseHarness(t, routerLearningProtectionOnlyTestConfig(config.RouterLearningScopeSession))
	h.send(releaseRequest{proposal: "frontier"})
	if model, _ := h.send(releaseRequest{proposal: "cheap", release: true}); model != "cheap" {
		t.Fatalf("first release not honored: %q", model)
	}
	h.send(releaseRequest{proposal: "frontier", toolLoop: true})
	model, ctx := h.send(releaseRequest{proposal: "frontier", release: true})
	if model != "cheap" {
		t.Fatalf("release honored one user turn later: %q", model)
	}
	if warnings := releaseWarnings(ctx); len(warnings) != 1 || warnings[0] != "release_interval" {
		t.Fatalf("warnings = %v", warnings)
	}
	if model, _ := h.send(releaseRequest{proposal: "frontier", release: true}); model != "frontier" {
		t.Fatalf("release not honored after two user turns: %q", model)
	}
}

func TestProtectionReleaseDisabled(t *testing.T) {
	cfg := routerLearningProtectionOnlyTestConfig(config.RouterLearningScopeSession)
	cfg.RouterLearning.Protection.Release.Enabled = extprocBoolPtr(false)
	h := newReleaseHarness(t, cfg)
	h.send(releaseRequest{proposal: "frontier"})
	model, ctx := h.send(releaseRequest{proposal: "cheap", release: true})
	if model != "frontier" {
		t.Fatalf("disabled release switched: %q", model)
	}
	if warnings := releaseWarnings(ctx); len(warnings) != 1 || warnings[0] != "release_disabled" {
		t.Fatalf("warnings = %v", warnings)
	}
}

func TestProtectionReleaseIgnoredWithoutHold(t *testing.T) {
	h := newReleaseHarness(t, routerLearningProtectionOnlyTestConfig(config.RouterLearningScopeSession))
	model, ctx := h.send(releaseRequest{proposal: "frontier", release: true})
	if model != "frontier" || len(releaseWarnings(ctx)) != 0 || releaseBlock(t, ctx) != nil {
		t.Fatalf("new session release: model=%q warnings=%v block=%#v", model, releaseWarnings(ctx), releaseBlock(t, ctx))
	}
	bypass := &config.Decision{Name: "choice", Adaptations: config.DecisionAdaptationsConfig{Mode: config.DecisionAdaptationModeBypass}}
	_, ctx = h.send(releaseRequest{proposal: "cheap", release: true, decision: bypass})
	if len(releaseWarnings(ctx)) != 0 || releaseBlock(t, ctx) != nil {
		t.Fatalf("bypass release: warnings=%v block=%#v", releaseWarnings(ctx), releaseBlock(t, ctx))
	}
}

func TestProtectionReleaseBypassesSwitchGateSoftRules(t *testing.T) {
	newConfig := func() *config.RouterConfig {
		cfg := routerLearningProtectionOnlyTestConfig(config.RouterLearningScopeConversation)
		cfg.RouterLearning.Protection.Tuning = config.RouterLearningProtectionTuning{
			MinTurnsBeforeSwitch: extprocIntPtr(0),
			SwitchMargin:         extprocFloat64Ptr(0),
			StabilityWeight:      extprocFloat64Ptr(0),
			ProgressGate:         &config.ProgressGateTuning{Enabled: extprocBoolPtr(true), Mode: "enforce"},
		}
		return cfg
	}
	h := newReleaseHarness(t, newConfig())
	h.send(releaseRequest{proposal: "frontier"})
	model, ctx := h.send(releaseRequest{proposal: "cheap"})
	gate := ctx.VSRLearningPolicy.Details.Protection.trace.SwitchGate
	if model != "frontier" || gate == nil || gate.Reason != selection.GateReasonColdStart {
		t.Fatalf("expected cold-start suppression without release: model=%q gate=%#v", model, gate)
	}

	model, ctx = h.send(releaseRequest{proposal: "cheap", release: true})
	gate = ctx.VSRLearningPolicy.Details.Protection.trace.SwitchGate
	if model != "cheap" {
		t.Fatalf("release suppressed by the switch gate: %q", model)
	}
	if gate == nil || gate.Reason != "bypassed_by_release" {
		t.Fatalf("gate trace = %#v", gate)
	}
}

func TestProtectionReleaseObserveModeRecordsOnly(t *testing.T) {
	h := newReleaseHarness(t, routerLearningProtectionOnlyTestConfig(config.RouterLearningScopeSession))
	h.send(releaseRequest{proposal: "frontier"})
	observe := &config.Decision{Name: "choice", Adaptations: config.DecisionAdaptationsConfig{
		Protection: &config.DecisionLearningProtectionConfig{Mode: config.DecisionAdaptationModeObserve},
	}}
	_, ctx := h.send(releaseRequest{proposal: "cheap", release: true, decision: observe})
	if ctx.VSRLearningPolicy.Action == routerLearningActionReleaseSwitch {
		t.Fatalf("observe mode applied a release: %s", ctx.VSRLearningPolicy.Action)
	}
	if block := releaseBlock(t, ctx); block["honored"] != true {
		t.Fatalf("observe release block = %#v", block)
	}
	snapshot, _ := sessiontelemetry.GetRouterSessionSnapshot(ctx.VSRLearningSessionID, time.Now())
	if snapshot.LastReleaseUserTurn != 0 {
		t.Fatalf("observe mode consumed the release interval: %d", snapshot.LastReleaseUserTurn)
	}
}

func TestProtectionReleaseIgnoredOnLooperHop(t *testing.T) {
	session := &selection.AgenticSessionContext{PreviousModel: "frontier"}
	ctx := &RequestContext{Headers: map[string]string{releaseTestHeader: "1"}}
	if verdict := evaluateProtectionRelease(ctx, config.RouterLearningProtectionConfig{}, session); !verdict.honored {
		t.Fatalf("client request release not honored: %+v", verdict)
	}
	ctx.Hop = &routing.Hop{}
	if verdict := evaluateProtectionRelease(ctx, config.RouterLearningProtectionConfig{}, session); verdict.requested {
		t.Fatalf("looper hop requested a release: %+v", verdict)
	}
}

func TestProtectionReleaseNotReportedWhenSelectionFails(t *testing.T) {
	cfg := routerLearningProtectionOnlyTestConfig(config.RouterLearningScopeSession)
	cfg.RouterLearning.Protection.Tuning.ProgressGate = &config.ProgressGateTuning{Enabled: extprocBoolPtr(true), Mode: "enforce"}
	h := newReleaseHarness(t, cfg)
	h.send(releaseRequest{proposal: "frontier"})
	h.send(releaseRequest{proposal: "frontier"})
	// No configured model passes the progress gate's hard filters.
	cfg.ModelConfig = map[string]config.ModelParams{}
	cfg.DefaultModel = ""
	_, ctx, _ := h.stage(releaseRequest{proposal: "cheap", release: true})
	if ctx.VSRProgressGateError == nil {
		t.Fatal("expected the progress gate to reject every candidate")
	}
	if ctx.VSRLearningPolicy.Action == routerLearningActionReleaseSwitch {
		t.Fatalf("failed selection reported as %s", ctx.VSRLearningPolicy.Action)
	}
	if block := releaseBlock(t, ctx); block != nil && block["honored"] == true {
		t.Fatalf("failed selection reported an honored release: %#v", block)
	}
}
