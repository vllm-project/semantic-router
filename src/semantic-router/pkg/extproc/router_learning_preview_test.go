package extproc

import (
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

func previewProtectionFixture() (*OpenAIRouter, *config.Decision) {
	cfg := routerLearningProtectionOnlyTestConfig(config.RouterLearningScopeConversation)
	cfg.DocumentHash = strings.Repeat("a", 64)
	cfg.ModelConfig["cheap"] = modelParamsWithTestQuality(0.1)
	cfg.ModelConfig["frontier"] = modelParamsWithTestQuality(0.9)
	router := &OpenAIRouter{Config: cfg}
	d := &config.Decision{Name: "preview-route", ModelRefs: []config.ModelRef{{Model: "cheap"}, {Model: "frontier"}}, Algorithm: &config.AlgorithmConfig{Type: "multi_factor", MultiFactor: &config.MultiFactorSelectionConfig{Weights: &config.MultiFactorWeightsConfig{Quality: 1}}}}
	return router, d
}

func TestLearningPreviewProtectionMatchesLiveWithoutUpdatingState(t *testing.T) {
	sessiontelemetry.ResetRouterSessionMemoryForTesting()
	t.Cleanup(sessiontelemetry.ResetRouterSessionMemoryForTesting)
	router, d := previewProtectionFixture()
	ctx := routerLearningRequestContext("preview-session", "preview-conversation")
	ctx.VSRSelectedDecision = d
	ctx.VSRConversationFacts = classification.ConversationFacts{LastMessageToolResult: true}
	identity, ok := router.protectionIdentity(ctx, router.Config.RouterLearning.Protection)
	if !ok {
		t.Fatal("identity missing")
	}
	at := time.Now()
	sessiontelemetry.RecordSessionDecision(sessiontelemetry.SessionDecisionParams{SessionID: identity.memoryKey, SelectedModel: "cheap", SelectedCandidate: &d.ModelRefs[0], DecisionName: d.Name, Timestamp: at})
	before, _ := sessiontelemetry.PeekRouterSessionSnapshot(identity.memoryKey, at)
	input := services.EvalModelSelectionInput{Decision: d, Query: "Continue tool execution", PreviewContext: &services.PreviewContext{SessionID: "preview-session", ConversationID: "preview-conversation"}, ConversationFacts: ctx.VSRConversationFacts}
	preview := router.SelectModelForEval(input)
	if preview.Status != services.EvalSelectionSelected || preview.SelectedModel != "cheap" {
		t.Fatalf("protection preview = %+v", preview)
	}
	if preview.Provenance == nil || preview.Provenance.Sampled || !preview.Provenance.StateDependent || len(preview.Provenance.StateHash) != 64 || preview.Provenance.ConfigHash != router.Config.DocumentHash {
		t.Fatalf("provenance = %+v", preview.Provenance)
	}
	after, _ := sessiontelemetry.PeekRouterSessionSnapshot(identity.memoryKey, at)
	if !reflect.DeepEqual(before, after) {
		t.Fatal("preview mutated session ownership/counters")
	}
	if router.routerLearningRuntime != nil {
		t.Fatal("preview initialized live learning state")
	}
	base := (&selection.SelectionResult{Score: 0.9, AllScores: map[string]float64{"cheap": 0.1, "frontier": 0.9}}).WithCandidate(d.ModelRefs[1])
	_, _, live, _, err := router.applyRouterLearning(&selection.SelectionContext{DecisionName: d.Name, CandidateModels: d.ModelRefs}, base, &d.ModelRefs[1], ctx)
	if err != nil || live == nil || live.Model != preview.SelectedModel {
		t.Fatalf("live selection = %v, %v; preview=%s", live, err, preview.SelectedModel)
	}
}

func TestLearningPreviewMissingIdentityIsActualProtectionNoop(t *testing.T) {
	router, d := previewProtectionFixture()
	output := router.SelectModelForEval(services.EvalModelSelectionInput{Decision: d})
	if output.SelectedModel != "frontier" || output.Status != services.EvalSelectionSelected || output.Provenance == nil || !output.Provenance.StateDependent {
		t.Fatalf("missing identity = %+v", output)
	}
}

func TestLearningPreviewSamplingUsesOnlyLocalSeed(t *testing.T) {
	router, d := previewProtectionFixture()
	router.Config.RouterLearning.Adaptation.Enabled = extprocBoolPtr(true)
	router.Config.RouterLearning.Protection.Enabled = extprocBoolPtr(false)
	original := routerLearningSamplingSeedSource
	routerLearningSamplingSeedSource = func() int64 { t.Fatal("preview consumed production random source"); return 0 }
	t.Cleanup(func() { routerLearningSamplingSeedSource = original })
	seed := int64(73)
	input := services.EvalModelSelectionInput{Decision: d, PreviewContext: &services.PreviewContext{SamplingSeed: &seed}}
	first := router.SelectModelForEval(input)
	second := router.SelectModelForEval(input)
	if first.Status != services.EvalSelectionSelected || first.SelectedModel != second.SelectedModel {
		t.Fatalf("preview sampling %v / %v", first, second)
	}
	p := first.Provenance
	if p == nil || !p.Sampled || p.SamplingSeed == nil || *p.SamplingSeed != seed || !strings.Contains(p.Caveat, "does not predict") {
		t.Fatalf("sampling provenance=%+v", p)
	}
	if router.routerLearningRuntime != nil {
		t.Fatal("preview initialized mutable learning runtime")
	}
}

func TestLearningPreviewExperienceRemainsFrozenAfterCapture(t *testing.T) {
	router, _ := previewProtectionFixture()
	runtime := router.routerLearningRuntimeState()
	runtime.recordModelExperience("preview-route", 0, "cheap", routerLearningOutcomeGoodFit, 1)
	snapshot, err := router.newLearningPreview(0)
	if err != nil {
		t.Fatal(err)
	}
	runtime.recordModelExperience("preview-route", 0, "cheap", routerLearningOutcomeFailed, 3)
	captured := router.learningExperience(&RequestContext{learningPreview: snapshot}, "preview-route", 0, "cheap")
	if captured.FailedCount != 0 || captured.GoodFitCount != 1 {
		t.Fatalf("captured state changed: %+v", captured)
	}
	if live := runtime.experienceSnapshot("preview-route", 0, "cheap"); live.FailedCount != 3 {
		t.Fatal("live state update missing")
	}
}

func TestLearningPreviewProgressGatePreservesLiveWindow(t *testing.T) {
	sessiontelemetry.ResetRouterSessionMemoryForTesting()
	t.Cleanup(sessiontelemetry.ResetRouterSessionMemoryForTesting)
	router, d := previewProtectionFixture()
	router.Config.RouterLearning.Protection.Tuning.ProgressGate = &config.ProgressGateTuning{
		Enabled: extprocBoolPtr(true), Mode: selection.GateModeEnforce, WindowSize: extprocIntPtr(1),
	}
	live := routerLearningRequestContext("gate-preview", "gate-preview")
	live.VSRSelectedDecision = d
	identity, ok := router.protectionIdentity(live, router.Config.RouterLearning.Protection)
	if !ok {
		t.Fatal("missing identity")
	}
	at := time.Now().Add(-time.Second)
	sessiontelemetry.RecordSessionDecision(sessiontelemetry.SessionDecisionParams{SessionID: identity.memoryKey, SelectedModel: "cheap", SelectedCandidate: &d.ModelRefs[0], Timestamp: at})
	for i := 0; i < 3; i++ {
		sessiontelemetry.RecordTurnOutcome(identity.memoryKey, sessiontelemetry.TurnOutcome{TurnIndex: i, Model: "cheap", Category: sessiontelemetry.TurnProgress}, at.Add(time.Duration(i)*time.Millisecond))
	}
	before, _ := sessiontelemetry.PeekRouterSessionSnapshot(identity.memoryKey, at)
	preview := router.SelectModelForEval(services.EvalModelSelectionInput{Decision: d, PreviewContext: &services.PreviewContext{SessionID: "gate-preview", ConversationID: "gate-preview"}})
	if preview.Status != services.EvalSelectionSelected || preview.SelectedModel != "cheap" || preview.Provenance == nil || !preview.Provenance.StateDependent {
		t.Fatalf("progress-gated preview = %+v", preview)
	}
	after, _ := sessiontelemetry.PeekRouterSessionSnapshot(identity.memoryKey, at)
	if !reflect.DeepEqual(before, after) {
		t.Fatal("preview changed the live window, policy or session state")
	}
	if router.routerLearningRuntime != nil {
		t.Fatal("protection-only preview enabled mutable adaptation")
	}
	base := (&selection.SelectionResult{Score: 0.9, AllScores: map[string]float64{"cheap": 0.1, "frontier": 0.9}}).WithCandidate(d.ModelRefs[1])
	_, _, actual, _, err := router.applyRouterLearning(&selection.SelectionContext{DecisionName: d.Name, CandidateModels: d.ModelRefs}, base, &d.ModelRefs[1], live)
	if err != nil || actual == nil || actual.Model != preview.SelectedModel {
		t.Fatalf("live=%+v error=%v preview=%+v", actual, err, preview)
	}
}

func TestLearningPreviewProgressEvidenceRemainsFrozen(t *testing.T) {
	sessiontelemetry.ResetRouterSessionMemoryForTesting()
	t.Cleanup(sessiontelemetry.ResetRouterSessionMemoryForTesting)
	router, _ := previewProtectionFixture()
	cfg := router.Config.RouterLearning.Protection
	cfg.Tuning.ProgressGate = &config.ProgressGateTuning{Enabled: extprocBoolPtr(true), Mode: selection.GateModeEnforce}
	live := &RequestContext{SessionID: "frozen-progress"}
	key := routingLearningStateKey(live)
	at := time.Now().Add(-time.Second)
	for i := 0; i < 3; i++ {
		sessiontelemetry.RecordTurnOutcome(key, sessiontelemetry.TurnOutcome{TurnIndex: i, Model: "cheap", Category: sessiontelemetry.TurnProgress}, at.Add(time.Duration(i)*time.Millisecond))
	}
	snapshot, err := router.newLearningPreview(0)
	if err != nil {
		t.Fatal(err)
	}
	ctx := &RequestContext{SessionID: live.SessionID, learningPreview: snapshot}
	first, trace, ran := router.switchGateVerdict(cfg, ctx, nil, "cheap", "frontier", false)
	if !ran || !first.Suppressed() || trace.WindowCount != 3 {
		t.Fatalf("initial gate=%+v trace=%+v", first, trace)
	}
	for i := 0; i < 3; i++ {
		sessiontelemetry.RecordTurnOutcome(key, sessiontelemetry.TurnOutcome{TurnIndex: 3 + i, Model: "cheap", Category: sessiontelemetry.TurnRegression}, time.Now())
	}
	second, secondTrace, _ := router.switchGateVerdict(cfg, ctx, nil, "cheap", "frontier", false)
	if !reflect.DeepEqual(first, second) || !reflect.DeepEqual(trace, secondTrace) {
		t.Fatal("preview consumed outcomes written after capture")
	}
	actual, liveTrace, _ := router.switchGateVerdict(cfg, live, nil, "cheap", "frontier", false)
	if actual.Suppressed() || liveTrace.RegressionStreak != 3 {
		t.Fatalf("live gate ignored new evidence: %+v %+v", actual, liveTrace)
	}
}

func TestLearningPreviewPropagatesProgressGateHardRejection(t *testing.T) {
	router, ctx := gateRejectedRouter(t, "preview-hard-rejection")
	snapshot, err := router.newLearningPreview(0)
	if err != nil {
		t.Fatal(err)
	}
	ctx.learningPreview = snapshot
	refs := []config.ModelRef{{Model: "cheap"}, {Model: "frontier"}}
	base := (&selection.SelectionResult{Score: 1}).WithCandidate(refs[1])
	output := router.finishEvalLearning(ctx, &selection.SelectionContext{CandidateModels: refs}, base, &refs[1], "multi_factor")
	if ctx.VSRProgressGateError == nil || output.Status == services.EvalSelectionSelected || !strings.Contains(output.Reason, "progress gate hard filters") {
		t.Fatalf("hard-rejected preview=%+v, gate error=%v", output, ctx.VSRProgressGateError)
	}
}
