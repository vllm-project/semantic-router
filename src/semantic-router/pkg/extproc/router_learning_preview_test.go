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
