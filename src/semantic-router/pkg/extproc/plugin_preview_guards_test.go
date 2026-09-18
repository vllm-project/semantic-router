package extproc

import (
	"context"
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/pluginruntime"
)

func TestPluginJailbreakPreviewIsOfflineUntilExplicitProbe(t *testing.T) {
	server, calls := newJailbreakPartialFailureServer(t, 0.95, 0.05)
	router, _ := newResponseStageRouter(t, server, "block", "block")
	input := pluginruntime.ResponseJailbreakPreviewRequest{Binding: pluginruntime.Binding{Recipe: config.DefaultRecipeName, Decision: responseStageRouteName}, Response: "unsafe response"}
	preview, err := router.PreviewResponseJailbreak(context.Background(), input)
	if err != nil {
		t.Fatal(err)
	}
	if calls.scored.Load() != 0 || preview.BackendCalls || preview.Resolved || preview.Reason != "probe_required" {
		t.Fatalf("offline preview called detector or invented evidence: %+v", preview)
	}
	input.Mode = pluginruntime.ModeProbe
	probe, err := router.PreviewResponseJailbreak(context.Background(), input)
	if err != nil {
		t.Fatal(err)
	}
	if calls.scored.Load() != 1 || !probe.BackendCalls || !probe.Resolved || !probe.Detected || probe.Action != "block" || probe.DetectionSource != "response_signal" || len(probe.MatchedRules) != 1 {
		t.Fatalf("probe did not reuse response-signal contract: %+v, calls=%d", probe, calls.scored.Load())
	}
	input.Binding.Recipe = "foreign"
	if _, err := router.PreviewResponseJailbreak(context.Background(), input); !errors.Is(err, pluginruntime.ErrInvalidBinding) {
		t.Fatalf("foreign recipe lookup: %v", err)
	}
	if calls.scored.Load() != 1 {
		t.Fatal("foreign recipe caused a backend call")
	}
}

func TestPluginHallucinationPreviewHonorsContextAndFactCheckGates(t *testing.T) {
	server, calls := newHallucinationEndpointServer(t, []string{"450 meters"}, false)
	router, _ := newHallucinationSignalRouter(t, server, "body")
	input := pluginruntime.HallucinationPreviewRequest{Binding: pluginruntime.Binding{Recipe: config.DefaultRecipeName, Decision: hallucinationRouteName}, Mode: pluginruntime.ModeProbe, Question: "How tall?", Response: hallucinationAnswer, Context: hallucinationContext}
	result, err := router.PreviewHallucination(context.Background(), input)
	if err != nil {
		t.Fatal(err)
	}
	if result.Reason != "fact_check_not_needed" || calls.Load() != 0 {
		t.Fatalf("fact-check gate ignored: %+v", result)
	}
	input.FactCheckNeeded = true
	input.Mode = pluginruntime.ModePreview
	result, err = router.PreviewHallucination(context.Background(), input)
	if err != nil {
		t.Fatal(err)
	}
	if result.Reason != "probe_required" || calls.Load() != 0 {
		t.Fatalf("offline preview called detector: %+v", result)
	}
	input.Mode = pluginruntime.ModeProbe
	result, err = router.PreviewHallucination(context.Background(), input)
	if err != nil {
		t.Fatal(err)
	}
	if !result.Resolved || !result.Detected || result.Action != "body" || len(result.UnsupportedSpans) != 1 || calls.Load() != 1 {
		t.Fatalf("grounded probe result: %+v, calls=%d", result, calls.Load())
	}
	input.Context = ""
	result, err = router.PreviewHallucination(context.Background(), input)
	if err != nil {
		t.Fatal(err)
	}
	if result.Resolved || result.Reason != "grounding_context_unavailable" || result.Action != "body" || calls.Load() != 1 {
		t.Fatalf("ungrounded answer fabricated detector evidence: %+v", result)
	}
}

func TestPluginBindingReadinessNeverFallsBackAcrossRecipes(t *testing.T) {
	server, _ := newJailbreakPartialFailureServer(t, 0.95, 0.05)
	router, _ := newResponseStageRouter(t, server, "block", "block")
	decisions := router.Config.Decisions
	router.Config.Recipes = []config.RoutingRecipe{
		{Name: config.DefaultRecipeName, Profile: config.RoutingProfile{Decisions: decisions}},
		{Name: "foreign", Profile: config.RoutingProfile{Decisions: decisions}},
	}
	for _, tc := range []struct {
		recipe       config.RecipeName
		availability string
	}{{config.DefaultRecipeName, "available"}, {"foreign", "unavailable"}} {
		dependencies, err := router.InspectPluginBinding(pluginruntime.Binding{Recipe: tc.recipe, Decision: responseStageRouteName}, config.DecisionPluginResponseJailbreak)
		if err != nil {
			t.Fatal(err)
		}
		if len(dependencies) != 1 || dependencies[0].Availability != tc.availability || dependencies[0].Health != "not_probed" {
			t.Fatalf("recipe %s used foreign readiness: %+v", tc.recipe, dependencies)
		}
	}
}
