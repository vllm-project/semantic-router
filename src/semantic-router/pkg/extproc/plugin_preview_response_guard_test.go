package extproc

import (
	"context"
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/pluginruntime"
)

func TestResponseGuardPreviewIsOfflineUntilExplicitProbe(t *testing.T) {
	server, calls := newCountingJailbreakDecider(0.95)
	router, _ := newResponseStageRouter(t, server, "", config.ResponseGuardBlock)
	input := pluginruntime.ResponseGuardPreviewRequest{Binding: pluginruntime.Binding{Recipe: config.DefaultRecipeName, Decision: responseStageRouteName}, Response: unsafeAnswer}

	preview, err := router.PreviewResponseGuard(context.Background(), input)
	if err != nil {
		t.Fatal(err)
	}
	if calls.Load() != 0 || preview.BackendCalls || preview.Reason != "probe_required" || preview.Action != config.ResponseGuardNone {
		t.Fatalf("offline preview called the model or invented an outcome: %+v", preview)
	}

	input.Mode = pluginruntime.ModeProbe
	probe, err := router.PreviewResponseGuard(context.Background(), input)
	if err != nil {
		t.Fatal(err)
	}
	if calls.Load() != 1 || !probe.BackendCalls || probe.Action != config.ResponseGuardBlock || len(probe.Rules) != 1 || probe.Rules[0].State != "matched" {
		t.Fatalf("probe = %+v (calls=%d), want the rule matched and the answer blocked", probe, calls.Load())
	}
	if len(probe.Signals) != 1 || probe.Signals[0].Signal != responseStageSignalKey || probe.Signals[0].Verdict != "detected" || probe.Signals[0].Score == nil {
		t.Fatalf("signals = %+v, want the response-direction rule detected with its score", probe.Signals)
	}

	input.Binding.Recipe = "foreign"
	if _, err := router.PreviewResponseGuard(context.Background(), input); !errors.Is(err, pluginruntime.ErrInvalidBinding) {
		t.Fatalf("foreign recipe lookup: %v", err)
	}
	if calls.Load() != 1 {
		t.Fatal("a foreign recipe caused a backend call")
	}
}

// The probe reads the guard's request-stage conditions on the question and
// its response-stage signals on the answer and the context it is given.
func TestResponseGuardPreviewChecksTheAnswerAgainstTheContext(t *testing.T) {
	server, calls := newHallucinationDecider([]string{"450 meters"}, false)
	guard := config.ResponseGuardPluginConfig{Rules: []config.ResponseGuardRule{{
		When: config.RuleNode{Type: config.SignalTypeHallucination, Name: hallucinationRuleName}, Action: config.ResponseGuardAnnotate,
	}}}
	router, _ := newHallucinationSignalRouter(t, server, guard)
	input := pluginruntime.ResponseGuardPreviewRequest{
		Binding: pluginruntime.Binding{Recipe: config.DefaultRecipeName, Decision: hallucinationRouteName}, Mode: pluginruntime.ModeProbe,
		Question: hallucinationQuestion, Context: hallucinationContext, Response: hallucinationAnswer,
	}

	result, err := router.PreviewResponseGuard(context.Background(), input)
	if err != nil {
		t.Fatal(err)
	}
	if result.Action != config.ResponseGuardAnnotate || len(result.Warnings) != 1 || result.Warnings[0] != headers.WarningHallucination || calls.Load() != 1 {
		t.Fatalf("grounded probe = %+v (calls=%d), want an annotation over the hallucination", result, calls.Load())
	}
	if len(result.Signals) != 1 || len(result.Signals[0].UnsupportedSpans) != 1 || result.Signals[0].UnsupportedSpans[0] != "450 meters" {
		t.Fatalf("signals = %+v, want the unsupported span", result.Signals)
	}

	input.Context = ""
	result, err = router.PreviewResponseGuard(context.Background(), input)
	if err != nil {
		t.Fatal(err)
	}
	if result.Rules[0].State != "unknown" || result.Signals[0].Verdict != "unavailable" || calls.Load() != 1 {
		t.Fatalf("ungrounded probe = %+v, want the rule unknown without asking the detector", result)
	}
}

func TestResponseGuardBindingReadinessNeverFallsBackAcrossRecipes(t *testing.T) {
	router, _ := newResponseStageRouter(t, newJailbreakDecider(0.95), "", config.ResponseGuardBlock)
	decisions := router.Config.Decisions
	router.Config.Recipes = []config.RoutingRecipe{
		{Name: config.DefaultRecipeName, Profile: config.RoutingProfile{Decisions: decisions}},
		{Name: "foreign", Profile: config.RoutingProfile{Decisions: decisions}},
	}
	for _, tc := range []struct {
		recipe       config.RecipeName
		availability string
	}{{config.DefaultRecipeName, "available"}, {"foreign", "unavailable"}} {
		dependencies, err := router.InspectPluginBinding(pluginruntime.Binding{Recipe: tc.recipe, Decision: responseStageRouteName}, config.DecisionPluginResponseGuard)
		if err != nil {
			t.Fatal(err)
		}
		if len(dependencies) != 1 || dependencies[0].Name != "response_signals" || dependencies[0].Availability != tc.availability {
			t.Fatalf("recipe %s readiness = %+v", tc.recipe, dependencies)
		}
	}
}
