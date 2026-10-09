package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

func TestBypassDispatchUpdatesProtectionOwnerBeforeToolLoop(t *testing.T) {
	sessiontelemetry.ResetRouterSessionMemoryForTesting()
	t.Cleanup(sessiontelemetry.ResetRouterSessionMemoryForTesting)

	router := &OpenAIRouter{Config: routerLearningProtectionOnlyTestConfig(config.RouterLearningScopeConversation)}
	refs := []config.ModelRef{{Model: "cheap"}, {Model: "frontier"}}
	dispatch := func(turn int, mode string, proposed int, activeToolLoop bool) (*selection.SelectionResult, *config.ModelRef) {
		t.Helper()
		ctx := routerLearningRequestContext("bypass-session", "conversation-a")
		ctx.TurnIndex = turn
		ctx.VSRSelectedDecision = &config.Decision{
			Name:        "bypass-owner",
			Adaptations: config.DecisionAdaptationsConfig{Mode: mode},
		}
		if activeToolLoop {
			ctx.VSRConversationFacts = classification.ConversationFacts{LastAssistantToolCall: true}
		}
		selCtx := &selection.SelectionContext{
			SessionID:       "bypass-session",
			DecisionName:    "bypass-owner",
			CandidateModels: refs,
		}
		base := (&selection.SelectionResult{Method: selection.MethodStatic}).WithCandidate(refs[proposed])
		finalCtx, result, selected, _, err := router.applyRouterLearning(selCtx, base, &refs[proposed], ctx)
		if err != nil {
			t.Fatal(err)
		}
		stageAgenticSessionDecision(finalCtx, result, selected, ctx)
		if err := commitAgenticSessionDecision(ctx); err != nil {
			t.Fatal(err)
		}
		return result, selected
	}

	dispatch(0, config.DecisionAdaptationModeApply, 0, false)
	dispatch(1, config.DecisionAdaptationModeBypass, 1, false)
	result, selected := dispatch(2, config.DecisionAdaptationModeApply, 1, true)
	if selected.Model != "frontier" {
		t.Fatalf("tool-loop continuation locked to stale owner: selected %q", selected.Model)
	}
	if result.SessionPolicy == nil || result.SessionPolicy.HardLockReason != "hard_lock=tool_loop" {
		t.Fatalf("expected tool-loop hard lock after bypass dispatch, got %#v", result.SessionPolicy)
	}
}

// Exercise the production preflight -> adaptation -> switch orchestration with
// maintained messages and actual selections, including a bypass control.
func TestRouterLearningSessionOrchestrationSuppressesActualSampling(t *testing.T) {
	corpus, _ := loadProtectionCorpus(t)
	for _, scenario := range corpus.Scenarios {
		if scenario.ID != "tool-loop-and-release" {
			continue
		}
		for _, mode := range []string{"apply", "bypass"} {
			t.Run(mode, func(t *testing.T) {
				scenario.Mode = mode
				runProtectionOrchestration(t, scenario)
			})
		}
		return
	}
	t.Fatal("missing tool-loop orchestration fixture")
}

func runProtectionOrchestration(t *testing.T, scenario protectionScenario) {
	t.Helper()
	sessiontelemetry.ResetRouterSessionMemoryForTesting()
	t.Cleanup(sessiontelemetry.ResetRouterSessionMemoryForTesting)
	calls := 0
	original := routerLearningSamplingSeedSource
	routerLearningSamplingSeedSource = func() int64 { calls++; return 424242 }
	t.Cleanup(func() { routerLearningSamplingSeedSource = original })
	cfg := routerLearningTestConfig(scenario.Scope)
	cfg.DefaultModel = "protection-cheap"
	cfg.ModelConfig = map[string]config.ModelParams{"protection-cheap": {}, "protection-frontier": {}}
	router := &OpenAIRouter{Config: cfg}
	request := &llmprotocol.Request{}
	for turn, step := range scenario.Steps {
		for _, message := range step.Messages {
			request.Messages = append(request.Messages, protectionNeutralMessage(message))
		}
		input := protectionScenarioInput(router, scenario, step, turn, request)
		before := calls
		ctx, result, ref, _, learningErr := router.applyRouterLearning(input.selCtx, input.baseResult, input.selectedModelRef, input.ctx)
		if learningErr != nil {
			t.Fatal(learningErr)
		}
		wantSampling := scenario.Mode == "bypass" || step.Expected.Sampling
		assertAdaptationSampled(t, input.ctx, wantSampling)
		if (calls > before) != wantSampling {
			t.Fatalf("%s: sampling invocation does not match permission", step.ID)
		}
		if scenario.Mode == "apply" && step.Expected.Category == "blocked" {
			previous := input.selCtx.AgenticSession.PreviousModel
			if previous == "" || ref.Model != previous || result.SelectedModel != previous {
				t.Fatalf("%s: protected continuation switched from %q to %q", step.ID, previous, ref.Model)
			}
		}
		// Each accepted corpus step represents a completed dispatch boundary.
		stageAgenticSessionDecision(ctx, result, ref, input.ctx)
		if err := commitAgenticSessionDecision(input.ctx); err != nil {
			t.Fatal(err)
		}
	}
}
