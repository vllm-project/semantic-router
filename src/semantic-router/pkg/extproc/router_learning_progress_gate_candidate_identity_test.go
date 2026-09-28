package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

func TestProgressGateSuppressionHoldsExactCurrentCandidate(t *testing.T) {
	router, ctx, learningCtx := gateReplayContext(t, "suppressed-exact-candidate")
	current := config.ModelRef{
		Model:    "cheap",
		LoRAName: "cheap-adapter",
		ModelReasoningControl: config.ModelReasoningControl{
			UseReasoning:    extprocBoolPtr(true),
			ReasoningEffort: "low",
		},
	}
	proposed := config.ModelRef{
		Model: "frontier",
		ModelReasoningControl: config.ModelReasoningControl{
			UseReasoning:    extprocBoolPtr(true),
			ReasoningEffort: "high",
		},
	}
	learningCtx.CandidateModels = []config.ModelRef{current, proposed}
	learningCtx.AgenticSession.PreviousModel = current.Model
	learningCtx.AgenticSession.PreviousCandidate = &current
	result := &selection.SelectionResult{
		SelectedModel:     proposed.Model,
		SelectedCandidate: &proposed,
		SessionPolicy: &selection.SessionPolicyTrace{
			CurrentModel:   current.Model,
			SelectedModel:  proposed.Model,
			DecisionReason: "switch_allowed",
		},
	}

	router.applySwitchGateToResult(
		gateReplayConfig(selection.GateModeEnforce),
		ctx,
		learningCtx,
		selection.NewSessionAwareSelector(nil),
		result,
	)

	if result.SelectedModel != current.Model || result.LoRAName != current.LoRAName {
		t.Fatalf("held result identity = model %q lora %q, want %q/%q",
			result.SelectedModel, result.LoRAName, current.Model, current.LoRAName)
	}
	if result.SelectedCandidate == nil ||
		selection.CandidateIdentity(*result.SelectedCandidate) != selection.CandidateIdentity(current) {
		t.Fatalf("held candidate = %+v, want %+v", result.SelectedCandidate, current)
	}
}

func TestProgressGateEligibilityKeepsExactCandidate(t *testing.T) {
	router, decision := candidateEffortTestRouter(t, config.DecisionAlgorithmMultiFactor, "high")
	floor := 75.0
	decision.Algorithm.MultiFactor.Quality.MinScore = &floor
	ctx := routerLearningRequestContext("exact-eligibility", "conversation")
	ctx.VSRSelectedDecision = decision
	selCtx := &selection.SelectionContext{
		DecisionName:    decision.Name,
		CandidateModels: decision.ModelRefs,
	}

	filtered := router.progressEligibleContext(ctx, selCtx)

	if filtered == nil || len(filtered.CandidateModels) != 1 {
		t.Fatalf("filtered candidates = %#v, want only the exact high-effort candidate", filtered)
	}
	if got := filtered.CandidateModels[0]; selection.CandidateIdentity(got) != selection.CandidateIdentity(decision.ModelRefs[1]) {
		t.Fatalf("filtered candidate = %+v, want %+v", got, decision.ModelRefs[1])
	}
}

func TestProtectionObserveSelectorErrorPreservesBaseResult(t *testing.T) {
	router, decision, ctx, input, _ := excludedExactOwnerFixture(
		t,
		config.RouterLearningScopeConversation,
		"tool_loop",
	)
	decision.Adaptations.Protection = &config.DecisionLearningProtectionConfig{
		Mode: config.DecisionAdaptationModeObserve,
	}
	input.CandidateModels = input.CandidateModels[:1]
	base := (&selection.SelectionResult{
		Score:  1,
		Method: selection.MethodMultiFactor,
	}).WithCandidate(input.CandidateModels[0])
	learningInput := routerLearningInput{
		selCtx:           input,
		baseResult:       base,
		selectedModelRef: base.SelectedCandidate,
		ctx:              ctx,
	}
	preflight := router.applyProtectionPreflight(learningInput)
	if !preflight.enabled || preflight.mode != config.DecisionAdaptationModeObserve {
		t.Fatalf("preflight = enabled:%t mode:%q, want enabled observe", preflight.enabled, preflight.mode)
	}

	result, err := router.applyProtectionSwitch(learningInput, preflight, routerLearningDecision{})
	if err != nil {
		t.Fatal(err)
	}
	if result.selectionResult != base || result.selectedModelRef != base.SelectedCandidate {
		t.Fatalf("observe selector error changed selection: result=%#v ref=%#v", result.selectionResult, result.selectedModelRef)
	}
	if result.policy.Action != routerLearningActionObserve || result.policy.Reason != "observe_only" {
		t.Fatalf("observe policy = action %q reason %q, want observe/observe_only", result.policy.Action, result.policy.Reason)
	}
}
