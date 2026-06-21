package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

func TestComposeRouterLearningHardResultWins(t *testing.T) {
	ctx := &RequestContext{}
	baseCtx, baseResult, baseRef := learningContractTestSelection("base")
	softCtx, softResult, softRef := learningContractTestSelection("soft")
	hardCtx, hardResult, hardRef := learningContractTestSelection("hard")

	composed := composeRouterLearning(routerLearningInput{
		selCtx:           baseCtx,
		baseResult:       baseResult,
		selectedModelRef: baseRef,
		ctx:              ctx,
	}, []routerLearningAdaptationResult{
		{
			method:           routerLearningMethodBandit,
			mode:             config.DecisionAdaptationModeApply,
			action:           routerLearningActionSwitch,
			reason:           "soft_score",
			changesModel:     true,
			selectionContext: softCtx,
			selectionResult:  softResult,
			selectedModelRef: softRef,
			policy:           routerLearningContractTestPolicy(routerLearningMethodBandit, routerLearningActionSwitch, "soft_score"),
		},
		{
			method:           routerLearningMethod("provider_health"),
			mode:             config.DecisionAdaptationModeApply,
			action:           routerLearningActionSwitch,
			reason:           "hard_guard",
			hard:             true,
			changesModel:     true,
			selectionContext: hardCtx,
			selectionResult:  hardResult,
			selectedModelRef: hardRef,
			policy:           routerLearningContractTestPolicy(routerLearningMethod("provider_health"), routerLearningActionSwitch, "hard_guard"),
		},
	})

	if !composed.applied || composed.selectedModelRef == nil || composed.selectedModelRef.Model != "hard" {
		t.Fatalf("expected hard result to win, got %#v", composed.selectedModelRef)
	}
	if got := ctx.VSRLearningPolicies[routerLearningMethodBandit].String("reason"); got != "soft_score" {
		t.Fatalf("expected bandit diagnostics recorded, got %#v", ctx.VSRLearningPolicies)
	}
	if got := ctx.VSRLearningPolicies[routerLearningMethod("provider_health")].String("reason"); got != "hard_guard" {
		t.Fatalf("expected hard diagnostics recorded, got %#v", ctx.VSRLearningPolicies)
	}
}

func TestComposeRouterLearningObserveDoesNotChangeModel(t *testing.T) {
	ctx := &RequestContext{}
	baseCtx, baseResult, baseRef := learningContractTestSelection("base")
	observeCtx, observeResult, observeRef := learningContractTestSelection("observed")

	composed := composeRouterLearning(routerLearningInput{
		selCtx:           baseCtx,
		baseResult:       baseResult,
		selectedModelRef: baseRef,
		ctx:              ctx,
	}, []routerLearningAdaptationResult{
		{
			method:           routerLearningMethodBandit,
			mode:             config.DecisionAdaptationModeObserve,
			action:           routerLearningActionSwitch,
			reason:           "shadow_win",
			changesModel:     true,
			selectionContext: observeCtx,
			selectionResult:  observeResult,
			selectedModelRef: observeRef,
			policy:           routerLearningContractTestPolicy(routerLearningMethodBandit, routerLearningActionSwitch, "shadow_win"),
		},
	})

	if !composed.applied {
		t.Fatal("expected observe result to count as learning-applied diagnostics")
	}
	if composed.selectedModelRef == nil || composed.selectedModelRef.Model != "base" {
		t.Fatalf("expected observe mode to keep base model, got %#v", composed.selectedModelRef)
	}
	if composed.selectionContext != observeCtx {
		t.Fatalf("expected observe mode to preserve adaptation context for telemetry")
	}
}

func TestAttachRouterLearningExperienceAddsMethodDiagnostics(t *testing.T) {
	snapshot := newRouterLearningExperienceSnapshot([]routerLearningExperienceView{
		{
			name:        "handoff_penalty",
			method:      routerLearningMethodSessionAware,
			status:      routerLearningExperienceStatusUsed,
			source:      routerLearningExperienceSourceLookupTable,
			version:     "v1",
			freshness:   "warm",
			sampleCount: 12,
		},
	})

	result := attachRouterLearningExperience(routerLearningAdaptationResult{
		method: routerLearningMethodSessionAware,
		policy: routerLearningContractTestPolicy(routerLearningMethodSessionAware, routerLearningActionStay, ""),
	}, snapshot)

	policy := result.policy.ToMap()
	experience, ok := policy["experience"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected experience diagnostics, got %#v", policy["experience"])
	}
	handoff, ok := experience["handoff_penalty"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected handoff_penalty diagnostics, got %#v", experience)
	}
	if got := replayPolicyString(handoff, "status"); got != string(routerLearningExperienceStatusUsed) {
		t.Fatalf("expected used experience status, got %#v", handoff)
	}
	if got := replayPolicyString(handoff, "source"); got != string(routerLearningExperienceSourceLookupTable) {
		t.Fatalf("expected lookup-table experience source, got %#v", handoff)
	}
	if got := replayNumericDiagnostic(handoff["sample_count"]); got != 12 {
		t.Fatalf("expected sample_count=12, got %#v", handoff)
	}
}

func TestRouterLearningPoliciesUseTypedDetails(t *testing.T) {
	banditPolicy := banditLearningPolicy(
		config.BanditLearningConfig{Goals: map[string]float64{"quality": 1}},
		config.DecisionAdaptationModeApply,
		routerLearningActionSwitch,
		banditReasonScoreWin,
		"decision:coding",
		[]routerLearningBanditScore{{model: "winner", score: 0.9}},
		routerLearningBanditScore{model: "winner", score: 0.9},
	)
	if banditPolicy.Details.Bandit == nil || len(banditPolicy.Details.ActiveMethods()) != 1 {
		t.Fatalf("expected one typed bandit detail, got %#v", banditPolicy.Details.ActiveMethods())
	}
	if got := banditPolicy.String("selected_model"); got != "winner" {
		t.Fatalf("expected selected model from typed bandit detail, got %q", got)
	}

	eloPolicy := eloLearningPolicy(
		config.EloLearningConfig{},
		config.DecisionAdaptationModeApply,
		routerLearningActionSwitch,
		eloReasonRatingWin,
		"decision:coding",
		[]routerLearningEloScore{{model: "winner", score: 0.7, rating: 1210}},
		routerLearningEloScore{model: "winner", score: 0.7, rating: 1210},
	)
	if eloPolicy.Details.Elo == nil || len(eloPolicy.Details.ActiveMethods()) != 1 {
		t.Fatalf("expected one typed Elo detail, got %#v", eloPolicy.Details.ActiveMethods())
	}
	if got := eloPolicy.String("selected_model"); got != "winner" {
		t.Fatalf("expected selected model from typed Elo detail, got %q", got)
	}

	personalizationPolicy := personalizationLearningPolicy(
		config.PersonalizationLearningConfig{},
		config.DecisionAdaptationModeApply,
		routerLearningActionSwitch,
		personalizationReasonPreferenceWin,
		"user:user-1/decision:coding",
		"user-1",
		[]routerLearningPersonalizationScore{{model: "winner", score: 0.8, preference: 0.9}},
		routerLearningPersonalizationScore{model: "winner", score: 0.8, preference: 0.9},
	)
	if personalizationPolicy.Details.Personalization == nil || len(personalizationPolicy.Details.ActiveMethods()) != 1 {
		t.Fatalf("expected one typed personalization detail, got %#v", personalizationPolicy.Details.ActiveMethods())
	}
	if got := personalizationPolicy.String("user_hash"); got == "" {
		t.Fatalf("expected user hash from typed personalization detail, got %q", got)
	}
}

func TestRouterLearningPolicySerializationKeepsCommonFieldsAuthoritative(t *testing.T) {
	policy := newRouterLearningPolicy(routerLearningMethodSessionAware)
	policy.Mode = config.DecisionAdaptationModeApply
	policy.Scope = config.RouterLearningScopeConversation
	policy.Action = routerLearningActionSwitch
	policy.Reason = "switch_allowed"
	policy.Details.SessionAware = newSessionAwareLearningDiagnostics(
		&selection.SessionPolicyTrace{
			Phase:          "provider_state",
			CurrentModel:   "qwen-small",
			SelectedModel:  "qwen-large",
			HardLocked:     true,
			HardLockReason: "tool_loop",
		},
		sessionAwareIdentityDiagnostics{},
	)

	serialized := policy.ToMap()
	if got := replayPolicyString(serialized, "learning"); got != routerLearningPolicyName {
		t.Fatalf("expected learning marker, got %#v", serialized)
	}
	if got := replayPolicyString(serialized, "adaptation"); got != string(routerLearningMethodSessionAware) {
		t.Fatalf("expected common adaptation field to be authoritative, got %#v", serialized)
	}
	if got := replayPolicyString(serialized, "action"); got != string(routerLearningActionSwitch) {
		t.Fatalf("expected common action field to be authoritative, got %#v", serialized)
	}
	if got := policy.SessionPhase(); got != "provider_state" {
		t.Fatalf("expected typed session phase accessor, got %q", got)
	}
	if !policy.HardLocked() {
		t.Fatalf("expected typed hard lock accessor")
	}
}

func TestRouterLearningAdapterRegistryIsMethodKeyed(t *testing.T) {
	registry := defaultRouterLearningAdapterRegistry()
	got := registry.Methods()
	want := []routerLearningMethod{
		routerLearningMethodSessionAware,
		routerLearningMethodBandit,
		routerLearningMethodElo,
		routerLearningMethodPersonalization,
	}
	if len(got) != len(want) {
		t.Fatalf("expected %d adapters, got %#v", len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("unexpected adapter order: got %#v want %#v", got, want)
		}
	}

	seen := map[routerLearningMethod]struct{}{}
	for _, method := range got {
		if _, exists := seen[method]; exists {
			t.Fatalf("adapter registry must not contain duplicate method %q", method)
		}
		seen[method] = struct{}{}
	}
	adapters := registry.Adapters(&OpenAIRouter{})
	if len(adapters) != len(want) {
		t.Fatalf("expected %d adapters, got %#v", len(want), adapters)
	}

	deduped := newRouterLearningAdapterRegistry([]routerLearningAdapterFactory{
		{method: routerLearningMethodBandit, build: newBanditLearningAdapter},
		{method: routerLearningMethodBandit, build: newEloLearningAdapter},
		{method: "", build: newSessionAwareLearningAdapter},
		{method: routerLearningMethodElo},
	})
	if got := deduped.Methods(); len(got) != 1 || got[0] != routerLearningMethodBandit {
		t.Fatalf("expected duplicate and invalid factories filtered, got %#v", got)
	}
}

func replayNumericDiagnostic(value interface{}) float64 {
	switch typed := value.(type) {
	case int:
		return float64(typed)
	case int64:
		return float64(typed)
	case float64:
		return typed
	default:
		return 0
	}
}

func learningContractTestSelection(model string) (*selection.SelectionContext, *selection.SelectionResult, *config.ModelRef) {
	modelRef := &config.ModelRef{Model: model}
	selCtx := &selection.SelectionContext{
		CandidateModels: []config.ModelRef{*modelRef},
	}
	result := &selection.SelectionResult{
		SelectedModel: model,
		Score:         1,
		Confidence:    1,
		Method:        selection.MethodStatic,
		Tier:          selection.TierSupported,
		AllScores:     map[string]float64{model: 1},
	}
	return selCtx, result, modelRef
}

func routerLearningContractTestPolicy(
	method routerLearningMethod,
	action routerLearningAction,
	reason string,
) routerLearningPolicy {
	policy := newRouterLearningPolicy(method)
	policy.Action = action
	policy.Reason = reason
	return policy
}
