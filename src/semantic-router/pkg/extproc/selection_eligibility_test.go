package extproc

import (
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

// Capability filtering runs before multi_factor. A capable sibling rejected
// by a hard quality/SLO policy cannot be restored by learning or dispatch.
func TestMultiFactorEligibilityIntersectsCapabilitiesBeforeDispatch(t *testing.T) {
	for _, policy := range []string{"slo", "quality_floor", "quality_missing"} {
		for _, qualifiedSibling := range []bool{false, true} {
			t.Run(policy+map[bool]string{false: "/reject", true: "/select"}[qualifiedSibling], func(t *testing.T) {
				router, primary := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
				params := addTestQuality(router.Config.ModelConfig[primary], .95)
				params.Pricing = config.ModelPricing{PromptPer1M: .1, CompletionPer1M: .1}
				router.Config.ModelConfig[primary] = params
				params.APIFormat = config.APIFormatResponses
				params.Capabilities = []string{"image_generation"}
				params.Pricing = config.ModelPricing{PromptPer1M: 20, CompletionPer1M: 20}
				params = addTestQuality(params, .2)
				if policy == "quality_missing" {
					params.IndexResults = nil
				}
				router.Config.ModelConfig["excluded-generator"] = params
				refs := []config.ModelRef{{Model: primary}, {Model: "excluded-generator"}}
				if qualifiedSibling {
					params.Pricing = config.ModelPricing{PromptPer1M: 1, CompletionPer1M: 1}
					params = addTestQuality(params, .8)
					router.Config.ModelConfig["qualified-generator"] = params
					refs = append(refs, config.ModelRef{Model: "qualified-generator"})
				}
				cfg := &config.MultiFactorSelectionConfig{Weights: &config.MultiFactorWeightsConfig{Cost: 1}, OnNoCandidates: "fail"}
				switch policy {
				case "slo":
					cfg.SLO = &config.MultiFactorSLOConfig{MaxCostPer1M: 2}
				case "quality_floor":
					floor := 70.0
					cfg.Quality = &config.QualityEvidenceConfig{Index: testIntelligenceIndex, MinScore: &floor}
				case "quality_missing":
					cfg.Weights = &config.MultiFactorWeightsConfig{Quality: 1}
					cfg.Quality = &config.QualityEvidenceConfig{Index: testIntelligenceIndex, OnMissing: config.QualityEvidenceOnMissingExclude}
				}
				decision := &config.Decision{Name: "hard-policy", ModelRefs: refs, Algorithm: &config.AlgorithmConfig{Type: config.DecisionAlgorithmMultiFactor, MultiFactor: cfg}}
				request := testNeutralRequest(primary, "draw a cat")
				request.ImageGeneration = &llmprotocol.ImageGenerationOptions{}
				ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
				ctx.VSRSelectedDecision = decision
				eligible, err := router.contextEligibleDecisionModelRefs(refs, decision.Name, 100, ctx)
				if err != nil {
					t.Fatal(err)
				}
				selected, _, err := router.selectModelFromCandidates(&selection.SelectionContext{DecisionName: decision.Name, CandidateModels: eligible}, decision.Algorithm, ctx)
				if !qualifiedSibling {
					if !errors.Is(err, selection.ErrNoEligibleCandidates) || selected != nil {
						t.Fatalf("selection=%+v err=%v, want closed rejection", selected, err)
					}
					return
				}
				if err != nil || selected == nil || selected.Model != "qualified-generator" {
					t.Fatalf("selection=%+v err=%v", selected, err)
				}
				if modelRefInEligibility(config.ModelRef{Model: "excluded-generator"}, ctx.VSREligibleModelRefs) {
					t.Fatal("selector did not narrow final inventory")
				}
				// Tier/global learning may start from a broader set, but cannot
				// resurrect a candidate the active hard policy excluded.
				learning := router.learningCandidateModels(&selection.SelectionContext{CandidateModels: refs}, ctx, config.RouterLearningCandidateSetGlobal)
				if modelRefInEligibility(config.ModelRef{Model: "excluded-generator"}, learning) {
					t.Fatal("global learning resurrected excluded candidate")
				}
				dispatch, err := router.prepareProviderDispatch(request, selected.Model, decision.Name, false, ctx)
				if err != nil || dispatch == nil || dispatch.logicalModel != "qualified-generator" {
					t.Fatalf("dispatch=%+v err=%v", dispatch, err)
				}
			})
		}
	}
}

func TestMultiFactorExplicitFallbackDoesNotAuthorizeOtherExcludedModels(t *testing.T) {
	for _, fallback := range []string{"first", "cheapest"} {
		router, primary := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
		params := router.Config.ModelConfig[primary]
		params.Pricing = config.ModelPricing{PromptPer1M: 10, CompletionPer1M: 10}
		params.APIFormat = config.APIFormatResponses
		params.Capabilities = []string{"image_generation"}
		router.Config.ModelConfig[primary] = params
		params.Pricing = config.ModelPricing{PromptPer1M: 20, CompletionPer1M: 20}
		params.Capabilities = []string{"image_generation", "image_input"}
		router.Config.ModelConfig["other-excluded"] = params
		refs := []config.ModelRef{{Model: primary}, {Model: "other-excluded"}}
		decision := &config.Decision{Name: "fallback", ModelRefs: refs, Algorithm: &config.AlgorithmConfig{Type: config.DecisionAlgorithmMultiFactor, MultiFactor: &config.MultiFactorSelectionConfig{SLO: &config.MultiFactorSLOConfig{MaxCostPer1M: 1}, OnNoCandidates: fallback}}}
		request := testNeutralRequest(primary, "draw a cat")
		request.ImageGeneration = &llmprotocol.ImageGenerationOptions{}
		ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
		ctx.VSRSelectedDecision = decision
		selected, _, err := router.selectModelFromCandidates(&selection.SelectionContext{DecisionName: decision.Name, CandidateModels: refs}, decision.Algorithm, ctx)
		if err != nil || selected == nil || selected.Model != primary {
			t.Fatalf("explicit %s fallback rejected: selected=%+v err=%v", fallback, selected, err)
		}
		if len(ctx.VSREligibleModelRefs) != 1 || ctx.VSREligibleModelRefs[0].Model != primary {
			t.Fatalf("fallback envelope=%+v", ctx.VSREligibleModelRefs)
		}
		// A later mutation cannot activate another candidate, even if that
		// candidate could handle the new task and appears in the full decision.
		request.Messages[0].Content = append(request.Messages[0].Content, llmprotocol.Content{
			Kind: llmprotocol.ContentImage, URL: "https://example.com/image.png",
		})
		if dispatch, err := router.prepareProviderDispatch(request, selected.Model, decision.Name, false, ctx); err == nil || dispatch != nil {
			t.Fatalf("fallback expanded policy: dispatch=%+v err=%v", dispatch, err)
		}
	}
}

func TestMultiFactorEligibilityPreservesExactEffort(t *testing.T) {
	router, decision := candidateEffortTestRouter(t, config.DecisionAlgorithmMultiFactor, "high")
	floor := 75.0
	decision.Algorithm.MultiFactor.Quality.MinScore = &floor
	router.Config.Decisions = []config.Decision{*decision}
	selCtx := &selection.SelectionContext{DecisionName: decision.Name, CandidateModels: decision.ModelRefs}
	ctx := &RequestContext{VSRSelectedDecision: decision}
	selected, _, err := router.selectModelFromCandidates(selCtx, decision.Algorithm, ctx)
	if err != nil || selected == nil || selected.ReasoningEffort != "high" {
		t.Fatalf("exact-effort selection = %+v, %v", selected, err)
	}
	if len(ctx.VSRPolicyEligibleModelRefs) != 1 || ctx.VSRPolicyEligibleModelRefs[0].ReasoningEffort != "high" {
		t.Fatalf("quality floor admitted the low-effort sibling: %+v", ctx.VSRPolicyEligibleModelRefs)
	}
	learning := router.learningCandidateModels(selCtx, ctx, config.RouterLearningCandidateSetGlobal)
	if len(learning) != 1 || learning[0].ReasoningEffort != "high" {
		t.Fatalf("global learning expanded exact-effort eligibility: %+v", learning)
	}
	_, err = applySelectionEligibility(selCtx, &selection.SelectionResult{
		SelectedModel: decision.ModelRefs[0].Model, SelectedCandidate: &decision.ModelRefs[0],
		EligibleModels: []config.ModelRef{decision.ModelRefs[1]},
	}, nil)
	if !errors.Is(err, selection.ErrNoEligibleCandidates) {
		t.Fatalf("ineligible exact winner must fail closed, got %v", err)
	}
}

func TestMultiFactorSoftRankingPreservesGlobalLearningInventory(t *testing.T) {
	router, primary := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
	router.Config.ModelConfig["global-sibling"] = router.Config.ModelConfig[primary]
	ctx := &RequestContext{}
	selCtx := &selection.SelectionContext{DecisionName: "soft", CandidateModels: []config.ModelRef{{Model: primary}}}
	_, _, err := router.selectModelFromCandidates(selCtx, &config.AlgorithmConfig{Type: config.DecisionAlgorithmMultiFactor}, ctx)
	if err != nil {
		t.Fatal(err)
	}
	if ctx.VSRPolicyEligibleModelRefs != nil {
		t.Fatal("soft ranking introduced a hard policy envelope")
	}
	learning := router.learningCandidateModels(selCtx, ctx, config.RouterLearningCandidateSetGlobal)
	if !modelRefInEligibility(config.ModelRef{Model: "global-sibling"}, learning) {
		t.Fatal("soft ranking unexpectedly narrowed configured global learning")
	}
}
