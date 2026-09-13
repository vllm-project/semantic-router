package extproc

import (
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

// Exercise actual multi_factor selection, the saved request inventory, and
// provider dispatch in sequence. The capable sibling is deliberately rejected
// by a different policy, so a 200 routed to that sibling would violate it.
func TestMultiFactorEligibilitySurvivesCapabilityReroute(t *testing.T) {
	for _, policy := range []string{"slo", "quality_floor", "quality_missing"} {
		for _, qualifiedSibling := range []bool{false, true} {
			t.Run(policy+map[bool]string{false: "/reject", true: "/reroute"}[qualifiedSibling], func(t *testing.T) {
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
				if err != nil || selected == nil || selected.Model != primary {
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
				if qualifiedSibling {
					if err != nil || dispatch == nil || dispatch.logicalModel != "qualified-generator" {
						t.Fatalf("dispatch=%+v err=%v", dispatch, err)
					}
				} else {
					var protocolErr *llmprotocol.ProtocolError
					if !errors.As(err, &protocolErr) || protocolErr.Code != "unsupported_capability" || dispatch != nil {
						t.Fatalf("dispatch=%+v err=%v, want closed rejection", dispatch, err)
					}
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
		router.Config.ModelConfig[primary] = params
		params.Pricing = config.ModelPricing{PromptPer1M: 20, CompletionPer1M: 20}
		params.APIFormat = config.APIFormatResponses
		params.Capabilities = []string{"image_generation"}
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
		if dispatch, err := router.prepareProviderDispatch(request, selected.Model, decision.Name, false, ctx); err == nil || dispatch != nil {
			t.Fatalf("fallback expanded policy: dispatch=%+v err=%v", dispatch, err)
		}
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
