package extproc

import (
	"math/rand"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

func TestRouterLearningPreservesEffortCandidatesAndModelLevelSampling(t *testing.T) {
	router, decision := candidateEffortTestRouter(t, config.DecisionAlgorithmMultiFactor, "high")
	copy := decision.ModelRefs[0]
	copy.UseReasoning = extprocBoolPtr(true)
	router.Config.Decisions = []config.Decision{
		*decision,
		{Name: "other", ModelRefs: []config.ModelRef{copy}},
	}
	refs := unionConfigDecisionModelRefs(router.Config, func(config.Decision) bool { return true })
	if len(refs) != 2 || refs[0].ReasoningEffort != "low" || refs[1].ReasoningEffort != "high" {
		t.Fatalf("effort variants collapsed or equal controls were duplicated: %+v", refs)
	}
	selCtx := &selection.SelectionContext{DecisionName: decision.Name, CandidateModels: refs}
	scores := router.scoreRoutingSamplingCandidates(selCtx, &RequestContext{VSRSelectedDecision: decision}, nil,
		config.RouterLearningCandidateSetDecision, true, rand.New(rand.NewSource(42)))
	if len(scores) != 2 || scores[0].score != scores[1].score {
		t.Fatalf("model-level experience was sampled separately for effort variants: %+v", scores)
	}
	if selection.CandidateIdentity(scores[0].candidate) == selection.CandidateIdentity(scores[1].candidate) {
		t.Fatal("sampling lost the exact candidate identities")
	}
}

func TestRouterLearningTierCandidatesStayInsideRecipe(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{
		Recipes: []config.RoutingRecipe{
			{
				Name: "recipe-a",
				Profile: config.RoutingProfile{Decisions: []config.Decision{{
					Name:      "a-tier-one",
					Tier:      1,
					ModelRefs: []config.ModelRef{{Model: "model-a"}, {Model: "model-a2"}},
				}}},
			},
			{
				Name: "recipe-b",
				Profile: config.RoutingProfile{Decisions: []config.Decision{{
					Name:      "b-tier-one",
					Tier:      1,
					ModelRefs: []config.ModelRef{{Model: "model-b"}},
				}}},
			},
		},
		BackendModels: config.BackendModels{
			ModelConfig: map[string]config.ModelParams{
				"model-a":  {},
				"model-a2": {},
				"model-b":  {},
			},
		},
	}}
	decision := &router.Config.Recipes[0].Profile.Decisions[0]
	ctx := &RequestContext{VSRSelectedDecision: decision}
	selCtx := &selection.SelectionContext{
		RecipeName:      "recipe-a",
		DecisionName:    decision.Name,
		CandidateModels: decision.ModelRefs,
	}

	candidates := router.learningCandidateModels(
		selCtx,
		ctx,
		config.RouterLearningCandidateSetTier,
	)

	if len(candidates) != 2 ||
		candidates[0].Model != "model-a" ||
		candidates[1].Model != "model-a2" {
		t.Fatalf("tier candidates escaped recipe-a: %#v", candidates)
	}
}
