package extproc

import (
	"context"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

func TestEloLearningKeepsBaseWithoutRatingState(t *testing.T) {
	router := eloLearningTestRouter(config.EloLearningConfig{Enabled: true})
	selCtx, baseResult, baseRef := banditLearningTestSelection("base", map[string]float64{
		"base":   0.4,
		"winner": 0.9,
	})
	ctx := banditLearningTestContext()

	_, result, selected, applied := router.applyRouterLearning(selCtx, baseResult, baseRef, ctx)
	if applied {
		t.Fatal("expected Elo diagnostics to leave applied=false without rating state")
	}
	if result.SelectedModel != "base" || selected.Model != "base" {
		t.Fatalf("expected base model, got result=%s ref=%s", result.SelectedModel, selected.Model)
	}
	if got := ctx.VSRLearningPolicies[routerLearningMethodElo].String("reason"); got != eloReasonStateMissing {
		t.Fatalf("expected state_missing diagnostics, got %#v", ctx.VSRLearningPolicies)
	}
}

func TestEloLearningSwitchesAfterFeedback(t *testing.T) {
	router := eloLearningTestRouter(config.EloLearningConfig{Enabled: true})
	runtime := router.routerLearningRuntimeState()
	updated := runtime.UpdateFeedback(context.Background(), &selection.Feedback{
		DecisionName: "coding",
		WinnerModel:  "winner",
		LoserModel:   "base",
		Confidence:   1,
	})
	if updated != 1 {
		t.Fatalf("expected Elo feedback update, got %d", updated)
	}

	selCtx, baseResult, baseRef := banditLearningTestSelection("base", map[string]float64{
		"base":   0.8,
		"winner": 0.4,
	})
	ctx := banditLearningTestContext()

	_, result, selected, applied := router.applyRouterLearning(selCtx, baseResult, baseRef, ctx)
	if !applied {
		t.Fatal("expected Elo learning to apply")
	}
	if result.SelectedModel != "winner" || selected.Model != "winner" {
		t.Fatalf("expected winner model, got result=%s ref=%s", result.SelectedModel, selected.Model)
	}
	policy := ctx.VSRLearningPolicies[routerLearningMethodElo]
	if policy.String("action") != string(routerLearningActionSwitch) || policy.String("reason") != eloReasonRatingWin {
		t.Fatalf("unexpected Elo policy: %#v", policy)
	}
	if ratings := runtime.EloLeaderboard("coding"); len(ratings) != 2 || ratings[0].Model != "winner" {
		t.Fatalf("unexpected Elo leaderboard: %#v", ratings)
	}
}

func TestEloLearningDecisionBypass(t *testing.T) {
	router := eloLearningTestRouter(config.EloLearningConfig{Enabled: true})
	selCtx, baseResult, baseRef := banditLearningTestSelection("base", map[string]float64{
		"base":   0.1,
		"winner": 0.9,
	})
	ctx := banditLearningTestContext()
	ctx.VSRSelectedDecision.Adaptations.Elo = &config.DecisionLearningAdaptationConfig{
		Mode: config.DecisionAdaptationModeBypass,
	}

	_, result, selected, applied := router.applyRouterLearning(selCtx, baseResult, baseRef, ctx)
	if applied {
		t.Fatal("expected bypass to leave applied=false")
	}
	if result.SelectedModel != "base" || selected.Model != "base" {
		t.Fatalf("expected base model, got result=%s ref=%s", result.SelectedModel, selected.Model)
	}
	if got := ctx.VSRLearningPolicies[routerLearningMethodElo].String("action"); got != string(routerLearningActionBypass) {
		t.Fatalf("expected bypass diagnostics, got %#v", ctx.VSRLearningPolicies)
	}
}

func eloLearningTestRouter(elo config.EloLearningConfig) *OpenAIRouter {
	return &OpenAIRouter{
		Config: &config.RouterConfig{
			RouterLearning: config.RouterLearningConfig{
				Enabled: true,
				Adaptations: config.RouterLearningAdaptations{
					Elo: elo,
				},
			},
		},
	}
}
