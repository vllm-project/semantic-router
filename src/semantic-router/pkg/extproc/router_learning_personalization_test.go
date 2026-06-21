package extproc

import (
	"context"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

func TestPersonalizationLearningRequiresUserIdentity(t *testing.T) {
	router := personalizationLearningTestRouter(config.PersonalizationLearningConfig{Enabled: true})
	selCtx, baseResult, baseRef := banditLearningTestSelection("base", map[string]float64{
		"base":   0.4,
		"winner": 0.9,
	})
	ctx := banditLearningTestContext()

	_, result, selected, applied := router.applyRouterLearning(selCtx, baseResult, baseRef, ctx)
	if applied {
		t.Fatal("expected missing user identity to leave applied=false")
	}
	if result.SelectedModel != "base" || selected.Model != "base" {
		t.Fatalf("expected base model, got result=%s ref=%s", result.SelectedModel, selected.Model)
	}
	if got := ctx.VSRLearningPolicies[routerLearningMethodPersonalization].String("reason"); got != personalizationReasonIdentityMissing {
		t.Fatalf("expected identity_missing diagnostics, got %#v", ctx.VSRLearningPolicies)
	}
}

func TestPersonalizationLearningSwitchesAfterUserFeedback(t *testing.T) {
	router := personalizationLearningTestRouter(config.PersonalizationLearningConfig{Enabled: true})
	runtime := router.routerLearningRuntimeState()
	updated := runtime.UpdateFeedback(context.Background(), &selection.Feedback{
		DecisionName: "coding",
		UserID:       "user-1",
		WinnerModel:  "winner",
		LoserModel:   "base",
		Confidence:   1,
	})
	if updated != 1 {
		t.Fatalf("expected personalization feedback update, got %d", updated)
	}

	selCtx, baseResult, baseRef := banditLearningTestSelection("base", map[string]float64{
		"base":   0.8,
		"winner": 0.4,
	})
	selCtx.UserID = "user-1"
	ctx := banditLearningTestContext()

	_, result, selected, applied := router.applyRouterLearning(selCtx, baseResult, baseRef, ctx)
	if !applied {
		t.Fatal("expected personalization learning to apply")
	}
	if result.SelectedModel != "winner" || selected.Model != "winner" {
		t.Fatalf("expected winner model, got result=%s ref=%s", result.SelectedModel, selected.Model)
	}
	policy := ctx.VSRLearningPolicies[routerLearningMethodPersonalization]
	if policy.String("action") != string(routerLearningActionSwitch) || policy.String("reason") != personalizationReasonPreferenceWin {
		t.Fatalf("unexpected personalization policy: %#v", policy)
	}
	if policy.String("user_hash") == "" {
		t.Fatalf("expected user hash in diagnostics: %#v", policy)
	}
}

func TestPersonalizationLearningDecisionBypass(t *testing.T) {
	router := personalizationLearningTestRouter(config.PersonalizationLearningConfig{Enabled: true})
	selCtx, baseResult, baseRef := banditLearningTestSelection("base", map[string]float64{
		"base":   0.1,
		"winner": 0.9,
	})
	selCtx.UserID = "user-1"
	ctx := banditLearningTestContext()
	ctx.VSRSelectedDecision.Adaptations.Personalization = &config.DecisionLearningAdaptationConfig{
		Mode: config.DecisionAdaptationModeBypass,
	}

	_, result, selected, applied := router.applyRouterLearning(selCtx, baseResult, baseRef, ctx)
	if applied {
		t.Fatal("expected bypass to leave applied=false")
	}
	if result.SelectedModel != "base" || selected.Model != "base" {
		t.Fatalf("expected base model, got result=%s ref=%s", result.SelectedModel, selected.Model)
	}
	if got := ctx.VSRLearningPolicies[routerLearningMethodPersonalization].String("action"); got != string(routerLearningActionBypass) {
		t.Fatalf("expected bypass diagnostics, got %#v", ctx.VSRLearningPolicies)
	}
}

func personalizationLearningTestRouter(personalization config.PersonalizationLearningConfig) *OpenAIRouter {
	return &OpenAIRouter{
		Config: &config.RouterConfig{
			RouterLearning: config.RouterLearningConfig{
				Enabled: true,
				Adaptations: config.RouterLearningAdaptations{
					Personalization: personalization,
				},
			},
		},
	}
}
