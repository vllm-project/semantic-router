package extproc

import (
	"context"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

func TestBanditLearningKeepsBaseWithoutState(t *testing.T) {
	router := banditLearningTestRouter(config.BanditLearningConfig{
		Enabled: true,
		Goals:   map[string]float64{"quality": 1},
	})
	selCtx, baseResult, baseRef := banditLearningTestSelection("base", map[string]float64{
		"base":   0.4,
		"winner": 0.9,
	})
	ctx := banditLearningTestContext()

	_, result, selected, applied := router.applyRouterLearning(selCtx, baseResult, baseRef, ctx)
	if applied {
		t.Fatal("expected diagnostics-only learning to leave applied=false")
	}
	if result.SelectedModel != "base" || selected.Model != "base" {
		t.Fatalf("expected base model, got result=%s ref=%s", result.SelectedModel, selected.Model)
	}
	if got := ctx.VSRLearningPolicies[routerLearningMethodBandit].String("reason"); got != banditReasonStateMissing {
		t.Fatalf("expected state_missing diagnostics, got %#v", ctx.VSRLearningPolicies)
	}
}

func TestBanditLearningSwitchesAfterFeedback(t *testing.T) {
	router := banditLearningTestRouter(config.BanditLearningConfig{
		Enabled: true,
		Goals:   map[string]float64{"quality": 1},
	})
	runtime := router.routerLearningRuntimeState()
	updated := runtime.UpdateFeedback(context.Background(), &selection.Feedback{
		DecisionName: "coding",
		WinnerModel:  "winner",
		LoserModel:   "base",
		Confidence:   1,
	})
	if updated != 1 {
		t.Fatalf("expected bandit feedback update, got %d", updated)
	}

	selCtx, baseResult, baseRef := banditLearningTestSelection("base", map[string]float64{
		"base":   0.8,
		"winner": 0.4,
	})
	ctx := banditLearningTestContext()

	_, result, selected, applied := router.applyRouterLearning(selCtx, baseResult, baseRef, ctx)
	if !applied {
		t.Fatal("expected bandit learning to apply")
	}
	if result.SelectedModel != "winner" || selected.Model != "winner" {
		t.Fatalf("expected winner model, got result=%s ref=%s", result.SelectedModel, selected.Model)
	}
	policy := ctx.VSRLearningPolicies[routerLearningMethodBandit]
	if policy.String("action") != string(routerLearningActionSwitch) || policy.String("reason") != banditReasonScoreWin {
		t.Fatalf("unexpected bandit policy: %#v", policy)
	}
}

func TestBanditLearningConversationScopeUsesFeedbackConversationID(t *testing.T) {
	router := banditLearningTestRouter(config.BanditLearningConfig{
		Enabled: true,
		Scope:   config.RouterLearningScopeConversation,
		Goals:   map[string]float64{"quality": 1},
	})
	runtime := router.routerLearningRuntimeState()
	updated := runtime.UpdateFeedback(context.Background(), &selection.Feedback{
		DecisionName:   "coding",
		SessionID:      "session-a",
		ConversationID: "conversation-a",
		WinnerModel:    "winner",
		LoserModel:     "base",
		Confidence:     1,
	})
	if updated != 1 {
		t.Fatalf("expected conversation-scoped bandit feedback update, got %d", updated)
	}

	selCtx, baseResult, baseRef := banditLearningTestSelection("base", map[string]float64{
		"base":   0.8,
		"winner": 0.4,
	})
	selCtx.SessionID = "session-a"
	ctx := banditLearningTestContext()
	ctx.SessionID = "session-a"
	ctx.VSRLearningConversationID = "conversation-a"

	_, result, selected, applied := router.applyRouterLearning(selCtx, baseResult, baseRef, ctx)
	if !applied {
		t.Fatal("expected conversation-scoped bandit learning to apply")
	}
	if result.SelectedModel != "winner" || selected.Model != "winner" {
		t.Fatalf("expected winner model, got result=%s ref=%s", result.SelectedModel, selected.Model)
	}
}

func TestBanditLearningCostGoalCanSwitchWithoutFeedback(t *testing.T) {
	router := banditLearningTestRouter(config.BanditLearningConfig{
		Enabled: true,
		Goals:   map[string]float64{"cost": 1},
	})
	router.Config.ModelConfig = map[string]config.ModelParams{
		"expensive": {Pricing: config.ModelPricing{PromptPer1M: 10, CompletionPer1M: 10}},
		"cheap":     {Pricing: config.ModelPricing{PromptPer1M: 1, CompletionPer1M: 1}},
	}
	selCtx, baseResult, baseRef := banditLearningTestSelection("expensive", map[string]float64{
		"expensive": 0.9,
		"cheap":     0.2,
	})
	ctx := banditLearningTestContext()

	_, result, selected, applied := router.applyRouterLearning(selCtx, baseResult, baseRef, ctx)
	if !applied {
		t.Fatal("expected cost-goal bandit learning to apply")
	}
	if result.SelectedModel != "cheap" || selected.Model != "cheap" {
		t.Fatalf("expected cheap model, got result=%s ref=%s", result.SelectedModel, selected.Model)
	}
}

func TestBanditLearningDecisionBypass(t *testing.T) {
	router := banditLearningTestRouter(config.BanditLearningConfig{
		Enabled: true,
		Goals:   map[string]float64{"cost": 1},
	})
	selCtx, baseResult, baseRef := banditLearningTestSelection("base", map[string]float64{
		"base":   0.1,
		"winner": 0.9,
	})
	ctx := banditLearningTestContext()
	ctx.VSRSelectedDecision.Adaptations.Bandit = &config.DecisionBanditAdaptationConfig{
		Mode: config.DecisionAdaptationModeBypass,
	}

	_, result, selected, applied := router.applyRouterLearning(selCtx, baseResult, baseRef, ctx)
	if applied {
		t.Fatal("expected bypass to leave applied=false")
	}
	if result.SelectedModel != "base" || selected.Model != "base" {
		t.Fatalf("expected base model, got result=%s ref=%s", result.SelectedModel, selected.Model)
	}
	if got := ctx.VSRLearningPolicies[routerLearningMethodBandit].String("action"); got != string(routerLearningActionBypass) {
		t.Fatalf("expected bypass diagnostics, got %#v", ctx.VSRLearningPolicies)
	}
}

func banditLearningTestRouter(bandit config.BanditLearningConfig) *OpenAIRouter {
	return &OpenAIRouter{
		Config: &config.RouterConfig{
			RouterLearning: config.RouterLearningConfig{
				Enabled: true,
				Adaptations: config.RouterLearningAdaptations{
					Bandit: bandit,
				},
			},
		},
	}
}

func banditLearningTestContext() *RequestContext {
	decision := &config.Decision{Name: "coding"}
	return &RequestContext{
		VSRSelectedDecision:     decision,
		VSRSelectedDecisionName: decision.Name,
	}
}

func banditLearningTestSelection(baseModel string, scores map[string]float64) (*selection.SelectionContext, *selection.SelectionResult, *config.ModelRef) {
	candidates := make([]config.ModelRef, 0, len(scores))
	for model := range scores {
		candidates = append(candidates, config.ModelRef{Model: model})
	}
	baseRef := &config.ModelRef{Model: baseModel}
	baseScore := scores[baseModel]
	return &selection.SelectionContext{
			DecisionName:    "coding",
			CandidateModels: candidates,
		}, &selection.SelectionResult{
			SelectedModel: baseModel,
			Score:         baseScore,
			Confidence:    1,
			Method:        selection.MethodStatic,
			Tier:          selection.TierSupported,
			AllScores:     scores,
		}, baseRef
}
