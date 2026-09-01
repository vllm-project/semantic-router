package extproc

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
)

func routeActionRouter(contextWindow int) *OpenAIRouter {
	return &OpenAIRouter{Config: &config.RouterConfig{
		BackendModels: config.BackendModels{
			ModelConfig: map[string]config.ModelParams{
				"safe-model": {ContextWindowSize: contextWindow},
			},
		},
	}}
}

func guardDecision(action *config.DecisionAction) *config.Decision {
	return &config.Decision{
		Name:   "guard",
		Rules:  config.RuleNode{Type: config.SignalTypeJailbreak, Name: "prompt_injection"},
		Action: action,
	}
}

func TestDecisionRouteActionDestination(t *testing.T) {
	router := routeActionRouter(0)
	ctx := &RequestContext{}

	destination, ok := router.decisionRouteActionDestination(
		guardDecision(&config.DecisionAction{Type: config.DecisionActionRoute, Destination: "safe-model"}),
		ctx,
	)
	assert.True(t, ok)
	assert.Equal(t, "safe-model", destination)

	_, ok = router.decisionRouteActionDestination(guardDecision(nil), ctx)
	assert.False(t, ok)

	_, ok = router.decisionRouteActionDestination(nil, ctx)
	assert.False(t, ok)
}

func TestFinalizeDecisionEvaluationRouteActionOverridesPinnedModel(t *testing.T) {
	router := routeActionRouter(0)
	ctx := &RequestContext{}
	result := &decision.DecisionResult{
		Decision: guardDecision(&config.DecisionAction{
			Type:        config.DecisionActionRoute,
			Destination: "safe-model",
		}),
		Confidence: 1,
	}

	decisionName, _, _, selectedModel, err := router.finalizeDecisionEvaluation(result, "pinned-model", "attack text", ctx)
	assert.NoError(t, err)
	assert.Equal(t, "guard", decisionName)
	assert.Equal(t, "safe-model", selectedModel)
}

func TestFinalizeDecisionEvaluationWithoutActionPreservesPinnedModel(t *testing.T) {
	router := routeActionRouter(0)
	ctx := &RequestContext{}
	result := &decision.DecisionResult{Decision: guardDecision(nil), Confidence: 1}

	_, _, _, selectedModel, err := router.finalizeDecisionEvaluation(result, "pinned-model", "attack text", ctx)
	assert.NoError(t, err)
	assert.Equal(t, "", selectedModel)
}

func TestDecisionRouteActionDestinationContextIneligible(t *testing.T) {
	router := routeActionRouter(100)
	ctx := &RequestContext{VSRContextTokenCount: 200}

	_, ok := router.decisionRouteActionDestination(
		guardDecision(&config.DecisionAction{Type: config.DecisionActionRoute, Destination: "safe-model"}),
		ctx,
	)
	assert.False(t, ok)

	ctx.VSRContextTokenCount = 50
	destination, ok := router.decisionRouteActionDestination(
		guardDecision(&config.DecisionAction{Type: config.DecisionActionRoute, Destination: "safe-model"}),
		ctx,
	)
	assert.True(t, ok)
	assert.Equal(t, "safe-model", destination)
}
