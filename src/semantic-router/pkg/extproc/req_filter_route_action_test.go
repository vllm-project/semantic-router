package extproc

import (
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
)

func routeActionRouter(modelWindows map[string]int) *OpenAIRouter {
	modelConfig := make(map[string]config.ModelParams, len(modelWindows))
	for model, window := range modelWindows {
		modelConfig[model] = config.ModelParams{ContextWindowSize: window}
	}
	return &OpenAIRouter{Config: &config.RouterConfig{
		BackendModels: config.BackendModels{ModelConfig: modelConfig},
	}}
}

func guardDecision(action *config.DecisionAction, modelRefs ...config.ModelRef) *config.Decision {
	return &config.Decision{
		Name:      "guard",
		Rules:     config.RuleNode{Type: config.SignalTypeJailbreak, Name: "prompt_injection"},
		Action:    action,
		ModelRefs: modelRefs,
	}
}

func routeAction(destination string) *config.DecisionAction {
	return &config.DecisionAction{Type: config.DecisionActionRoute, Destination: destination}
}

func TestDecisionRouteActionDestinationEligible(t *testing.T) {
	router := routeActionRouter(map[string]int{"safe-model": 0})
	ctx := &RequestContext{}

	destination, terminal, err := router.decisionRouteActionDestination(
		guardDecision(routeAction("safe-model")),
		ctx,
	)
	assert.NoError(t, err)
	assert.True(t, terminal)
	assert.Equal(t, "safe-model", destination)
}

func TestDecisionRouteActionDestinationNoActionIsNotTerminal(t *testing.T) {
	router := routeActionRouter(map[string]int{"safe-model": 0})
	ctx := &RequestContext{}

	for name, decisionValue := range map[string]*config.Decision{
		"nil decision":      nil,
		"nil action":        guardDecision(nil),
		"empty destination": guardDecision(routeAction("  ")),
	} {
		t.Run(name, func(t *testing.T) {
			destination, terminal, err := router.decisionRouteActionDestination(decisionValue, ctx)
			assert.NoError(t, err)
			assert.False(t, terminal)
			assert.Empty(t, destination)
		})
	}
}

func TestDecisionRouteActionFallsBackToEligibleCandidate(t *testing.T) {
	router := routeActionRouter(map[string]int{"safe-model": 100, "large-safe-model": 100000})
	ctx := &RequestContext{VSRContextTokenCount: 200}

	destination, terminal, err := router.decisionRouteActionDestination(
		guardDecision(
			routeAction("safe-model"),
			config.ModelRef{Model: "large-safe-model"},
		),
		ctx,
	)
	assert.NoError(t, err)
	assert.True(t, terminal)
	assert.Equal(t, "large-safe-model", destination)
}

func TestDecisionRouteActionFailsClosedWithoutEligibleModel(t *testing.T) {
	router := routeActionRouter(map[string]int{"safe-model": 100, "small-safe-model": 100})
	ctx := &RequestContext{VSRContextTokenCount: 200}

	tests := map[string]*config.Decision{
		"no candidates": guardDecision(routeAction("safe-model")),
		"only ineligible candidates": guardDecision(
			routeAction("safe-model"),
			config.ModelRef{Model: "small-safe-model"},
		),
	}
	for name, decisionValue := range tests {
		t.Run(name, func(t *testing.T) {
			destination, terminal, err := router.decisionRouteActionDestination(decisionValue, ctx)
			assert.ErrorIs(t, err, errNoContextEligibleDecisionModel)
			assert.False(t, terminal)
			assert.Empty(t, destination)
		})
	}
}

func TestDecisionRouteActionDestinationEligibleAfterContextShrinks(t *testing.T) {
	router := routeActionRouter(map[string]int{"safe-model": 100})
	ctx := &RequestContext{VSRContextTokenCount: 50}

	destination, terminal, err := router.decisionRouteActionDestination(
		guardDecision(routeAction("safe-model")),
		ctx,
	)
	assert.NoError(t, err)
	assert.True(t, terminal)
	assert.Equal(t, "safe-model", destination)
}

func TestFinalizeDecisionEvaluationRouteActionOverridesPinnedModel(t *testing.T) {
	router := routeActionRouter(map[string]int{"safe-model": 0})
	ctx := &RequestContext{}
	result := &decision.DecisionResult{
		Decision:   guardDecision(routeAction("safe-model")),
		Confidence: 1,
	}

	decisionName, _, _, selectedModel, err := router.finalizeDecisionEvaluation(result, "pinned-model", "attack text", ctx)
	assert.NoError(t, err)
	assert.Equal(t, "guard", decisionName)
	assert.Equal(t, "safe-model", selectedModel)
}

func TestFinalizeDecisionEvaluationRouteActionNeverFallsBackToPinnedModel(t *testing.T) {
	router := routeActionRouter(map[string]int{"safe-model": 100})
	ctx := &RequestContext{VSRContextTokenCount: 200}
	result := &decision.DecisionResult{
		Decision:   guardDecision(routeAction("safe-model")),
		Confidence: 1,
	}

	_, _, _, selectedModel, err := router.finalizeDecisionEvaluation(result, "pinned-model", "attack text", ctx)
	if !errors.Is(err, errNoContextEligibleDecisionModel) {
		t.Fatalf("err = %v, want errNoContextEligibleDecisionModel", err)
	}
	assert.Empty(t, selectedModel)
}

func TestFinalizeDecisionEvaluationWithoutActionPreservesPinnedModel(t *testing.T) {
	router := routeActionRouter(map[string]int{"safe-model": 0})
	ctx := &RequestContext{}
	result := &decision.DecisionResult{Decision: guardDecision(nil), Confidence: 1}

	_, _, _, selectedModel, err := router.finalizeDecisionEvaluation(result, "pinned-model", "attack text", ctx)
	assert.NoError(t, err)
	assert.Equal(t, "", selectedModel)
}
