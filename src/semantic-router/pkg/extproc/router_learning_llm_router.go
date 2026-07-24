package extproc

import (
	"context"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

func routeRouterLearningLLMRouterQuery(
	input routerLearningInput,
	serverURL string,
	query string,
) (*selection.LLMRouterResponse, error) {
	client := selection.NewLLMRouterClient(serverURL)
	return client.Route(routerLearningRouteContext(input), query)
}

func routerLearningRouteContext(input routerLearningInput) context.Context {
	if input.ctx != nil && input.ctx.TraceContext != nil {
		return input.ctx.TraceContext
	}
	return context.Background()
}

func newRouterLearningLLMRouterSelectionResult(
	baseResult *selection.SelectionResult,
	ref *config.ModelRef,
) *selection.SelectionResult {
	result := cloneSelectionResult(baseResult)
	if result == nil {
		result = &selection.SelectionResult{}
	}
	result.SelectedModel = ref.Model
	result.LoRAName = ref.LoRAName
	result.Method = baseSelectionMethod(baseResult)
	result.Tier = baseSelectionTier(baseResult)
	result.Reasoning = "router_learning adaptation: llm_router"
	if result.AllScores == nil {
		result.AllScores = map[string]float64{}
	}
	return result
}

func routerLearningLLMRouterPolicy(
	mode string,
	candidateSet string,
	input routerLearningInput,
	learningCtx *selection.SelectionContext,
	selectedModel string,
) routerLearningPolicy {
	return adaptationPolicy(mode, routerLearningActionProposeSwitch, "llm_router_selected", &routerLearningAdaptationDiagnostics{
		candidateSet:  candidateSet,
		strategy:      "llm_router",
		baseModel:     selectedModelName(input.baseResult),
		proposalModel: selectedModel,
		decision:      strings.TrimSpace(learningCtx.DecisionName),
		decisionTier:  decisionTier(input.ctx),
	})
}

func observeLLMRouterAdaptationPolicy(policy routerLearningPolicy) routerLearningPolicy {
	policy.Action = routerLearningActionObserve
	policy.Reason = "llm_router_observe"
	return policy
}

func logRouterLearningLLMRouterWarning(
	event string,
	err error,
	serverURL string,
	decisionName string,
	extra map[string]interface{},
) {
	fields := map[string]interface{}{
		"server_url":    serverURL,
		"decision_name": decisionName,
	}
	if err != nil {
		fields["error"] = err.Error()
	}
	for key, value := range extra {
		fields[key] = value
	}
	logging.ComponentWarnEvent("extproc", event, fields)
}
