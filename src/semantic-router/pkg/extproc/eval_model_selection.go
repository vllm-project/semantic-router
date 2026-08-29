package extproc

import (
	"context"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

// SelectModelForEval previews only selectors whose result exists before model
// execution. Looper algorithms deliberately report execution_required: their
// final model is part of the executed multi-model algorithm, not a candidate
// list ordering that Eval can honestly present as final.
func (r *OpenAIRouter) SelectModelForEval(
	input services.EvalModelSelectionInput,
) services.EvalModelSelection {
	decision := input.Decision
	if r == nil || r.Config == nil || decision == nil {
		return evalSelectionUnavailable("router selection runtime is unavailable")
	}
	if r.contextIneligibleAlgorithmModelCount(decision, input.ContextTokenCount) > 0 {
		return evalSelectionUnavailable("an explicitly configured algorithm model cannot satisfy the request context")
	}
	eligibleModelRefs, excluded := r.contextEligibleModelRefs(decision.ModelRefs, input.ContextTokenCount)
	if len(eligibleModelRefs) == 0 && excluded > 0 {
		return evalSelectionUnavailable("no decision model can satisfy the request context")
	}
	if excluded > 0 {
		eligibleDecision := *decision
		eligibleDecision.ModelRefs = eligibleModelRefs
		decision = &eligibleDecision
	}
	algorithmType := evalAlgorithmType(decision)
	if selectionResult, resolved := r.evalSelectionBeforeDryRun(decision, algorithmType); resolved {
		return selectionResult
	}

	method := r.getSelectionMethod(decision.Algorithm)
	if !evalSupportsDryRunSelection(method) {
		return services.EvalModelSelection{
			Status: services.EvalSelectionExecutionRequired,
			Method: string(method),
			Reason: "selector depends on request-time state that Eval does not mutate",
		}
	}
	return r.selectEvalCandidate(input, decision, method)
}

func evalAlgorithmType(decision *config.Decision) string {
	if decision.Algorithm != nil && decision.Algorithm.Type != "" {
		return decision.Algorithm.Type
	}
	return config.DecisionAlgorithmStatic
}

func (r *OpenAIRouter) evalSelectionBeforeDryRun(
	decision *config.Decision,
	algorithmType string,
) (services.EvalModelSelection, bool) {
	if model, ok := configuredLooperFinalModel(decision); ok {
		return services.EvalModelSelection{
			SelectedModel: model,
			Status:        services.EvalSelectionPlannedFinal,
			Method:        algorithmType,
			Reason:        "configured final-output model for the multi-model algorithm",
		}, true
	}
	if config.IsLooperAlgorithmType(algorithmType) {
		return services.EvalModelSelection{
			Status: services.EvalSelectionExecutionRequired,
			Method: algorithmType,
			Reason: "final model is produced only when the multi-model algorithm executes",
		}, true
	}
	if r.evalSelectionCanChangeAtExecution(decision) {
		return services.EvalModelSelection{
			Status: services.EvalSelectionExecutionRequired,
			Method: algorithmType,
			Reason: "Router Learning can adapt or protect the base selector only during request execution",
		}, true
	}
	return services.EvalModelSelection{}, false
}

func (r *OpenAIRouter) selectEvalCandidate(
	input services.EvalModelSelectionInput,
	decision *config.Decision,
	method selection.SelectionMethod,
) services.EvalModelSelection {
	defaultCandidate := firstConfiguredEvalCandidate(decision.ModelRefs)
	if defaultCandidate == nil {
		return evalSelectionUnavailable("decision has no selectable model")
	}
	if len(decision.ModelRefs) == 1 {
		return selectedEvalModel(defaultCandidate, "single", "single declared candidate")
	}

	requestContext := &RequestContext{
		Headers:              map[string]string{},
		TraceContext:         context.Background(),
		VSRContextTokenCount: input.ContextTokenCount,
		VSRSelectedDecision:  decision,
	}
	if recipe, ok := r.Config.RecipeByName(input.Recipe); ok {
		requestContext.Routing.SelectRecipe(recipe)
	}
	costWeight, qualityWeight := r.getSelectionWeights(decision.Algorithm)
	tpot, ttft := r.getLatencyAwarePercentiles(decision.Algorithm)
	selectionContext := &selection.SelectionContext{
		Query:                      input.Query,
		DecisionName:               decision.Name,
		RecipeName:                 input.Recipe,
		CategoryName:               input.Category,
		CandidateModels:            decision.ModelRefs,
		CandidateIterations:        decision.CandidateIterations,
		CostWeight:                 costWeight,
		QualityWeight:              qualityWeight,
		LatencyAwareTPOTPercentile: tpot,
		LatencyAwareTTFTPercentile: ttft,
	}
	selector := r.selectorForDecisionMethod(method, decision.Algorithm, requestContext)
	if selector == nil {
		return fallbackEvalModel(defaultCandidate, method, "selector is unavailable")
	}
	result, err := selector.Select(context.Background(), selectionContext)
	if err != nil {
		return fallbackEvalModel(defaultCandidate, method, "selector failed during dry-run")
	}
	if err := selection.ValidateSelectionResult(selectionContext, result); err != nil {
		return fallbackEvalModel(defaultCandidate, method, "selector returned an invalid candidate")
	}
	selected := selectedModelRefFromResult(selectionContext, result)
	if selected == nil {
		return fallbackEvalModel(defaultCandidate, method, "selected candidate is not declared")
	}
	reason := strings.TrimSpace(result.Reasoning)
	if reason == "" {
		reason = "selected by the live runtime selector"
	}
	return selectedEvalModel(selected, string(method), boundedSelectionReasoning(reason))
}

func (r *OpenAIRouter) evalSelectionCanChangeAtExecution(decision *config.Decision) bool {
	if r == nil || r.Config == nil || decision == nil || !r.Config.RouterLearning.Enabled {
		return false
	}
	adaptationEnabled := r.Config.RouterLearning.Adaptation.EffectiveEnabled() &&
		decision.Adaptations.AdaptationMode() == config.DecisionAdaptationModeApply
	protectionEnabled := r.Config.RouterLearning.Protection.EffectiveEnabled() &&
		decision.Adaptations.ProtectionMode() == config.DecisionAdaptationModeApply
	return adaptationEnabled || protectionEnabled
}

func configuredLooperFinalModel(decision *config.Decision) (string, bool) {
	if decision == nil || decision.Algorithm == nil {
		return "", false
	}
	var model string
	switch decision.Algorithm.Type {
	case config.DecisionAlgorithmFusion:
		if decision.Algorithm.Fusion != nil {
			model = decision.Algorithm.Fusion.Model
		}
	case config.DecisionAlgorithmWorkflows:
		if decision.Algorithm.Workflows != nil {
			model = decision.Algorithm.Workflows.Final.Model
		}
	case config.DecisionAlgorithmReMoM:
		if decision.Algorithm.ReMoM != nil {
			model = decision.Algorithm.ReMoM.SynthesisModel
		}
	}
	model = strings.TrimSpace(model)
	return model, model != ""
}

func evalSupportsDryRunSelection(method selection.SelectionMethod) bool {
	switch method {
	case selection.MethodStatic, selection.MethodMultiFactor, selection.MethodLatencyAware:
		return true
	default:
		return false
	}
}

func firstConfiguredEvalCandidate(modelRefs []config.ModelRef) *config.ModelRef {
	for index := range modelRefs {
		if strings.TrimSpace(modelRefs[index].Model) != "" {
			return &modelRefs[index]
		}
	}
	return nil
}

func selectedEvalModel(
	modelRef *config.ModelRef,
	method string,
	reason string,
) services.EvalModelSelection {
	selected := modelRef.Model
	if modelRef.LoRAName != "" {
		selected = modelRef.LoRAName
	}
	return services.EvalModelSelection{
		SelectedModel: selected,
		Status:        services.EvalSelectionSelected,
		Method:        method,
		Reason:        reason,
	}
}

func fallbackEvalModel(
	modelRef *config.ModelRef,
	method selection.SelectionMethod,
	reason string,
) services.EvalModelSelection {
	selection := selectedEvalModel(modelRef, string(method), reason)
	selection.Status = services.EvalSelectionFallback
	return selection
}

func evalSelectionUnavailable(reason string) services.EvalModelSelection {
	return services.EvalModelSelection{
		Status: services.EvalSelectionUnavailable,
		Reason: reason,
	}
}
