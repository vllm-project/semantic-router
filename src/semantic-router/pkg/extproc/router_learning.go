package extproc

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

func (r *OpenAIRouter) applyRouterLearning(
	selCtx *selection.SelectionContext,
	baseResult *selection.SelectionResult,
	selectedModelRef *config.ModelRef,
	ctx *RequestContext,
) (*selection.SelectionContext, *selection.SelectionResult, *config.ModelRef, bool) {
	input := routerLearningInput{
		selCtx:           selCtx,
		baseResult:       baseResult,
		selectedModelRef: selectedModelRef,
		ctx:              ctx,
		experience:       r.routerLearningExperienceSnapshot(),
	}
	var results []routerLearningAdaptationResult
	if result, ok := r.applySessionAwareLearning(input); ok {
		results = append(results, result)
	}
	if result, ok := r.applyBanditLearning(input); ok {
		results = append(results, result)
	}
	if result, ok := r.applyEloLearning(input); ok {
		results = append(results, result)
	}
	if result, ok := r.applyPersonalizationLearning(input); ok {
		results = append(results, result)
	}
	composed := composeRouterLearning(input, results)
	r.observeBanditSelection(input, composed)
	return composed.selectionContext, composed.selectionResult, composed.selectedModelRef, composed.applied
}
