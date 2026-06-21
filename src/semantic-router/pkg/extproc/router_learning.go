package extproc

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

type routerLearningAdapter interface {
	Method() routerLearningMethod
	Apply(routerLearningInput) (routerLearningAdaptationResult, bool)
	Observe(routerLearningInput, routerLearningComposition)
}

type routerLearningBaseAdapter struct {
	router *OpenAIRouter
}

func (a routerLearningBaseAdapter) Observe(routerLearningInput, routerLearningComposition) {}

type sessionAwareLearningAdapter struct {
	routerLearningBaseAdapter
}

func newSessionAwareLearningAdapter(router *OpenAIRouter) routerLearningAdapter {
	return sessionAwareLearningAdapter{
		routerLearningBaseAdapter: routerLearningBaseAdapter{router: router},
	}
}

func (a sessionAwareLearningAdapter) Method() routerLearningMethod {
	return routerLearningMethodSessionAware
}

func (a sessionAwareLearningAdapter) Apply(input routerLearningInput) (routerLearningAdaptationResult, bool) {
	if a.router == nil {
		return routerLearningAdaptationResult{}, false
	}
	return a.router.applySessionAwareLearning(input)
}

type banditLearningAdapter struct {
	routerLearningBaseAdapter
}

func newBanditLearningAdapter(router *OpenAIRouter) routerLearningAdapter {
	return banditLearningAdapter{
		routerLearningBaseAdapter: routerLearningBaseAdapter{router: router},
	}
}

func (a banditLearningAdapter) Method() routerLearningMethod {
	return routerLearningMethodBandit
}

func (a banditLearningAdapter) Apply(input routerLearningInput) (routerLearningAdaptationResult, bool) {
	if a.router == nil {
		return routerLearningAdaptationResult{}, false
	}
	return a.router.applyBanditLearning(input)
}

func (a banditLearningAdapter) Observe(input routerLearningInput, composed routerLearningComposition) {
	if a.router != nil {
		a.router.observeBanditSelection(input, composed)
	}
}

type eloLearningAdapter struct {
	routerLearningBaseAdapter
}

func newEloLearningAdapter(router *OpenAIRouter) routerLearningAdapter {
	return eloLearningAdapter{
		routerLearningBaseAdapter: routerLearningBaseAdapter{router: router},
	}
}

func (a eloLearningAdapter) Method() routerLearningMethod {
	return routerLearningMethodElo
}

func (a eloLearningAdapter) Apply(input routerLearningInput) (routerLearningAdaptationResult, bool) {
	if a.router == nil {
		return routerLearningAdaptationResult{}, false
	}
	return a.router.applyEloLearning(input)
}

type personalizationLearningAdapter struct {
	routerLearningBaseAdapter
}

func newPersonalizationLearningAdapter(router *OpenAIRouter) routerLearningAdapter {
	return personalizationLearningAdapter{
		routerLearningBaseAdapter: routerLearningBaseAdapter{router: router},
	}
}

func (a personalizationLearningAdapter) Method() routerLearningMethod {
	return routerLearningMethodPersonalization
}

func (a personalizationLearningAdapter) Apply(input routerLearningInput) (routerLearningAdaptationResult, bool) {
	if a.router == nil {
		return routerLearningAdaptationResult{}, false
	}
	return a.router.applyPersonalizationLearning(input)
}

func (r *OpenAIRouter) routerLearningAdapters() []routerLearningAdapter {
	return []routerLearningAdapter{
		newSessionAwareLearningAdapter(r),
		newBanditLearningAdapter(r),
		newEloLearningAdapter(r),
		newPersonalizationLearningAdapter(r),
	}
}

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
	adapters := r.routerLearningAdapters()
	for _, adapter := range adapters {
		if result, ok := adapter.Apply(input); ok {
			results = append(results, result)
		}
	}
	composed := composeRouterLearning(input, results)
	for _, adapter := range adapters {
		adapter.Observe(input, composed)
	}
	return composed.selectionContext, composed.selectionResult, composed.selectedModelRef, composed.applied
}
