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

type routerLearningAdapterFactory struct {
	method routerLearningMethod
	build  func(*OpenAIRouter) routerLearningAdapter
}

type routerLearningAdapterRegistry struct {
	factories []routerLearningAdapterFactory
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
	return defaultRouterLearningAdapterRegistry().Adapters(r)
}

func defaultRouterLearningAdapterRegistry() routerLearningAdapterRegistry {
	return newRouterLearningAdapterRegistry([]routerLearningAdapterFactory{
		{method: routerLearningMethodSessionAware, build: newSessionAwareLearningAdapter},
		{method: routerLearningMethodBandit, build: newBanditLearningAdapter},
		{method: routerLearningMethodElo, build: newEloLearningAdapter},
		{method: routerLearningMethodPersonalization, build: newPersonalizationLearningAdapter},
	})
}

func newRouterLearningAdapterRegistry(factories []routerLearningAdapterFactory) routerLearningAdapterRegistry {
	registry := routerLearningAdapterRegistry{
		factories: make([]routerLearningAdapterFactory, 0, len(factories)),
	}
	seen := map[routerLearningMethod]struct{}{}
	for _, factory := range factories {
		if factory.method == "" || factory.build == nil {
			continue
		}
		if _, ok := seen[factory.method]; ok {
			continue
		}
		seen[factory.method] = struct{}{}
		registry.factories = append(registry.factories, factory)
	}
	return registry
}

func (r routerLearningAdapterRegistry) Methods() []routerLearningMethod {
	methods := make([]routerLearningMethod, 0, len(r.factories))
	for _, factory := range r.factories {
		methods = append(methods, factory.method)
	}
	return methods
}

func (r routerLearningAdapterRegistry) Adapters(router *OpenAIRouter) []routerLearningAdapter {
	adapters := make([]routerLearningAdapter, 0, len(r.factories))
	for _, factory := range r.factories {
		adapter := factory.build(router)
		if adapter == nil || adapter.Method() != factory.method {
			continue
		}
		adapters = append(adapters, adapter)
	}
	return adapters
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
