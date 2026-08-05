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
	}

	preflight := r.applyProtectionPreflight(input)
	adaptation := r.applyLearningAdaptation(input, preflight)
	protection := r.applyProtectionSwitch(input, preflight, adaptation)
	recordRouterLearningPolicies(ctx, preflight, adaptation, protection)

	finalCtx := firstNonNilSelectionContext(protection.selectionContext, adaptation.selectionContext, selCtx)
	finalResult := firstNonNilSelectionResult(protection.selectionResult, adaptation.selectionResult, baseResult)
	finalRef := firstNonNilModelRef(protection.selectedModelRef, adaptation.selectedModelRef, selectedModelRef)
	applied := learningChangesModel(baseResult, finalResult)
	return finalCtx, finalResult, finalRef, applied
}

func (r *OpenAIRouter) applyLearningAdaptation(
	input routerLearningInput,
	preflight routerLearningProtectionPreflight,
) routerLearningDecision {
	cfg, ok := r.adaptationConfig(input.selCtx, input.baseResult, input.selectedModelRef, input.ctx)
	if !ok {
		return routerLearningDecision{}
	}
	mode := adaptationMode(input.ctx)
	strategy, ok := routerLearningAdaptationStrategies.Strategy(cfg)
	if !ok {
		return baseAdaptationDecision(
			input,
			adaptationPolicy(mode, routerLearningActionKeepBase, "strategy_unavailable", nil),
		)
	}
	return strategy.Select(r, input, preflight, cfg)
}

func recordRouterLearningPolicies(
	ctx *RequestContext,
	preflight routerLearningProtectionPreflight,
	adaptation routerLearningDecision,
	protection routerLearningDecision,
) {
	if ctx == nil {
		return
	}
	var policies routerLearningPolicies
	if !preflight.policy.Empty() {
		policies.Set(preflight.policy)
		ctx.VSRLearningProtectionPreflight = preflight.policy.toReplayProtection()
	}
	if !adaptation.policy.Empty() {
		policies.Set(adaptation.policy)
	}
	if !protection.policy.Empty() {
		policies.Set(protection.policy)
	}
	if policies.Empty() {
		return
	}
	ctx.VSRLearningPolicies = policies
	if policy, ok := policies.Policy(routerLearningMethodProtection); ok {
		ctx.VSRLearningPolicy = &policy
		return
	}
	if policy, ok := policies.Policy(routerLearningMethodAdaptation); ok {
		ctx.VSRLearningPolicy = &policy
	}
}
