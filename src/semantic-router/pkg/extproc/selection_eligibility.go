package extproc

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

// applySelectionEligibility carries hard selector filters to every subsequent
// model choice. The result is explicit: score maps are diagnostics, not policy.
func applySelectionEligibility(selCtx *selection.SelectionContext, result *selection.SelectionResult, ctx *RequestContext) (*selection.SelectionContext, error) {
	if result.EligibleModels == nil {
		return selCtx, nil
	}
	for _, ref := range result.EligibleModels {
		if !modelRefInEligibility(ref, selCtx.CandidateModels) {
			return nil, fmt.Errorf("%w: selector eligibility contains an undeclared model %q", selection.ErrNoEligibleCandidates, ref.Model)
		}
	}
	eligible := make([]config.ModelRef, 0, len(result.EligibleModels))
	selectedAllowed := false
	for _, ref := range selCtx.CandidateModels {
		if !modelRefInEligibility(ref, result.EligibleModels) ||
			(ctx != nil && ctx.VSREligibleModelRefs != nil && !modelRefInEligibility(ref, ctx.VSREligibleModelRefs)) {
			continue
		}
		eligible = append(eligible, ref)
		if (ref.Model == result.SelectedModel && ref.LoRAName == result.LoRAName) ||
			(ref.LoRAName != "" && ref.LoRAName == result.SelectedModel) {
			selectedAllowed = true
		}
	}
	if !selectedAllowed {
		return nil, fmt.Errorf("%w: selected model %q is outside selector eligibility", selection.ErrNoEligibleCandidates, result.SelectedModel)
	}
	if ctx != nil {
		ctx.VSRPolicyEligibleModelRefs = cloneModelRefs(eligible)
		ctx.VSREligibleModelRefs = cloneModelRefs(eligible)
	}
	clone := *selCtx
	clone.CandidateModels = eligible
	return &clone, nil
}

func modelRefInEligibility(ref config.ModelRef, refs []config.ModelRef) bool {
	for _, candidate := range refs {
		if ref.Model == candidate.Model && ref.LoRAName == candidate.LoRAName {
			return true
		}
	}
	return false
}
