package extproc

import (
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

func (r *OpenAIRouter) candidateRequirements(ctx *RequestContext) *config.CandidateRequirements {
	if r == nil || r.Config == nil {
		return nil
	}
	if ctx != nil {
		if ctx.Routing.IsPassthrough() {
			return nil
		}
		if recipe := ctx.Routing.SelectedRecipe(); recipe != nil {
			return recipe.Profile.CandidateRequirements
		}
	}
	return r.Config.CandidateRequirements
}

func (r *OpenAIRouter) validateModelDemand(requirements *config.CandidateRequirements, model string, demand selection.CandidateDemand) error {
	if !selection.CandidateRequirementsEnabled(requirements) {
		return nil
	}
	params, _ := selection.CandidateModelParams(r.Config.ModelConfig, nil, model)
	if err := selection.ValidateCandidateRequirements(requirements, model, params, demand); err != nil {
		return err
	}
	format, err := wireFormatForModel(r.Config.GetModelAPIFormat(model))
	if err != nil {
		return fmt.Errorf("%w: %w", selection.ErrNoEligibleCandidates, err)
	}
	supported, ok := r.codecCapabilitiesForFormat(format)
	if !ok {
		return fmt.Errorf("%w: model %q has no request codec", selection.ErrNoEligibleCandidates, model)
	}
	return selection.ValidateCandidateCodec(requirements, model, supported, demand)
}

func (r *OpenAIRouter) eligibleDemandModelRefs(requirements *config.CandidateRequirements, refs []config.ModelRef, demand selection.CandidateDemand) ([]config.ModelRef, error) {
	eligible := make([]config.ModelRef, 0, len(refs))
	for _, ref := range refs {
		if strings.TrimSpace(ref.Model) == "" {
			continue
		}
		if err := r.validateModelDemand(requirements, ref.Model, demand); err == nil {
			eligible = append(eligible, ref)
		}
	}
	if len(eligible) == 0 {
		return nil, fmt.Errorf("%w: no assigned model satisfies the effective request capabilities and budget", selection.ErrNoEligibleCandidates)
	}
	return eligible, nil
}

func (r *OpenAIRouter) decisionEligibleModelRefs(decision *config.Decision, ctx *RequestContext) ([]config.ModelRef, error) {
	requirements := r.candidateRequirements(ctx)
	if !selection.CandidateRequirementsEnabled(requirements) {
		return r.contextEligibleDecisionModelRefs(decision.ModelRefs, decision.Name, ctx.VSRContextTokenCount, ctx)
	}
	demand, err := selection.EffectiveCandidateDemand(ctx.SemanticRequest, decision)
	if err != nil {
		return nil, err
	}
	eligible, err := r.eligibleDemandModelRefs(requirements, decision.ModelRefs, demand)
	if err != nil {
		return nil, err
	}
	// Both filters survive adaptation, fallback, and single-candidate selection.
	ctx.VSREligibleModelRefs = cloneModelRefs(eligible)
	ctx.VSRPolicyEligibleModelRefs = cloneModelRefs(eligible)
	return eligible, nil
}

func (r *OpenAIRouter) modelInRefs(model string, refs []config.ModelRef) bool {
	for _, ref := range refs {
		if model == ref.LoRAName || r.Config.ModelNameMatches(ref.Model, model) {
			return true
		}
	}
	return false
}

// Strict dispatch remains inside the assigned inventory. Internal algorithm
// helpers are allowed only when explicitly declared by this decision.
func (r *OpenAIRouter) validateDispatchRequirements(request *llmprotocol.Request, dispatch *providerDispatch, ctx *RequestContext) error {
	requirements := r.candidateRequirements(ctx)
	if !selection.CandidateRequirementsEnabled(requirements) {
		return nil
	}
	decision := ctx.VSRSelectedDecision
	allowed := false
	if decision != nil {
		refs := decision.ModelRefs
		if ctx.VSREligibleModelRefs != nil {
			refs = ctx.VSREligibleModelRefs
		}
		allowed = r.modelInRefs(dispatch.logicalModel, refs)
		if decision.Action != nil && decision.Action.Type == config.DecisionActionRoute {
			allowed = allowed || r.Config.ModelNameMatches(decision.Action.Destination, dispatch.logicalModel)
		}
		if r.isLooperRequest(ctx) {
			for _, model := range explicitAlgorithmModels(decision.Algorithm) {
				allowed = allowed || (model != "" && r.Config.ModelNameMatches(model, dispatch.logicalModel))
			}
		}
	}
	if !allowed {
		return fmt.Errorf("%w: model %q is outside the selected decision's permission scope", selection.ErrNoEligibleCandidates, dispatch.logicalModel)
	}
	model := dispatch.logicalModel
	if decision != nil {
		for _, ref := range decision.ModelRefs {
			if ref.LoRAName == model && ref.LoRAName != "" {
				model = ref.Model
				break
			}
		}
	}
	return r.validateModelDemand(requirements, model, selection.DemandForRequest(request))
}

func (r *OpenAIRouter) strictRouteActionDestination(decision *config.Decision, demand selection.CandidateDemand, requirements *config.CandidateRequirements) (string, error) {
	refs := []config.ModelRef{{Model: strings.TrimSpace(decision.Action.Destination)}}
	refs = append(refs, decision.ModelRefs...)
	eligible, err := r.eligibleDemandModelRefs(requirements, refs, demand)
	if err != nil {
		return "", err
	}
	return eligible[0].Model, nil
}
