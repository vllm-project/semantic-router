package extproc

import (
	"errors"
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

func decisionUsesAutomaticOutput(request *llmprotocol.Request, decision *config.Decision) bool {
	if request == nil || decision == nil {
		return false
	}
	if request.Sampling.AutomaticOutput {
		return true
	}
	params := decision.GetRequestParamsConfig()
	if params == nil || !params.DefaultMaxTokens.IsAuto() {
		return false
	}
	if request.Sampling.MaxOutputTokens == nil {
		return true
	}
	// Request params remove blocked fields before applying defaults. Recognize
	// that policy here without mutating ingress; EffectiveCandidateRequest still
	// validates and applies the complete policy before any provider render.
	for _, field := range params.BlockedParams {
		switch strings.TrimSpace(field) {
		case "max_tokens", "max_completion_tokens", "max_output_tokens":
			return true
		}
	}
	return false
}

// prepareAutomaticCandidates preserves the full input whenever any candidate
// fits. Only a provider-confirmed overflow of the entire pool can activate the
// operator's existing compression policy. A failed renderer never means overflow.
func (r *OpenAIRouter) prepareAutomaticCandidates(ctx *RequestContext, refs []config.ModelRef) error {
	for attempt := 0; attempt < 2; attempt++ {
		demands, overflows, err := r.renderAutomaticCandidates(ctx, refs)
		if err != nil {
			return err
		}
		if len(demands) > 0 {
			ctx.AutomaticCandidateDemands = demands
			return nil
		}
		if len(overflows) == 0 {
			return automaticOutputUnsupported("at least one compatible configured candidate")
		}
		cfg := contextOverflowConfig(ctx)
		if attempt > 0 || cfg == nil {
			return overflowBudgetError("The provider-rendered input exceeds every candidate's context window")
		}
		model, available := "", 0
		for _, ref := range overflows {
			budget := r.automaticOverflowInputBudget(ref.Model, cfg)
			if budget > available {
				model, available = ref.Model, budget
			}
		}
		if available <= 0 {
			return overflowBudgetError("The configured compression reserve leaves no usable input capacity")
		}
		if err := r.applyContextOverflow(ctx, ctx.SemanticRequest, cfg, ctx.VSRSelectedDecision, model, available); err != nil {
			return err
		}
	}
	return overflowBudgetError("The compressed input still exceeds the provider context window")
}

func (r *OpenAIRouter) renderAutomaticCandidates(ctx *RequestContext, refs []config.ModelRef) (map[string]selection.CandidateDemand, []config.ModelRef, error) {
	demands := make(map[string]selection.CandidateDemand, len(refs))
	overflow := []config.ModelRef{}
	for _, ref := range refs {
		view, err := selection.EffectiveCandidateRequest(ctx.SemanticRequest, ctx.VSRSelectedDecision)
		if err != nil {
			return nil, nil, err
		}
		// Preserve the existing capability/known-metadata filter before the
		// provider call. A one-token minimum tests admission, not a forecast.
		admission := selection.DemandForRequest(view)
		admission.InputTokens = 0
		admission.MaxOutputTokens = llmprotocol.Int64(1)
		if admissionErr := r.validateModelDemand(r.candidateRequirements(ctx), ref.Model, admission); admissionErr != nil {
			continue
		}
		if ref.LoRAName != "" {
			return nil, nil, automaticOutputUnsupported("a non-LoRA single-backend candidate")
		}
		useReasoning := ref.UseReasoning != nil && *ref.UseReasoning
		dispatch, err := r.resolveProviderDispatch(ref.Model, ctx.VSRSelectedDecision.Name, useReasoning)
		if err != nil {
			return nil, nil, err
		}
		if err := r.resolveAutomaticOutput(view, dispatch, ctx); err != nil {
			var budget *selection.RequestBudgetError
			if errors.As(err, &budget) && budget.Code == "context_length_exceeded" {
				overflow = append(overflow, ref)
				continue
			}
			return nil, nil, err
		}
		demand := selection.DemandForRequest(view)
		if err := r.validateModelDemand(r.candidateRequirements(ctx), ref.Model, demand); err != nil {
			continue
		}
		demands[ref.Model] = demand
	}
	return demands, overflow, nil
}

func (r *OpenAIRouter) automaticOverflowInputBudget(model string, cfg *config.ContextCompressionPluginConfig) int {
	params := r.Config.ModelConfig[model]
	reserve := 1
	if cfg.Budget != nil && cfg.Budget.ReserveOutputTokens != nil && !cfg.Budget.ReserveOutputTokens.Auto {
		reserve = max(reserve, cfg.Budget.ReserveOutputTokens.Value)
	}
	return max(0, params.ContextWindowSize-reserve)
}

func (r *OpenAIRouter) automaticEligibleRefs(refs []config.ModelRef, ctx *RequestContext) ([]config.ModelRef, error) {
	if ctx.AutomaticCandidateDemands == nil {
		if err := r.prepareAutomaticCandidates(ctx, refs); err != nil {
			return nil, err
		}
	}
	eligible := make([]config.ModelRef, 0, len(refs))
	for _, ref := range refs {
		if _, ok := ctx.AutomaticCandidateDemands[ref.Model]; ok {
			eligible = append(eligible, ref)
		}
	}
	if len(eligible) == 0 {
		return nil, fmt.Errorf("%w: no candidate has a provider-rendered output budget", selection.ErrNoEligibleCandidates)
	}
	ctx.VSREligibleModelRefs = cloneModelRefs(eligible)
	ctx.VSRPolicyEligibleModelRefs = cloneModelRefs(eligible)
	return eligible, nil
}

// Re-run after all provider-facing mutations. A second render is bounded and
// only follows the configured compression of an explicitly confirmed overflow.
func (r *OpenAIRouter) prepareAutomaticDispatch(ctx *RequestContext, request *llmprotocol.Request, dispatch *providerDispatch) error {
	err := r.resolveAutomaticOutput(request, dispatch, ctx)
	var budget *selection.RequestBudgetError
	if !errors.As(err, &budget) || budget.Code != "context_length_exceeded" {
		return err
	}
	cfg := contextOverflowConfig(ctx)
	if cfg == nil {
		return err
	}
	available := r.automaticOverflowInputBudget(dispatch.logicalModel, cfg)
	if available <= 0 {
		return err
	}
	if err = r.applyContextOverflow(ctx, request, cfg, nil, dispatch.logicalModel, available); err != nil {
		return err
	}
	return r.resolveAutomaticOutput(request, dispatch, ctx)
}
