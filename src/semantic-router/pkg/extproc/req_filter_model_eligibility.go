package extproc

import (
	"errors"
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

var errNoContextEligibleDecisionModel = errors.New("no decision model can satisfy the request context")

// contextEligibleModelRefs applies only contracts that can be established from
// local configuration. Missing or zero context-window metadata remains
// eligible so a partial inventory does not turn into an accidental outage.
func (r *OpenAIRouter) contextEligibleModelRefs(
	refs []config.ModelRef,
	contextTokens int,
) (eligible []config.ModelRef, excluded int) {
	if len(refs) == 0 {
		return nil, 0
	}
	eligible = make([]config.ModelRef, 0, len(refs))
	for _, ref := range refs {
		if r.modelRefExceedsContextWindow(ref, contextTokens) {
			excluded++
			continue
		}
		eligible = append(eligible, ref)
	}
	return eligible, excluded
}

func (r *OpenAIRouter) contextEligibleDecisionModelRefs(
	refs []config.ModelRef,
	decisionName string,
	contextTokens int,
	minimum int,
	ctx *RequestContext,
) ([]config.ModelRef, error) {
	eligible, excluded := r.contextEligibleModelRefs(refs, contextTokens)
	if excluded > 0 {
		if fallback, fallbackExcluded, ok := r.reserveFallbackModelRefs(refs, eligible, contextTokensWithoutReserves(ctx, contextTokens), minimum, ctx); ok {
			dropContextReserves(ctx, decisionName)
			eligible, excluded, contextTokens = fallback, fallbackExcluded, ctx.VSRContextTokenCount
		}
	}
	if len(eligible) == 0 && excluded > 0 {
		return nil, fmt.Errorf(
			"%w: decision %q requires %d request tokens but every configured candidate has a smaller context window",
			errNoContextEligibleDecisionModel,
			decisionName,
			contextTokens,
		)
	}
	ctx.VSREligibleModelRefs = cloneModelRefs(eligible)
	if excluded > 0 {
		logging.ComponentEvent("extproc", "decision_models_context_filtered", map[string]interface{}{
			"request_id":          ctx.RequestID,
			"decision":            decisionName,
			"context_tokens":      contextTokens,
			"excluded_candidates": excluded,
			"eligible_candidates": len(eligible),
		})
	}
	return eligible, nil
}

// The estimate keeps conservative reserves (llmprotocol.InputEstimate
// ReserveTokens) so that candidates are chosen safely, but a reserve is no
// evidence that a request does not fit. A refusal rests on the estimate
// without reserves, and when a decision has to fall back to that count, the
// request continues with it.
func contextTokensWithoutReserves(ctx *RequestContext, contextTokens int) int {
	return max(0, contextTokens-ctx.VSRContextReserveTokens)
}

// reserveFallbackModelRefs is the window filter's fallback, shared by routing
// and eval: when the conservative count leaves fewer capable candidates than
// the decision needs, it returns the refs the count without reserves admits,
// provided they hold more capable candidates.
func (r *OpenAIRouter) reserveFallbackModelRefs(
	refs, eligible []config.ModelRef,
	withoutReserves, minimum int,
	ctx *RequestContext,
) ([]config.ModelRef, int, bool) {
	usable := r.capableModelRefCount(eligible, ctx)
	if usable >= minimum {
		return nil, 0, false
	}
	fallback, excluded := r.contextEligibleModelRefs(refs, withoutReserves)
	if r.capableModelRefCount(fallback, ctx) <= usable {
		return nil, 0, false
	}
	return fallback, excluded, true
}

// capableModelRefCount counts the distinct candidates, as the minimum check
// counts them, that also declare what the request needs on the request the
// capability filter checks, so the fallback does not keep a model that
// filter drops while excluding one that would serve.
func (r *OpenAIRouter) capableModelRefCount(refs []config.ModelRef, ctx *RequestContext) int {
	request, err := r.selectionCapabilityRequest(ctx)
	if err != nil || request == nil {
		return distinctModelRefCount(refs)
	}
	requirements := r.candidateRequirements(ctx)
	capable := make([]config.ModelRef, 0, len(refs))
	for _, ref := range refs {
		if r.candidateCapabilityMismatch(ref, request, ctx.VSRSelectedDecision, requirements, ctx.AutomaticCandidateDemands, contextReservesCounted(ctx)) == nil {
			capable = append(capable, ref)
		}
	}
	return distinctModelRefCount(capable)
}

// distinctModelRefCount counts candidates by model and LoRA.
func distinctModelRefCount(refs []config.ModelRef) int {
	seen := make(map[string]struct{}, len(refs))
	for _, ref := range refs {
		if model := strings.TrimSpace(ref.Model); model != "" {
			seen[model+"\x00"+strings.TrimSpace(ref.LoRAName)] = struct{}{}
		}
	}
	return len(seen)
}

// contextReservesCounted reports whether checks after the window filter still
// count the estimate's reserves, which they do until a fallback drops them.
func contextReservesCounted(ctx *RequestContext) bool {
	return ctx != nil && ctx.VSRContextReserveTokens > 0
}

// candidateSelectionDemand is the demand live selection scores candidates on.
// It counts reserves exactly when the eligibility filters still do, so
// selection and preview rank on the count the filters used.
func candidateSelectionDemand(ctx *RequestContext, decision *config.Decision) selection.CandidateDemand {
	demand, _ := selection.EffectiveCandidateDemand(ctx.SemanticRequest, decision)
	if !contextReservesCounted(ctx) {
		demand = demand.WithoutReserves()
	}
	return demand
}

// minimumContextCandidates is how many candidates a decision needs to keep
// before the window filter falls back to the count without reserves.
func minimumContextCandidates(decision *config.Decision) int {
	if decision != nil && decision.Algorithm != nil && decision.Algorithm.MinimumCandidates > 1 {
		return decision.Algorithm.MinimumCandidates
	}
	return 1
}

func dropContextReserves(ctx *RequestContext, decisionName string) {
	logging.ComponentEvent("extproc", "context_reserves_dropped", map[string]interface{}{
		"request_id":     ctx.RequestID,
		"decision":       decisionName,
		"context_tokens": ctx.VSRContextTokenCount,
		"reserve_tokens": ctx.VSRContextReserveTokens,
	})
	ctx.VSRContextTokenCount = contextTokensWithoutReserves(ctx, ctx.VSRContextTokenCount)
	ctx.VSRContextReserveTokens = 0
}

func (r *OpenAIRouter) modelRefExceedsContextWindow(ref config.ModelRef, contextTokens int) bool {
	return r.modelNameExceedsContextWindow(ref.Model, contextTokens)
}

// decisionRouteActionDestination resolves a matched decision's route action.
// The action is terminal: the destination, or an eligible decision candidate
// when the destination cannot satisfy the request context, overrides a
// caller-pinned model so a detected prompt attack can never fall back to the
// caller's choice. With no eligible safe model at all the request fails
// closed.
func (r *OpenAIRouter) decisionRouteActionDestination(
	decision *config.Decision,
	ctx *RequestContext,
) (string, bool, error) {
	if decision == nil || decision.Action == nil ||
		decision.Action.Type != config.DecisionActionRoute {
		return "", false, nil
	}
	destination := strings.TrimSpace(decision.Action.Destination)
	if destination == "" {
		return "", false, nil
	}
	if decisionUsesAutomaticOutput(ctx.SemanticRequest, decision) {
		refs := append([]config.ModelRef{{Model: destination}}, decision.ModelRefs...)
		eligible, err := r.automaticEligibleRefs(refs, ctx)
		if err != nil {
			return "", false, err
		}
		return eligible[0].Model, true, nil
	}
	if requirements := r.candidateRequirements(ctx); selection.CandidateRequirementsEnabled(requirements) {
		refs := append([]config.ModelRef{{Model: destination}}, decision.ModelRefs...)
		eligible, err := r.eligibleRequestModelRefs(requirements, refs, ctx.SemanticRequest, decision, 1, ctx)
		if err != nil {
			return "", false, err
		}
		return eligible[0].Model, true, nil
	}
	if !r.modelNameExceedsContextWindow(destination, ctx.VSRContextTokenCount) {
		logging.ComponentEvent("extproc", "route_action_applied", map[string]interface{}{
			"request_id":  ctx.RequestID,
			"decision":    decision.Name,
			"destination": destination,
		})
		return destination, true, nil
	}
	eligible, _ := r.contextEligibleModelRefs(decision.ModelRefs, ctx.VSRContextTokenCount)
	if len(eligible) > 0 {
		logging.ComponentEvent("extproc", "route_action_destination_ineligible", map[string]interface{}{
			"request_id":     ctx.RequestID,
			"decision":       decision.Name,
			"destination":    destination,
			"fallback":       eligible[0].Model,
			"context_tokens": ctx.VSRContextTokenCount,
		})
		return eligible[0].Model, true, nil
	}
	withoutReserves := contextTokensWithoutReserves(ctx, ctx.VSRContextTokenCount)
	if !r.modelNameExceedsContextWindow(destination, withoutReserves) {
		dropContextReserves(ctx, decision.Name)
		return destination, true, nil
	}
	if eligible, _ := r.contextEligibleModelRefs(decision.ModelRefs, withoutReserves); len(eligible) > 0 {
		dropContextReserves(ctx, decision.Name)
		return eligible[0].Model, true, nil
	}
	return "", false, fmt.Errorf(
		"%w: route action destination %q and every candidate of decision %q have a smaller context window than the %d request tokens",
		errNoContextEligibleDecisionModel,
		destination,
		decision.Name,
		ctx.VSRContextTokenCount,
	)
}

func validateMinimumEligibleDecisionModels(
	decision *config.Decision,
	eligible []config.ModelRef,
	contextTokens int,
) error {
	if decision == nil || decision.Algorithm == nil ||
		decision.Algorithm.MinimumCandidates <= 0 {
		return nil
	}
	count := distinctModelRefCount(eligible)
	if count >= decision.Algorithm.MinimumCandidates {
		return nil
	}
	return fmt.Errorf(
		"%w: decision %q requires at least %d eligible candidates for %d request tokens, got %d",
		errNoContextEligibleDecisionModel,
		decision.Name,
		decision.Algorithm.MinimumCandidates,
		contextTokens,
		count,
	)
}

func (r *OpenAIRouter) modelNameExceedsContextWindow(model string, contextTokens int) bool {
	if r == nil || r.Config == nil || contextTokens <= 0 {
		return false
	}
	params, ok := r.Config.ModelConfig[strings.TrimSpace(model)]
	return ok && params.ContextWindowSize > 0 && contextTokens > params.ContextWindowSize
}

// contextIneligibleAlgorithmModelCount covers explicit multi-model control
// models that can bypass decision.modelRefs during execution. The algorithm is
// rejected instead of rewritten so planner, judge, and synthesis semantics stay
// visible and deterministic.
func (r *OpenAIRouter) contextIneligibleAlgorithmModelCount(
	decision *config.Decision,
	contextTokens int,
) int {
	if decision == nil || decision.Algorithm == nil || contextTokens <= 0 {
		return 0
	}
	models := explicitAlgorithmModels(decision.Algorithm)
	seen := make(map[string]struct{}, len(models))
	count := 0
	for _, model := range models {
		model = strings.TrimSpace(model)
		if model == "" {
			continue
		}
		if _, ok := seen[model]; ok {
			continue
		}
		seen[model] = struct{}{}
		if r.modelNameExceedsContextWindow(model, contextTokens) {
			count++
		}
	}
	return count
}

func explicitAlgorithmModels(algorithm *config.AlgorithmConfig) []string {
	if algorithm == nil {
		return nil
	}
	var models []string
	if fusion := algorithm.Fusion; fusion != nil {
		models = append(models, fusion.Model)
		models = append(models, fusion.AnalysisModels...)
	}
	if remom := algorithm.ReMoM; remom != nil {
		models = append(models, remom.SynthesisModel)
	}
	if workflows := algorithm.Workflows; workflows != nil {
		models = append(models, workflows.Planner.Model, workflows.Final.Model)
		for _, role := range workflows.Roles {
			models = append(models, role.Models...)
		}
	}
	return models
}
