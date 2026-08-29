package extproc

import (
	"errors"
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
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
	ctx *RequestContext,
) ([]config.ModelRef, error) {
	eligible, excluded := r.contextEligibleModelRefs(refs, contextTokens)
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

func (r *OpenAIRouter) modelRefExceedsContextWindow(ref config.ModelRef, contextTokens int) bool {
	return r.modelNameExceedsContextWindow(ref.Model, contextTokens)
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
	seen := make(map[string]struct{}, len(eligible))
	for _, ref := range eligible {
		model := strings.TrimSpace(ref.Model)
		if model == "" {
			continue
		}
		seen[model+"\x00"+strings.TrimSpace(ref.LoRAName)] = struct{}{}
	}
	if len(seen) >= decision.Algorithm.MinimumCandidates {
		return nil
	}
	return fmt.Errorf(
		"%w: decision %q requires at least %d eligible candidates for %d request tokens, got %d",
		errNoContextEligibleDecisionModel,
		decision.Name,
		decision.Algorithm.MinimumCandidates,
		contextTokens,
		len(seen),
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
