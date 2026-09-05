package extproc

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

func (r *OpenAIRouter) attachSuccessEstimateObserveDiagnostics(
	input routerLearningInput,
	cfg config.RouterLearningAdaptationConfig,
	decision *routerLearningDecision,
) {
	if r == nil || decision == nil || decision.policy.Empty() {
		return
	}
	if adaptationMode(input.ctx) == config.DecisionAdaptationModeBypass {
		return
	}
	selCtx := firstNonNilSelectionContext(decision.selectionContext, input.selCtx)
	candidates := r.learningCandidateModels(selCtx, input.ctx, cfg.EffectiveCandidateSet())
	models := successEstimateModelNames(candidates)
	if len(models) == 0 {
		return
	}

	snap := r.routerLearningRuntimeState().freezeEvidenceSnapshot(
		selectionDecisionStateKey(selCtx),
		decisionTier(input.ctx),
		models,
		"",
	)
	estimates := estimateCandidateSuccess(snap, models, successEstimateConfig{Now: snap.takenAt})

	diag := decision.policy.Details.Adaptation
	if diag == nil {
		diag = &routerLearningAdaptationDiagnostics{
			candidateSet: cfg.EffectiveCandidateSet(),
			strategy:     cfg.EffectiveStrategy(),
			baseModel:    selectedModelName(input.baseResult),
			decision:     strings.TrimSpace(selCtxDecisionName(selCtx)),
			decisionTier: decisionTier(input.ctx),
		}
		decision.policy.Details.Adaptation = diag
	}
	diag.snapshotIdentity = snap.identity
	diag.successEstimates = estimates
}

func successEstimateModelNames(refs []config.ModelRef) []string {
	if len(refs) == 0 {
		return nil
	}
	names := make([]string, 0, len(refs))
	seen := map[string]struct{}{}
	for _, ref := range refs {
		model := strings.TrimSpace(ref.Model)
		if model == "" {
			continue
		}
		if _, ok := seen[model]; ok {
			continue
		}
		seen[model] = struct{}{}
		names = append(names, model)
	}
	return names
}

func selCtxDecisionName(selCtx *selection.SelectionContext) string {
	if selCtx == nil {
		return ""
	}
	return strings.TrimSpace(selCtx.DecisionName)
}
