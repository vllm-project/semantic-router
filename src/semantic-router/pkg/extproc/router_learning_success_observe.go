package extproc

import (
	"strings"
	"time"

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
	models := successEstimateCandidateNames(candidates)
	if len(models) == 0 {
		return
	}

	snap := r.routerLearningRuntimeState().freezeEvidenceSnapshot(
		selectionDecisionStateKey(selCtx),
		decisionTier(input.ctx),
		models,
		"",
	)
	estimates := estimateCandidateSuccess(snap, models, successEstimateObserveConfig(snap, cfg))

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

func successEstimateCandidateNames(refs []config.ModelRef) []string {
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
		lora := strings.TrimSpace(ref.LoRAName)
		key := model + "\x00" + lora
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		names = append(names, routedCandidateName(ref))
	}
	return names
}

func routedCandidateName(ref config.ModelRef) string {
	if name := strings.TrimSpace(ref.LoRAName); name != "" {
		return name
	}
	return strings.TrimSpace(ref.Model)
}

func successEstimateObserveConfig(
	snap routerLearningEvidenceSnapshot,
	cfg config.RouterLearningAdaptationConfig,
) successEstimateConfig {
	return successEstimateConfig{
		Now:        snap.takenAt,
		StaleAfter: time.Duration(cfg.Success.EffectiveStaleAfterSeconds()) * time.Second,
		Outcome:    cfg.Success.EffectiveOutcome(),
	}
}

func selCtxDecisionName(selCtx *selection.SelectionContext) string {
	if selCtx == nil {
		return ""
	}
	return strings.TrimSpace(selCtx.DecisionName)
}
