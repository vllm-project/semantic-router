package extproc

import (
	"fmt"
	"sort"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

func (r *OpenAIRouter) applyPersonalizationLearning(
	input routerLearningInput,
) (routerLearningAdaptationResult, bool) {
	cfg, ok := r.personalizationLearningConfig(input.ctx)
	if !ok {
		return routerLearningAdaptationResult{}, false
	}

	mode := personalizationAdaptationMode(input.ctx)
	if mode == config.DecisionAdaptationModeBypass {
		result := personalizationNoChangeResult(input, cfg, mode, routerLearningActionBypass, personalizationReasonDecisionBypass, "")
		return attachRouterLearningExperience(result, input.experience), true
	}

	userID := learningUserIDFromRequest(input)
	stateKey, stateKeyOK := personalizationStateKeyFromRequest(cfg.EffectiveScope(), input)
	if userID == "" || !stateKeyOK {
		result := personalizationNoChangeResult(input, cfg, mode, routerLearningActionNoop, personalizationReasonIdentityMissing, "")
		return attachRouterLearningExperience(result, input.experience), true
	}
	if input.selCtx == nil || len(input.selCtx.CandidateModels) == 0 {
		result := personalizationNoChangeResult(input, cfg, mode, routerLearningActionNoop, personalizationReasonNoCandidates, stateKey)
		return attachRouterLearningExperience(result, input.experience), true
	}

	runtime := r.routerLearningRuntimeState()
	scores := r.scorePersonalizationCandidates(runtime, input, stateKey)
	if len(scores) == 0 {
		result := personalizationNoChangeResult(input, cfg, mode, routerLearningActionNoop, personalizationReasonNoCandidates, stateKey)
		return attachRouterLearningExperience(result, input.experience), true
	}
	winner := scores[0]
	if !personalizationHasKnownState(scores) {
		result := personalizationScoreResult(input, cfg, mode, stateKey, userID, scores, winner, personalizationReasonStateMissing, false)
		return attachRouterLearningExperience(result, input.experience), true
	}
	if input.baseResult == nil || winner.model == input.baseResult.SelectedModel {
		result := personalizationScoreResult(input, cfg, mode, stateKey, userID, scores, winner, personalizationReasonBaseBest, false)
		return attachRouterLearningExperience(result, input.experience), true
	}
	result := personalizationScoreResult(input, cfg, mode, stateKey, userID, scores, winner, personalizationReasonPreferenceWin, true)
	return attachRouterLearningExperience(result, input.experience), true
}

func (r *OpenAIRouter) personalizationLearningConfig(ctx *RequestContext) (config.PersonalizationLearningConfig, bool) {
	if r == nil || r.Config == nil || !r.Config.RouterLearning.Enabled {
		return config.PersonalizationLearningConfig{}, false
	}
	cfg := r.Config.RouterLearning.Adaptations.Personalization
	return cfg, cfg.Enabled
}

func personalizationAdaptationMode(ctx *RequestContext) string {
	if ctx != nil && ctx.VSRSelectedDecision != nil {
		return ctx.VSRSelectedDecision.Adaptations.PersonalizationMode()
	}
	return config.DecisionAdaptationModeApply
}

func (r *OpenAIRouter) scorePersonalizationCandidates(
	runtime *routerLearningRuntime,
	input routerLearningInput,
	stateKey string,
) []routerLearningPersonalizationScore {
	if input.selCtx == nil {
		return nil
	}
	preferences := personalizationPreferencesSnapshot(runtime, stateKey)
	baseScores := cloneSelectionScores(nil)
	if input.baseResult != nil {
		baseScores = cloneSelectionScores(input.baseResult.AllScores)
	}

	scores := make([]routerLearningPersonalizationScore, 0, len(input.selCtx.CandidateModels))
	for _, candidate := range input.selCtx.CandidateModels {
		score, ok := r.personalizationCandidateScore(candidate, input, baseScores, preferences)
		if !ok {
			continue
		}
		scores = append(scores, score)
	}
	sortPersonalizationCandidateScores(scores, baseSelectedModel(input))
	return scores
}

func personalizationPreferencesSnapshot(
	runtime *routerLearningRuntime,
	stateKey string,
) map[string]routerLearningPersonalizationModelState {
	preferences := map[string]routerLearningPersonalizationModelState{}
	if runtime == nil {
		return preferences
	}
	runtime.mu.Lock()
	defer runtime.mu.Unlock()

	storedPreferences := runtime.personalization.preferences[stateKey]
	if storedPreferences == nil {
		return preferences
	}
	preferences = make(map[string]routerLearningPersonalizationModelState, len(storedPreferences))
	for model, preference := range storedPreferences {
		if preference != nil {
			preferences[model] = *preference
		}
	}
	return preferences
}

func (r *OpenAIRouter) personalizationCandidateScore(
	candidate config.ModelRef,
	input routerLearningInput,
	baseScores map[string]float64,
	preferences map[string]routerLearningPersonalizationModelState,
) (routerLearningPersonalizationScore, bool) {
	model := strings.TrimSpace(candidate.Model)
	if model == "" {
		return routerLearningPersonalizationScore{}, false
	}
	baseScore, ok := baseScores[model]
	if !ok {
		baseScore = r.banditQualityPrior(input, model)
	}
	metrics := personalizationPreferenceMetrics(preferences[model])
	score := 0.4*clamp01(baseScore) + 0.6*clamp01(metrics.preference)
	return routerLearningPersonalizationScore{
		model:        model,
		baseScore:    clamp01(baseScore),
		preference:   metrics.preference,
		score:        score,
		positive:     metrics.positive,
		negative:     metrics.negative,
		interactions: metrics.interactions,
		known:        metrics.known,
	}, true
}

type personalizationPreferenceSnapshot struct {
	preference   float64
	positive     float64
	negative     float64
	interactions int
	known        bool
}

func personalizationPreferenceMetrics(
	preference routerLearningPersonalizationModelState,
) personalizationPreferenceSnapshot {
	metrics := personalizationPreferenceSnapshot{preference: 0.5}
	if preference.Interactions <= 0 {
		return metrics
	}
	total := preference.Positive + preference.Negative
	if total > 0 {
		metrics.preference = preference.Positive / total
	}
	metrics.positive = preference.Positive
	metrics.negative = preference.Negative
	metrics.interactions = preference.Interactions
	metrics.known = true
	return metrics
}

func sortPersonalizationCandidateScores(scores []routerLearningPersonalizationScore, baseModel string) {
	sort.SliceStable(scores, func(i, j int) bool {
		if scores[i].score == scores[j].score {
			return candidateTieBreaksBefore(scores[i].model, scores[j].model, baseModel)
		}
		return scores[i].score > scores[j].score
	})
}

func personalizationScoreResult(
	input routerLearningInput,
	cfg config.PersonalizationLearningConfig,
	mode string,
	stateKey string,
	userID string,
	scores []routerLearningPersonalizationScore,
	winner routerLearningPersonalizationScore,
	reason string,
	changesModel bool,
) routerLearningAdaptationResult {
	selectionCtx := input.selCtx
	selectionResult := input.baseResult
	selectedModelRef := input.selectedModelRef
	action := routerLearningActionStay
	if changesModel {
		action = routerLearningActionSwitch
		modelRef := selectedModelRefByModel(selectionCtx, winner.model)
		if modelRef != nil {
			selectedModelRef = modelRef
		}
		selectionResult = personalizationSelectionResult(input.baseResult, winner, scores)
		if selectedModelRef != nil {
			selectionResult.LoRAName = selectedModelRef.LoRAName
		}
	}
	if mode == config.DecisionAdaptationModeObserve && changesModel {
		action = routerLearningActionSwitch
	}
	policy := personalizationLearningPolicy(cfg, mode, action, reason, stateKey, userID, scores, winner)
	return routerLearningAdaptationResult{
		method:           routerLearningMethodPersonalization,
		mode:             mode,
		scope:            cfg.EffectiveScope(),
		action:           action,
		reason:           reason,
		changesModel:     changesModel && mode != config.DecisionAdaptationModeObserve,
		selectionContext: selectionCtx,
		selectionResult:  selectionResult,
		selectedModelRef: selectedModelRef,
		policy:           policy,
	}
}

func personalizationNoChangeResult(
	input routerLearningInput,
	cfg config.PersonalizationLearningConfig,
	mode string,
	action routerLearningAction,
	reason string,
	stateKey string,
) routerLearningAdaptationResult {
	if action == "" {
		action = routerLearningActionNoop
	}
	return routerLearningAdaptationResult{
		method: routerLearningMethodPersonalization,
		mode:   mode,
		scope:  cfg.EffectiveScope(),
		action: action,
		reason: reason,
		policy: personalizationLearningPolicy(
			cfg,
			mode,
			action,
			reason,
			stateKey,
			"",
			nil,
			routerLearningPersonalizationScore{},
		),
	}
}

func personalizationSelectionResult(
	baseResult *selection.SelectionResult,
	winner routerLearningPersonalizationScore,
	scores []routerLearningPersonalizationScore,
) *selection.SelectionResult {
	result := &selection.SelectionResult{
		SelectedModel: winner.model,
		Score:         winner.score,
		Confidence:    clamp01(0.5 + winner.preference/2),
		Method:        selection.MethodStatic,
		Tier:          selection.TierSupported,
		Reasoning:     fmt.Sprintf("Router Learning personalization selected %s", winner.model),
		AllScores:     personalizationAllScores(scores),
	}
	if baseResult != nil {
		result.LoRAName = baseResult.LoRAName
		result.Method = baseResult.Method
		result.Tier = baseResult.Tier
	}
	return result
}

func personalizationAllScores(scores []routerLearningPersonalizationScore) map[string]float64 {
	result := make(map[string]float64, len(scores))
	for _, score := range scores {
		result[score.model] = score.score
	}
	return result
}

func personalizationHasKnownState(scores []routerLearningPersonalizationScore) bool {
	for _, score := range scores {
		if score.known {
			return true
		}
	}
	return false
}
