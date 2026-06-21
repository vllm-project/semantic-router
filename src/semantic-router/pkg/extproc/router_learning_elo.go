package extproc

import (
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

func (r *OpenAIRouter) applyEloLearning(
	input routerLearningInput,
) (routerLearningAdaptationResult, bool) {
	cfg, ok := r.eloLearningConfig(input.ctx)
	if !ok {
		return routerLearningAdaptationResult{}, false
	}

	mode := eloAdaptationMode(input.ctx)
	if mode == config.DecisionAdaptationModeBypass {
		result := eloNoChangeResult(input, cfg, mode, routerLearningActionBypass, eloReasonDecisionBypass, "")
		return attachRouterLearningExperience(result, input.experience), true
	}

	stateKey, stateKeyOK := learningStateKeyFromRequest(cfg.EffectiveScope(), input)
	if !stateKeyOK {
		result := eloNoChangeResult(input, cfg, mode, routerLearningActionNoop, eloReasonIdentityMissing, "")
		return attachRouterLearningExperience(result, input.experience), true
	}
	if input.selCtx == nil || len(input.selCtx.CandidateModels) == 0 {
		result := eloNoChangeResult(input, cfg, mode, routerLearningActionNoop, eloReasonNoCandidates, stateKey)
		return attachRouterLearningExperience(result, input.experience), true
	}

	runtime := r.routerLearningRuntimeState()
	scores := runtime.scoreEloCandidates(input, cfg, stateKey)
	if len(scores) == 0 {
		result := eloNoChangeResult(input, cfg, mode, routerLearningActionNoop, eloReasonNoCandidates, stateKey)
		return attachRouterLearningExperience(result, input.experience), true
	}
	winner := scores[0]
	if !eloHasKnownState(scores) {
		result := eloScoreResult(input, cfg, mode, stateKey, scores, winner, eloReasonStateMissing, false)
		return attachRouterLearningExperience(result, input.experience), true
	}
	if input.baseResult == nil || winner.model == input.baseResult.SelectedModel {
		result := eloScoreResult(input, cfg, mode, stateKey, scores, winner, eloReasonBaseBest, false)
		return attachRouterLearningExperience(result, input.experience), true
	}

	result := eloScoreResult(input, cfg, mode, stateKey, scores, winner, eloReasonRatingWin, true)
	return attachRouterLearningExperience(result, input.experience), true
}

func (r *OpenAIRouter) eloLearningConfig(ctx *RequestContext) (config.EloLearningConfig, bool) {
	if r == nil || r.Config == nil || !r.Config.RouterLearning.Enabled {
		return config.EloLearningConfig{}, false
	}
	cfg := r.Config.RouterLearning.Adaptations.Elo
	return cfg, cfg.Enabled
}

func eloAdaptationMode(ctx *RequestContext) string {
	if ctx != nil && ctx.VSRSelectedDecision != nil {
		return ctx.VSRSelectedDecision.Adaptations.EloMode()
	}
	return config.DecisionAdaptationModeApply
}

func (rt *routerLearningRuntime) scoreEloCandidates(
	input routerLearningInput,
	cfg config.EloLearningConfig,
	stateKey string,
) []routerLearningEloScore {
	if rt != nil {
		rt.mu.Lock()
		defer rt.mu.Unlock()
	}
	if input.selCtx == nil {
		return nil
	}
	ratings := map[string]*routerLearningEloRating{}
	if rt != nil && rt.elo.ratings != nil && rt.elo.ratings[stateKey] != nil {
		ratings = rt.elo.ratings[stateKey]
	}

	raw, total := buildEloCandidateScores(input.selCtx.CandidateModels, ratings, cfg)
	normalizeEloCandidateScores(raw, total)
	sortEloCandidateScores(raw, baseSelectedModel(input))
	return raw
}

func buildEloCandidateScores(
	candidates []config.ModelRef,
	ratings map[string]*routerLearningEloRating,
	cfg config.EloLearningConfig,
) ([]routerLearningEloScore, float64) {
	raw := make([]routerLearningEloScore, 0, len(candidates))
	total := 0.0
	for _, candidate := range candidates {
		score, ok := eloCandidateScore(candidate, ratings, cfg)
		if !ok {
			continue
		}
		total += score.score
		raw = append(raw, score)
	}
	return raw, total
}

func eloCandidateScore(
	candidate config.ModelRef,
	ratings map[string]*routerLearningEloRating,
	cfg config.EloLearningConfig,
) (routerLearningEloScore, bool) {
	model := strings.TrimSpace(candidate.Model)
	if model == "" {
		return routerLearningEloScore{}, false
	}
	rating := routerLearningEloRating{
		Rating: eloInitialRating(cfg),
	}
	known := false
	if stored := ratings[model]; stored != nil {
		rating = *stored
		known = stored.Comparisons > 0
	}
	weight := math.Pow(10, rating.Rating/400.0)
	return routerLearningEloScore{
		model:       model,
		rating:      rating.Rating,
		score:       weight,
		comparisons: rating.Comparisons,
		wins:        rating.Wins,
		losses:      rating.Losses,
		ties:        rating.Ties,
		known:       known,
	}, true
}

func normalizeEloCandidateScores(scores []routerLearningEloScore, total float64) {
	if total <= 0 {
		return
	}
	for i := range scores {
		scores[i].score /= total
	}
}

func sortEloCandidateScores(scores []routerLearningEloScore, baseModel string) {
	sort.SliceStable(scores, func(i, j int) bool {
		if scores[i].score == scores[j].score {
			return candidateTieBreaksBefore(scores[i].model, scores[j].model, baseModel)
		}
		return scores[i].score > scores[j].score
	})
}

func baseSelectedModel(input routerLearningInput) string {
	if input.baseResult == nil {
		return ""
	}
	return input.baseResult.SelectedModel
}

func candidateTieBreaksBefore(left string, right string, baseModel string) bool {
	if left == baseModel {
		return true
	}
	if right == baseModel {
		return false
	}
	return left < right
}

func eloScoreResult(
	input routerLearningInput,
	cfg config.EloLearningConfig,
	mode string,
	stateKey string,
	scores []routerLearningEloScore,
	winner routerLearningEloScore,
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
		selectionResult = eloSelectionResult(input.baseResult, winner, scores)
		if selectedModelRef != nil {
			selectionResult.LoRAName = selectedModelRef.LoRAName
		}
	}
	if mode == config.DecisionAdaptationModeObserve && changesModel {
		action = routerLearningActionSwitch
	}
	policy := eloLearningPolicy(cfg, mode, action, reason, stateKey, scores, winner)
	return routerLearningAdaptationResult{
		method:           routerLearningMethodElo,
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

func eloNoChangeResult(
	input routerLearningInput,
	cfg config.EloLearningConfig,
	mode string,
	action routerLearningAction,
	reason string,
	stateKey string,
) routerLearningAdaptationResult {
	if action == "" {
		action = routerLearningActionNoop
	}
	return routerLearningAdaptationResult{
		method: routerLearningMethodElo,
		mode:   mode,
		scope:  cfg.EffectiveScope(),
		action: action,
		reason: reason,
		policy: eloLearningPolicy(
			cfg,
			mode,
			action,
			reason,
			stateKey,
			nil,
			routerLearningEloScore{},
		),
	}
}

func eloSelectionResult(
	baseResult *selection.SelectionResult,
	winner routerLearningEloScore,
	scores []routerLearningEloScore,
) *selection.SelectionResult {
	result := &selection.SelectionResult{
		SelectedModel: winner.model,
		Score:         winner.score,
		Confidence:    eloConfidence(winner),
		Method:        selection.MethodStatic,
		Tier:          selection.TierSupported,
		Reasoning:     fmt.Sprintf("Router Learning Elo selected %s", winner.model),
		AllScores:     eloAllScores(scores),
	}
	if baseResult != nil {
		result.LoRAName = baseResult.LoRAName
		result.Method = baseResult.Method
		result.Tier = baseResult.Tier
	}
	return result
}

func eloAllScores(scores []routerLearningEloScore) map[string]float64 {
	result := make(map[string]float64, len(scores))
	for _, score := range scores {
		result[score.model] = score.score
	}
	return result
}

func eloHasKnownState(scores []routerLearningEloScore) bool {
	for _, score := range scores {
		if score.known {
			return true
		}
	}
	return false
}

func eloConfidence(score routerLearningEloScore) float64 {
	if score.comparisons <= 0 {
		return 0.5
	}
	confidence := 1.0 / (1.0 + math.Exp(-0.2*(float64(score.comparisons)-5)))
	return clamp01(confidence)
}

func eloInitialRating(cfg config.EloLearningConfig) float64 {
	if cfg.InitialRating != nil {
		return *cfg.InitialRating
	}
	return selection.DefaultEloRating
}

func eloKFactor(cfg config.EloLearningConfig) float64 {
	if cfg.KFactor != nil {
		return *cfg.KFactor
	}
	return selection.EloKFactor
}
