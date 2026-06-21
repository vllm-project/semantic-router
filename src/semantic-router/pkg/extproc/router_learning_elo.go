package extproc

import (
	"fmt"
	"math"
	"sort"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

const (
	routerLearningMethodElo routerLearningMethod = "elo"

	eloReasonBaseBest        = "base_best"
	eloReasonDecisionBypass  = "decision_bypass"
	eloReasonIdentityMissing = "identity_missing"
	eloReasonNoCandidates    = "no_candidates"
	eloReasonRatingWin       = "rating_win"
	eloReasonStateMissing    = "state_missing"
)

type routerLearningEloState struct {
	ratings map[string]map[string]*routerLearningEloRating
}

type routerLearningEloRating struct {
	Model       string
	Rating      float64
	Comparisons int
	Wins        int
	Losses      int
	Ties        int
	LastUpdated time.Time
}

type routerLearningEloScore struct {
	model       string
	rating      float64
	score       float64
	comparisons int
	wins        int
	losses      int
	ties        int
	known       bool
}

type routerLearningEloScoreDiagnostic struct {
	Model       string  `json:"model"`
	Score       float64 `json:"score"`
	Rating      float64 `json:"rating"`
	Comparisons int     `json:"comparisons"`
	Wins        int     `json:"wins"`
	Losses      int     `json:"losses"`
	Ties        int     `json:"ties"`
}

func (rt *routerLearningRuntime) updateEloFeedback(feedback *selection.Feedback) bool {
	cfg, ok := rt.eloFeedbackConfig()
	if !ok {
		return false
	}
	stateKey, ok := learningStateKeyFromParts(
		cfg.EffectiveScope(),
		feedback.DecisionName,
		feedback.SessionID,
		feedback.ConversationID,
	)
	if !ok {
		return false
	}
	rt.updateEloRatings(stateKey, cfg, feedback)
	return true
}

func (rt *routerLearningRuntime) eloFeedbackConfig() (config.EloLearningConfig, bool) {
	if rt == nil || rt.config == nil || !rt.config.RouterLearning.Enabled {
		return config.EloLearningConfig{}, false
	}
	cfg := rt.config.RouterLearning.Adaptations.Elo
	return cfg, cfg.Enabled
}

func (rt *routerLearningRuntime) updateEloRatings(
	stateKey string,
	cfg config.EloLearningConfig,
	feedback *selection.Feedback,
) {
	if rt == nil || strings.TrimSpace(stateKey) == "" || feedback == nil {
		return
	}
	rt.mu.Lock()
	defer rt.mu.Unlock()

	ratings := rt.eloRatings(stateKey)
	switch {
	case feedback.WinnerModel != "" && feedback.LoserModel != "":
		rt.applyEloPairwiseFeedbackLocked(ratings, cfg, feedback)
	case feedback.WinnerModel != "":
		rt.applyEloWinnerOnlyFeedbackLocked(ratings, cfg, feedback.WinnerModel)
	case feedback.LoserModel != "":
		rt.applyEloLoserOnlyFeedbackLocked(ratings, cfg, feedback.LoserModel)
	}
}

func (rt *routerLearningRuntime) applyEloPairwiseFeedbackLocked(
	ratings map[string]*routerLearningEloRating,
	cfg config.EloLearningConfig,
	feedback *selection.Feedback,
) {
	winner := eloRatingLocked(ratings, cfg, feedback.WinnerModel)
	loser := eloRatingLocked(ratings, cfg, feedback.LoserModel)

	expectedWinner := 1.0 / (1.0 + math.Pow(10, (loser.Rating-winner.Rating)/400.0))
	expectedLoser := 1.0 - expectedWinner
	actualWinner, actualLoser := 1.0, 0.0
	if feedback.Tie {
		actualWinner = 0.5
		actualLoser = 0.5
	}
	k := eloKFactor(cfg)
	winner.Rating += k * (actualWinner - expectedWinner)
	loser.Rating += k * (actualLoser - expectedLoser)
	winner.Comparisons++
	loser.Comparisons++
	if feedback.Tie {
		winner.Ties++
		loser.Ties++
	} else {
		winner.Wins++
		loser.Losses++
	}
	now := time.Now()
	winner.LastUpdated = now
	loser.LastUpdated = now
}

func (rt *routerLearningRuntime) applyEloWinnerOnlyFeedbackLocked(
	ratings map[string]*routerLearningEloRating,
	cfg config.EloLearningConfig,
	model string,
) {
	rating := eloRatingLocked(ratings, cfg, model)
	rating.Rating += eloKFactor(cfg) * 0.1
	rating.Comparisons++
	rating.Wins++
	rating.LastUpdated = time.Now()
}

func (rt *routerLearningRuntime) applyEloLoserOnlyFeedbackLocked(
	ratings map[string]*routerLearningEloRating,
	cfg config.EloLearningConfig,
	model string,
) {
	rating := eloRatingLocked(ratings, cfg, model)
	rating.Rating -= eloKFactor(cfg) * 0.1
	rating.Comparisons++
	rating.Losses++
	rating.LastUpdated = time.Now()
}

func (rt *routerLearningRuntime) eloRatings(stateKey string) map[string]*routerLearningEloRating {
	if rt.elo.ratings == nil {
		rt.elo.ratings = map[string]map[string]*routerLearningEloRating{}
	}
	if rt.elo.ratings[stateKey] == nil {
		rt.elo.ratings[stateKey] = map[string]*routerLearningEloRating{}
	}
	return rt.elo.ratings[stateKey]
}

func eloRatingLocked(
	ratings map[string]*routerLearningEloRating,
	cfg config.EloLearningConfig,
	model string,
) *routerLearningEloRating {
	model = strings.TrimSpace(model)
	if ratings[model] == nil {
		ratings[model] = &routerLearningEloRating{
			Model:  model,
			Rating: eloInitialRating(cfg),
		}
	}
	return ratings[model]
}

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

func eloLearningPolicy(
	cfg config.EloLearningConfig,
	mode string,
	action routerLearningAction,
	reason string,
	stateKey string,
	scores []routerLearningEloScore,
	winner routerLearningEloScore,
) routerLearningPolicy {
	policy := newRouterLearningPolicy(routerLearningMethodElo)
	policy.Mode = mode
	policy.Scope = cfg.EffectiveScope()
	policy.Action = action
	policy.Reason = reason
	policy.Set("initial_rating", roundLearningFloat(eloInitialRating(cfg)))
	policy.Set("k_factor", roundLearningFloat(eloKFactor(cfg)))
	if stateKey != "" {
		policy.Set("state_key_hash", shortLearningIdentityHash(stateKey))
	}
	if winner.model != "" {
		policy.Set("selected_model", winner.model)
		policy.Set("selected_score", roundLearningFloat(winner.score))
		policy.Set("selected_rating", roundLearningFloat(winner.rating))
	}
	if len(scores) > 0 {
		policy.Set("ratings", eloScoreDiagnostics(scores))
	}
	return policy
}

func eloScoreDiagnostics(scores []routerLearningEloScore) []routerLearningEloScoreDiagnostic {
	result := make([]routerLearningEloScoreDiagnostic, 0, len(scores))
	for _, score := range scores {
		result = append(result, routerLearningEloScoreDiagnostic{
			Model:       score.model,
			Score:       roundLearningFloat(score.score),
			Rating:      roundLearningFloat(score.rating),
			Comparisons: score.comparisons,
			Wins:        score.wins,
			Losses:      score.losses,
			Ties:        score.ties,
		})
	}
	return result
}

func (rt *routerLearningRuntime) EloLeaderboard(category string) []selection.ModelRating {
	if !rt.EloLearningEnabled() {
		return nil
	}
	stateKey, ok := learningStateKeyFromParts(config.RouterLearningScopeDecision, category, "", "")
	if !ok {
		return nil
	}
	if rt != nil {
		rt.mu.Lock()
		defer rt.mu.Unlock()
	}
	if rt == nil || rt.elo.ratings == nil || rt.elo.ratings[stateKey] == nil {
		return nil
	}
	leaderboard := make([]selection.ModelRating, 0, len(rt.elo.ratings[stateKey]))
	for _, rating := range rt.elo.ratings[stateKey] {
		if rating == nil {
			continue
		}
		leaderboard = append(leaderboard, selection.ModelRating{
			Model:       rating.Model,
			Rating:      rating.Rating,
			Comparisons: rating.Comparisons,
			Wins:        rating.Wins,
			Losses:      rating.Losses,
			Ties:        rating.Ties,
		})
	}
	sort.SliceStable(leaderboard, func(i, j int) bool {
		if leaderboard[i].Rating == leaderboard[j].Rating {
			return leaderboard[i].Model < leaderboard[j].Model
		}
		return leaderboard[i].Rating > leaderboard[j].Rating
	})
	return leaderboard
}

func (rt *routerLearningRuntime) EloLearningEnabled() bool {
	if rt == nil || rt.config == nil || !rt.config.RouterLearning.Enabled {
		return false
	}
	return rt.config.RouterLearning.Adaptations.Elo.Enabled
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
