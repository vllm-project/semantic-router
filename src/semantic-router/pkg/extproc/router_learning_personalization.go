package extproc

import (
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

const (
	routerLearningMethodPersonalization routerLearningMethod = "personalization"

	personalizationReasonBaseBest        = "base_best"
	personalizationReasonDecisionBypass  = "decision_bypass"
	personalizationReasonIdentityMissing = "identity_missing"
	personalizationReasonNoCandidates    = "no_candidates"
	personalizationReasonPreferenceWin   = "preference_win"
	personalizationReasonStateMissing    = "state_missing"
)

type routerLearningPersonalizationState struct {
	preferences map[string]map[string]*routerLearningPersonalizationModelState
}

type routerLearningPersonalizationModelState struct {
	Model        string
	Positive     float64
	Negative     float64
	Interactions int
	LastUpdated  time.Time
}

type routerLearningPersonalizationScore struct {
	model        string
	baseScore    float64
	preference   float64
	score        float64
	positive     float64
	negative     float64
	interactions int
	known        bool
}

type routerLearningPersonalizationScoreDiagnostic struct {
	Model        string  `json:"model"`
	Score        float64 `json:"score"`
	BaseScore    float64 `json:"base_score"`
	Preference   float64 `json:"preference"`
	Positive     float64 `json:"positive"`
	Negative     float64 `json:"negative"`
	Interactions int     `json:"interactions"`
}

func (rt *routerLearningRuntime) updatePersonalizationFeedback(feedback *selection.Feedback) bool {
	cfg, ok := rt.personalizationFeedbackConfig()
	if !ok || feedback == nil || strings.TrimSpace(feedback.UserID) == "" {
		return false
	}
	stateKey, ok := personalizationStateKeyFromFeedback(cfg.EffectiveScope(), feedback)
	if !ok {
		return false
	}
	rt.updatePersonalizationPreferences(stateKey, feedback)
	return true
}

func (rt *routerLearningRuntime) personalizationFeedbackConfig() (config.PersonalizationLearningConfig, bool) {
	if rt == nil || rt.config == nil || !rt.config.RouterLearning.Enabled {
		return config.PersonalizationLearningConfig{}, false
	}
	cfg := rt.config.RouterLearning.Adaptations.Personalization
	return cfg, cfg.Enabled
}

func personalizationStateKeyFromFeedback(scope string, feedback *selection.Feedback) (string, bool) {
	baseKey, ok := learningStateKeyFromParts(
		scope,
		feedback.DecisionName,
		feedback.SessionID,
		feedback.ConversationID,
	)
	if !ok {
		return "", false
	}
	userID := strings.TrimSpace(feedback.UserID)
	if userID == "" {
		return "", false
	}
	return "user:" + userID + "/" + baseKey, true
}

func personalizationStateKeyFromRequest(scope string, input routerLearningInput) (string, bool) {
	baseKey, ok := learningStateKeyFromRequest(scope, input)
	if !ok {
		return "", false
	}
	userID := learningUserIDFromRequest(input)
	if userID == "" {
		return "", false
	}
	return "user:" + userID + "/" + baseKey, true
}

func (rt *routerLearningRuntime) updatePersonalizationPreferences(stateKey string, feedback *selection.Feedback) {
	if rt == nil || strings.TrimSpace(stateKey) == "" || feedback == nil {
		return
	}
	rt.mu.Lock()
	defer rt.mu.Unlock()

	if feedback.Tie {
		if feedback.WinnerModel != "" {
			pref := rt.personalizationPreference(stateKey, feedback.WinnerModel)
			pref.Positive += 0.5
			pref.Interactions++
			pref.LastUpdated = time.Now()
		}
		if feedback.LoserModel != "" {
			pref := rt.personalizationPreference(stateKey, feedback.LoserModel)
			pref.Positive += 0.5
			pref.Interactions++
			pref.LastUpdated = time.Now()
		}
		return
	}

	reward := 1.0
	if feedback.Confidence > 0 {
		reward = clamp01(feedback.Confidence)
	}
	if feedback.WinnerModel != "" {
		pref := rt.personalizationPreference(stateKey, feedback.WinnerModel)
		pref.Positive += reward
		pref.Interactions++
		pref.LastUpdated = time.Now()
	}
	if feedback.LoserModel != "" {
		pref := rt.personalizationPreference(stateKey, feedback.LoserModel)
		pref.Negative += reward
		pref.Interactions++
		pref.LastUpdated = time.Now()
	}
}

func (rt *routerLearningRuntime) personalizationPreference(
	stateKey string,
	model string,
) *routerLearningPersonalizationModelState {
	if rt.personalization.preferences == nil {
		rt.personalization.preferences = map[string]map[string]*routerLearningPersonalizationModelState{}
	}
	if rt.personalization.preferences[stateKey] == nil {
		rt.personalization.preferences[stateKey] = map[string]*routerLearningPersonalizationModelState{}
	}
	model = strings.TrimSpace(model)
	if rt.personalization.preferences[stateKey][model] == nil {
		rt.personalization.preferences[stateKey][model] = &routerLearningPersonalizationModelState{Model: model}
	}
	return rt.personalization.preferences[stateKey][model]
}

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

func personalizationLearningPolicy(
	cfg config.PersonalizationLearningConfig,
	mode string,
	action routerLearningAction,
	reason string,
	stateKey string,
	userID string,
	scores []routerLearningPersonalizationScore,
	winner routerLearningPersonalizationScore,
) routerLearningPolicy {
	policy := newRouterLearningPolicy(routerLearningMethodPersonalization)
	policy.Mode = mode
	policy.Scope = cfg.EffectiveScope()
	policy.Action = action
	policy.Reason = reason
	if stateKey != "" {
		policy.Set("state_key_hash", shortLearningIdentityHash(stateKey))
	}
	if userID != "" {
		policy.Set("user_hash", shortLearningIdentityHash(userID))
	}
	if winner.model != "" {
		policy.Set("selected_model", winner.model)
		policy.Set("selected_score", roundLearningFloat(winner.score))
		policy.Set("selected_preference", roundLearningFloat(winner.preference))
	}
	if len(scores) > 0 {
		policy.Set("preferences", personalizationScoreDiagnostics(scores))
	}
	return policy
}

func personalizationScoreDiagnostics(scores []routerLearningPersonalizationScore) []routerLearningPersonalizationScoreDiagnostic {
	result := make([]routerLearningPersonalizationScoreDiagnostic, 0, len(scores))
	for _, score := range scores {
		result = append(result, routerLearningPersonalizationScoreDiagnostic{
			Model:        score.model,
			Score:        roundLearningFloat(score.score),
			BaseScore:    roundLearningFloat(score.baseScore),
			Preference:   roundLearningFloat(score.preference),
			Positive:     roundLearningFloat(score.positive),
			Negative:     roundLearningFloat(score.negative),
			Interactions: score.interactions,
		})
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
