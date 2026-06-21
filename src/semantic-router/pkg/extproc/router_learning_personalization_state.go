package extproc

import (
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

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
