package extproc

import (
	"math"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

func (rt *routerLearningRuntime) updateBanditFeedback(feedback *selection.Feedback) bool {
	if rt == nil || feedback == nil || strings.TrimSpace(feedback.WinnerModel) == "" {
		return false
	}
	cfg, ok := rt.banditFeedbackConfig()
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
	reward := 1.0
	if feedback.Confidence > 0 {
		reward = math.Min(1, feedback.Confidence)
	}
	if feedback.Tie {
		reward = 0.5
	}
	rt.recordBanditReward(stateKey, feedback.WinnerModel, reward)
	if loser := strings.TrimSpace(feedback.LoserModel); loser != "" && !feedback.Tie {
		rt.recordBanditReward(stateKey, loser, 0)
	}
	return true
}

func (rt *routerLearningRuntime) banditFeedbackConfig() (config.BanditLearningConfig, bool) {
	if rt == nil || rt.config == nil || !rt.config.RouterLearning.Enabled {
		return config.BanditLearningConfig{}, false
	}
	cfg := rt.config.RouterLearning.Adaptations.Bandit
	return cfg, cfg.Enabled
}

func (rt *routerLearningRuntime) recordBanditImpression(stateKey string, model string) {
	if rt == nil || strings.TrimSpace(stateKey) == "" || strings.TrimSpace(model) == "" {
		return
	}
	rt.mu.Lock()
	defer rt.mu.Unlock()
	arm := rt.banditArm(stateKey, model)
	arm.Impressions++
	arm.LastUpdated = time.Now()
}

func (rt *routerLearningRuntime) recordBanditReward(stateKey string, model string, reward float64) {
	if rt == nil || strings.TrimSpace(stateKey) == "" || strings.TrimSpace(model) == "" {
		return
	}
	rt.mu.Lock()
	defer rt.mu.Unlock()
	arm := rt.banditArm(stateKey, model)
	arm.FeedbackCount++
	arm.RewardSum += clamp01(reward)
	arm.LastUpdated = time.Now()
}

func (rt *routerLearningRuntime) banditArm(stateKey string, model string) *routerLearningBanditArmState {
	if rt.bandit.arms == nil {
		rt.bandit.arms = map[string]map[string]*routerLearningBanditArmState{}
	}
	if rt.bandit.arms[stateKey] == nil {
		rt.bandit.arms[stateKey] = map[string]*routerLearningBanditArmState{}
	}
	if rt.bandit.arms[stateKey][model] == nil {
		rt.bandit.arms[stateKey][model] = &routerLearningBanditArmState{}
	}
	return rt.bandit.arms[stateKey][model]
}

func (rt *routerLearningRuntime) banditSnapshot(stateKey string, model string) routerLearningBanditArmState {
	if rt != nil {
		rt.mu.Lock()
		defer rt.mu.Unlock()
	}
	if rt == nil || rt.bandit.arms == nil || rt.bandit.arms[stateKey] == nil || rt.bandit.arms[stateKey][model] == nil {
		return routerLearningBanditArmState{}
	}
	return *rt.bandit.arms[stateKey][model]
}

func (rt *routerLearningRuntime) banditTotalImpressions(stateKey string) int {
	if rt != nil {
		rt.mu.Lock()
		defer rt.mu.Unlock()
	}
	if rt == nil || rt.bandit.arms == nil || rt.bandit.arms[stateKey] == nil {
		return 0
	}
	total := 0
	for _, arm := range rt.bandit.arms[stateKey] {
		if arm != nil {
			total += arm.Impressions
		}
	}
	return total
}

func (r *OpenAIRouter) observeBanditSelection(
	input routerLearningInput,
	composed routerLearningComposition,
) {
	cfg, ok := r.banditLearningConfig(input.ctx)
	if !ok || composed.selectedModelRef == nil {
		return
	}
	stateKey, ok := banditStateKeyFromRequest(cfg.EffectiveScope(), input)
	if !ok {
		return
	}
	runtime := r.routerLearningRuntimeState()
	runtime.recordBanditImpression(stateKey, composed.selectedModelRef.Model)
}

func banditStateKeyFromRequest(scope string, input routerLearningInput) (string, bool) {
	return learningStateKeyFromRequest(scope, input)
}
