package extproc

import (
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

func (r *OpenAIRouter) applyBanditLearning(
	input routerLearningInput,
) (routerLearningAdaptationResult, bool) {
	cfg, ok := r.banditLearningConfig(input.ctx)
	if !ok {
		return routerLearningAdaptationResult{}, false
	}

	mode := banditAdaptationMode(input.ctx)
	scope := cfg.EffectiveScope()
	if mode == config.DecisionAdaptationModeBypass {
		result := banditNoChangeResult(input, cfg, mode, routerLearningActionBypass, banditReasonDecisionBypass, "")
		return attachRouterLearningExperience(result, input.experience), true
	}

	stateKey, stateKeyOK := banditStateKeyFromRequest(scope, input)
	if !stateKeyOK {
		result := banditNoChangeResult(input, cfg, mode, routerLearningActionNoop, banditReasonIdentityMissing, "")
		return attachRouterLearningExperience(result, input.experience), true
	}
	if input.selCtx == nil || len(input.selCtx.CandidateModels) == 0 {
		result := banditNoChangeResult(input, cfg, mode, routerLearningActionNoop, banditReasonNoCandidates, stateKey)
		return attachRouterLearningExperience(result, input.experience), true
	}

	runtime := r.routerLearningRuntimeState()
	scores := r.scoreBanditCandidates(runtime, input, cfg, stateKey)
	if len(scores) == 0 {
		result := banditNoChangeResult(input, cfg, mode, routerLearningActionNoop, banditReasonNoCandidates, stateKey)
		return attachRouterLearningExperience(result, input.experience), true
	}

	winner := scores[0]
	if input.baseResult == nil || winner.model == input.baseResult.SelectedModel {
		result := banditScoreResult(input, cfg, mode, stateKey, scores, winner, banditReasonBaseBest, false)
		return attachRouterLearningExperience(result, input.experience), true
	}
	if !winner.knownRewardState && !banditExplorationEnabled(cfg) && !banditHasNonQualityGoal(cfg.Goals) {
		result := banditScoreResult(input, cfg, mode, stateKey, scores, winner, banditReasonStateMissing, false)
		return attachRouterLearningExperience(result, input.experience), true
	}
	result := banditScoreResult(input, cfg, mode, stateKey, scores, winner, banditReasonScoreWin, true)
	return attachRouterLearningExperience(result, input.experience), true
}

func (r *OpenAIRouter) banditLearningConfig(ctx *RequestContext) (config.BanditLearningConfig, bool) {
	if r == nil || r.Config == nil || !r.Config.RouterLearning.Enabled {
		return config.BanditLearningConfig{}, false
	}
	cfg := r.Config.RouterLearning.Adaptations.Bandit
	if !cfg.Enabled {
		return config.BanditLearningConfig{}, false
	}
	cfg = banditConfigWithDecisionOverrides(cfg, ctx)
	return cfg, true
}

func banditConfigWithDecisionOverrides(base config.BanditLearningConfig, ctx *RequestContext) config.BanditLearningConfig {
	if ctx == nil || ctx.VSRSelectedDecision == nil || ctx.VSRSelectedDecision.Adaptations.Bandit == nil {
		return base
	}
	override := ctx.VSRSelectedDecision.Adaptations.Bandit
	if strings.TrimSpace(override.Scope) != "" {
		base.Scope = override.Scope
	}
	if len(override.Goals) > 0 {
		base.Goals = cloneLearningGoals(override.Goals)
	}
	if override.Tuning.ExplorationBudget != nil {
		base.Tuning.ExplorationBudget = override.Tuning.ExplorationBudget
	}
	return base
}

func banditAdaptationMode(ctx *RequestContext) string {
	if ctx != nil && ctx.VSRSelectedDecision != nil {
		return ctx.VSRSelectedDecision.Adaptations.BanditMode()
	}
	return config.DecisionAdaptationModeApply
}

func (r *OpenAIRouter) scoreBanditCandidates(
	runtime *routerLearningRuntime,
	input routerLearningInput,
	cfg config.BanditLearningConfig,
	stateKey string,
) []routerLearningBanditScore {
	if input.selCtx == nil {
		return nil
	}
	weights := normalizedLearningGoals(cfg.Goals)
	costScores := r.banditCostScores(input.selCtx.CandidateModels)
	totalImpressions := runtime.banditTotalImpressions(stateKey)
	scores := make([]routerLearningBanditScore, 0, len(input.selCtx.CandidateModels))
	for _, candidate := range input.selCtx.CandidateModels {
		model := strings.TrimSpace(candidate.Model)
		if model == "" {
			continue
		}
		quality := r.banditQualityPrior(input, model)
		cost := costScores[model]
		latency := 0.5
		arm := runtime.banditSnapshot(stateKey, model)
		meanReward := quality
		knownRewardState := false
		if arm.FeedbackCount > 0 {
			meanReward = arm.RewardSum / float64(arm.FeedbackCount)
			knownRewardState = true
		}
		exploration := banditExplorationBonus(cfg, totalImpressions, arm.Impressions)
		score := weights["quality"]*meanReward +
			weights["cost"]*cost +
			weights["latency"]*latency +
			exploration
		scores = append(scores, routerLearningBanditScore{
			model:            model,
			quality:          quality,
			cost:             cost,
			latency:          latency,
			meanReward:       meanReward,
			exploration:      exploration,
			score:            score,
			impressions:      arm.Impressions,
			feedbackCount:    arm.FeedbackCount,
			knownRewardState: knownRewardState,
		})
	}
	sort.SliceStable(scores, func(i, j int) bool {
		if scores[i].score == scores[j].score {
			return scores[i].model < scores[j].model
		}
		return scores[i].score > scores[j].score
	})
	return scores
}

func banditExplorationBonus(cfg config.BanditLearningConfig, totalImpressions int, armImpressions int) float64 {
	if !banditExplorationEnabled(cfg) {
		return 0
	}
	budget := clamp01(*cfg.Tuning.ExplorationBudget)
	return budget * math.Sqrt(math.Log(float64(totalImpressions+2))/float64(armImpressions+1))
}

func banditExplorationEnabled(cfg config.BanditLearningConfig) bool {
	return cfg.Tuning.ExplorationBudget != nil && *cfg.Tuning.ExplorationBudget > 0
}

func banditHasNonQualityGoal(goals map[string]float64) bool {
	for goal, weight := range goals {
		if strings.TrimSpace(goal) != "quality" && weight > 0 {
			return true
		}
	}
	return false
}

func (r *OpenAIRouter) banditQualityPrior(input routerLearningInput, model string) float64 {
	if input.baseResult != nil && input.baseResult.AllScores != nil {
		if score, ok := input.baseResult.AllScores[model]; ok {
			return clamp01(score)
		}
	}
	if r != nil && r.Config != nil && r.Config.ModelConfig != nil {
		if params, ok := r.Config.ModelConfig[model]; ok && params.QualityScore > 0 {
			return clamp01(params.QualityScore)
		}
	}
	if input.baseResult != nil && input.baseResult.SelectedModel == model {
		return clamp01(input.baseResult.Score)
	}
	return 0.5
}

func (r *OpenAIRouter) banditCostScores(candidates []config.ModelRef) map[string]float64 {
	costs := map[string]float64{}
	minCost := math.Inf(1)
	maxCost := math.Inf(-1)
	for _, candidate := range candidates {
		model := strings.TrimSpace(candidate.Model)
		if model == "" {
			continue
		}
		cost := 0.0
		if r != nil && r.Config != nil {
			prompt, completion, _, ok := r.Config.GetModelPricing(model)
			if ok {
				cost = prompt + completion
			}
		}
		costs[model] = cost
		if cost < minCost {
			minCost = cost
		}
		if cost > maxCost {
			maxCost = cost
		}
	}
	result := make(map[string]float64, len(costs))
	for model, cost := range costs {
		if !math.IsInf(minCost, 0) && maxCost > minCost {
			result[model] = 1 - ((cost - minCost) / (maxCost - minCost))
			continue
		}
		result[model] = 0.5
	}
	return result
}

func banditScoreResult(
	input routerLearningInput,
	cfg config.BanditLearningConfig,
	mode string,
	stateKey string,
	scores []routerLearningBanditScore,
	winner routerLearningBanditScore,
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
		selectionResult = banditSelectionResult(input.baseResult, winner, scores)
		if selectedModelRef != nil {
			selectionResult.LoRAName = selectedModelRef.LoRAName
		}
	}
	if mode == config.DecisionAdaptationModeObserve && changesModel {
		action = routerLearningActionSwitch
	}
	policy := banditLearningPolicy(cfg, mode, action, reason, stateKey, scores, winner)
	return routerLearningAdaptationResult{
		method:           routerLearningMethodBandit,
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

func banditNoChangeResult(
	input routerLearningInput,
	cfg config.BanditLearningConfig,
	mode string,
	action routerLearningAction,
	reason string,
	stateKey string,
) routerLearningAdaptationResult {
	if action == "" {
		action = routerLearningActionNoop
	}
	return routerLearningAdaptationResult{
		method: routerLearningMethodBandit,
		mode:   mode,
		scope:  cfg.EffectiveScope(),
		action: action,
		reason: reason,
		policy: banditLearningPolicy(
			cfg,
			mode,
			action,
			reason,
			stateKey,
			nil,
			routerLearningBanditScore{},
		),
	}
}

func banditSelectionResult(
	baseResult *selection.SelectionResult,
	winner routerLearningBanditScore,
	scores []routerLearningBanditScore,
) *selection.SelectionResult {
	result := &selection.SelectionResult{
		SelectedModel: winner.model,
		Score:         winner.score,
		Confidence:    clamp01(0.5 + winner.score/2),
		Method:        selection.MethodStatic,
		Tier:          selection.TierSupported,
		Reasoning:     fmt.Sprintf("Router Learning bandit selected %s", winner.model),
		AllScores:     banditAllScores(scores),
	}
	if baseResult != nil {
		result.LoRAName = baseResult.LoRAName
		result.Method = baseResult.Method
		result.Tier = baseResult.Tier
	}
	return result
}

func banditAllScores(scores []routerLearningBanditScore) map[string]float64 {
	result := make(map[string]float64, len(scores))
	for _, score := range scores {
		result[score.model] = score.score
	}
	return result
}

func selectedModelRefByModel(selCtx *selection.SelectionContext, model string) *config.ModelRef {
	if selCtx == nil {
		return nil
	}
	for i := range selCtx.CandidateModels {
		if selCtx.CandidateModels[i].Model == model {
			return &selCtx.CandidateModels[i]
		}
	}
	return nil
}

func normalizedLearningGoals(goals map[string]float64) map[string]float64 {
	if len(goals) == 0 {
		return map[string]float64{"quality": 1}
	}
	result := map[string]float64{}
	total := 0.0
	for key, value := range goals {
		key = strings.TrimSpace(key)
		if key == "" || value <= 0 {
			continue
		}
		result[key] = value
		total += value
	}
	if total <= 0 {
		return map[string]float64{"quality": 1}
	}
	for key, value := range result {
		result[key] = value / total
	}
	return result
}

func cloneLearningGoals(goals map[string]float64) map[string]float64 {
	if len(goals) == 0 {
		return nil
	}
	result := make(map[string]float64, len(goals))
	for key, value := range goals {
		result[key] = value
	}
	return result
}

func clamp01(value float64) float64 {
	if value < 0 {
		return 0
	}
	if value > 1 {
		return 1
	}
	return value
}

func roundLearningFloat(value float64) float64 {
	return math.Round(value*10000) / 10000
}
