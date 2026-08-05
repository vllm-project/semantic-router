package extproc

import (
	"math"
	"sort"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

// linUCBAdaptationStrategy implements the LinUCB contextual bandit
// (Li et al. 2010, "A Contextual-Bandit Approach to Personalized News
// Article Recommendation").
//
// At score time, for each candidate model m, given a feature vector x:
//
//	score(m) = theta_m · x + alpha · sqrt(x^T A_m^{-1} x)
//	         = posterior mean prediction + UCB exploration bonus
//
// State per arm m is a ridge-regularized matrix (A_m, b_m). After observing
// reward r for arm m on feature vector x:
//
//	A_m ← A_m + x x^T
//	b_m ← b_m + r · x
//
// Both updates run in O(d^2). Matrix state is owned by routerLearningRuntime
// (see router_learning_contextual_state.go) and is local to one router
// process — distributed-state semantics are out of scope for this strategy
// and tracked separately under #2243.
type linUCBAdaptationStrategy struct{}

func (linUCBAdaptationStrategy) Name() string {
	return config.RouterLearningStrategyLinUCB
}

func (linUCBAdaptationStrategy) Select(
	router *OpenAIRouter,
	input routerLearningInput,
	preflight routerLearningProtectionPreflight,
	cfg config.RouterLearningAdaptationConfig,
) routerLearningDecision {
	if router == nil {
		return routerLearningDecision{}
	}
	return router.applyLinUCBAdaptation(input, preflight, cfg)
}

func (r *OpenAIRouter) applyLinUCBAdaptation(
	input routerLearningInput,
	preflight routerLearningProtectionPreflight,
	cfg config.RouterLearningAdaptationConfig,
) routerLearningDecision {
	mode := adaptationMode(input.ctx)
	candidateSet := cfg.EffectiveCandidateSet()
	strategyName := cfg.EffectiveStrategy()
	if mode == config.DecisionAdaptationModeBypass {
		return baseAdaptationDecision(input, adaptationPolicy(mode, routerLearningActionBypass, "decision_bypass", nil))
	}

	learningCtx := r.adaptationSelectionContext(input.selCtx, input.ctx, candidateSet)
	if learningCtx == nil || len(learningCtx.CandidateModels) == 0 {
		return baseAdaptationDecision(input, adaptationPolicy(mode, routerLearningActionKeepBase, "candidate_set_empty", nil))
	}

	linucbCfg := config.RouterLearningLinUCBConfig{}
	if cfg.LinUCB != nil {
		linucbCfg = *cfg.LinUCB
	}
	dim := linucbCfg.EffectiveDim()
	alpha := linucbCfg.EffectiveAlpha()
	lambda := linucbCfg.EffectiveLambda()
	usedSampling := adaptationSamplingAllowed(mode, preflight)

	state := r.routerLearningRuntimeState().contextualState(strategyName, dim, lambda)
	x := extractContextFeatures(learningCtx, dim)
	r.routerLearningRuntimeState().recordPendingContextualUpdate(input.ctx.RouterReplayID, strategyName, x)
	scores := r.scoreLinUCBCandidates(learningCtx, input.ctx, input.baseResult, candidateSet, state, strategyName, x, alpha)
	if len(scores) == 0 {
		return baseAdaptationDecision(input, adaptationPolicy(mode, routerLearningActionKeepBase, "scores_missing", nil))
	}

	baseModel := selectedModelName(input.baseResult)
	winner := selectRoutingSamplingWinner(scores, baseModel, candidateSet)
	diag := newContextualDiagnostics(learningCtx, input.ctx, candidateSet, strategyName, baseModel, winner, usedSampling, scores, dim, alpha, lambda, 0)
	action, reason := contextualAction(mode, usedSampling, winner.model, baseModel)
	policy := adaptationPolicy(mode, action, reason, diag)
	if mode == config.DecisionAdaptationModeObserve || winner.model == baseModel {
		return baseAdaptationDecision(input, policy)
	}

	ref := modelRefForName(learningCtx.CandidateModels, winner.model)
	if ref == nil {
		return baseAdaptationDecision(input, adaptationPolicy(mode, routerLearningActionKeepBase, "selected_model_missing", diag))
	}
	result := proposalSelectionResult(input.baseResult, *ref, winner, scores)
	result.Reasoning = "router_learning adaptation: " + strategyName
	return routerLearningDecision{
		selectionContext: learningCtx,
		selectionResult:  result,
		selectedModelRef: ref,
		changesModel:     learningChangesModel(input.baseResult, result),
		policy:           policy,
	}
}

func (r *OpenAIRouter) scoreLinUCBCandidates(
	selCtx *selection.SelectionContext,
	ctx *RequestContext,
	baseResult *selection.SelectionResult,
	candidateSet string,
	state *routerLearningContextualState,
	strategyName string,
	x []float64,
	alpha float64,
) []routerLearningCandidateScore {
	if selCtx == nil || state == nil {
		return nil
	}
	maxCost := r.maxCandidateCost(selCtx.CandidateModels)
	tier := decisionTier(ctx)
	scores := make([]routerLearningCandidateScore, 0, len(selCtx.CandidateModels))
	for _, ref := range selCtx.CandidateModels {
		model := strings.TrimSpace(ref.Model)
		if model == "" {
			continue
		}
		mean := 0.0
		bonus := 0.0
		if len(x) == state.dimension() {
			arm := state.arm(contextualBanditKey(strategyName, selCtx.DecisionName, tier, model))
			if arm != nil {
				mean = arm.dotTheta(x)
				if alpha > 0 {
					bonus = alpha * math.Sqrt(arm.quadInv(x))
				}
			}
		}
		predicted := mean + bonus
		// Reuse the same penalty/bonus shape as routing_sampling so cost,
		// reliability, latency, and cache adjustments behave identically
		// across strategies. This keeps strategy comparison apples-to-apples.
		exp := r.routerLearningRuntimeState().experienceSnapshot(selCtx.DecisionName, tier, model)
		costPenalty := r.costPenalty(model, maxCost, candidateSet) +
			0.03*clamp01(exp.InputCostMultiplierEWMA)
		total := float64(exp.GoodFitCount + exp.UnderpoweredCount + exp.OverprovisionedCount + exp.FailedCount + 1)
		overusePenalty := 0.03 * float64(exp.OverprovisionedCount) / total
		reliabilityPenalty := 0.10 * float64(exp.FailedCount) / total
		latencyAdjustment := -0.02 * clamp01(exp.LatencyEWMA)
		cacheAdjustment := 0.02 * clamp01(exp.CacheHitEWMA)
		score := predicted - costPenalty - overusePenalty - reliabilityPenalty + latencyAdjustment + cacheAdjustment
		if baseResult != nil && baseResult.SelectedModel == model {
			score += 0.001
		}
		scores = append(scores, routerLearningCandidateScore{
			model:              model,
			score:              score,
			posteriorMean:      mean,
			predictedQuality:   predicted,
			costPenalty:        costPenalty,
			overusePenalty:     overusePenalty,
			reliabilityPenalty: reliabilityPenalty,
			latencyAdjustment:  latencyAdjustment,
			cacheAdjustment:    cacheAdjustment,
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

// contextualAction reports the typed action + reason for LinUCB / Linear
// Thompson decisions. The action set is identical to routing_sampling's
// (so replay diagnostics stay homogeneous), but the reasons document the
// contextual nature of the win.
func contextualAction(mode string, usedSampling bool, winnerModel, baseModel string) (routerLearningAction, string) {
	if mode == config.DecisionAdaptationModeObserve {
		return routerLearningActionObserve, "observe_only"
	}
	if winnerModel == baseModel {
		return routerLearningActionKeepBase, "base_best"
	}
	if usedSampling {
		return routerLearningActionProposeSwitch, "contextual_sampled_win"
	}
	return routerLearningActionProposeSwitch, "contextual_posterior_win"
}
