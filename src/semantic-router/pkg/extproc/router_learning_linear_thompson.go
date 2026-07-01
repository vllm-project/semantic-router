package extproc

import (
	"math/rand"
	"sort"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

// linearThompsonAdaptationStrategy implements Linear Thompson Sampling
// (Agrawal & Goyal 2013, "Thompson Sampling for Contextual Bandits with
// Linear Payoffs"). It is a Bayesian sibling of LinUCB: instead of adding
// the deterministic UCB bonus alpha · sqrt(x^T A^{-1} x), it samples a
// weight vector theta_tilde from the posterior:
//
//	theta_tilde ~ N(theta, sigma^2 · A^{-1})
//	score(m)    = theta_tilde · x
//
// Posterior updates are identical to LinUCB (rank-1 update on A, weighted
// accumulation on b), so the matrix backend is fully shared. The only
// per-request difference is the score formula and the diagnostic field
// emitted (sigma instead of alpha).
type linearThompsonAdaptationStrategy struct{}

func (linearThompsonAdaptationStrategy) Name() string {
	return config.RouterLearningStrategyLinearThompson
}

func (linearThompsonAdaptationStrategy) Select(
	router *OpenAIRouter,
	input routerLearningInput,
	preflight routerLearningProtectionPreflight,
	cfg config.RouterLearningAdaptationConfig,
) routerLearningDecision {
	if router == nil {
		return routerLearningDecision{}
	}
	return router.applyLinearThompsonAdaptation(input, preflight, cfg)
}

func (r *OpenAIRouter) applyLinearThompsonAdaptation(
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

	tsCfg := config.RouterLearningLinearThompsonConfig{}
	if cfg.LinearThompson != nil {
		tsCfg = *cfg.LinearThompson
	}
	dim := tsCfg.EffectiveDim()
	sigma := tsCfg.EffectiveSigma()
	lambda := tsCfg.EffectiveLambda()
	usedSampling := adaptationSamplingAllowed(mode, preflight)

	state := r.routerLearningRuntimeState().contextualState(strategyName, dim, lambda)
	x := extractContextFeatures(learningCtx, dim)
	r.routerLearningRuntimeState().recordPendingContextualUpdate(input.ctx.RouterReplayID, strategyName, x)

	seed := int64(0)
	effectiveSigma := sigma
	if usedSampling {
		seed = routerLearningSamplingSeedSource()
	} else {
		// Protection or observe-mode suppresses sampling — fall back to the
		// posterior mean by zeroing sigma.
		effectiveSigma = 0
	}
	rng := rand.New(rand.NewSource(seed))

	scores := r.scoreLinearThompsonCandidates(learningCtx, input.ctx, input.baseResult, candidateSet, state, strategyName, x, effectiveSigma, rng)
	if len(scores) == 0 {
		return baseAdaptationDecision(input, adaptationPolicy(mode, routerLearningActionKeepBase, "scores_missing", nil))
	}

	baseModel := selectedModelName(input.baseResult)
	winner := selectRoutingSamplingWinner(scores, baseModel, candidateSet)
	diag := newContextualDiagnostics(learningCtx, input.ctx, candidateSet, strategyName, baseModel, winner, usedSampling, scores, dim, 0, lambda, sigma)
	if diag != nil {
		diag.sampling.seed = seed
	}
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

func (r *OpenAIRouter) scoreLinearThompsonCandidates(
	selCtx *selection.SelectionContext,
	ctx *RequestContext,
	baseResult *selection.SelectionResult,
	candidateSet string,
	state *routerLearningContextualState,
	strategyName string,
	x []float64,
	sigma float64,
	rng *rand.Rand,
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
		predicted := 0.0
		if len(x) == state.dimension() {
			arm := state.arm(contextualBanditKey(strategyName, selCtx.DecisionName, tier, model))
			if arm != nil {
				mean = arm.dotTheta(x)
				thetaTilde := arm.sampleTheta(sigma, rng)
				if len(thetaTilde) == len(x) {
					var dot float64
					for i, v := range x {
						dot += v * thetaTilde[i]
					}
					predicted = dot
				} else {
					predicted = mean
				}
			}
		}
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
