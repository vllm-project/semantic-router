package extproc

import (
	"math"
	"math/rand"
	"sort"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

type routerLearningAdaptationDiagnostics struct {
	candidateSet  string
	strategy      string
	baseModel     string
	proposalModel string
	decision      string
	decisionTier  int
	sampling      routerLearningSamplingDiagnostics
	scores        []routerLearningCandidateScore
}

type routerLearningSamplingDiagnostics struct {
	used bool
	seed int64
}

type routerLearningCandidateScore struct {
	model              string
	score              float64
	posteriorMean      float64
	predictedQuality   float64
	costPenalty        float64
	overusePenalty     float64
	reliabilityPenalty float64
	latencyAdjustment  float64
	cacheAdjustment    float64
}

var routerLearningSamplingSeedSource = func() int64 {
	return time.Now().UnixNano()
}

func (r *OpenAIRouter) applyRoutingSamplingAdaptation(
	input routerLearningInput,
	preflight routerLearningProtectionPreflight,
	cfg config.RouterLearningAdaptationConfig,
) routerLearningDecision {
	mode := adaptationMode(input.ctx)
	candidateSet := cfg.EffectiveCandidateSet()
	strategy := cfg.EffectiveStrategy()
	if mode == config.DecisionAdaptationModeBypass {
		return baseAdaptationDecision(input, adaptationPolicy(mode, routerLearningActionBypass, "decision_bypass", nil))
	}

	learningCtx := r.adaptationSelectionContext(input.selCtx, input.ctx, candidateSet)
	if learningCtx == nil || len(learningCtx.CandidateModels) == 0 {
		return baseAdaptationDecision(input, adaptationPolicy(mode, routerLearningActionKeepBase, "candidate_set_empty", nil))
	}

	usedSampling := adaptationSamplingAllowed(mode, preflight)
	seed := int64(0)
	if usedSampling {
		seed = routerLearningSamplingSeedSource()
	}
	rng := rand.New(rand.NewSource(seed))
	scores := r.scoreRoutingSamplingCandidates(learningCtx, input.ctx, input.baseResult, candidateSet, usedSampling, rng)
	if len(scores) == 0 {
		return baseAdaptationDecision(input, adaptationPolicy(mode, routerLearningActionKeepBase, "scores_missing", nil))
	}

	baseModel := selectedModelName(input.baseResult)
	winner := selectRoutingSamplingWinner(scores, baseModel, candidateSet)
	diag := newRoutingSamplingDiagnostics(learningCtx, input.ctx, candidateSet, strategy, baseModel, winner, usedSampling, seed, scores)
	action, reason := routingSamplingAction(mode, usedSampling, winner.model, baseModel)
	policy := adaptationPolicy(mode, action, reason, diag)
	if mode == config.DecisionAdaptationModeObserve || winner.model == baseModel {
		return baseAdaptationDecision(input, policy)
	}

	ref := modelRefForName(learningCtx.CandidateModels, winner.model)
	if ref == nil {
		return baseAdaptationDecision(input, adaptationPolicy(mode, routerLearningActionKeepBase, "selected_model_missing", diag))
	}
	result := proposalSelectionResult(input.baseResult, *ref, winner, scores)
	return routerLearningDecision{
		selectionContext: learningCtx,
		selectionResult:  result,
		selectedModelRef: ref,
		changesModel:     learningChangesModel(input.baseResult, result),
		policy:           policy,
	}
}

func baseAdaptationDecision(input routerLearningInput, policy routerLearningPolicy) routerLearningDecision {
	return routerLearningDecision{
		selectionContext: input.selCtx,
		selectionResult:  input.baseResult,
		selectedModelRef: input.selectedModelRef,
		policy:           policy,
	}
}

func adaptationSamplingAllowed(mode string, preflight routerLearningProtectionPreflight) bool {
	if mode != config.DecisionAdaptationModeApply {
		return false
	}
	if preflight.enabled && protectionPreflightMode(preflight) == config.DecisionAdaptationModeApply {
		return preflight.samplingAllowed
	}
	return true
}

func protectionPreflightMode(preflight routerLearningProtectionPreflight) string {
	if preflight.mode == "" {
		return config.DecisionAdaptationModeApply
	}
	return preflight.mode
}

func selectRoutingSamplingWinner(
	scores []routerLearningCandidateScore,
	baseModel string,
	candidateSet string,
) routerLearningCandidateScore {
	winner := scores[0]
	baseScore := scoreForModel(scores, baseModel)
	requiredMargin := routingSamplingMargin(candidateSet) + candidateCostMargin(scores, baseModel, winner.model)
	if winner.model != baseModel && winner.score < baseScore+requiredMargin {
		return scoreByModel(scores, baseModel)
	}
	return winner
}

func newRoutingSamplingDiagnostics(
	learningCtx *selection.SelectionContext,
	ctx *RequestContext,
	candidateSet string,
	strategy string,
	baseModel string,
	winner routerLearningCandidateScore,
	usedSampling bool,
	seed int64,
	scores []routerLearningCandidateScore,
) *routerLearningAdaptationDiagnostics {
	return &routerLearningAdaptationDiagnostics{
		candidateSet:  candidateSet,
		strategy:      strategy,
		baseModel:     baseModel,
		proposalModel: winner.model,
		decision:      strings.TrimSpace(learningCtx.DecisionName),
		decisionTier:  decisionTier(ctx),
		sampling: routerLearningSamplingDiagnostics{
			used: usedSampling,
			seed: seed,
		},
		scores: scores,
	}
}

func routingSamplingAction(
	mode string,
	usedSampling bool,
	winnerModel string,
	baseModel string,
) (routerLearningAction, string) {
	if mode == config.DecisionAdaptationModeObserve {
		return routerLearningActionObserve, "observe_only"
	}
	if winnerModel == baseModel {
		return routerLearningActionKeepBase, "base_best"
	}
	if usedSampling {
		return routerLearningActionProposeSwitch, "sampled_win"
	}
	return routerLearningActionProposeSwitch, "posterior_win"
}

func (r *OpenAIRouter) adaptationConfig(
	selCtx *selection.SelectionContext,
	baseResult *selection.SelectionResult,
	selectedModelRef *config.ModelRef,
	ctx *RequestContext,
) (config.RouterLearningAdaptationConfig, bool) {
	if r == nil || r.Config == nil || selCtx == nil || baseResult == nil || selectedModelRef == nil || ctx == nil {
		return config.RouterLearningAdaptationConfig{}, false
	}
	cfg := r.Config.RouterLearning.Adaptation
	if ctx.VSRSelectedDecision != nil {
		cfg.CandidateSet = ctx.VSRSelectedDecision.Adaptations.AdaptationCandidateSet(cfg.EffectiveCandidateSet())
	}
	return cfg, r.Config.RouterLearning.Enabled && cfg.EffectiveEnabled()
}

func adaptationMode(ctx *RequestContext) string {
	if ctx != nil && ctx.VSRSelectedDecision != nil {
		return ctx.VSRSelectedDecision.Adaptations.AdaptationMode()
	}
	return config.DecisionAdaptationModeApply
}

func (r *OpenAIRouter) adaptationSelectionContext(
	selCtx *selection.SelectionContext,
	ctx *RequestContext,
	candidateSet string,
) *selection.SelectionContext {
	if selCtx == nil {
		return nil
	}
	candidates := r.learningCandidateModels(selCtx, ctx, candidateSet)
	if len(candidates) == 0 {
		return nil
	}
	clone := *selCtx
	clone.CandidateModels = candidates
	clone.CacheAffinityCtx = r.buildCacheAffinityContext(ctx, candidates)
	if clone.AgenticSession != nil {
		clone.AgenticSession.ModelContextWindows = r.modelContextWindows(candidates)
	}
	return &clone
}

func (r *OpenAIRouter) learningCandidateModels(
	selCtx *selection.SelectionContext,
	ctx *RequestContext,
	candidateSet string,
) []config.ModelRef {
	switch candidateSet {
	case config.RouterLearningCandidateSetTier:
		if tier := decisionTier(ctx); tier > 0 {
			return r.eligibleLearningModelRefs(r.unionDecisionModelRefs(func(decision config.Decision) bool {
				return decision.Tier == tier
			}))
		}
	case config.RouterLearningCandidateSetGlobal:
		return r.eligibleLearningModelRefs(r.deployedLearningModelRefs())
	}
	return r.eligibleLearningModelRefs(cloneModelRefs(selCtx.CandidateModels))
}

func (r *OpenAIRouter) deployedLearningModelRefs() []config.ModelRef {
	if r == nil || r.Config == nil {
		return nil
	}
	seen := map[string]struct{}{}
	refs := []config.ModelRef{}
	add := func(ref config.ModelRef) {
		ref.Model = strings.TrimSpace(ref.Model)
		if ref.Model == "" {
			return
		}
		key := ref.Model + "|" + ref.LoRAName
		if _, ok := seen[key]; ok {
			return
		}
		seen[key] = struct{}{}
		refs = append(refs, ref)
	}
	add(config.ModelRef{Model: r.Config.DefaultModel})
	for _, ref := range r.unionDecisionModelRefs(func(config.Decision) bool { return true }) {
		add(ref)
	}
	for _, endpoint := range r.Config.VLLMEndpoints {
		add(config.ModelRef{Model: endpoint.Model})
	}
	modelNames := make([]string, 0, len(r.Config.ModelConfig))
	for modelName := range r.Config.ModelConfig {
		modelNames = append(modelNames, modelName)
	}
	sort.Strings(modelNames)
	for _, modelName := range modelNames {
		add(config.ModelRef{Model: modelName})
	}
	return refs
}

func (r *OpenAIRouter) unionDecisionModelRefs(match func(config.Decision) bool) []config.ModelRef {
	if r == nil || r.Config == nil || match == nil {
		return nil
	}
	seen := map[string]struct{}{}
	var refs []config.ModelRef
	for _, decision := range r.Config.Decisions {
		if !match(decision) {
			continue
		}
		for _, ref := range decision.ModelRefs {
			key := ref.Model + "|" + ref.LoRAName
			if strings.TrimSpace(ref.Model) == "" {
				continue
			}
			if _, ok := seen[key]; ok {
				continue
			}
			seen[key] = struct{}{}
			refs = append(refs, ref)
		}
	}
	return refs
}

func cloneModelRefs(refs []config.ModelRef) []config.ModelRef {
	if len(refs) == 0 {
		return nil
	}
	clone := make([]config.ModelRef, len(refs))
	copy(clone, refs)
	return clone
}

func (r *OpenAIRouter) scoreRoutingSamplingCandidates(
	selCtx *selection.SelectionContext,
	ctx *RequestContext,
	baseResult *selection.SelectionResult,
	candidateSet string,
	useSampling bool,
	rng *rand.Rand,
) []routerLearningCandidateScore {
	if selCtx == nil {
		return nil
	}
	maxCost := r.maxCandidateCost(selCtx.CandidateModels)
	scores := make([]routerLearningCandidateScore, 0, len(selCtx.CandidateModels))
	for _, ref := range selCtx.CandidateModels {
		model := strings.TrimSpace(ref.Model)
		if model == "" {
			continue
		}
		exp := r.routerLearningRuntimeState().experienceSnapshot(selCtx.DecisionName, decisionTier(ctx), model)
		if params, ok := r.Config.ModelConfig[model]; ok && params.QualityScore > 0 && exp.GoodFitCount+exp.UnderpoweredCount == 0 {
			exp.QualitySeed = clamp01(params.QualityScore)
			exp.SeedWeight = 2
		}
		alpha := exp.SeedWeight*exp.QualitySeed + float64(exp.GoodFitCount) + 1
		beta := exp.SeedWeight*(1-exp.QualitySeed) + float64(exp.UnderpoweredCount) + 1
		mean := alpha / (alpha + beta)
		predicted := mean
		if useSampling && rng != nil {
			predicted = sampleBeta(alpha, beta, rng)
		}
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

func (r *OpenAIRouter) maxCandidateCost(refs []config.ModelRef) float64 {
	maxCost := 0.0
	for _, ref := range refs {
		if cost := r.modelInputCost(ref.Model); cost > maxCost {
			maxCost = cost
		}
	}
	return maxCost
}

func (r *OpenAIRouter) costPenalty(model string, maxCost float64, candidateSet string) float64 {
	if maxCost <= 0 {
		return 0
	}
	multiplier := 0.04
	switch candidateSet {
	case config.RouterLearningCandidateSetTier:
		multiplier = 0.06
	case config.RouterLearningCandidateSetGlobal:
		multiplier = 0.10
	}
	return multiplier * clamp01(r.modelInputCost(model)/maxCost)
}

func (r *OpenAIRouter) modelInputCost(model string) float64 {
	if r == nil || r.Config == nil || r.Config.ModelConfig == nil {
		return 0
	}
	params, ok := r.Config.ModelConfig[model]
	if !ok {
		return 0
	}
	return params.Pricing.PromptPer1M + params.Pricing.CompletionPer1M
}

func routingSamplingMargin(candidateSet string) float64 {
	switch candidateSet {
	case config.RouterLearningCandidateSetTier:
		return 0.03
	case config.RouterLearningCandidateSetGlobal:
		return 0.08
	default:
		return 0
	}
}

func candidateCostMargin(scores []routerLearningCandidateScore, baseModel string, winnerModel string) float64 {
	if baseModel == "" || winnerModel == "" || baseModel == winnerModel {
		return 0
	}
	base := scoreByModel(scores, baseModel)
	winner := scoreByModel(scores, winnerModel)
	extra := winner.costPenalty - base.costPenalty
	if extra <= 0 {
		return 0
	}
	return extra
}

func scoreForModel(scores []routerLearningCandidateScore, model string) float64 {
	return scoreByModel(scores, model).score
}

func scoreByModel(scores []routerLearningCandidateScore, model string) routerLearningCandidateScore {
	for _, score := range scores {
		if score.model == model {
			return score
		}
	}
	if len(scores) > 0 {
		return scores[0]
	}
	return routerLearningCandidateScore{}
}

func proposalSelectionResult(
	baseResult *selection.SelectionResult,
	ref config.ModelRef,
	winner routerLearningCandidateScore,
	scores []routerLearningCandidateScore,
) *selection.SelectionResult {
	result := &selection.SelectionResult{
		SelectedModel: ref.Model,
		LoRAName:      ref.LoRAName,
		Score:         winner.score,
		Confidence:    clamp01(winner.posteriorMean),
		Method:        selection.MethodStatic,
		Tier:          selection.TierSupported,
		Reasoning:     "router_learning adaptation: routing_sampling",
		AllScores:     map[string]float64{},
	}
	if baseResult != nil {
		result.Method = baseResult.Method
		result.Tier = baseResult.Tier
		result.AllScores = cloneSelectionScores(baseResult.AllScores)
		if result.AllScores == nil {
			result.AllScores = map[string]float64{}
		}
	}
	for _, score := range scores {
		result.AllScores[score.model] = score.score
	}
	return result
}

func adaptationPolicy(
	mode string,
	action routerLearningAction,
	reason string,
	diagnostics *routerLearningAdaptationDiagnostics,
) routerLearningPolicy {
	policy := newRouterLearningPolicy(routerLearningMethodAdaptation)
	policy.Mode = mode
	policy.Action = action
	policy.Reason = reason
	policy.Details.Adaptation = diagnostics
	return policy
}

func (d *routerLearningAdaptationDiagnostics) toPolicyMap() map[string]interface{} {
	if d == nil {
		return nil
	}
	out := map[string]interface{}{}
	setLearningPolicyString(out, learningPolicyFieldCandidateSet, d.candidateSet)
	setLearningPolicyString(out, learningPolicyFieldStrategy, d.strategy)
	setLearningPolicyString(out, learningPolicyFieldBaseModel, d.baseModel)
	setLearningPolicyString(out, learningPolicyFieldProposalModel, d.proposalModel)
	setLearningPolicyString(out, learningPolicyFieldDecision, d.decision)
	if d.decisionTier > 0 {
		setLearningPolicyInt(out, learningPolicyFieldDecisionTier, d.decisionTier)
	}
	sampling := map[string]interface{}{"used": d.sampling.used}
	if d.sampling.seed > 0 {
		sampling["seed"] = d.sampling.seed
	}
	setLearningPolicyValue(out, learningPolicyFieldSampling, sampling)
	scoreDiagnostics := make(map[string]interface{}, len(d.scores))
	for _, score := range d.scores {
		scoreDiagnostics[score.model] = map[string]interface{}{
			"score":               roundLearningFloat(score.score),
			"posterior_mean":      roundLearningFloat(score.posteriorMean),
			"predicted_quality":   roundLearningFloat(score.predictedQuality),
			"cost_penalty":        roundLearningFloat(score.costPenalty),
			"overuse_penalty":     roundLearningFloat(score.overusePenalty),
			"reliability_penalty": roundLearningFloat(score.reliabilityPenalty),
			"latency_adjustment":  roundLearningFloat(score.latencyAdjustment),
			"cache_adjustment":    roundLearningFloat(score.cacheAdjustment),
		}
	}
	if len(scoreDiagnostics) > 0 {
		setLearningPolicyValue(out, learningPolicyFieldScores, scoreDiagnostics)
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func (d *routerLearningAdaptationDiagnostics) stringField(field routerLearningPolicyField) string {
	if d == nil {
		return ""
	}
	switch field {
	case learningPolicyFieldCandidateSet:
		return strings.TrimSpace(d.candidateSet)
	case learningPolicyFieldStrategy:
		return strings.TrimSpace(d.strategy)
	case learningPolicyFieldBaseModel:
		return strings.TrimSpace(d.baseModel)
	case learningPolicyFieldProposalModel:
		return strings.TrimSpace(d.proposalModel)
	case learningPolicyFieldDecision:
		return strings.TrimSpace(d.decision)
	default:
		return ""
	}
}

func (d *routerLearningAdaptationDiagnostics) boolField(routerLearningPolicyField) bool {
	return false
}

func decisionTier(ctx *RequestContext) int {
	if ctx != nil && ctx.VSRSelectedDecision != nil {
		return ctx.VSRSelectedDecision.Tier
	}
	return 0
}

func modelRefForName(candidates []config.ModelRef, model string) *config.ModelRef {
	for i := range candidates {
		if candidates[i].Model == model || candidates[i].LoRAName == model {
			return &candidates[i]
		}
	}
	return nil
}

func clamp01(value float64) float64 {
	if math.IsNaN(value) || math.IsInf(value, 0) {
		return 0
	}
	if value < 0 {
		return 0
	}
	if value > 1 {
		return 1
	}
	return value
}

func sampleBeta(alpha float64, beta float64, rng *rand.Rand) float64 {
	if rng == nil || alpha <= 0 || beta <= 0 {
		return alpha / (alpha + beta)
	}
	x := sampleGamma(alpha, rng)
	y := sampleGamma(beta, rng)
	if x <= 0 || y <= 0 {
		return alpha / (alpha + beta)
	}
	return clamp01(x / (x + y))
}

func sampleGamma(shape float64, rng *rand.Rand) float64 {
	if shape < 1 {
		u := rng.Float64()
		return sampleGamma(shape+1, rng) * math.Pow(u, 1/shape)
	}
	d := shape - 1.0/3.0
	c := 1 / math.Sqrt(9*d)
	for {
		x := rng.NormFloat64()
		v := 1 + c*x
		if v <= 0 {
			continue
		}
		v = v * v * v
		u := rng.Float64()
		if u < 1-0.0331*x*x*x*x || math.Log(u) < 0.5*x*x+d*(1-v+math.Log(v)) {
			return d * v
		}
	}
}

func roundLearningFloat(value float64) float64 {
	if math.IsNaN(value) || math.IsInf(value, 0) {
		return 0
	}
	return math.Round(value*10000) / 10000
}
