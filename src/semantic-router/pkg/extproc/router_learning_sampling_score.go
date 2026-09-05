package extproc

import "sort"

// scoreRoutingSamplingExperience is the production equation shared with replay parity tests.
// A nil sampler selects the posterior mean (observe/protected path).
func scoreRoutingSamplingExperience(model string, exp routerLearningModelExperience,
	catalogCostPenalty float64, isBase bool,
	sample func(float64, float64) float64,
) routerLearningCandidateScore {
	alpha := exp.SeedWeight*exp.QualitySeed + float64(exp.GoodFitCount) + 1
	beta := exp.SeedWeight*(1-exp.QualitySeed) + float64(exp.UnderpoweredCount) + 1
	mean := alpha / (alpha + beta)
	predicted := mean
	if sample != nil {
		predicted = sample(alpha, beta)
	}
	costPenalty := catalogCostPenalty + 0.03*clamp01(exp.InputCostMultiplierEWMA)
	total := float64(exp.GoodFitCount + exp.UnderpoweredCount + exp.OverprovisionedCount + exp.FailedCount + 1)
	overusePenalty := 0.03 * float64(exp.OverprovisionedCount) / total
	reliabilityPenalty := 0.10 * float64(exp.FailedCount) / total
	latencyAdjustment := -0.02 * clamp01(exp.LatencyEWMA)
	cacheAdjustment := 0.02 * clamp01(exp.CacheHitEWMA)
	score := predicted - costPenalty - overusePenalty - reliabilityPenalty + latencyAdjustment + cacheAdjustment
	if isBase {
		score += 0.001
	}
	return routerLearningCandidateScore{
		model: model, score: score, posteriorMean: mean, predictedQuality: predicted,
		costPenalty: costPenalty, overusePenalty: overusePenalty,
		reliabilityPenalty: reliabilityPenalty, latencyAdjustment: latencyAdjustment,
		cacheAdjustment: cacheAdjustment, coldStart: exp.LastUpdated.IsZero(),
	}
}

func sortRoutingSamplingScores(scores []routerLearningCandidateScore) {
	sort.SliceStable(scores, func(i, j int) bool {
		if scores[i].score == scores[j].score {
			return scores[i].model < scores[j].model
		}
		return scores[i].score > scores[j].score
	})
}

func routingSamplingCostPenalty(cost, maxCost float64, candidateSet string) float64 {
	if maxCost <= 0 {
		return 0
	}
	multiplier := 0.04
	switch candidateSet {
	case "tier":
		multiplier = 0.06
	case "global":
		multiplier = 0.10
	}
	return multiplier * clamp01(cost/maxCost)
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
	coldStart          bool
}

func routingSamplingWinner(
	scores []routerLearningCandidateScore,
	baseModel string,
	candidateSet string,
	usedSampling bool,
) routerLearningCandidateScore {
	winner := selectRoutingSamplingWinner(scores, baseModel, candidateSet)
	if !usedSampling {
		return winner
	}
	if coldStartWinner, ok := firstColdStartCandidate(scores); ok {
		return coldStartWinner
	}
	return winner
}

func firstColdStartCandidate(scores []routerLearningCandidateScore) (routerLearningCandidateScore, bool) {
	for _, score := range scores {
		if score.coldStart {
			return score, true
		}
	}
	return routerLearningCandidateScore{}, false
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

func routingSamplingMargin(candidateSet string) float64 {
	switch candidateSet {
	case "tier":
		return 0.03
	case "global":
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
