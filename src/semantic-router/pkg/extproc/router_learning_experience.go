package extproc

import "time"

type routerLearningModelExperience struct {
	QualitySeed             float64
	SeedWeight              float64
	GoodFitCount            int
	UnderpoweredCount       int
	OverprovisionedCount    int
	FailedCount             int
	LatencyEWMA             float64
	CacheHitEWMA            float64
	CacheWriteEWMA          float64
	InputCostMultiplierEWMA float64
	LastUpdated             time.Time
}

const routerLearningTelemetryEWMAAlpha = 0.20

type routerLearningTelemetryObservation struct {
	LatencySeconds          float64
	LatencyObserved         bool
	CacheHitRatio           float64
	CacheWritePressure      float64
	CacheObserved           bool
	InputCostMultiplier     float64
	InputCostObserved       bool
	ProviderFailureObserved bool
}

func applyRouterLearningOutcome(exp *routerLearningModelExperience, verdict routerLearningOutcomeVerdict, score float64) {
	switch verdict {
	case routerLearningOutcomeGoodFit:
		exp.GoodFitCount += outcomeCount(score)
	case routerLearningOutcomeUnderpowered:
		exp.UnderpoweredCount += outcomeCount(score)
	case routerLearningOutcomeOverprovisioned:
		exp.OverprovisionedCount += outcomeCount(score)
	case routerLearningOutcomeFailed:
		exp.FailedCount += outcomeCount(score)
	}
}

func outcomeCount(score float64) int {
	if score <= 0 {
		return 1
	}
	if score < 1 {
		return 1
	}
	return int(score)
}

func applyRouterLearningTelemetry(exp *routerLearningModelExperience, observation routerLearningTelemetryObservation) {
	if observation.LatencyObserved {
		exp.LatencyEWMA = updateRouterLearningEWMA(exp.LatencyEWMA, observation.LatencySeconds)
	}
	if observation.CacheObserved {
		exp.CacheHitEWMA = updateRouterLearningEWMA(exp.CacheHitEWMA, observation.CacheHitRatio)
		exp.CacheWriteEWMA = updateRouterLearningEWMA(exp.CacheWriteEWMA, observation.CacheWritePressure)
	}
	if observation.InputCostObserved {
		exp.InputCostMultiplierEWMA = updateRouterLearningEWMA(
			exp.InputCostMultiplierEWMA,
			observation.InputCostMultiplier,
		)
	}
	if observation.ProviderFailureObserved {
		exp.FailedCount++
	}
}

func updateRouterLearningEWMA(previous float64, observed float64) float64 {
	if observed < 0 {
		return previous
	}
	if previous <= 0 {
		return observed
	}
	return previous*(1-routerLearningTelemetryEWMAAlpha) + observed*routerLearningTelemetryEWMAAlpha
}
