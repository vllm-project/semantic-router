package extproc

import (
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
)

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

func (r *OpenAIRouter) observeRouterLearningUsageTelemetry(
	ctx *RequestContext,
	completionLatency time.Duration,
	usage responseUsageMetrics,
	_ routerreplay.UsageCost,
) {
	if !r.shouldObserveRouterLearningTelemetry(ctx) {
		return
	}
	promptTokens := usage.promptTokens
	cacheHitRatio := 0.0
	cacheWritePressure := 0.0
	cacheObserved := promptTokens > 0
	if promptTokens > 0 {
		cacheHitRatio = float64(usage.cachedPromptTokens) / float64(promptTokens)
		cacheWritePressure = float64(promptTokens-usage.cachedPromptTokens) / float64(promptTokens)
	}
	inputCostMultiplier := r.learningInputCostMultiplier(ctx.RequestModel, usage)
	r.routerLearningRuntimeState().recordModelTelemetry(
		ctx.VSRSelectedDecisionName,
		decisionTier(ctx),
		ctx.RequestModel,
		routerLearningTelemetryObservation{
			LatencySeconds:      completionLatency.Seconds(),
			LatencyObserved:     completionLatency > 0,
			CacheHitRatio:       clamp01(cacheHitRatio),
			CacheWritePressure:  clamp01(cacheWritePressure),
			CacheObserved:       cacheObserved,
			InputCostMultiplier: inputCostMultiplier,
			InputCostObserved:   inputCostMultiplier > 0,
		},
	)
}

func (r *OpenAIRouter) observeRouterLearningProviderStatus(ctx *RequestContext, statusCode int) {
	if statusCode != 429 && statusCode < 500 {
		return
	}
	if !r.shouldObserveRouterLearningTelemetry(ctx) {
		return
	}
	r.routerLearningRuntimeState().recordModelTelemetry(
		ctx.VSRSelectedDecisionName,
		decisionTier(ctx),
		ctx.RequestModel,
		routerLearningTelemetryObservation{ProviderFailureObserved: true},
	)
}

func (r *OpenAIRouter) shouldObserveRouterLearningTelemetry(ctx *RequestContext) bool {
	if r == nil || r.Config == nil || ctx == nil || ctx.RequestModel == "" {
		return false
	}
	if !r.Config.RouterLearning.Enabled || !r.Config.RouterLearning.Adaptation.EffectiveEnabled() {
		return false
	}
	if ctx.VSRSelectedDecision != nil &&
		ctx.VSRSelectedDecision.Adaptations.AdaptationMode() == config.DecisionAdaptationModeBypass {
		return false
	}
	return true
}

func (r *OpenAIRouter) learningInputCostMultiplier(model string, usage responseUsageMetrics) float64 {
	if r == nil || r.Config == nil || model == "" || usage.promptTokens <= 0 {
		return 0
	}
	pricing, ok := r.Config.GetFullModelPricing(model)
	if !ok || pricing.PromptPer1M <= 0 {
		return 0
	}
	cached := clampCachedPromptTokensInt(usage.promptTokens, usage.cachedPromptTokens)
	uncached := usage.promptTokens - cached
	actualPromptCost := float64(uncached)*pricing.PromptPer1M +
		float64(cached)*pricing.CachedInputPer1M
	fullPromptCost := float64(usage.promptTokens) * pricing.PromptPer1M
	if fullPromptCost <= 0 {
		return 0
	}
	return clamp01(actualPromptCost / fullPromptCost)
}

func (rt *routerLearningRuntime) recordModelTelemetry(
	decisionName string,
	decisionTier int,
	model string,
	observation routerLearningTelemetryObservation,
) {
	if rt == nil || model == "" {
		return
	}
	rt.mu.Lock()
	defer rt.mu.Unlock()
	rt.recordModelTelemetryLocked(decisionName, decisionTier, model, observation)
	if decisionName != "" {
		rt.recordModelTelemetryLocked("", decisionTier, model, observation)
	}
	if decisionTier != 0 {
		rt.recordModelTelemetryLocked("", 0, model, observation)
	}
}

func (rt *routerLearningRuntime) recordModelTelemetryLocked(
	decisionName string,
	decisionTier int,
	model string,
	observation routerLearningTelemetryObservation,
) {
	key := modelExperienceKey(decisionName, decisionTier, model)
	exp := rt.experience[key]
	if exp == nil {
		exp = &routerLearningModelExperience{
			QualitySeed: 0.5,
			SeedWeight:  2,
		}
		rt.experience[key] = exp
	}
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
	exp.LastUpdated = time.Now()
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
