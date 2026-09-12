package extproc

import (
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelpricing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
)

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
		if usage.cacheWriteTokensReported {
			cacheWritePressure = float64(usage.cacheWriteTokens) / float64(promptTokens)
		} else {
			cacheWritePressure = float64(promptTokens-usage.cachedPromptTokens) / float64(promptTokens)
		}
	}
	inputCostMultiplier := r.learningInputCostMultiplier(ctx.RequestModel, usage)
	r.routerLearningRuntimeState().recordModelTelemetry(
		requestDecisionStateKey(ctx),
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
		requestDecisionStateKey(ctx),
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
	return clamp01(modelpricing.InputCostMultiplier(modelPricingUsage(usage), modelPricingRates(pricing)))
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
	rt.shared.mu.Lock()
	defer rt.shared.mu.Unlock()
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
	exp := rt.shared.experience[key]
	if exp == nil {
		exp = &routerLearningModelExperience{
			QualitySeed: 0.5,
			SeedWeight:  2,
		}
		rt.shared.experience[key] = exp
	}
	applyRouterLearningTelemetry(exp, observation)
	exp.LastUpdated = time.Now()
}
