package extproc

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelpricing"
)

func normalizeResponseUsage(usage responseUsageMetrics) responseUsageMetrics {
	breakdown := modelpricing.Normalize(modelPricingUsage(usage))
	usage.promptTokens = breakdown.PromptTokens
	usage.cachedPromptTokens = breakdown.CachedInputTokens
	usage.cacheWriteTokens = breakdown.CacheWriteTokens
	usage.cacheWrite5mTokens = breakdown.CacheWrite5mTokens
	usage.cacheWrite1hTokens = breakdown.CacheWrite1hTokens
	usage.completionTokens = breakdown.CompletionTokens
	return usage
}

func modelPricingUsage(usage responseUsageMetrics) modelpricing.Usage {
	return modelpricing.Usage{
		PromptTokens:       usage.promptTokens,
		CachedInputTokens:  usage.cachedPromptTokens,
		CacheWriteTokens:   usage.cacheWriteTokens,
		CacheWrite5mTokens: usage.cacheWrite5mTokens,
		CacheWrite1hTokens: usage.cacheWrite1hTokens,
		CompletionTokens:   usage.completionTokens,
	}
}

func modelPricingRates(pricing config.ModelPricing) modelpricing.Rates {
	return modelpricing.Rates{
		Currency:          pricing.Currency,
		PromptPer1M:       pricing.PromptPer1M,
		CachedInputPer1M:  pricing.CachedInputPer1M,
		CacheWritePer1M:   pricing.CacheWritePer1M,
		CacheWrite1hPer1M: pricing.CacheWrite1hPer1M,
		CompletionPer1M:   pricing.CompletionPer1M,
	}
}

func costForResponseUsage(usage responseUsageMetrics, pricing config.ModelPricing) float64 {
	return modelpricing.Cost(modelPricingUsage(usage), modelPricingRates(pricing))
}

func effectiveCacheWriteRate(pricing config.ModelPricing) float64 {
	return modelPricingRates(pricing).EffectiveCacheWritePer1M()
}

func effectiveCacheWrite1hRate(pricing config.ModelPricing) float64 {
	return modelPricingRates(pricing).EffectiveCacheWrite1hPer1M()
}
