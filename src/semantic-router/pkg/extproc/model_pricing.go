package extproc

import (
	"time"

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

// modelRates resolves the configured prices of a candidate pool the way usage
// is priced; a model without prices is left out.
func (r *OpenAIRouter) modelRates(refs []config.ModelRef) map[string]modelpricing.Rates {
	if r == nil || r.Config == nil || len(refs) == 0 {
		return nil
	}
	rates := make(map[string]modelpricing.Rates, len(refs))
	for _, ref := range refs {
		if pricing, ok := r.Config.GetFullModelPricing(ref.Model); ok {
			rates[ref.Model] = modelPricingRates(pricing)
		}
	}
	return rates
}

func cacheReceiptAge(receipt *modelpricing.Receipt, now time.Time) time.Duration {
	if receipt == nil || now.Before(receipt.At) {
		return 0
	}
	return now.Sub(receipt.At)
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

func estimateDispatchCost(input int, output int64, pricing config.ModelPricing) float64 {
	inputRate := max(pricing.PromptPer1M, effectiveCacheWriteRate(pricing), effectiveCacheWrite1hRate(pricing))
	return (float64(input)*inputRate + float64(output)*max(pricing.CompletionPer1M, 0)) / 1e6
}
