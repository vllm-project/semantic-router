package config

// GetFullModelPricing returns the complete ModelPricing entry for the given model,
// including cached-input and cache-write rates. Returns (p, true) when at least one rate is non-zero
// or Currency is explicitly set (currency-only counts as configured so that free/
// self-hosted models produce cost=0 telemetry). Returns (zero, false) when the model
// has no pricing entry at all. Accepts both short names and provider model IDs.
func (c *RouterConfig) GetFullModelPricing(modelName string) (ModelPricing, bool) {
	if modelConfig, ok := c.resolveModelConfig(modelName); ok {
		p := modelConfig.Pricing
		if p.PromptPer1M != 0 || p.CompletionPer1M != 0 || p.CachedInputPer1M != 0 || p.CacheWritePer1M != nil || p.Currency != "" {
			if p.Currency == "" {
				p.Currency = "USD"
			}
			return p, true
		}
	}
	return ModelPricing{}, false
}

// GetMostExpensiveFullModelPricing returns the configured model with the
// highest combined peak-input and completion rate.
func (c *RouterConfig) GetMostExpensiveFullModelPricing() (string, ModelPricing, bool) {
	if c == nil || c.ModelConfig == nil {
		return "", ModelPricing{}, false
	}

	bestModel := ""
	bestPricing := ModelPricing{}
	bestScore := 0.0
	found := false
	for candidate := range c.ModelConfig {
		pricing, ok := c.GetFullModelPricing(candidate)
		if !ok {
			continue
		}
		peakInputRate := pricing.PromptPer1M
		if pricing.CacheWritePer1M != nil && *pricing.CacheWritePer1M > peakInputRate {
			peakInputRate = *pricing.CacheWritePer1M
		}
		score := peakInputRate + pricing.CompletionPer1M
		if !found || score > bestScore {
			bestModel = candidate
			bestPricing = pricing
			bestScore = score
			found = true
		}
	}
	return bestModel, bestPricing, found
}
