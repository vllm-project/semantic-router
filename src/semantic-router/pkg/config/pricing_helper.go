package config

// GetFullModelPricing returns the complete ModelPricing entry for the given model,
// including cached-input and cache-write rates. Returns (p, true) when at least one rate is non-zero
// or Currency is explicitly set (currency-only counts as configured so that free/
// self-hosted models produce cost=0 telemetry). Returns (zero, false) when the model
// has no pricing entry at all. Accepts both short names and provider model IDs.
func (c *RouterConfig) GetFullModelPricing(modelName string) (ModelPricing, bool) {
	if modelConfig, ok := c.resolveModelConfig(modelName); ok {
		if pricing, configured := normalizeConfiguredPricing(modelConfig.Pricing); configured {
			return pricing, true
		}
	}

	// A routed LoRA name is the model identity returned by the backend, but
	// transport and billing belong to its base provider model. An unpriced LoRA
	// therefore inherits the base model's rates; an explicitly priced alias above
	// remains an override.
	if _, baseModel, ok := c.resolveLoRABaseModel(modelName); ok {
		return normalizeConfiguredPricing(baseModel.Pricing)
	}
	return ModelPricing{}, false
}

func normalizeConfiguredPricing(pricing ModelPricing) (ModelPricing, bool) {
	if pricing.PromptPer1M == 0 && pricing.CompletionPer1M == 0 && pricing.CachedInputPer1M == 0 && pricing.CacheWritePer1M == nil && pricing.Currency == "" {
		return ModelPricing{}, false
	}
	if pricing.Currency == "" {
		pricing.Currency = "USD"
	}
	return pricing, true
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
