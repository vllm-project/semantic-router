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

// PricingProvenance says where the rates GetFullModelPricing returns for a
// model come from.
type PricingProvenance struct {
	// CatalogListPrice is true when the operator set no pricing on the model
	// and the rates are its provider's list price in the built-in catalog.
	CatalogListPrice bool
	// CallerLogin is true when the model's backends take the client's own
	// provider login (auth.credential: caller), so the provider bills the
	// client's plan, whatever the rates say.
	CallerLogin bool
}

// ModelPricingProvenance reports the provenance of a model's rates the way
// GetFullModelPricing resolves them: a LoRA without rates of its own takes
// its base model's. ok is false for a model without rates; CallerLogin is
// reported either way, since a model on the client's login often has none.
func (c *RouterConfig) ModelPricingProvenance(modelName string) (PricingProvenance, bool) {
	name, params, ok := c.resolveModelConfigEntry(modelName)
	_, priced := normalizeConfiguredPricing(params.Pricing)
	if !ok || !priced {
		if base, baseParams, isLoRA := c.resolveLoRABaseModel(modelName); isLoRA {
			name, params = base, baseParams
			_, priced = normalizeConfiguredPricing(params.Pricing)
		}
	}
	_, mode, takesRequestCredential := c.CredentialBackendOf(name)
	provenance := PricingProvenance{CallerLogin: takesRequestCredential && mode == CredentialCaller}
	if !priced {
		return provenance, false
	}
	// Materialization fills an empty authored pricing from the catalog.
	provenance.CatalogListPrice = params.AuthoredModel != nil && params.AuthoredModel.Pricing == (ModelPricing{})
	return provenance, true
}

func normalizeConfiguredPricing(pricing ModelPricing) (ModelPricing, bool) {
	if pricing.PromptPer1M == 0 && pricing.CompletionPer1M == 0 && pricing.CachedInputPer1M == 0 && pricing.CacheWritePer1M == nil && pricing.CacheWrite1hPer1M == nil && pricing.Currency == "" {
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
		if pricing.CacheWrite1hPer1M != nil && *pricing.CacheWrite1hPer1M > peakInputRate {
			peakInputRate = *pricing.CacheWrite1hPer1M
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
