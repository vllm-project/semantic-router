package config

// DeploymentPricing resolves explicit deployment rates before the provider
// model it names. An explicit empty block declares a free deployment, and so
// does a model_runtime deployment with neither.
func (c *RouterConfig) DeploymentPricing(name string) (ModelPricing, bool) {
	if c == nil {
		return ModelPricing{}, false
	}
	d, ok := c.ModelDeployments[name]
	if !ok {
		return ModelPricing{}, false
	}
	if d.Pricing != nil {
		pricing := *d.Pricing
		if pricing.Currency == "" {
			pricing.Currency = "USD"
		}
		return pricing, true
	}
	if d.Model != "" {
		if pricing, priced := c.GetFullModelPricing(d.Model); priced {
			return pricing, true
		}
	}
	if d.IsModelRuntime() {
		// The built-in runtime runs on the operator's own hardware and bills no
		// tokens, so an unpriced one is free, as an empty pricing block says.
		return ModelPricing{Currency: "USD"}, true
	}
	return ModelPricing{}, false
}

// cloneDeploymentPricing keeps exported declarations independent of runtime config.
func cloneDeploymentPricing(pricing *ModelPricing) *ModelPricing {
	if pricing == nil {
		return nil
	}
	cloned := *pricing
	if pricing.CacheWritePer1M != nil {
		value := *pricing.CacheWritePer1M
		cloned.CacheWritePer1M = &value
	}
	if pricing.CacheWrite1hPer1M != nil {
		value := *pricing.CacheWrite1hPer1M
		cloned.CacheWrite1hPer1M = &value
	}
	return &cloned
}
