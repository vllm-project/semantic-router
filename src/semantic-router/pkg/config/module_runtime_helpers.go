package config

// GetCacheSimilarityThreshold returns the effective threshold for the semantic cache.
func (c *RouterConfig) GetCacheSimilarityThreshold() float32 {
	if c.SimilarityThreshold != nil {
		return *c.SimilarityThreshold
	}
	if threshold := c.MinSimilarityThreshold(); threshold > 0 {
		return threshold
	}
	return 0.5
}

// IsHallucinationMitigationEnabled checks if hallucination mitigation is enabled.
func (c *RouterConfig) IsHallucinationMitigationEnabled() bool {
	return c.HallucinationMitigation.Enabled
}

// IsFactCheckClassifierEnabled reports whether the configured fact-check model
// has a default API or request-reachable routing consumer.
func (c *RouterConfig) IsFactCheckClassifierEnabled() bool {
	return c != nil &&
		(c.NeedsFactCheckModelForAPI() || c.NeedsFactCheckModelForRouting())
}

// GetFactCheckRules returns all configured fact_check_rules.
func (c *RouterConfig) GetFactCheckRules() []FactCheckRule {
	return c.FactCheckRules
}

// IsHallucinationModelEnabled reports whether the configured detector has a
// default-runtime or request-reachable routing consumer.
func (c *RouterConfig) IsHallucinationModelEnabled() bool {
	return c != nil &&
		(c.NeedsHallucinationDetectorForDefaultRuntime() || c.NeedsHallucinationDetectorForRouting())
}

// GetFactCheckThreshold returns the configured or default fact-check threshold.
func (c *RouterConfig) GetFactCheckThreshold() float32 {
	if c.HallucinationMitigation.FactCheckModel.Threshold > 0 {
		return c.HallucinationMitigation.FactCheckModel.Threshold
	}
	return 0.7
}

// GetHallucinationModelThreshold returns the configured or default hallucination threshold.
func (c *RouterConfig) GetHallucinationModelThreshold() float32 {
	if c.HallucinationMitigation.HallucinationModel.Threshold > 0 {
		return c.HallucinationMitigation.HallucinationModel.Threshold
	}
	return 0.5
}

// GetHallucinationAction returns the supported hallucination action.
func (c *RouterConfig) GetHallucinationAction() string {
	if c.HallucinationMitigation.OnHallucinationDetected == "" {
		return "warn"
	}
	return "warn"
}

// IsFeedbackDetectorEnabled reports whether the configured feedback detector
// has a default API or request-reachable routing consumer.
func (c *RouterConfig) IsFeedbackDetectorEnabled() bool {
	return c != nil &&
		(c.NeedsFeedbackModelForAPI() || c.NeedsFeedbackModelForRouting())
}
