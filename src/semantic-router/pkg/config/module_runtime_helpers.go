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

// IsFactCheckClassifierEnabled reports whether a fact-check model is bound.
func (c *RouterConfig) IsFactCheckClassifierEnabled() bool {
	if c.HallucinationMitigation.FactCheckModel.ModelID == "" {
		return false
	}
	if c.HallucinationMitigation.Enabled {
		return true
	}
	return len(c.FactCheckRules) > 0
}

// GetFactCheckRules returns all configured fact_check_rules.
func (c *RouterConfig) GetFactCheckRules() []FactCheckRule {
	return c.FactCheckRules
}

// IsHallucinationModelEnabled reports whether hallucination detection should run.
func (c *RouterConfig) IsHallucinationModelEnabled() bool {
	if c.HallucinationMitigation.HallucinationModel.ModelID == "" {
		return false
	}
	if c.HallucinationMitigation.Enabled {
		return true
	}
	for _, decision := range c.Decisions {
		halConfig := decision.GetHallucinationConfig()
		if halConfig != nil && halConfig.Enabled {
			return true
		}
	}
	return false
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

// IsFeedbackDetectorEnabled reports whether feedback detection is configured.
func (c *RouterConfig) IsFeedbackDetectorEnabled() bool {
	return c.FeedbackDetector.Enabled &&
		c.FeedbackDetector.ModelID != "" &&
		len(c.UserFeedbackRules) > 0
}
