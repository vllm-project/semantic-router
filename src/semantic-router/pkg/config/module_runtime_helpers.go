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
