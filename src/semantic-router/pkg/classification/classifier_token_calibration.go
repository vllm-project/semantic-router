package classification

func withCalibratedContextClassifier(contextClassifier *ContextClassifier, calibrator *CalibratedTokenCounter) option {
	return func(c *Classifier) {
		c.contextClassifier = contextClassifier
		c.tokenCalibrator = calibrator
	}
}

// ObserveTokenUsage feeds provider-reported prompt token usage back into the
// request-length estimator used by context routing.
func (c *Classifier) ObserveTokenUsage(category string, byteLen int, actualTokens int) {
	if c == nil || c.tokenCalibrator == nil {
		return
	}
	c.tokenCalibrator.Observe(category, byteLen, actualTokens)
}

// TokenCalibrationRatio returns a calibration snapshot for tests and runtime
// diagnostics.
func (c *Classifier) TokenCalibrationRatio(category string) (mean, conservative float64, samples int64, calibrated bool) {
	if c == nil || c.tokenCalibrator == nil {
		return defaultBytesPerToken, defaultBytesPerToken, 0, false
	}
	return c.tokenCalibrator.GetRatio(category)
}
