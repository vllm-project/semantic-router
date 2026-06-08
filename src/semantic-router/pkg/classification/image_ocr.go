package classification

// getImageOCR is the package-level hook to call an OCR backend over an
// image payload. Tests may stub this variable to inject deterministic OCR
// outputs. The default implementation returns empty text and no error so the
// feature is opt-in and does not change behavior unless callers enable it.
var getImageOCR = func(imagePayload string) (string, error) {
	return "", nil
}

// ExtractImageOCR provides a cache-aware wrapper around getImageOCR.
// Placed on the EmbeddingClassifier to make it easy for signal evaluators to
// obtain OCR'd text using the same request-scoped cache pattern as image
// embeddings.
func (c *EmbeddingClassifier) ExtractImageOCR(payload string, cache *requestImageOCRCache) (string, error) {
	if payload == "" {
		return "", nil
	}
	return cache.resolve(payload, func() (string, error) {
		return getImageOCR(payload)
	})
}
