package services

// HasFactCheckClassifier returns true when the fact-check classifier has been initialized.
func (s *ClassificationService) HasFactCheckClassifier() bool {
	return s.classifier != nil &&
		s.classifier.GetFactCheckClassifier() != nil &&
		s.classifier.GetFactCheckClassifier().IsInitialized()
}

// HasHallucinationDetector returns true when the hallucination detector has been initialized.
func (s *ClassificationService) HasHallucinationDetector() bool {
	return s.classifier != nil && s.classifier.IsHallucinationDetectorReady()
}

// HasHallucinationExplainer returns true when the hallucination NLI explainer is initialized.
func (s *ClassificationService) HasHallucinationExplainer() bool {
	return s.classifier != nil && s.classifier.IsHallucinationExplainerReady()
}

// HasFeedbackDetector returns true when the feedback detector has been initialized.
func (s *ClassificationService) HasFeedbackDetector() bool {
	return s.classifier != nil &&
		s.classifier.GetFeedbackDetector() != nil &&
		s.classifier.GetFeedbackDetector().IsInitialized()
}
