package services

// HasFactCheckClassifier returns true when the fact-check classifier has been initialized.
func (s *ClassificationService) HasFactCheckClassifier() bool {
	classifier := s.classifierSnapshot()
	return classifier != nil &&
		classifier.GetFactCheckClassifier() != nil &&
		classifier.GetFactCheckClassifier().IsInitialized()
}

// HasHallucinationDetector returns true when the hallucination detector has been initialized.
func (s *ClassificationService) HasHallucinationDetector() bool {
	classifier := s.classifierSnapshot()
	return classifier != nil && classifier.IsHallucinationDetectorReady()
}

// HasHallucinationExplainer returns true when the hallucination NLI explainer is initialized.
func (s *ClassificationService) HasHallucinationExplainer() bool {
	classifier := s.classifierSnapshot()
	return classifier != nil && classifier.IsHallucinationExplainerReady()
}

// HasFeedbackDetector returns true when the feedback detector has been initialized.
func (s *ClassificationService) HasFeedbackDetector() bool {
	classifier := s.classifierSnapshot()
	return classifier != nil &&
		classifier.GetFeedbackDetector() != nil &&
		classifier.GetFeedbackDetector().IsInitialized()
}
