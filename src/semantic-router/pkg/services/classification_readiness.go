package services

// HasFactCheckClassifier returns true when the fact-check classifier has been initialized.
func (s *ClassificationService) HasFactCheckClassifier() bool {
	if s == nil {
		return false
	}
	classifier := s.classifierSnapshot()
	return classifier != nil &&
		classifier.GetFactCheckClassifier() != nil &&
		classifier.GetFactCheckClassifier().IsInitialized()
}

// HasHallucinationDetector returns true when the hallucination detector has been initialized.
func (s *ClassificationService) HasHallucinationDetector() bool {
	if s == nil {
		return false
	}
	classifier := s.classifierSnapshot()
	return classifier != nil && classifier.IsHallucinationDetectorReady()
}

// HasHallucinationExplainer returns true when the hallucination NLI explainer is initialized.
func (s *ClassificationService) HasHallucinationExplainer() bool {
	if s == nil {
		return false
	}
	classifier := s.classifierSnapshot()
	return classifier != nil && classifier.IsHallucinationExplainerReady()
}

// HasFeedbackDetector returns true when the feedback detector has been initialized.
func (s *ClassificationService) HasFeedbackDetector() bool {
	if s == nil {
		return false
	}
	classifier := s.classifierSnapshot()
	return classifier != nil &&
		classifier.GetFeedbackDetector() != nil &&
		classifier.GetFeedbackDetector().IsInitialized()
}

// HasAnyFactCheckClassifier reports aggregate reachable-recipe inventory
// readiness while HasFactCheckClassifier remains scoped to the default API.
func (s *ClassificationService) HasAnyFactCheckClassifier() bool {
	if s == nil {
		return false
	}
	if s.recipeClassifiers != nil {
		return s.recipeClassifiers.HasAnyFactCheckClassifier()
	}
	return s.HasFactCheckClassifier()
}

// HasAnyHallucinationDetector reports aggregate reachable-recipe inventory
// readiness while HasHallucinationDetector remains scoped to the default API.
func (s *ClassificationService) HasAnyHallucinationDetector() bool {
	if s == nil {
		return false
	}
	if s.recipeClassifiers != nil {
		return s.recipeClassifiers.HasAnyHallucinationDetector()
	}
	return s.HasHallucinationDetector()
}

// HasAnyHallucinationExplainer reports aggregate reachable-recipe inventory
// readiness while HasHallucinationExplainer remains scoped to the default API.
func (s *ClassificationService) HasAnyHallucinationExplainer() bool {
	if s == nil {
		return false
	}
	if s.recipeClassifiers != nil {
		return s.recipeClassifiers.HasAnyHallucinationExplainer()
	}
	return s.HasHallucinationExplainer()
}

// HasAnyFeedbackDetector reports aggregate reachable-recipe inventory readiness
// while HasFeedbackDetector remains scoped to the default API.
func (s *ClassificationService) HasAnyFeedbackDetector() bool {
	if s == nil {
		return false
	}
	if s.recipeClassifiers != nil {
		return s.recipeClassifiers.HasAnyFeedbackDetector()
	}
	return s.HasFeedbackDetector()
}
