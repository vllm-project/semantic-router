package services

// HasFactCheckClassifier returns true when the fact-check classifier has been initialized.
func (s *ClassificationService) HasFactCheckClassifier() bool {
	if s == nil {
		return false
	}
	s.runtimeMutex.RLock()
	defer s.runtimeMutex.RUnlock()
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
	s.runtimeMutex.RLock()
	defer s.runtimeMutex.RUnlock()
	classifier := s.classifierSnapshot()
	return classifier != nil && classifier.IsHallucinationDetectorReady()
}

// HasFeedbackDetector returns true when the feedback detector has been initialized.
func (s *ClassificationService) HasFeedbackDetector() bool {
	if s == nil {
		return false
	}
	s.runtimeMutex.RLock()
	defer s.runtimeMutex.RUnlock()
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
	s.runtimeMutex.RLock()
	if s.recipeClassifiers != nil {
		ready := s.recipeClassifiers.HasAnyFactCheckClassifier()
		s.runtimeMutex.RUnlock()
		return ready
	}
	s.runtimeMutex.RUnlock()
	return s.HasFactCheckClassifier()
}

// HasAnyHallucinationDetector reports aggregate reachable-recipe inventory
// readiness while HasHallucinationDetector remains scoped to the default API.
func (s *ClassificationService) HasAnyHallucinationDetector() bool {
	if s == nil {
		return false
	}
	s.runtimeMutex.RLock()
	if s.recipeClassifiers != nil {
		ready := s.recipeClassifiers.HasAnyHallucinationDetector()
		s.runtimeMutex.RUnlock()
		return ready
	}
	s.runtimeMutex.RUnlock()
	return s.HasHallucinationDetector()
}

// HasAnyFeedbackDetector reports aggregate reachable-recipe inventory readiness
// while HasFeedbackDetector remains scoped to the default API.
func (s *ClassificationService) HasAnyFeedbackDetector() bool {
	if s == nil {
		return false
	}
	s.runtimeMutex.RLock()
	if s.recipeClassifiers != nil {
		ready := s.recipeClassifiers.HasAnyFeedbackDetector()
		s.runtimeMutex.RUnlock()
		return ready
	}
	s.runtimeMutex.RUnlock()
	return s.HasFeedbackDetector()
}
