package services

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"

// PreparedBindings reads the published generation. Standalone replacement and
// close cannot retire its classifiers while their inventory is copied.
func (s *ClassificationService) PreparedBindings() ([]binding.PreparedBinding, bool) {
	if s == nil {
		return nil, false
	}
	s.runtimeMutex.RLock()
	defer s.runtimeMutex.RUnlock()
	if s.closed {
		return nil, true
	}
	s.configMutex.RLock()
	defer s.configMutex.RUnlock()
	return s.classifier.PreparedBindings()
}
