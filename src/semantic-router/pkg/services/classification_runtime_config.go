package services

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// RefreshRuntimeConfig updates the live service config and refreshes the legacy
// classifier so signal evaluation uses the new routing rules immediately.
func (s *ClassificationService) RefreshRuntimeConfig(newConfig *config.RouterConfig) {
	s.configMutex.Lock()
	defer s.configMutex.Unlock()

	s.config = newConfig
	if s.classifier != nil {
		rebuiltClassifier, err := createLegacyClassifier(newConfig)
		if err != nil {
			logging.Warnf("Failed to rebuild classifier during config update, falling back to in-place config swap: %v", err)
			s.classifier.Config = newConfig
		} else {
			s.classifier = rebuiltClassifier
		}
	}
}
