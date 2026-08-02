package services

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// RefreshRuntimeConfig updates the live service config and refreshes the legacy
// classifier so signal evaluation uses the new routing rules immediately.
func (s *ClassificationService) RefreshRuntimeConfig(newConfig *config.RouterConfig) {
	if err := s.TryRefreshRuntimeConfig(newConfig); err != nil {
		logging.Errorf(
			"Classifier config reload rejected; retaining the previous runtime snapshot: %v",
			err,
		)
	}
}

// TryRefreshRuntimeConfig builds the complete classifier before atomically
// publishing the classifier/config pair.
func (s *ClassificationService) TryRefreshRuntimeConfig(
	newConfig *config.RouterConfig,
) error {
	rebuiltClassifier, err := classification.NewLegacyClassifierFromConfig(newConfig)
	if err != nil {
		return fmt.Errorf("rebuild classifier: %w", err)
	}
	s.configMutex.Lock()
	s.classifier = rebuiltClassifier
	s.config = newConfig
	s.configMutex.Unlock()
	return nil
}
