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

// TryRefreshRuntimeConfig builds the complete classifier graph before
// atomically publishing the classifier/config pair.
func (s *ClassificationService) TryRefreshRuntimeConfig(
	newConfig *config.RouterConfig,
) error {
	s.configMutex.RLock()
	currentRecipes := s.recipeClassifiers
	currentClassifier := s.classifier
	s.configMutex.RUnlock()
	if currentRecipes != nil {
		return s.refreshRecipeClassifiers(
			newConfig,
			currentClassifier,
		)
	}

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

func (s *ClassificationService) refreshRecipeClassifiers(
	newConfig *config.RouterConfig,
	current *classification.Classifier,
) error {
	var (
		categoryMapping  *classification.CategoryMapping
		piiMapping       *classification.PIIMapping
		jailbreakMapping *classification.JailbreakMapping
	)
	if current != nil {
		categoryMapping = current.CategoryMapping
		piiMapping = current.PIIMapping
		jailbreakMapping = current.JailbreakMapping
	}
	rebuilt, err := classification.BuildRecipeClassifiers(
		newConfig,
		categoryMapping,
		piiMapping,
		jailbreakMapping,
	)
	if err != nil {
		return fmt.Errorf("rebuild recipe classifiers: %w", err)
	}
	if err := rebuilt.InitializeRuntime(); err != nil {
		return fmt.Errorf("initialize recipe classifiers: %w", err)
	}
	defaultClassifier := rebuilt.Default()
	if defaultClassifier == nil {
		return fmt.Errorf("default routing recipe classifier is unavailable")
	}

	s.configMutex.Lock()
	s.classifier = defaultClassifier
	s.recipeClassifiers = rebuilt
	s.config = newConfig
	s.configMutex.Unlock()
	return nil
}
