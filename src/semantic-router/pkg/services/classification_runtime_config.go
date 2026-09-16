package services

import (
	"fmt"
	"io"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/native"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// AcquireRuntimeSnapshot protects direct consumers of prepared classifier
// resources, such as embeddings, in standalone API mode. Release before calling
// a ClassificationService method: those methods acquire their own runtime lock.
// Router-backed consumers should prefer the existing router generation lease.
func (s *ClassificationService) AcquireRuntimeSnapshot() (*config.RouterConfig, *classification.Classifier, func()) {
	if s == nil {
		return nil, nil, func() {}
	}
	s.runtimeMutex.RLock()
	if s.closed {
		s.runtimeMutex.RUnlock()
		return nil, nil, func() {}
	}
	s.configMutex.RLock()
	cfg, classifier := s.config, s.classifier
	s.configMutex.RUnlock()
	var once sync.Once
	return cfg, classifier, func() { once.Do(s.runtimeMutex.RUnlock) }
}

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
	s.reloadMutex.Lock()
	defer s.reloadMutex.Unlock()
	if s.closed {
		return binding.ErrClosed
	}
	if s.modelPool == nil {
		s.modelPool = binding.NewPool()
	}
	options := classification.RecipeRuntimeOptions{Runtime: native.New(s.modelPool)}
	s.configMutex.RLock()
	currentRecipes := s.recipeClassifiers
	currentClassifier := s.classifier
	s.configMutex.RUnlock()
	if currentRecipes != nil {
		return s.refreshRecipeClassifiers(
			newConfig,
			currentClassifier,
			options,
		)
	}

	rebuiltClassifier, err := classification.NewLegacyClassifierFromConfig(newConfig, options)
	if err != nil {
		return fmt.Errorf("rebuild classifier: %w", err)
	}
	s.publishClassifiers(newConfig, rebuiltClassifier, nil, rebuiltClassifier)
	return nil
}

func (s *ClassificationService) refreshRecipeClassifiers(
	newConfig *config.RouterConfig,
	current *classification.Classifier,
	options classification.RecipeRuntimeOptions,
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
		options,
	)
	if err != nil {
		return fmt.Errorf("rebuild recipe classifiers: %w", err)
	}
	if err := rebuilt.InitializeRuntime(); err != nil {
		return fmt.Errorf("initialize recipe classifiers: %w", err)
	}
	defaultClassifier := rebuilt.Default()
	if defaultClassifier == nil {
		_ = rebuilt.Close()
		return fmt.Errorf("default routing recipe classifier is unavailable")
	}

	s.publishClassifiers(newConfig, defaultClassifier, rebuilt, rebuilt)
	return nil
}

// Preparation leaves old requests running. Only publication/retirement waits
// for standalone service calls; normal router replacement uses its generation
// lease and never mutates this service's classifier graph.
func (s *ClassificationService) publishClassifiers(cfg *config.RouterConfig, classifier *classification.Classifier, recipes *classification.RecipeClassifiers, owner io.Closer) {
	s.runtimeMutex.Lock()
	defer s.runtimeMutex.Unlock()
	s.configMutex.Lock()
	previousOwner, previousUnified := s.runtimeOwner, s.unifiedClassifier
	s.classifier, s.recipeClassifiers, s.config = classifier, recipes, cfg
	s.runtimeOwner = owner
	s.unifiedClassifier = classification.NewUnifiedClassifierFromRecipe(classifier)
	s.configMutex.Unlock()
	if err := previousUnified.Close(); err != nil {
		logging.Warnf("Close retired unified classifier: %v", err)
	}
	if previousOwner != nil {
		if err := previousOwner.Close(); err != nil {
			logging.Warnf("Close retired service classifiers: %v", err)
		}
	}
}
