package services

import (
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Global classification service instance
var (
	globalClassificationService *ClassificationService
	globalClassificationMu      sync.RWMutex
)

// ClassificationService provides classification functionality
type ClassificationService struct {
	classifier        *classification.Classifier
	unifiedClassifier *classification.UnifiedClassifier // New unified classifier
	config            *config.RouterConfig
	configMutex       sync.RWMutex // Protects config access
}

// NewClassificationService creates a new classification service
func NewClassificationService(classifier *classification.Classifier, config *config.RouterConfig) *ClassificationService {
	return &ClassificationService{
		classifier:        classifier,
		unifiedClassifier: nil, // Will be initialized separately
		config:            config,
	}
}

// NewUnifiedClassificationService creates a new service with unified classifier
func NewUnifiedClassificationService(unifiedClassifier *classification.UnifiedClassifier, legacyClassifier *classification.Classifier, config *config.RouterConfig) *ClassificationService {
	return &ClassificationService{
		classifier:        legacyClassifier,
		unifiedClassifier: unifiedClassifier,
		config:            config,
	}
}

// SetGlobalClassificationService publishes the compatibility global service used
// by legacy API-server startup paths that do not yet receive a runtime registry.
func SetGlobalClassificationService(service *ClassificationService) {
	globalClassificationMu.Lock()
	globalClassificationService = service
	globalClassificationMu.Unlock()
}

// GetGlobalClassificationService returns the global classification service instance
func GetGlobalClassificationService() *ClassificationService {
	globalClassificationMu.RLock()
	defer globalClassificationMu.RUnlock()
	return globalClassificationService
}

// HasClassifier returns true if the service has a real classifier (not placeholder)
func (s *ClassificationService) HasClassifier() bool {
	return s.classifier != nil
}

// NewPlaceholderClassificationService creates a placeholder service for API-only mode
func NewPlaceholderClassificationService() *ClassificationService {
	return &ClassificationService{
		classifier: nil, // No classifier - will return placeholder responses
		config:     nil,
	}
}

// GetClassifier returns the classifier instance (for signal-driven methods)
func (s *ClassificationService) GetClassifier() *classification.Classifier {
	return s.classifier
}

// GetConfig returns the current configuration
func (s *ClassificationService) GetConfig() *config.RouterConfig {
	s.configMutex.RLock()
	defer s.configMutex.RUnlock()
	return s.config
}

// UpdateConfig updates the configuration
func (s *ClassificationService) UpdateConfig(newConfig *config.RouterConfig) {
	s.configMutex.Lock()
	defer s.configMutex.Unlock()
	s.config = newConfig
}
